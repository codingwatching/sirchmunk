#!/usr/bin/env python3
"""Create dynamic G_n/D_n artifacts for HotpotQA evaluation.

This CLI prepares publication protocol artifacts around auditable sample/corpus
stage bindings. System execution can then use those bindings through
run_evaluation.py or queue-based frozen runs.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
import time
from pathlib import Path

_SCRIPT_DIR = Path(__file__).parent.resolve()
_PROJECT_ROOT = _SCRIPT_DIR.parent
_SRC = _PROJECT_ROOT / "src"
for _p in (str(_SCRIPT_DIR), str(_SRC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from evaluation.golden_set import GoldenSetManager  # noqa: E402
from evaluation.sampling_protocol import (  # noqa: E402
    DEFAULT_HOTPOTQA_POPULATION_SIZE,
    DEFAULT_HOTPOTQA_STRATA,
    compute_sample_id_checksum,
    create_sampling_protocol,
)
from framework.answer_policy import policy_from_env  # noqa: E402
from framework.registry import load_benchmark_adapter, supported_benchmarks  # noqa: E402
from framework.dynamic_stage_runner import (  # noqa: E402
    StageExecutionRecord,
    StalenessEvaluationRecord,
    build_stage_bindings,
    save_stage_bindings,
    validate_result_reuse,
)
from hotpotqa.dynamic_corpus import (  # noqa: E402
    build_dynamic_corpus_snapshot,
    compute_frozen_order_checksum,
    derive_nested_sample_sets,
)
from hotpotqa.title_resolver import HotpotQATitleResolver  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build dynamic G_n/D_n artifacts")
    parser.add_argument("--benchmark", default="hotpotqa", choices=supported_benchmarks())
    parser.add_argument("--env", required=True, help="Benchmark env file")
    parser.add_argument("--output-dir", default="", help="Default: benchmarks/{benchmark}/output/dynamic_eval")
    parser.add_argument("--golden-n", type=int, default=500, help="Parent sampled set size; 500 is the recommended maximum for the sampled protocol")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stages", default="125,250,500", help="Comma separated nested stage sizes, must not exceed --golden-n")
    parser.add_argument("--strata", default=",".join(DEFAULT_HOTPOTQA_STRATA))
    parser.add_argument("--materialize", choices=["symlink", "copy", "manifest"], default="symlink")
    parser.add_argument("--background-ratio", type=float, default=3.0)
    parser.add_argument("--background-seed", type=int, default=42)
    parser.add_argument("--allow-missing-evidence", action="store_true", help="Allow snapshots with unresolved evidence titles; not for main-table runs")
    parser.add_argument("--rebuild-snapshots", action="store_true", help="Force rebuilding D_n snapshots even when a matching frozen snapshot manifest already exists")
    parser.add_argument("--force-recreate-golden", action="store_true")
    parser.add_argument("--run-baselines", action="store_true", help="Run built-in baselines for each G/D stage")
    parser.add_argument("--baselines", default="bm25_rag,hybrid_rag,react", help="Comma separated: bm25_rag,hybrid_rag,react,lens_full,lens_no_prior,lens_no_seq,lightrag_v136,lightrag_v136_<mode>")
    parser.add_argument("--baseline-max-files", type=int, default=50000, help="Max corpus files indexed by BM25-RAG/Hybrid-RAG per stage; must cover the largest D_n snapshot so evidence files are never truncated out of the index")
    parser.add_argument("--lightrag-query-mode", default="hybrid", choices=["naive", "local", "global", "hybrid", "mix"], help="Default LightRAG v1.3.6 query mode")
    parser.add_argument("--lightrag-max-files", type=int, default=0, help="LightRAG v1.3.6 max indexed files, 0=unlimited")
    parser.add_argument("--lightrag-max-file-chars", type=int, default=300000, help="LightRAG v1.3.6 max chars per indexed file")
    parser.add_argument("--skip-existing", action="store_true", help="Reuse existing per-stage baseline JSONL when present")
    parser.add_argument("--stale-index-arm", action="store_true", help="Also answer newly added questions with the previous stage index to measure staleness cost")
    parser.add_argument("--staleness-max-samples", type=int, default=0, help="Cap delta questions per transition in the stale-index arm, 0=all newly added questions")
    parser.add_argument("--allow-corpus-desync", action="store_true", help="Build stages even when referenced evidence articles are missing from the raw corpus")
    parser.add_argument("--rebuild-corpus-index", action="store_true", help="Ignore the cached raw-corpus title index and rescan the dump")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.benchmark != "hotpotqa":
        raise ValueError("Dynamic G_n/D_n artifacts are currently defined for HotpotQA only")

    env_file = str(Path(args.env).expanduser().resolve())
    adapter = load_benchmark_adapter(args.benchmark, env_file)
    output_dir = Path(args.output_dir or (_SCRIPT_DIR / args.benchmark / "output" / "dynamic_eval")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    stages = [int(part.strip()) for part in args.stages.split(",") if part.strip()]
    if not stages:
        raise ValueError("--stages must contain at least one stage size")
    if max(stages) > args.golden_n:
        raise ValueError("max(--stages) must be <= --golden-n")

    protocol = create_sampling_protocol(
        benchmark=args.benchmark,
        split=str(adapter.get_run_config().get("split") or "validation"),
        population_size=_population_size(adapter),
        method="stratified",
        seed=args.seed,
        target_n=args.golden_n,
        strata=args.strata,
        expected_population_size=DEFAULT_HOTPOTQA_POPULATION_SIZE,
    )
    manager = GoldenSetManager(str(_SCRIPT_DIR / args.benchmark))
    golden_set = manager.get_or_create(
        adapter=adapter,
        seed=args.seed,
        n=args.golden_n,
        force_recreate=args.force_recreate_golden,
        sampling_protocol=protocol,
    )

    # Gate before deriving stages or building snapshots: an unresolvable evidence
    # article must surface here as an explicit sync failure, so no stage artifact
    # is written for a parent set the raw corpus cannot support.
    corpus_sync = _check_corpus_sync(args, adapter, golden_set)

    sampling_dir = output_dir / "sampling"
    nested = derive_nested_sample_sets(
        golden_set,
        stages=stages,
        output_dir=sampling_dir,
        strata=[key.strip() for key in args.strata.split(",") if key.strip()],
    )

    wiki_dir = _wiki_dir(adapter)
    resolver = HotpotQATitleResolver(
        wiki_dir,
        index_cache_dir=_corpus_index_cache_dir(adapter),
    )
    parent_samples = golden_set.to_benchmark_samples()
    corpus_manifests = []
    for stage in nested.stages:
        stage_n = int(stage["stage_n"])
        sample_ids = stage["sample_ids_file"]
        ids = _load_sample_ids(sample_ids)
        d_stage = f"D_{stage_n}"
        snapshot_dir = output_dir / "corpus" / d_stage
        reused = None if args.rebuild_snapshots else _load_reusable_snapshot_manifest(
            snapshot_dir,
            stage_name=d_stage,
            sample_ids=ids,
            materialize_mode=args.materialize,
            background_ratio=args.background_ratio,
            background_seed=args.background_seed,
        )
        if reused is not None:
            print(f"[snapshot-gate] {d_stage}: frozen snapshot reused (corpus_checksum={reused.get('corpus_checksum','')})", flush=True)
            corpus_manifests.append(reused)
            continue
        manifest = build_dynamic_corpus_snapshot(
            parent_samples,
            sample_ids=ids,
            wiki_dir=wiki_dir,
            output_dir=snapshot_dir,
            stage_name=d_stage,
            materialize_mode=args.materialize,
            background_ratio=args.background_ratio,
            background_seed=args.background_seed,
            resolver=resolver,
            strict_evidence=not args.allow_missing_evidence,
        )
        print(f"[snapshot-gate] {d_stage}: snapshot built (corpus_checksum={manifest.corpus_checksum})", flush=True)
        corpus_manifests.append(manifest.to_dict())

    bindings = build_stage_bindings(
        nested_sample_manifest=nested.to_dict(),
        corpus_manifests=corpus_manifests,
        base_work_path=adapter.get_work_path(),
        base_output_dir=output_dir,
    )
    bindings_path = save_stage_bindings(bindings, output_dir / "stage_bindings.json")
    table_paths = _generate_snapshot_table(corpus_manifests, output_dir)
    baseline_runs = {}
    if args.run_baselines:
        baseline_runs = asyncio.run(_run_baselines_for_bindings(args, adapter, golden_set, bindings, output_dir))
        table_paths.update(baseline_runs.get("tables", {}))
    summary = {
        "benchmark": args.benchmark,
        "env_file": env_file,
        "golden_set_checksum": golden_set.checksum,
        "sample_id_checksum": golden_set.sample_id_checksum(),
        "sampling_protocol": golden_set.sampling_protocol,
        "golden_set_sampling_manifest": golden_set.sampling_manifest,
        "nested_sample_manifest": nested.to_dict(),
        "corpus_snapshots": corpus_manifests,
        "stage_bindings_path": bindings_path,
        "table_paths": table_paths,
        "stale_index_arm": bool(args.stale_index_arm),
        "staleness_max_samples": int(args.staleness_max_samples),
        "corpus_sync": corpus_sync,
        "baseline_runs": baseline_runs,
    }
    summary_path = output_dir / "dynamic_eval_manifest.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    printed = {"dynamic_eval_manifest": str(summary_path), "stage_bindings": bindings_path}
    if baseline_runs.get("staleness_summary"):
        printed["staleness_summary"] = baseline_runs["staleness_summary"]
    print(json.dumps(printed, indent=2, ensure_ascii=False))
    return 0


def _population_size(adapter) -> int:
    try:
        return int(adapter.describe_split().get("population_size", 0) or 0)
    except Exception:
        return 0


def _corpus_index_cache_dir(adapter) -> str:
    getter = getattr(adapter, "get_corpus_index_cache_dir", None)
    if callable(getter):
        try:
            return str(getter())
        except Exception:
            return ""
    return ""


def _check_corpus_sync(args, adapter, golden_set) -> dict:
    """Verify the frozen parent set is fully resolvable in the raw corpus.

    Returns the report as a dict for the run manifest. Missing evidence articles
    abort the run unless ``--allow-corpus-desync`` is set, in which case the
    stages are still built but the manifest records that they are not
    main-table eligible.
    """
    checker = getattr(adapter, "evaluate_corpus_sync", None)
    if not callable(checker):
        return {"corpus_sync": "unsupported"}
    samples = golden_set.to_benchmark_samples() if hasattr(golden_set, "to_benchmark_samples") else golden_set.samples
    report = checker(samples, force_rebuild=args.rebuild_corpus_index)
    print(f"[corpus-sync] {report.summary_line()}", file=sys.stderr)
    payload = report.to_dict()
    payload["allow_corpus_desync"] = bool(args.allow_corpus_desync)
    if not report.passed:
        if not args.allow_corpus_desync:
            raise SystemExit(
                "Raw corpus is out of sync with the frozen sample set: "
                f"missing_evidence_titles={report.missing_evidence_titles[:10]} "
                f"unresolvable_questions={len(report.unresolvable_sample_ids)}. "
                "Fix the wiki dump, or pass --allow-corpus-desync to proceed without main-table eligibility."
            )
        payload["accepted_for_main_table"] = False
    return payload


def _wiki_dir(adapter) -> str:
    try:
        manifest = adapter.get_dataset_manifest()
        if manifest.get("wiki_dir"):
            return str(manifest["wiki_dir"])
    except Exception:
        pass
    if hasattr(adapter, "_wiki_dir"):
        return str(adapter._wiki_dir())
    raise RuntimeError("Cannot resolve HotpotQA wiki corpus directory")


def _load_sample_ids(path: str | Path) -> list[str]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        return [str(item) for item in data.get("sample_ids", [])]
    if isinstance(data, list):
        return [str(item) for item in data]
    raise ValueError(f"Invalid sample ids file: {path}")


async def _run_baselines_for_bindings(args, base_adapter, golden_set, bindings, output_dir: Path) -> dict:
    from evaluation.suite import BaselineEvaluationSuite
    from evaluation.dynamic_table_generator import DynamicPaperTableGenerator

    baseline_specs = [spec.strip() for spec in args.baselines.split(",") if spec.strip()]
    runs = {}
    dynamic_rows = []
    update_rows = []
    staleness_rows = []
    previous_binding = None
    previous_baselines = {}
    previous_sample_ids: list[str] = []
    for binding in bindings:
        stage_adapter = _StageAdapter(base_adapter, binding)
        stage_sample_ids = _load_sample_ids(binding.sample_ids_file)
        stage_samples = _select_sample_dicts(golden_set.samples, stage_sample_ids)
        stage_golden = _StageGoldenSet(stage_samples, binding)
        baselines = [_baseline_by_name(spec, stage_adapter, args) for spec in baseline_specs]
        if previous_binding is not None and previous_baselines:
            for baseline in previous_baselines.values():
                update_rows.append(await _measure_transition_cost(
                    from_binding=previous_binding,
                    to_binding=binding,
                    baseline=baseline,
                    to_stage_adapter=stage_adapter,
                ))
        baseline_dir = Path(binding.output_dir) / "baselines"
        records_dir = Path(binding.output_dir) / "stage_records"
        records_dir.mkdir(parents=True, exist_ok=True)
        _prepare_reuse_gate(args, binding, stage_adapter, baselines, baseline_dir, records_dir)
        suite = BaselineEvaluationSuite(
            bm_adapter=stage_adapter,
            baselines=baselines,
            output_dir=str(baseline_dir),
        )
        with _stage_environment(binding):
            result_map = await suite.run(
                stage_golden,
                skip_existing=args.skip_existing,
                skip_cleanup=args.stale_index_arm,
            )
        stage_records = {}
        for baseline in baselines:
            results = result_map.get(baseline.name, [])
            record = _stage_execution_record(binding, stage_adapter, baseline, baseline_dir / f"baseline_{baseline.name}.jsonl")
            record_payload = {**record.to_dict(), "reuse_fingerprint": record.reuse_fingerprint(), "result_count": len(results)}
            record_path = records_dir / f"{baseline.name}_stage_execution_record.json"
            record_path.write_text(json.dumps(record_payload, indent=2, ensure_ascii=False), encoding="utf-8")
            stage_records[baseline.name] = str(record_path)
            dynamic_rows.append(_dynamic_result_row(binding, baseline, results))
        stage_entry = {
            "output_dir": binding.output_dir,
            "corpus_checksum": binding.corpus_checksum,
            "sample_id_checksum": binding.sample_id_checksum,
            "stage_records": stage_records,
            "systems": {name: len(results) for name, results in result_map.items()},
        }
        if args.stale_index_arm and previous_binding is not None and previous_baselines:
            stale_rows, stale_records = await _run_staleness_arm(
                args=args,
                from_binding=previous_binding,
                to_binding=binding,
                previous_baselines=previous_baselines,
                previous_sample_ids=previous_sample_ids,
                current_sample_ids=stage_sample_ids,
                golden_set=golden_set,
                stage_adapter=stage_adapter,
                fresh_results=result_map,
            )
            staleness_rows.extend(stale_rows)
            if stale_records:
                stage_entry["staleness_records"] = stale_records
        if args.stale_index_arm and previous_baselines:
            # Previous-stage instances were kept alive only to hold a stale index
            # for the transition just measured; release them before advancing.
            for baseline in previous_baselines.values():
                await baseline.cleanup()
        runs[binding.stage_name] = stage_entry
        previous_binding = binding
        previous_baselines = {baseline.name: baseline for baseline in baselines}
        previous_sample_ids = stage_sample_ids
    if args.stale_index_arm and previous_baselines:
        for baseline in previous_baselines.values():
            await baseline.cleanup()
    tables_dir = output_dir / "tables"
    generator = DynamicPaperTableGenerator()
    runs["tables"] = {}
    runs["tables"].update({f"dynamic_{k}": v for k, v in generator.generate_dynamic_main_table(dynamic_rows, tables_dir).items()})
    runs["tables"].update({f"lifecycle_{k}": v for k, v in generator.generate_lifecycle_main_table(dynamic_rows, tables_dir).items()})
    runs["tables"].update({f"budget_{k}": v for k, v in generator.generate_budget_quality_table(dynamic_rows, tables_dir).items()})
    runs["tables"].update({f"update_{k}": v for k, v in generator.generate_update_readiness_table(update_rows, tables_dir).items()})
    if staleness_rows:
        from evaluation.staleness import summarize_staleness_rows

        runs["tables"].update({f"staleness_{k}": v for k, v in generator.generate_staleness_table(staleness_rows, tables_dir).items()})
        runs["staleness_rows"] = staleness_rows
        runs["staleness_summary"] = summarize_staleness_rows(staleness_rows)
    return runs


class _StageGoldenSet:
    def __init__(
        self,
        samples: list[dict],
        binding,
        *,
        stage_name: str = "",
        sample_id_checksum: str = "",
        frozen_order_checksum: str = "",
    ) -> None:
        self.samples = samples
        self.n_questions = len(samples)
        self.seed = 0
        self.sampling_protocol = {"method": "fixed_ids", "stage_name": stage_name or binding.stage_name}
        self.sampling_manifest = {
            "sample_ids": [sample["sample_id"] for sample in samples],
            "sample_id_checksum": sample_id_checksum or binding.sample_id_checksum,
            "frozen_order_checksum": frozen_order_checksum or binding.frozen_order_checksum,
        }

    def sample_ids(self) -> list[str]:
        return [str(sample["sample_id"]) for sample in self.samples]

    def sample_id_checksum(self) -> str:
        return str(self.sampling_manifest.get("sample_id_checksum", ""))


class _StageAdapter:
    def __init__(self, base_adapter, binding) -> None:
        self._base = base_adapter
        self._binding = binding

    def __getattr__(self, name):
        return getattr(self._base, name)

    @property
    def name(self) -> str:
        return getattr(self._base, "name", "hotpotqa")

    @property
    def env_file(self) -> str:
        return getattr(self._base, "env_file", "")

    def get_search_paths(self, sample) -> list[str]:
        return [self._binding.search_corpus_dir]

    def get_work_path(self) -> str:
        return self._binding.work_path

    def get_output_dir(self) -> str:
        return self._binding.output_dir

    def get_run_config(self) -> dict:
        config = dict(self._base.get_run_config())
        config.update({
            "dynamic_stage_name": self._binding.stage_name,
            "sample_id_checksum": self._binding.sample_id_checksum,
            "frozen_order_checksum": self._binding.frozen_order_checksum,
            "corpus_checksum": self._binding.corpus_checksum,
            "HOTPOT_WIKI_CORPUS_DIR": self._binding.search_corpus_dir,
            "SIRCHMUNK_WORK_PATH": self._binding.work_path,
        })
        return config

    def get_dataset_manifest(self) -> dict:
        manifest = dict(self._base.get_dataset_manifest())
        manifest.update({
            "dynamic_stage_name": self._binding.stage_name,
            "wiki_dir": self._binding.search_corpus_dir,
            "corpus_snapshot_dir": self._binding.corpus_snapshot_dir,
            "search_corpus_dir": self._binding.search_corpus_dir,
            "corpus_checksum": self._binding.corpus_checksum,
            "sample_id_checksum": self._binding.sample_id_checksum,
            "frozen_order_checksum": self._binding.frozen_order_checksum,
        })
        return manifest

    def build_searcher(self):
        from sirchmunk.llm.openai_chat import OpenAIChat
        from sirchmunk.search import AgenticSearch

        getter = getattr(self._base, "_get", None)
        def _get(key: str, default: str = "") -> str:
            if callable(getter):
                return getter(key, default)
            return os.environ.get(key, default)

        llm = OpenAIChat(
            api_key=_get("LLM_API_KEY", ""),
            base_url=_get("LLM_BASE_URL", "https://api.openai.com/v1"),
            model=_get("LLM_MODEL_NAME", "gpt-4o-mini"),
        )
        return AgenticSearch(
            llm=llm,
            work_path=self.get_work_path(),
            reuse_knowledge=str(self.get_run_config().get("reuse_knowledge", False)).lower() in {"1", "true", "yes"},
            verbose=False,
            # Dynamic arms are scored like the main experiment, so they share
            # the evaluation's no-abstention reporting decision.
            answer_policy=policy_from_env(),
        )


def _baseline_by_name(spec: str, bm_adapter, args=None):
    lower = spec.strip().lower()
    max_files = int(getattr(args, "baseline_max_files", 50000) or 50000) if args is not None else 50000
    if lower == "bm25_rag":
        from baselines import BM25RAGBaseline
        return BM25RAGBaseline(max_files=max_files)
    if lower == "hybrid_rag":
        from baselines import HybridRAGBaseline
        return HybridRAGBaseline(max_files=max_files)
    if lower == "react":
        from baselines import ReActSearchBaseline
        return ReActSearchBaseline()
    if lower in {"closed_book", "closedbook", "no_retrieval"}:
        # Reference arm, not a competitor: it reads nothing, so its score is the
        # part of the benchmark answerable from the model's parameters alone.
        from baselines import ClosedBookBaseline
        return ClosedBookBaseline()
    if lower in {"lens_full", "lens_no_prior", "lens_no_seq"}:
        from ablations import build_single_lens_ablation
        return build_single_lens_ablation(bm_adapter, profile_name=lower)
    lightrag_modes = ("naive", "local", "global", "hybrid", "mix")
    if lower == "lightrag_v136" or lower in {f"lightrag_v136_{mode}" for mode in lightrag_modes}:
        from baselines import LightRAGV136Baseline
        mode = getattr(args, "lightrag_query_mode", "hybrid") if args is not None else "hybrid"
        for candidate in lightrag_modes:
            if lower == f"lightrag_v136_{candidate}":
                mode = candidate
                break
        return LightRAGV136Baseline(
            query_mode=mode,
            max_files=getattr(args, "lightrag_max_files", 0) if args is not None else 0,
            max_file_chars=getattr(args, "lightrag_max_file_chars", 300000) if args is not None else 300000,
        )
    raise ValueError(f"Unsupported dynamic baseline: {spec}")


def _select_sample_dicts(samples: list[dict], sample_ids: list[str]) -> list[dict]:
    by_id = {str(sample.get("sample_id")): sample for sample in samples}
    missing = [sample_id for sample_id in sample_ids if sample_id not in by_id]
    if missing:
        raise ValueError(f"Missing stage sample ids: {missing[:10]} total_missing={len(missing)}")
    return [by_id[sample_id] for sample_id in sample_ids]


def _load_reusable_snapshot_manifest(
    snapshot_dir: Path,
    *,
    stage_name: str,
    sample_ids: list,
    materialize_mode: str,
    background_ratio: float,
    background_seed: int,
) -> dict | None:
    """Return the persisted snapshot manifest when it matches this request.

    Frozen ``D_n`` snapshots must be built once and then reused: rebuilding on
    every invocation both wastes time and breaks result-reuse fingerprints
    whenever any non-determinism slips into materialization. A snapshot is
    reusable only when its manifest binds the same stage name, sample-ID
    checksum, frozen order, materialization mode, and background selection
    parameters, and its search corpus directory still exists on disk.
    """
    manifest_path = snapshot_dir / "corpus_snapshot_manifest.json"
    if not manifest_path.exists():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    expected_ids = [str(sample_id) for sample_id in sample_ids]
    checks = [
        (str(manifest.get("stage_name", "")), stage_name),
        (str(manifest.get("sample_id_checksum", "")), compute_sample_id_checksum(expected_ids)),
        (str(manifest.get("frozen_order_checksum", "")), compute_frozen_order_checksum(expected_ids)),
        (str(manifest.get("materialize_mode", "")), str(materialize_mode)),
    ]
    background = manifest.get("background_selection_manifest") or {}
    checks.append((str(background.get("seed", "")), str(background_seed)))
    checks.append((str(background.get("background_ratio", "")), str(background_ratio)))
    if any(str(observed) != str(expected) for observed, expected in checks):
        return None
    search_dir = manifest.get("search_corpus_dir", "")
    if not search_dir or not Path(search_dir).exists():
        return None
    if not str(manifest.get("corpus_checksum", "")):
        return None
    return manifest


def _generate_snapshot_table(corpus_manifests: list[dict], output_dir: Path) -> dict:
    from evaluation.dynamic_table_generator import DynamicPaperTableGenerator
    return {f"snapshot_{k}": v for k, v in DynamicPaperTableGenerator().generate_snapshot_audit_table(corpus_manifests, output_dir / "tables").items()}


def _prepare_reuse_gate(args, binding, stage_adapter, baselines, baseline_dir: Path, records_dir: Path) -> None:
    if not args.skip_existing:
        return
    baseline_dir.mkdir(parents=True, exist_ok=True)
    for baseline in baselines:
        result_path = baseline_dir / f"baseline_{baseline.name}.jsonl"
        record_path = records_dir / f"{baseline.name}_stage_execution_record.json"
        if not result_path.exists():
            continue
        expected = _stage_execution_record(binding, stage_adapter, baseline, result_path)
        if not record_path.exists():
            print(f"[reuse-gate] {baseline.name}: no stage execution record, dropping cached results", flush=True)
            result_path.unlink(missing_ok=True)
            continue
        try:
            record = json.loads(record_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            print(f"[reuse-gate] {baseline.name}: stage record unreadable, dropping cached results", flush=True)
            result_path.unlink(missing_ok=True)
            continue
        reuse = validate_result_reuse(expected, record)
        if not reuse.get("reusable"):
            mismatches = {
                key: item for key, item in (reuse.get("checks") or {}).items()
                if not item.get("matched")
            }
            print(f"[reuse-gate] {baseline.name}: cache rejected, mismatched fields: {json.dumps(mismatches, ensure_ascii=False)}", flush=True)
            result_path.unlink(missing_ok=True)
        else:
            print(f"[reuse-gate] {baseline.name}: cached stage results accepted for reuse", flush=True)


class _StageEnv:
    def __init__(self, binding) -> None:
        self.binding = binding
        self.old_values = {}

    def __enter__(self):
        updates = {
            "HOTPOT_WIKI_CORPUS_DIR": self.binding.search_corpus_dir,
            "SIRCHMUNK_WORK_PATH": self.binding.work_path,
        }
        for key, value in updates.items():
            self.old_values[key] = os.environ.get(key)
            os.environ[key] = str(value)
        return self

    def __exit__(self, exc_type, exc, tb):
        for key, value in self.old_values.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        return False


def _stage_environment(binding) -> _StageEnv:
    return _StageEnv(binding)


def _stage_execution_record(binding, stage_adapter, baseline, results_path: Path) -> StageExecutionRecord:
    config = stage_adapter.get_run_config()
    return StageExecutionRecord(
        stage_name=binding.stage_name,
        system_name=baseline.citation_name,
        sample_id_checksum=binding.sample_id_checksum,
        frozen_order_checksum=binding.frozen_order_checksum,
        corpus_checksum=binding.corpus_checksum,
        system_config_hash=_hash_payload({"config": config, "baseline": baseline.name}),
        baseline_version=f"{baseline.__class__.__module__}.{baseline.__class__.__name__}",
        cache_mode=str(config.get("cache_mode", "")),
        results_path=str(results_path),
        output_dir=binding.output_dir,
        work_path=binding.work_path,
        metadata={"baseline_name": baseline.name, "stage_binding": binding.to_dict()},
    )


def _dynamic_result_row(binding, baseline, results: list) -> dict:
    n = len(results)
    metric_payloads = [getattr(result, "telemetry", {}) or {} for result in results]
    setup = baseline.collect_setup_metrics()
    query_budgets = [_query_budget_of_result(result) for result in results]
    query_budget_summary = _summarize_query_budgets(query_budgets)
    evidence_trace_count = sum(1 for result in results if _evidence_traces_of_result(result))
    return {
        "system_name": baseline.citation_name,
        "stage_name": binding.stage_name,
        "official_em": _avg_metric(metric_payloads, "official_em") * 100,
        "official_f1": _avg_metric(metric_payloads, "official_f1") * 100,
        "evidence_recall": _avg_metric(metric_payloads, "evidence_recall") * 100,
        # Source-grounded answer rate: the fraction of samples whose answer is
        # backed by retrieved gold evidence. The previous "has any trace" count
        # was trivially 100% for every system that read at least one file and
        # carried no discriminative signal for the paper table.
        "evidence_trace_coverage": _avg_metric(metric_payloads, "answer_source_grounded") * 100,
        "evidence_trace_count": evidence_trace_count,
        "avg_latency": sum(float(getattr(result, "elapsed", 0.0) or 0.0) for result in results) / max(n, 1),
        "avg_tokens": sum(int(getattr(result, "tokens_used", 0) or 0) + int(getattr(result, "judge_tokens", 0) or 0) for result in results) / max(n, 1),
        "avg_oracle_calls": query_budget_summary.get("avg_oracle_calls", 0.0),
        "setup_update": setup.get("setup_seconds", 0.0),
        "sample_id_checksum": binding.sample_id_checksum,
        "frozen_order_checksum": binding.frozen_order_checksum,
        "corpus_checksum": binding.corpus_checksum,
        "setup_metrics": setup,
        "query_budget_summary": query_budget_summary,
    }


async def _measure_transition_cost(from_binding, to_binding, baseline, to_stage_adapter) -> dict:
    try:
        from framework.dynamic_update import CorpusMutation, UpdateOperation
        mutation = CorpusMutation(
            mutation_id=f"{from_binding.d_stage}_to_{to_binding.d_stage}",
            operation=UpdateOperation.MIXED,
            metadata={
                "from_stage": from_binding.to_dict(),
                "to_stage": to_binding.to_dict(),
                "transition": f"{from_binding.stage_name}->{to_binding.stage_name}",
            },
        )
        started = time.monotonic()
        result = baseline.update_index(mutation, bm_adapter=to_stage_adapter)
        if asyncio.iscoroutine(result):
            result = await result
        elapsed = time.monotonic() - started
        metadata = result if isinstance(result, dict) else {}
        update_supported = bool(metadata.get("update_supported", True))
        rebuild_required = bool(metadata.get("rebuild_required", not update_supported))
        return {
            "system_name": baseline.citation_name,
            "baseline_name": baseline.name,
            "transition": f"{from_binding.stage_name}->{to_binding.stage_name}",
            "from_stage": from_binding.stage_name,
            "to_stage": to_binding.stage_name,
            "update_completed": update_supported and not rebuild_required,
            "update_time_seconds": elapsed,
            "rebuild_required": rebuild_required,
            "query_ready_immediately": bool(metadata.get("query_ready_immediately", False)),
            "failure_reason": str(metadata.get("failure_reason") or ("full_rebuild_required" if rebuild_required else "none")),
            "from_corpus_checksum": from_binding.corpus_checksum,
            "to_corpus_checksum": to_binding.corpus_checksum,
            "corpus_checksum": to_binding.corpus_checksum,
            "sample_id_checksum": to_binding.sample_id_checksum,
            "frozen_order_checksum": to_binding.frozen_order_checksum,
            "metadata": metadata,
        }
    except Exception as exc:
        return {
            "system_name": getattr(baseline, "citation_name", getattr(baseline, "name", "unknown")),
            "baseline_name": getattr(baseline, "name", "unknown"),
            "transition": f"{from_binding.stage_name}->{to_binding.stage_name}",
            "from_stage": from_binding.stage_name,
            "to_stage": to_binding.stage_name,
            "update_completed": False,
            "update_time_seconds": 0.0,
            "rebuild_required": True,
            "query_ready_immediately": False,
            "failure_reason": "update_error",
            "failure_message": str(exc),
            "from_corpus_checksum": from_binding.corpus_checksum,
            "to_corpus_checksum": to_binding.corpus_checksum,
            "corpus_checksum": to_binding.corpus_checksum,
            "sample_id_checksum": to_binding.sample_id_checksum,
            "frozen_order_checksum": to_binding.frozen_order_checksum,
        }


async def _run_staleness_arm(
    *,
    args,
    from_binding,
    to_binding,
    previous_baselines: dict,
    previous_sample_ids: list[str],
    current_sample_ids: list[str],
    golden_set,
    stage_adapter,
    fresh_results: dict,
) -> tuple[list[dict], dict]:
    """Answer newly added questions with the previous stage index.

    The arm isolates index staleness from every other variable: the questions are
    exactly the ones added between the two stages, the corpus path already points
    at the current snapshot, and ``skip_prepare`` keeps each system on the index
    it built for the previous snapshot. Systems that retrieve from an internal
    index therefore cannot reach the newly added evidence, while systems that
    read the corpus at query time can.
    """
    from evaluation.suite import BaselineEvaluationSuite
    from evaluation.staleness import build_staleness_row, compute_delta_sample_ids

    delta_ids = compute_delta_sample_ids(previous_sample_ids, current_sample_ids)
    if args.staleness_max_samples > 0:
        delta_ids = delta_ids[: args.staleness_max_samples]
    if not delta_ids:
        return [], {}

    delta_samples = _select_sample_dicts(golden_set.samples, delta_ids)
    delta_checksum = compute_sample_id_checksum(delta_ids)
    delta_golden = _StageGoldenSet(
        delta_samples,
        to_binding,
        stage_name=f"{to_binding.stage_name}_stale_delta",
        sample_id_checksum=delta_checksum,
        frozen_order_checksum=compute_frozen_order_checksum(delta_ids),
    )

    baselines = list(previous_baselines.values())
    stale_dir = Path(to_binding.output_dir) / "staleness"
    stale_dir.mkdir(parents=True, exist_ok=True)

    # Rebuild the stale index on the PREVIOUS snapshot before answering.
    # When the main stage reused cached JSONL under --skip-existing, the kept
    # baseline instances never ran prepare(), so their in-memory index is
    # empty; querying that with skip_prepare would not measure staleness but a
    # broken index. Index-required systems are therefore prepared here against
    # D_{n-1} under the from-stage environment, which is exactly the stale index
    # definition: built on the old snapshot, queried on the new delta. Index-
    # free systems (e.g. ReAct) declare no index and are left untouched.
    prev_samples = _select_sample_dicts(golden_set.samples, previous_sample_ids)
    prev_golden = _StageGoldenSet(
        prev_samples,
        from_binding,
        stage_name=f"{from_binding.stage_name}_stale_source",
        sample_id_checksum=compute_sample_id_checksum(previous_sample_ids),
        frozen_order_checksum=compute_frozen_order_checksum(previous_sample_ids),
    )
    with _stage_environment(from_binding):
        for baseline in baselines:
            if not baseline.is_index_required():
                continue
            setup = await baseline.prepare(golden_set=prev_golden, bm_adapter=stage_adapter)
            logger.info(
                "[Staleness] rebuilt stale index for '%s' on %s: docs=%d",
                baseline.name, from_binding.d_stage, setup.indexed_documents,
            )

    suite = BaselineEvaluationSuite(
        bm_adapter=stage_adapter,
        baselines=baselines,
        output_dir=str(stale_dir),
        corpus_metadata={
            "index_state": "stale",
            "stale_index_prepared_on": from_binding.d_stage,
            "query_corpus_stage": to_binding.d_stage,
            "corpus_checksum": to_binding.corpus_checksum,
        },
    )
    with _stage_environment(to_binding):
        stale_map = await suite.run(
            delta_golden,
            skip_existing=False,
            skip_prepare=True,
            skip_cleanup=True,
        )

    delta_set = {str(sample_id) for sample_id in delta_ids}
    records_dir = Path(to_binding.output_dir) / "stage_records"
    records_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    record_paths: dict = {}
    for baseline in baselines:
        stale_results = stale_map.get(baseline.name, []) or []
        fresh_delta = [
            result for result in (fresh_results.get(baseline.name, []) or [])
            if str(getattr(result, "sample_id", "")) in delta_set
        ]
        mode = "measured" if stale_results else "no_stale_results"
        row = build_staleness_row(
            system_name=baseline.citation_name,
            baseline_name=baseline.name,
            from_stage=from_binding.stage_name,
            to_stage=to_binding.stage_name,
            delta_sample_ids=delta_ids,
            fresh_results=fresh_delta,
            stale_results=stale_results,
            index_required=bool(_baseline_flag(baseline, "is_index_required", True)),
            query_ready_immediately=bool(_baseline_flag(baseline, "is_query_ready_immediately", False)),
            stale_index_setup_metrics=_safe_setup_metrics(baseline),
            from_corpus_checksum=from_binding.corpus_checksum,
            to_corpus_checksum=to_binding.corpus_checksum,
            delta_sample_id_checksum=delta_checksum,
            stale_arm_mode=mode,
        )
        rows.append(row)
        record = StalenessEvaluationRecord(
            transition=row["transition"],
            from_stage=from_binding.stage_name,
            to_stage=to_binding.stage_name,
            system_name=baseline.citation_name,
            baseline_name=baseline.name,
            delta_sample_ids=list(delta_ids),
            delta_sample_id_checksum=delta_checksum,
            from_corpus_checksum=from_binding.corpus_checksum,
            to_corpus_checksum=to_binding.corpus_checksum,
            stale_index_prepared_on=from_binding.d_stage,
            query_corpus_dir=to_binding.search_corpus_dir,
            stale_arm_mode=mode,
            fresh_results_path=str(Path(to_binding.output_dir) / "baselines" / f"baseline_{baseline.name}.jsonl"),
            stale_results_path=str(stale_dir / f"baseline_{baseline.name}.jsonl"),
            metrics=row,
            metadata={"baseline_version": f"{baseline.__class__.__module__}.{baseline.__class__.__name__}"},
        )
        record_path = records_dir / f"{baseline.name}_staleness_record.json"
        record_path.write_text(json.dumps(record.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        record_paths[baseline.name] = str(record_path)
    return rows, record_paths


def _baseline_flag(baseline, method_name: str, default: bool) -> bool:
    getter = getattr(baseline, method_name, None)
    if not callable(getter):
        return default
    try:
        return bool(getter())
    except Exception:
        return default


def _safe_setup_metrics(baseline) -> dict:
    getter = getattr(baseline, "collect_setup_metrics", None)
    if not callable(getter):
        return {}
    try:
        metrics = getter()
    except Exception:
        return {}
    return metrics if isinstance(metrics, dict) else {}


def _query_budget_of_result(result) -> dict:
    telemetry = getattr(result, "telemetry", {}) or {}
    metadata = getattr(result, "metadata", {}) or {}
    query_budget = telemetry.get("query_budget") if isinstance(telemetry.get("query_budget"), dict) else {}
    if not query_budget and isinstance(metadata.get("query_budget"), dict):
        query_budget = metadata.get("query_budget", {})
    return {
        "oracle_calls": _num(query_budget.get("oracle_calls", telemetry.get("oracle_calls", telemetry.get("loop_count", 0.0)))),
        "llm_calls": _num(query_budget.get("llm_calls", telemetry.get("llm_calls", telemetry.get("total_llm_calls", 0.0)))),
        "search_calls": _num(query_budget.get("search_calls", telemetry.get("search_calls", len(telemetry.get("search_history", []) or [])))),
        "read_calls": _num(query_budget.get("read_calls", telemetry.get("read_calls", len(telemetry.get("read_file_ids", []) or [])))),
        "total_tokens": _num(query_budget.get("total_tokens", getattr(result, "tokens_used", 0))),
        "latency_seconds": _num(query_budget.get("latency_seconds", getattr(result, "elapsed", 0.0))),
        "budget_exceeded": bool(query_budget.get("budget_exceeded") or telemetry.get("failure_reason") == "budget_exceeded"),
    }


def _summarize_query_budgets(query_budgets: list[dict]) -> dict:
    oracle_calls = [_num(row.get("oracle_calls")) for row in query_budgets]
    llm_calls = [_num(row.get("llm_calls")) for row in query_budgets]
    search_calls = [_num(row.get("search_calls")) for row in query_budgets]
    read_calls = [_num(row.get("read_calls")) for row in query_budgets]
    total_tokens = [_num(row.get("total_tokens")) for row in query_budgets]
    latency_seconds = [_num(row.get("latency_seconds")) for row in query_budgets]
    return {
        "avg_oracle_calls": _avg(oracle_calls),
        "std_oracle_calls": _std(oracle_calls),
        "max_oracle_calls": max(oracle_calls, default=0.0),
        "avg_llm_calls": _avg(llm_calls),
        "avg_search_calls": _avg(search_calls),
        "avg_read_calls": _avg(read_calls),
        "avg_total_tokens": _avg(total_tokens),
        "std_total_tokens": _std(total_tokens),
        "max_total_tokens": max(total_tokens, default=0.0),
        "avg_latency_seconds": _avg(latency_seconds),
        "std_latency_seconds": _std(latency_seconds),
        "max_latency_seconds": max(latency_seconds, default=0.0),
        "budget_exceeded_count": sum(1 for row in query_budgets if row.get("budget_exceeded")),
    }


def _evidence_traces_of_result(result) -> list:
    telemetry = getattr(result, "telemetry", {}) or {}
    metadata = getattr(result, "metadata", {}) or {}
    for source in (telemetry, metadata):
        if not isinstance(source, dict):
            continue
        traces = source.get("evidence_traces")
        if isinstance(traces, list) and traces:
            return traces
        for key in ("evidence_sources", "read_file_ids", "retrieval_logs"):
            values = source.get(key)
            if isinstance(values, list) and values:
                return values
    return []


def _avg(values: list[float]) -> float:
    return sum(values) / max(len(values), 1)


def _std(values: list[float]) -> float:
    if not values:
        return 0.0
    mean = _avg(values)
    return (sum((value - mean) ** 2 for value in values) / len(values)) ** 0.5


def _num(value, default: float = 0.0) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return default


def _avg_metric(payloads: list[dict], key: str) -> float:
    values = []
    for payload in payloads:
        try:
            values.append(float(payload.get(key, 0.0) or 0.0))
        except (TypeError, ValueError):
            values.append(0.0)
    return sum(values) / max(len(values), 1)


def _hash_payload(payload: dict) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


if __name__ == "__main__":
    raise SystemExit(main())
