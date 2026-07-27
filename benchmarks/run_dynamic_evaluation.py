#!/usr/bin/env python3
"""Create v4 dynamic G_n/D_n artifacts for HotpotQA evaluation.

This CLI prepares the publication protocol artifacts described in
sirchmunk_experiment_design_v4_20260727.md. It intentionally focuses on
creating auditable sample/corpus stage bindings; system execution can then use
those bindings through run_evaluation.py or queue-based frozen runs.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
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
    create_sampling_protocol,
)
from framework.registry import load_benchmark_adapter, supported_benchmarks  # noqa: E402
from framework.v4_stage_runner import (  # noqa: E402
    StageExecutionRecord,
    build_stage_bindings,
    save_stage_bindings,
    validate_result_reuse,
)
from hotpotqa.dynamic_corpus import build_dynamic_corpus_snapshot, derive_nested_sample_sets  # noqa: E402
from hotpotqa.title_resolver import HotpotQATitleResolver  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build v4 dynamic G_n/D_n artifacts")
    parser.add_argument("--benchmark", default="hotpotqa", choices=supported_benchmarks())
    parser.add_argument("--env", required=True, help="Benchmark env file")
    parser.add_argument("--output-dir", default="", help="Default: benchmarks/{benchmark}/output/dynamic_eval_v4")
    parser.add_argument("--golden-n", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stages", default="500,1000,2000", help="Comma separated nested stage sizes")
    parser.add_argument("--strata", default=",".join(DEFAULT_HOTPOTQA_STRATA))
    parser.add_argument("--materialize", choices=["symlink", "copy", "manifest"], default="symlink")
    parser.add_argument("--background-ratio", type=float, default=3.0)
    parser.add_argument("--background-seed", type=int, default=42)
    parser.add_argument("--allow-missing-evidence", action="store_true", help="Allow snapshots with unresolved evidence titles; not for main-table runs")
    parser.add_argument("--force-recreate-golden", action="store_true")
    parser.add_argument("--run-baselines", action="store_true", help="Run built-in baselines for each G/D stage")
    parser.add_argument("--baselines", default="bm25_rag,react", help="Comma separated: bm25_rag,react,lens_full,lens_no_prior,lens_no_seq")
    parser.add_argument("--skip-existing", action="store_true", help="Reuse existing per-stage baseline JSONL when present")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.benchmark != "hotpotqa":
        raise ValueError("v4 dynamic G_n/D_n artifacts are currently defined for HotpotQA only")

    env_file = str(Path(args.env).expanduser().resolve())
    adapter = load_benchmark_adapter(args.benchmark, env_file)
    output_dir = Path(args.output_dir or (_SCRIPT_DIR / args.benchmark / "output" / "dynamic_eval_v4")).resolve()
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

    sampling_dir = output_dir / "sampling"
    nested = derive_nested_sample_sets(golden_set, stages=stages, output_dir=sampling_dir)

    wiki_dir = _wiki_dir(adapter)
    resolver = HotpotQATitleResolver(wiki_dir)
    parent_samples = golden_set.to_benchmark_samples()
    corpus_manifests = []
    for stage in nested.stages:
        stage_n = int(stage["stage_n"])
        sample_ids = stage["sample_ids_file"]
        ids = _load_sample_ids(sample_ids)
        d_stage = f"D_{stage_n}"
        manifest = build_dynamic_corpus_snapshot(
            parent_samples,
            sample_ids=ids,
            wiki_dir=wiki_dir,
            output_dir=output_dir / "corpus" / d_stage,
            stage_name=d_stage,
            materialize_mode=args.materialize,
            background_ratio=args.background_ratio,
            background_seed=args.background_seed,
            resolver=resolver,
            strict_evidence=not args.allow_missing_evidence,
        )
        corpus_manifests.append(manifest.to_dict())

    bindings = build_stage_bindings(
        nested_sample_manifest=nested.to_dict(),
        corpus_manifests=corpus_manifests,
        base_work_path=adapter.get_work_path(),
        base_output_dir=output_dir.parent,
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
        "baseline_runs": baseline_runs,
    }
    summary_path = output_dir / "dynamic_eval_manifest.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"dynamic_eval_manifest": str(summary_path), "stage_bindings": bindings_path}, indent=2))
    return 0


def _population_size(adapter) -> int:
    try:
        return int(adapter.describe_split().get("population_size", 0) or 0)
    except Exception:
        return 0


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
    from evaluation.v4_table_generator import V4PaperTableGenerator

    baseline_specs = [spec.strip() for spec in args.baselines.split(",") if spec.strip()]
    runs = {}
    dynamic_rows = []
    update_rows = []
    for binding in bindings:
        stage_adapter = _StageAdapter(base_adapter, binding)
        stage_samples = _select_sample_dicts(golden_set.samples, _load_sample_ids(binding.sample_ids_file))
        stage_golden = _StageGoldenSet(stage_samples, binding)
        baselines = [_baseline_by_name(spec, stage_adapter) for spec in baseline_specs]
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
            result_map = await suite.run(stage_golden, skip_existing=args.skip_existing)
        stage_records = {}
        for baseline in baselines:
            results = result_map.get(baseline.name, [])
            record = _stage_execution_record(binding, stage_adapter, baseline, baseline_dir / f"baseline_{baseline.name}.jsonl")
            record_payload = {**record.to_dict(), "reuse_fingerprint": record.reuse_fingerprint(), "result_count": len(results)}
            record_path = records_dir / f"{baseline.name}_stage_execution_record.json"
            record_path.write_text(json.dumps(record_payload, indent=2, ensure_ascii=False), encoding="utf-8")
            stage_records[baseline.name] = str(record_path)
            dynamic_rows.append(_dynamic_result_row(binding, baseline, results))
            update_rows.append(_update_readiness_row(binding, baseline))
        runs[binding.stage_name] = {
            "output_dir": binding.output_dir,
            "corpus_checksum": binding.corpus_checksum,
            "sample_id_checksum": binding.sample_id_checksum,
            "stage_records": stage_records,
            "systems": {name: len(results) for name, results in result_map.items()},
        }
    tables_dir = output_dir / "tables"
    generator = V4PaperTableGenerator()
    runs["tables"] = {}
    runs["tables"].update({f"dynamic_{k}": v for k, v in generator.generate_dynamic_main_table(dynamic_rows, tables_dir).items()})
    runs["tables"].update({f"update_{k}": v for k, v in generator.generate_update_readiness_table(update_rows, tables_dir).items()})
    return runs


class _StageGoldenSet:
    def __init__(self, samples: list[dict], binding) -> None:
        self.samples = samples
        self.n_questions = len(samples)
        self.seed = 0
        self.sampling_protocol = {"method": "fixed_ids", "stage_name": binding.stage_name}
        self.sampling_manifest = {
            "sample_ids": [sample["sample_id"] for sample in samples],
            "sample_id_checksum": binding.sample_id_checksum,
            "frozen_order_checksum": binding.frozen_order_checksum,
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
            "v4_stage_name": self._binding.stage_name,
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
            "v4_stage_name": self._binding.stage_name,
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
        )


def _baseline_by_name(spec: str, bm_adapter):
    lower = spec.strip().lower().replace("-", "_")
    if lower in {"bm25_rag", "rag_bm25"}:
        from baselines import BM25RAGBaseline
        return BM25RAGBaseline()
    if lower in {"react", "react_search"}:
        from baselines import ReActSearchBaseline
        return ReActSearchBaseline()
    if lower in {"lens_full", "full", "lens_no_prior", "no_prior", "lens_no_seq", "no_seq"}:
        from ablations import build_single_lens_ablation
        return build_single_lens_ablation(bm_adapter, profile_name=lower)
    raise ValueError(f"Unsupported v4 baseline: {spec}")


def _select_sample_dicts(samples: list[dict], sample_ids: list[str]) -> list[dict]:
    by_id = {str(sample.get("sample_id")): sample for sample in samples}
    missing = [sample_id for sample_id in sample_ids if sample_id not in by_id]
    if missing:
        raise ValueError(f"Missing stage sample ids: {missing[:10]} total_missing={len(missing)}")
    return [by_id[sample_id] for sample_id in sample_ids]


def _generate_snapshot_table(corpus_manifests: list[dict], output_dir: Path) -> dict:
    from evaluation.v4_table_generator import V4PaperTableGenerator
    return {f"snapshot_{k}": v for k, v in V4PaperTableGenerator().generate_snapshot_audit_table(corpus_manifests, output_dir / "tables").items()}


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
            result_path.unlink(missing_ok=True)
            continue
        try:
            record = json.loads(record_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            result_path.unlink(missing_ok=True)
            continue
        reuse = validate_result_reuse(expected, record)
        if not reuse.get("reusable"):
            result_path.unlink(missing_ok=True)


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
    return {
        "system_name": baseline.citation_name,
        "stage_name": binding.stage_name,
        "official_em": _avg_metric(metric_payloads, "official_em") * 100,
        "official_f1": _avg_metric(metric_payloads, "official_f1") * 100,
        "evidence_recall": _avg_metric(metric_payloads, "evidence_recall") * 100,
        "avg_latency": sum(float(getattr(result, "elapsed", 0.0) or 0.0) for result in results) / max(n, 1),
        "avg_tokens": sum(int(getattr(result, "tokens_used", 0) or 0) + int(getattr(result, "judge_tokens", 0) or 0) for result in results) / max(n, 1),
        "setup_update": setup.get("setup_seconds", 0.0),
        "sample_id_checksum": binding.sample_id_checksum,
        "frozen_order_checksum": binding.frozen_order_checksum,
        "corpus_checksum": binding.corpus_checksum,
        "setup_metrics": setup,
    }


def _update_readiness_row(binding, baseline) -> dict:
    setup = baseline.collect_setup_metrics()
    return {
        "system_name": baseline.citation_name,
        "baseline_name": baseline.name,
        "transition": binding.stage_name,
        "update_time_seconds": setup.get("index_build_seconds", 0.0) if setup.get("rebuild_required") else 0.0,
        "rebuild_required": bool(setup.get("rebuild_required", False)),
        "query_ready_immediately": bool(setup.get("query_ready_immediately", False)),
        "corpus_checksum": binding.corpus_checksum,
    }


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
