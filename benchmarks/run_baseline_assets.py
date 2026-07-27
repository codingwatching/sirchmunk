#!/usr/bin/env python3
"""Unified baseline/competitor asset management for ResearchOps P1.

This entry point owns offline baseline assets only: preprocessing, indexing,
embedding/graph construction readiness, lifecycle feasibility, and asset
registry updates.  It deliberately does not run paper QA evaluation; formal
main/ablation evaluation stays in ``run_paper_experiment.py`` and
``run_evaluation.py``.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib
import json
import logging
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

_SCRIPT_DIR = Path(__file__).parent.resolve()
_PROJECT_ROOT = _SCRIPT_DIR.parent
_SRC = _PROJECT_ROOT / "src"
for _p in (str(_SCRIPT_DIR), str(_SRC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from baselines.base_adapter import BaselineAdapter  # noqa: E402
from evaluation.table_generator import PaperTableGenerator  # noqa: E402
from framework.asset_registry import AssetRecord, AssetRegistry  # noqa: E402
from framework.baseline_lifecycle import BaselineLifecycleManager  # noqa: E402
from framework.control_gates import gate_0_params, gate_1_assets  # noqa: E402
from framework.control_phase import ControlBlock, ExperimentStage  # noqa: E402
from framework.lifecycle_schema import BaselineLifecycleRecord, BaselinePhase, ResourceBudget  # noqa: E402
from framework.param_schema import AssetsConfig, ControlRunConfig, validate_control_config  # noqa: E402
from framework.registry import load_benchmark_adapter, supported_benchmarks  # noqa: E402
from framework.run_summary import (  # noqa: E402
    ControlRunStatus,
    GateSummary,
    StageSummary,
    create_control_run_summary,
    save_summary,
    summarize_assets,
)
from framework.time_utils import local_timestamp, now_local_iso  # noqa: E402

logger = logging.getLogger("run_baseline_assets")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Manage baseline preprocessing/index assets and asset registry.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser("prepare", help="Build/validate baseline assets and update registry")
    _add_common_args(prepare)
    prepare.add_argument("--methods", "--baselines", dest="methods", default="bm25,naive_rag")
    prepare.add_argument("--limit", type=int, default=20, help="Small sample count for baseline path inference")
    prepare.add_argument("--seed", type=int, default=42)
    prepare.add_argument("--corpus-scale", default="fullwiki")
    prepare.add_argument("--corpus-dir", default="")
    prepare.add_argument("--corpus-id", default="")
    prepare.add_argument("--corpus-hash", default="")
    prepare.add_argument("--force-rebuild", action="store_true")
    prepare.add_argument("--no-reuse-assets", action="store_true")
    prepare.add_argument("--validate-only", action="store_true")
    prepare.add_argument("--dry-run", action="store_true")
    prepare.add_argument("--strict", action="store_true")
    prepare.add_argument("--run-id", default="")
    prepare.add_argument("--stage", default="exploration", choices=["exploration", "frozen"])
    prepare.add_argument("--build-timeout", type=float, default=0.0)
    prepare.add_argument("--max-ram-bytes", type=int, default=0)
    prepare.add_argument("--max-disk-bytes", type=int, default=0)
    prepare.add_argument("--max-llm-tokens", type=int, default=0)
    prepare.add_argument("--max-api-cost-usd", type=float, default=0.0)
    prepare.add_argument("--retry-count", type=int, default=0)
    prepare.add_argument("--bm25-max-files", type=int, default=20000)
    prepare.add_argument("--naive-rag-max-files", type=int, default=5000)
    prepare.add_argument("--lightrag-predictions", default="")
    prepare.add_argument("--lightrag-setup-metrics", default="")
    prepare.add_argument("--graphrag-predictions", default="")
    prepare.add_argument("--graphrag-setup-metrics", default="")
    prepare.add_argument("--no-sirchmunk-row", action="store_true")

    validate = sub.add_parser("validate", help="Validate/query an existing asset registry")
    _add_registry_query_args(validate)
    validate.add_argument("--strict", action="store_true")

    status = sub.add_parser("status", help="Print compact registry status")
    _add_registry_query_args(status)

    scaling = sub.add_parser("scaling", help="Run multi-scale lifecycle/scaling study")
    _add_common_args(scaling)
    scaling.add_argument("--methods", "--baselines", dest="methods", default="bm25,naive_rag")
    scaling.add_argument("--scales", default="10k:10000,100k:100000,fullwiki:0")
    scaling.add_argument("--limit", type=int, default=20)
    scaling.add_argument("--seed", type=int, default=42)
    scaling.add_argument("--subset-output-dir", default="")
    scaling.add_argument("--subset-strategy", default="random_shard", choices=["random_shard", "prefix"])
    scaling.add_argument("--materialize", default="symlink", choices=["manifest", "symlink", "copy"])
    scaling.add_argument("--build-timeout", type=float, default=0.0)
    scaling.add_argument("--max-ram-bytes", type=int, default=0)
    scaling.add_argument("--max-disk-bytes", type=int, default=0)
    scaling.add_argument("--max-llm-tokens", type=int, default=0)
    scaling.add_argument("--max-api-cost-usd", type=float, default=0.0)
    scaling.add_argument("--q-values", default="1,10,100,1000")

    update = sub.add_parser("update-readiness", help="Measure baseline incremental-update readiness")
    _add_common_args(update)
    update.add_argument("--methods", "--baselines", dest="methods", default="bm25,naive_rag")
    update.add_argument("--base-corpus-dir", default="", help="Base corpus dir; defaults to adapter wiki/corpus dir when available")
    update.add_argument("--work-dir", default="", help="Work dir for mutated corpus versions")
    update.add_argument("--operation", default="mixed", choices=["add", "delete", "update", "mixed"])
    update.add_argument("--doc-ids", default="", help="Comma-separated relative doc ids affected by the mutation")
    update.add_argument("--delta-docs-dir", default="", help="Directory containing add/update docs")
    update.add_argument("--mutation-id", default="")
    update.add_argument("--mutation-ratio", type=float, default=0.0)
    update.add_argument("--materialize", default="symlink", choices=["symlink", "copy"])
    update.add_argument("--limit", type=int, default=20)
    update.add_argument("--seed", type=int, default=42)
    update.add_argument("--bm25-max-files", type=int, default=20000)
    update.add_argument("--naive-rag-max-files", type=int, default=5000)
    update.add_argument("--lightrag-predictions", default="")
    update.add_argument("--lightrag-setup-metrics", default="")
    update.add_argument("--graphrag-predictions", default="")
    update.add_argument("--graphrag-setup-metrics", default="")
    update.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--benchmark", "-b", required=True, choices=supported_benchmarks())
    parser.add_argument("--env", "-e", required=True)
    parser.add_argument("--output-dir", default="", help="Benchmark output base or assets directory")
    parser.add_argument("--assets-dir", default="", help="Explicit assets directory override")
    parser.add_argument("--asset-registry", default="")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])


def _add_registry_query_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--asset-registry", required=True)
    parser.add_argument("--benchmark", default="")
    parser.add_argument("--method", default="")
    parser.add_argument("--methods", default="")
    parser.add_argument("--stage", default="")
    parser.add_argument("--status", default="")
    parser.add_argument("--type", default="")
    parser.add_argument("--corpus-hash", default="")
    parser.add_argument("--config-hash", default="")
    parser.add_argument("--reusable-only", action="store_true")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s  %(name)-25s  %(levelname)-7s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


async def _prepare(args: argparse.Namespace) -> int:
    _setup_logging(args.log_level)
    env_file = str(Path(args.env).expanduser().resolve())
    if not Path(env_file).exists():
        logger.error("env file does not exist: %s", env_file)
        return 1

    adapter = load_benchmark_adapter(args.benchmark, env_file)
    output_base = Path(args.output_dir or adapter.get_output_dir()).expanduser().resolve()
    assets_dir = _resolve_assets_dir(output_base, args.assets_dir)
    assets_dir.mkdir(parents=True, exist_ok=True)
    registry_path = Path(args.asset_registry).expanduser().resolve() if args.asset_registry else assets_dir / "asset_registry.jsonl"
    registry = AssetRegistry(registry_path)

    methods = _split_csv(args.methods)
    budget = ResourceBudget(
        wall_clock_seconds=args.build_timeout,
        max_ram_bytes=args.max_ram_bytes,
        max_disk_bytes=args.max_disk_bytes,
        max_llm_tokens=args.max_llm_tokens,
        max_api_cost_usd=args.max_api_cost_usd,
        retry_count=args.retry_count,
    )
    corpus_manifest = adapter.get_dataset_manifest()
    if args.corpus_dir:
        corpus_manifest["corpus_dir"] = str(Path(args.corpus_dir).expanduser().resolve())
    if args.corpus_id:
        corpus_manifest["corpus_id"] = args.corpus_id
    corpus_hash = args.corpus_hash or _stable_hash(corpus_manifest)
    config_hash = _stable_hash(
        {
            "benchmark": args.benchmark,
            "methods": methods,
            "corpus_scale": args.corpus_scale,
            "corpus_hash": corpus_hash,
            "resource_budget": budget.to_dict(),
        }
    )
    reuse_assets = not args.no_reuse_assets and not args.force_rebuild
    control_config = ControlRunConfig(
        benchmark=args.benchmark,
        block=ControlBlock.ASSETS,
        stage=ExperimentStage(args.stage),
        env_file=env_file,
        output_dir=str(output_base),
        run_id=args.run_id or f"assets_{args.benchmark}_{local_timestamp()}",
        seed=args.seed,
        dry_run=args.dry_run,
        assets=AssetsConfig(
            methods=methods,
            corpus_scale=args.corpus_scale,
            corpus_dir=str(corpus_manifest.get("corpus_dir") or corpus_manifest.get("wiki_dir") or ""),
            corpus_id=str(corpus_manifest.get("corpus_id") or corpus_manifest.get("id") or ""),
            corpus_hash=corpus_hash,
            config_hash=config_hash,
            asset_registry=str(registry_path),
            force_rebuild=args.force_rebuild,
            reuse_assets=reuse_assets,
            validate_only=args.validate_only,
        ),
        resource_budget=budget,
        metadata={"assets_dir": str(assets_dir)},
    )
    validation = validate_control_config(control_config)
    if not validation.ok:
        print(json.dumps({"validation": validation.to_dict()}, indent=2, ensure_ascii=False))
        return 1
    if args.dry_run:
        print(json.dumps({"dry_run": True, "config": control_config.to_dict(), "validation": validation.to_dict()}, indent=2, ensure_ascii=False))
        return 0

    records: List[BaselineLifecycleRecord] = []
    manager = BaselineLifecycleManager(assets_dir, resource_budget=budget)
    if not args.no_sirchmunk_row:
        records.append(_sirchmunk_no_index_record(control_config.run_id, args.benchmark, corpus_manifest, args.corpus_scale))
        manager.save_record(records[-1])
    if not args.validate_only:
        golden_like = _build_golden_like(adapter, limit=args.limit, seed=args.seed)
        for baseline in _load_baselines(methods, args):
            logger.info("Preparing baseline asset: %s", baseline.citation_name)
            record = await manager.run_build(
                baseline,
                run_id=control_config.run_id,
                benchmark=args.benchmark,
                corpus_manifest=corpus_manifest,
                golden_set=golden_like,
                bm_adapter=adapter,
                corpus_scale=args.corpus_scale,
            )
            records.append(record)

    asset_records = [
        registry.append(
            AssetRecord.from_lifecycle_record(
                record,
                stage=args.stage,
                block=ControlBlock.ASSETS.value,
                corpus_hash=corpus_hash,
                config_hash=config_hash,
            )
        )
        for record in records
    ]
    table_paths = PaperTableGenerator(benchmark_name=f"{args.benchmark} assets").generate_feasibility_table(
        records,
        str(assets_dir / "tables"),
    )
    gate_results = [gate_0_params(control_config), gate_1_assets(control_config, asset_registry=registry)]
    summary = create_control_run_summary(
        control_run_id=control_config.run_id,
        benchmark=args.benchmark,
        block=ControlBlock.ASSETS,
        stage=ExperimentStage(args.stage),
        env_file=env_file,
        output_dir=str(output_base),
        paths={
            "assets_dir": str(assets_dir),
            "asset_registry": str(registry_path),
            **{f"table_{key}": value for key, value in table_paths.items()},
        },
        metadata={"corpus_hash": corpus_hash, "config_hash": config_hash},
    )
    summary.status = ControlRunStatus.SUCCESS
    summary.config_hash = config_hash
    for gate in gate_results:
        summary.add_gate(GateSummary.from_gate_result(gate))
    summary.add_stage(
        StageSummary(
            block=ControlBlock.ASSETS.value,
            stage=args.stage,
            status=ControlRunStatus.SUCCESS if not summary.blocked else ControlRunStatus.BLOCKED,
            started_at=now_local_iso(),
            ended_at=now_local_iso(),
            run_ids=[control_config.run_id],
            artifact_paths={"assets_dir": str(assets_dir), "registry": str(registry_path)},
            metrics=summarize_assets(asset_records),
        )
    )
    for asset in asset_records:
        summary.add_asset(asset)
    if summary.blocked:
        summary.status = ControlRunStatus.BLOCKED
    summary_path = assets_dir / "asset_summary.json"
    save_summary(summary, summary_path)

    payload = {
        "run_id": control_config.run_id,
        "asset_registry": str(registry_path),
        "summary": str(summary_path),
        "assets": [asset.to_dict() for asset in asset_records],
        "asset_counts": summarize_assets(asset_records),
        "table_paths": table_paths,
        "gates": [gate.to_dict() for gate in gate_results],
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    failed_assets = [asset for asset in asset_records if not asset.query_eligible]
    return 1 if args.strict and (summary.blocked or failed_assets) else 0


def _validate_or_status(args: argparse.Namespace, *, compact: bool) -> int:
    _setup_logging(args.log_level)
    registry_path = Path(args.asset_registry).expanduser().resolve()
    if not registry_path.exists():
        print(json.dumps({"asset_registry": str(registry_path), "exists": False, "summary": {}}, indent=2, ensure_ascii=False))
        return 1 if getattr(args, "strict", False) else 0
    registry = AssetRegistry(registry_path)
    methods = _split_csv(args.methods) or ([args.method] if args.method else [])
    records = registry.list(
        benchmark=args.benchmark,
        method=args.method,
        asset_type=args.type or None,
        status=args.status or None,
        stage=args.stage,
        corpus_hash=args.corpus_hash,
        config_hash=args.config_hash,
        reusable_only=args.reusable_only,
    )
    payload: Dict[str, Any] = {
        "asset_registry": str(registry.path),
        "summary": summarize_assets(records),
    }
    if compact:
        payload["assets"] = [
            {
                "asset_id": row.asset_id,
                "type": row.asset_type.value,
                "status": row.status.value,
                "benchmark": row.benchmark,
                "method": row.method,
                "path": row.path,
                "query_eligible": row.query_eligible,
                "failure_reason": row.failure_reason,
            }
            for row in records
        ]
    else:
        payload["assets"] = [row.to_dict() for row in records]
    if args.benchmark and methods:
        config = ControlRunConfig(
            benchmark=args.benchmark,
            block=ControlBlock.ASSETS,
            stage=ExperimentStage(args.stage or "frozen"),
            assets=AssetsConfig(
                methods=methods,
                corpus_hash=args.corpus_hash,
                config_hash=args.config_hash,
                asset_registry=str(registry.path),
                reuse_assets=True,
            ),
        )
        gate = gate_1_assets(config, asset_registry=registry)
        payload["gate_1_assets"] = gate.to_dict()
        exit_code = 1 if getattr(args, "strict", False) and not gate.passed else 0
    else:
        exit_code = 0
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return exit_code


def _scaling(args: argparse.Namespace) -> int:
    _setup_logging(args.log_level)
    output_base = Path(args.output_dir).expanduser().resolve() if args.output_dir else Path("benchmarks") / args.benchmark / "output"
    scaling_dir = output_base / "scaling"
    cmd = [
        sys.executable,
        str(_SCRIPT_DIR / "run_scaling_study.py"),
        "--benchmark",
        args.benchmark,
        "--env",
        args.env,
        "--baselines",
        args.methods,
        "--scales",
        args.scales,
        "--limit",
        str(args.limit),
        "--seed",
        str(args.seed),
        "--output-dir",
        str(scaling_dir),
        "--subset-strategy",
        args.subset_strategy,
        "--materialize",
        args.materialize,
        "--build-timeout",
        str(args.build_timeout),
        "--max-ram-bytes",
        str(args.max_ram_bytes),
        "--max-disk-bytes",
        str(args.max_disk_bytes),
        "--max-llm-tokens",
        str(args.max_llm_tokens),
        "--max-api-cost-usd",
        str(args.max_api_cost_usd),
        "--q-values",
        args.q_values,
        "--log-level",
        args.log_level,
    ]
    if args.subset_output_dir:
        cmd.extend(["--subset-output-dir", args.subset_output_dir])
    return _run_subprocess(cmd)


async def _update_readiness(args: argparse.Namespace) -> int:
    _setup_logging(args.log_level)
    env_file = str(Path(args.env).expanduser().resolve())
    if not Path(env_file).exists():
        logger.error("env file does not exist: %s", env_file)
        return 1
    adapter = load_benchmark_adapter(args.benchmark, env_file)
    output_base = Path(args.output_dir or adapter.get_output_dir()).expanduser().resolve()
    assets_dir = _resolve_assets_dir(output_base, args.assets_dir)
    update_dir = Path(args.work_dir).expanduser().resolve() if args.work_dir else assets_dir / "update_readiness"
    update_dir.mkdir(parents=True, exist_ok=True)
    base_corpus_dir = _resolve_base_corpus_dir(args, adapter)
    mutation_id = args.mutation_id or f"mutation_{args.operation}_{local_timestamp()}"
    mutation_payload = {
        "mutation_id": mutation_id,
        "operation": args.operation,
        "doc_ids": _split_csv(args.doc_ids),
        "delta_docs_dir": args.delta_docs_dir,
        "mutation_ratio": args.mutation_ratio,
        "base_corpus_dir": str(base_corpus_dir),
    }
    if args.dry_run:
        print(json.dumps({"dry_run": True, "mutation": mutation_payload, "work_dir": str(update_dir)}, indent=2, ensure_ascii=False))
        return 0
    from framework.dynamic_update import CorpusMutation, DynamicUpdateManager, UpdateOperation

    manager = DynamicUpdateManager(base_corpus_dir, update_dir / "corpus_versions", materialize_mode=args.materialize)
    mutation = CorpusMutation(
        mutation_id=mutation_id,
        operation=UpdateOperation(args.operation),
        doc_ids=_split_csv(args.doc_ids),
        delta_docs_dir=args.delta_docs_dir,
        mutation_ratio=args.mutation_ratio,
        metadata={"benchmark": args.benchmark},
    )
    version_manifest = manager.create_version(mutation)
    baselines = _load_baselines(_split_csv(args.methods), args)
    results = []
    for baseline in baselines:
        results.append(await manager.evaluate_baseline_update(baseline, mutation, bm_adapter=adapter))
    results_path = update_dir / "update_readiness.jsonl"
    manager.save_update_results(results, results_path)
    summary = {
        "benchmark": args.benchmark,
        "env_file": env_file,
        "version_manifest": version_manifest.to_dict(),
        "mutation": mutation.to_dict(),
        "results_path": str(results_path),
        "results": [result.to_dict() for result in results],
    }
    summary_path = update_dir / "update_readiness_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"summary": str(summary_path), "results_path": str(results_path), "results": summary["results"]}, indent=2, ensure_ascii=False))
    return 0


def _resolve_base_corpus_dir(args: argparse.Namespace, adapter: Any) -> Path:
    if args.base_corpus_dir:
        raw_path = args.base_corpus_dir
    else:
        manifest = adapter.get_dataset_manifest()
        raw_path = str(manifest.get("wiki_dir") or manifest.get("corpus_dir") or "")
    if not raw_path:
        raise ValueError("Base corpus directory is not declared; pass --base-corpus-dir")
    path = Path(raw_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Base corpus directory not found: {path}")
    return path


def _run_subprocess(cmd: List[str]) -> int:
    logger.info("Running: %s", " ".join(cmd))
    return int(subprocess.run(cmd, check=False).returncode)


def _resolve_assets_dir(output_base: Path, assets_dir: str) -> Path:
    if assets_dir:
        return Path(assets_dir).expanduser().resolve()
    if output_base.name == "assets":
        return output_base
    return output_base / "assets"


def _build_golden_like(adapter: Any, *, limit: int, seed: int) -> SimpleNamespace:
    samples = adapter.load_samples(limit=max(int(limit), 0), seed=seed)
    return SimpleNamespace(samples=[
        {
            "sample_id": sample.sample_id,
            "question": sample.question,
            "gold_answer": sample.gold_answer,
            "metadata": sample.metadata,
        }
        for sample in samples
    ])


def _load_baselines(methods: List[str], args: argparse.Namespace) -> List[BaselineAdapter]:
    baselines: List[BaselineAdapter] = []
    for raw_name in methods:
        lower = raw_name.strip().lower().replace("-", "_")
        if not lower:
            continue
        if lower in {"bm25", "bm25_local"}:
            from baselines import LocalBM25Baseline
            baselines.append(LocalBM25Baseline(max_files=args.bm25_max_files))
        elif lower in {"bm25_rag", "rag_bm25"}:
            from baselines import BM25RAGBaseline
            baselines.append(BM25RAGBaseline(max_files=args.bm25_max_files))
        elif lower in {"naive_rag", "naive_rag_local"}:
            from baselines import NaiveRAGBaseline
            baselines.append(NaiveRAGBaseline(max_files=args.naive_rag_max_files))
        elif lower in {"react", "react_search"}:
            from baselines import ReActSearchBaseline
            baselines.append(ReActSearchBaseline())
        elif lower in {"lightrag", "lightrag_v1"}:
            from baselines import LightRAGV1Baseline
            baselines.append(LightRAGV1Baseline(args.lightrag_predictions, args.lightrag_setup_metrics))
        elif lower == "graphrag":
            from baselines import GraphRAGBaseline
            baselines.append(GraphRAGBaseline(args.graphrag_predictions, args.graphrag_setup_metrics))
        elif ":" in raw_name:
            module_name, _, factory_name = raw_name.partition(":")
            module = importlib.import_module(module_name)
            baseline = getattr(module, factory_name)()
            if not isinstance(baseline, BaselineAdapter):
                raise TypeError(f"Factory {raw_name} did not return BaselineAdapter")
            baselines.append(baseline)
        else:
            raise ValueError(
                "Unknown baseline. Use bm25, bm25_rag, naive_rag, react, "
                "lightrag_v1, graphrag, or module:factory."
            )
    return baselines


def _sirchmunk_no_index_record(
    run_id: str,
    benchmark: str,
    corpus_manifest: Dict[str, Any],
    corpus_scale: str,
) -> BaselineLifecycleRecord:
    return BaselineLifecycleRecord(
        run_id=run_id,
        benchmark=benchmark,
        baseline_name="sirchmunk",
        citation_name="Sirchmunk / LENS (no index required)",
        corpus_id=str(corpus_manifest.get("corpus_id") or corpus_manifest.get("id") or ""),
        corpus_scale=corpus_scale,
        corpus_size_docs=int(corpus_manifest.get("doc_count") or corpus_manifest.get("total_documents") or 0),
        index_required=False,
        phase=BaselinePhase.READY,
        build_completed=True,
        index_ready=True,
        query_eligible=True,
        build_time_seconds=0.0,
        preprocessing_seconds=0.0,
        index_build_seconds=0.0,
        disk_bytes=0,
        preprocess_llm_tokens=0,
        metadata={"index_required": False},
    )


def _split_csv(raw: str) -> List[str]:
    return [part.strip() for part in str(raw or "").split(",") if part.strip()]


def _stable_hash(payload: Dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def main() -> None:
    args = _parse_args()
    if args.command == "prepare":
        raise SystemExit(asyncio.run(_prepare(args)))
    if args.command == "validate":
        raise SystemExit(_validate_or_status(args, compact=False))
    if args.command == "status":
        raise SystemExit(_validate_or_status(args, compact=True))
    if args.command == "scaling":
        raise SystemExit(_scaling(args))
    if args.command == "update-readiness":
        raise SystemExit(asyncio.run(_update_readiness(args)))
    raise SystemExit(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
