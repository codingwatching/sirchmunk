#!/usr/bin/env python3
"""Formal paper-experiment orchestration for ResearchOps P2.

This script composes existing benchmark capabilities into paper-facing flows:
sampling/GoldenSet governance, optional frozen Sirchmunk execution, baseline
comparison, paper table/report generation, and quality gates.  It does not
reimplement search, judging, baseline prediction, or report formatting logic.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_SCRIPT_DIR = Path(__file__).parent.resolve()
_PROJECT_ROOT = _SCRIPT_DIR.parent
_SRC = _PROJECT_ROOT / "src"
for _p in (str(_SCRIPT_DIR), str(_SRC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from evaluation.report_generator import ReportGenerator  # noqa: E402
from evaluation.sampling_protocol import write_sample_ids  # noqa: E402
from framework.asset_registry import AssetRegistry  # noqa: E402
from framework.control_gates import evaluate_control_gates, failed_gate_names  # noqa: E402
from framework.control_phase import ControlBlock, ExperimentStage, for_benchmark_output_dir  # noqa: E402
from framework.param_schema import (  # noqa: E402
    AssetsConfig,
    ControlRunConfig,
    EvaluationConfig,
    ReportConfig,
    SamplingConfig,
    validate_control_config,
)
from framework.registry import load_benchmark_adapter, supported_benchmarks  # noqa: E402
from framework.run_summary import (  # noqa: E402
    ControlRunStatus,
    GateSummary,
    StageSummary,
    create_control_run_summary,
    save_summary,
)
from framework.runner import UnifiedExperimentRunner  # noqa: E402
from framework.time_utils import local_timestamp, now_local_iso  # noqa: E402

logger = logging.getLogger("run_paper_experiment")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run formal ResearchOps paper experiments.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    main = sub.add_parser("main", help="Run/assemble formal main experiment artifacts")
    _add_common_main_args(main)

    report = sub.add_parser("report", help="Generate report and run report gate")
    report.add_argument("--run-dir", default="")
    report.add_argument("--table-json", default="")
    report.add_argument("--output-dir", default="")
    report.add_argument("--title", default="Sirchmunk ResearchOps Report")
    report.add_argument("--stage", default="frozen", choices=["exploration", "frozen"])
    report.add_argument("--strict", action="store_true")
    report.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    ablation_spec = sub.add_parser("ablation-spec", help="Create frozen ablation spec/variants only")
    _add_ablation_args(ablation_spec, require_env=False)

    ablation = sub.add_parser("ablation", help="Queue and optionally run frozen ablation variants")
    _add_ablation_args(ablation, require_env=True)
    return parser.parse_args()


def _add_common_main_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--benchmark", "-b", required=True, choices=supported_benchmarks())
    parser.add_argument("--env", "-e", required=True)
    parser.add_argument("--output-dir", default="", help="Benchmark output base")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--stage", default="frozen", choices=["frozen"])
    parser.add_argument("--cache-mode", default="cold", choices=["cold", "compiled"])
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    parser.add_argument("--sampling-method", default="stratified", choices=["simple_random", "stratified", "full", "diagnostic_rare", "fixed_ids"])
    parser.add_argument("--golden-n", type=int, default=500)
    parser.add_argument("--strata", default="type,supporting_fact_bucket")
    parser.add_argument("--sample-ids-file", default="")
    parser.add_argument("--sampling-protocol", default="")
    parser.add_argument("--expected-population-size", type=int, default=0)
    parser.add_argument("--create-golden-only", action="store_true")

    parser.add_argument("--run-sirchmunk", action="store_true", help="Execute frozen Sirchmunk run with UnifiedExperimentRunner")
    parser.add_argument("--sirchmunk-results", default="")
    parser.add_argument("--run-artifact-dir", default="")
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", action="store_false", dest="resume")

    parser.add_argument("--systems", default="sirchmunk")
    parser.add_argument("--baselines", default="")
    parser.add_argument("--asset-registry", default="")
    parser.add_argument("--import-baseline", action="append", dest="import_baseline", metavar="NAME=PATH")
    parser.add_argument("--import-baseline-setup", action="append", dest="import_baseline_setup", metavar="NAME=PATH")
    parser.add_argument("--import-published", action="append", dest="import_published", metavar="'Name:acc=XX,cov=XX,lat=XX'")
    parser.add_argument("--lightrag-predictions", default="")
    parser.add_argument("--lightrag-setup-metrics", default="")
    parser.add_argument("--graphrag-predictions", default="")
    parser.add_argument("--graphrag-setup-metrics", default="")
    parser.add_argument("--table-only", action="store_true")
    parser.add_argument("--ours-name", default="")
    parser.add_argument("--caption", default="")

    parser.add_argument("--baseline-sample-timeout", type=float, default=0.0)
    parser.add_argument("--baseline-max-runtime", type=float, default=0.0)
    parser.add_argument("--baseline-max-total-tokens", type=int, default=0)
    parser.add_argument("--baseline-max-api-cost-usd", type=float, default=0.0)
    parser.add_argument("--baseline-max-disk-bytes", type=int, default=0)
    parser.add_argument("--baseline-min-free-disk-bytes", type=int, default=0)
    parser.add_argument("--generate-report", action="store_true")
    parser.add_argument("--title", default="")


def _add_ablation_args(parser: argparse.ArgumentParser, *, require_env: bool) -> None:
    parser.add_argument("--benchmark", "-b", required=True, choices=supported_benchmarks())
    if require_env:
        parser.add_argument("--env", "-e", required=True)
    else:
        parser.add_argument("--env", "-e", default="")
    parser.add_argument("--output-dir", default="", help="Benchmark output base")
    parser.add_argument("--design", default="orthogonal", choices=["orthogonal", "cartesian"])
    parser.add_argument("--max-combinations", type=int, default=16)
    parser.add_argument("--benchmark-prefix", default="HOTPOT")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stage", default="frozen", choices=["frozen"])
    parser.add_argument("--cache-mode", default="cold", choices=["cold", "compiled"])
    parser.add_argument("--sample-ids-file", default="")
    parser.add_argument("--sampling-protocol", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--queue-path", default="")
    parser.add_argument("--registry-path", default="")
    parser.add_argument("--priority", type=int, default=100)
    parser.add_argument("--task-max-attempts", type=int, default=1)
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--run", action="store_true", help="Execute queued ablation tasks immediately")
    parser.add_argument("--max-concurrent", type=int, default=1)
    parser.add_argument("--max-tasks", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s  %(name)-26s  %(levelname)-7s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


async def _main_experiment(args: argparse.Namespace) -> int:
    _setup_logging(args.log_level)
    env_file = str(Path(args.env).expanduser().resolve())
    if not Path(env_file).exists():
        logger.error("env file does not exist: %s", env_file)
        return 1
    if args.sample_ids_file:
        os.environ["HOTPOT_SAMPLE_IDS_FILE"] = str(Path(args.sample_ids_file).expanduser().resolve())

    adapter = load_benchmark_adapter(args.benchmark, env_file)
    output_base = Path(args.output_dir or adapter.get_output_dir()).expanduser().resolve()
    layout = for_benchmark_output_dir(output_base)
    layout.ensure(blocks=(ControlBlock.MAIN,))
    for path in (layout.main_sampling_dir, layout.main_runs_dir, layout.main_evaluation_dir, layout.main_report_dir):
        path.mkdir(parents=True, exist_ok=True)

    run_id = args.run_id or f"main_{args.benchmark}_{local_timestamp()}"
    config = _control_config(args, output_base=output_base, run_id=run_id)
    validation = validate_control_config(config)
    if not validation.ok:
        print(json.dumps({"validation": validation.to_dict()}, indent=2, ensure_ascii=False))
        return 1
    if args.dry_run:
        print(json.dumps({"dry_run": True, "config": config.to_dict(), "validation": validation.to_dict()}, indent=2, ensure_ascii=False))
        return 0

    sirchmunk_results = Path(args.sirchmunk_results).expanduser().resolve() if args.sirchmunk_results else None
    run_artifact_dir = Path(args.run_artifact_dir).expanduser().resolve() if args.run_artifact_dir else None
    runner_meta: Dict[str, Any] = {}
    if args.run_sirchmunk:
        sirchmunk_results, run_artifact_dir, runner_meta = await _run_sirchmunk(args, adapter, run_id)
    elif sirchmunk_results:
        run_artifact_dir = run_artifact_dir or _infer_run_dir_from_results(sirchmunk_results)
    elif not args.create_golden_only and not args.table_only:
        logger.error("main requires --sirchmunk-results, --run-sirchmunk, --table-only, or --create-golden-only")
        return 1

    if sirchmunk_results and not args.sample_ids_file and not args.sampling_protocol:
        args.sample_ids_file = _write_sample_ids_from_results(
            sirchmunk_results,
            layout.main_sampling_dir / f"{run_id}_actual_sample_ids.json",
            run_id=run_id,
        )
        config = _control_config(args, output_base=output_base, run_id=run_id)
        validation = validate_control_config(config)
        if not validation.ok:
            print(json.dumps({"validation": validation.to_dict()}, indent=2, ensure_ascii=False))
            return 1

    eval_result = _run_evaluation(args, env_file, layout, sirchmunk_results, run_artifact_dir)
    table_json = layout.main_evaluation_dir / "paper_table.json"
    report_dir = layout.main_report_dir if args.generate_report else None
    metrics = _read_json((run_artifact_dir / "results" / "metrics.json") if run_artifact_dir else None)
    registry = AssetRegistry(args.asset_registry) if args.asset_registry and Path(args.asset_registry).exists() else None
    gate_report = evaluate_control_gates(
        config,
        asset_registry=registry,
        run_dir=run_artifact_dir,
        table_json=table_json if table_json.exists() else None,
        metrics=metrics,
    )
    summary = create_control_run_summary(
        control_run_id=run_id,
        benchmark=args.benchmark,
        block=ControlBlock.MAIN,
        stage=ExperimentStage.FROZEN,
        env_file=env_file,
        output_dir=str(output_base),
        paths={
            "main_dir": str(layout.main_dir),
            "sampling_dir": str(layout.main_sampling_dir),
            "evaluation_dir": str(layout.main_evaluation_dir),
            "report_dir": str(layout.main_report_dir),
            "sirchmunk_results": str(sirchmunk_results or ""),
            "run_artifact_dir": str(run_artifact_dir or ""),
            "paper_table_json": str(table_json if table_json.exists() else ""),
        },
        metadata={"runner_meta": runner_meta, "evaluation_exit_code": eval_result},
    )
    summary.status = ControlRunStatus.SUCCESS if eval_result == 0 and gate_report.passed else ControlRunStatus.BLOCKED
    summary.config_hash = runner_meta.get("config_hash", "")
    summary.sample_id_checksum = runner_meta.get("sample_id_checksum", "")
    summary.metrics = metrics
    for gate in gate_report.results:
        summary.add_gate(GateSummary.from_gate_result(gate))
    summary.add_stage(
        StageSummary(
            block=ControlBlock.MAIN.value,
            stage=ExperimentStage.FROZEN.value,
            status=summary.status,
            started_at=now_local_iso(),
            ended_at=now_local_iso(),
            run_ids=[run_id],
            artifact_paths=summary.paths,
            metrics={"evaluation_exit_code": eval_result, "gate_passed": gate_report.passed},
        )
    )
    save_summary(summary, layout.main_summary_path)

    print(json.dumps({
        "run_id": run_id,
        "summary": str(layout.main_summary_path),
        "paper_table_json": str(table_json if table_json.exists() else ""),
        "report_dir": str(report_dir or ""),
        "gates_passed": gate_report.passed,
        "failed_gates": failed_gate_names(gate_report.results),
    }, indent=2, ensure_ascii=False))
    return 1 if args.strict and (eval_result != 0 or not gate_report.passed) else eval_result


def _control_config(args: argparse.Namespace, *, output_base: Path, run_id: str) -> ControlRunConfig:
    return ControlRunConfig(
        benchmark=args.benchmark,
        block=ControlBlock.MAIN,
        stage=ExperimentStage.FROZEN,
        env_file=str(Path(args.env).expanduser().resolve()),
        output_dir=str(output_base),
        run_id=run_id,
        seed=args.seed,
        sampling=SamplingConfig(
            method="fixed_ids" if args.sample_ids_file else args.sampling_method,
            seed=args.seed,
            target_n=0 if args.sampling_method in {"full", "diagnostic_rare", "fixed_ids"} else args.golden_n,
            strata=_split_csv(args.strata) if args.sampling_method == "stratified" else [],
            sample_ids_file=str(Path(args.sample_ids_file).expanduser().resolve()) if args.sample_ids_file else "",
            sampling_protocol=str(Path(args.sampling_protocol).expanduser().resolve()) if args.sampling_protocol else "",
            expected_population_size=args.expected_population_size,
        ),
        assets=AssetsConfig(
            methods=_split_csv(args.baselines),
            asset_registry=str(Path(args.asset_registry).expanduser().resolve()) if args.asset_registry else "",
            reuse_assets=bool(args.asset_registry),
        ),
        evaluation=EvaluationConfig(
            systems=_split_csv(args.systems) or ["sirchmunk"],
            baselines=_split_csv(args.baselines),
            cache_mode=args.cache_mode,
            run_evaluation=True,
            resume=args.resume,
            imported_predictions_dir="",
            table_json="",
            run_dir=str(Path(args.run_artifact_dir).expanduser().resolve()) if args.run_artifact_dir else "",
        ),
        report=ReportConfig(generate=args.generate_report, report_dir="", report_title=args.title),
    )


async def _run_sirchmunk(
    args: argparse.Namespace,
    adapter: Any,
    run_id: str,
) -> Tuple[Path, Path, Dict[str, Any]]:
    runner = UnifiedExperimentRunner(adapter)
    limit = 0 if args.sample_ids_file or args.sampling_method in {"full", "diagnostic_rare", "fixed_ids"} else args.golden_n
    _, meta = await runner.run(
        limit=limit,
        seed=args.seed,
        run_id=run_id,
        resume=args.resume,
        stage=ExperimentStage.FROZEN.value,
        system_name="sirchmunk",
        config_overrides={
            "cache_mode": args.cache_mode,
            "sampling": {
                "method": args.sampling_method,
                "target_n": args.golden_n,
                "sample_ids_file": args.sample_ids_file,
                "sampling_protocol": args.sampling_protocol,
            },
        },
    )
    return Path(meta["results_path"]).resolve(), Path(meta["artifact_dir"]).resolve(), meta


def _run_evaluation(
    args: argparse.Namespace,
    env_file: str,
    layout: Any,
    sirchmunk_results: Optional[Path],
    run_artifact_dir: Optional[Path],
) -> int:
    cmd = [
        sys.executable,
        str(_SCRIPT_DIR / "run_evaluation.py"),
        "--benchmark",
        args.benchmark,
        "--env",
        env_file,
        "--output-dir",
        str(layout.main_evaluation_dir),
        "--golden-seed",
        str(args.seed),
        "--sampling-report-dir",
        str(layout.main_sampling_dir),
        "--expected-population-size",
        str(args.expected_population_size),
        "--baseline-sample-timeout",
        str(args.baseline_sample_timeout),
        "--baseline-max-runtime",
        str(args.baseline_max_runtime),
        "--baseline-max-total-tokens",
        str(args.baseline_max_total_tokens),
        "--baseline-max-api-cost-usd",
        str(args.baseline_max_api_cost_usd),
        "--baseline-max-disk-bytes",
        str(args.baseline_max_disk_bytes),
        "--baseline-min-free-disk-bytes",
        str(args.baseline_min_free_disk_bytes),
    ]
    if sirchmunk_results:
        cmd.extend(["--sirchmunk-results", str(sirchmunk_results)])
    if args.create_golden_only:
        cmd.append("--create-golden-only")
    if args.table_only:
        cmd.append("--table-only")
    if args.ours_name:
        cmd.extend(["--ours-name", args.ours_name])
    if args.caption:
        cmd.extend(["--caption", args.caption])
    _extend_sampling_args(cmd, args)
    if args.baselines:
        cmd.extend(["--baselines", args.baselines])
    for spec in args.import_baseline or []:
        cmd.extend(["--import-baseline", spec])
    for spec in args.import_baseline_setup or []:
        cmd.extend(["--import-baseline-setup", spec])
    for spec in args.import_published or []:
        cmd.extend(["--import-published", spec])
    if args.lightrag_predictions:
        cmd.extend(["--lightrag-predictions", args.lightrag_predictions])
    if args.lightrag_setup_metrics:
        cmd.extend(["--lightrag-setup-metrics", args.lightrag_setup_metrics])
    if args.graphrag_predictions:
        cmd.extend(["--graphrag-predictions", args.graphrag_predictions])
    if args.graphrag_setup_metrics:
        cmd.extend(["--graphrag-setup-metrics", args.graphrag_setup_metrics])
    if args.generate_report:
        cmd.append("--generate-report")
        cmd.extend(["--report-output-dir", str(layout.main_report_dir)])
        if run_artifact_dir:
            cmd.extend(["--run-artifact-dir", str(run_artifact_dir)])
    return _run_cmd(cmd)


def _extend_sampling_args(cmd: List[str], args: argparse.Namespace) -> None:
    if args.sampling_protocol:
        cmd.extend(["--sampling-protocol", str(Path(args.sampling_protocol).expanduser().resolve())])
        return
    if args.sample_ids_file:
        cmd.extend(["--sample-ids-file", str(Path(args.sample_ids_file).expanduser().resolve()), "--sampling-method", "fixed_ids"])
        return
    cmd.extend([
        "--sampling-method",
        args.sampling_method,
        "--golden-n",
        str(args.golden_n),
        "--strata",
        args.strata,
    ])


def _report(args: argparse.Namespace) -> int:
    _setup_logging(args.log_level)
    if not args.run_dir and not args.table_json:
        logger.error("report requires --run-dir or --table-json")
        return 1
    report_out = Path(args.output_dir).expanduser().resolve() if args.output_dir else None
    paths = ReportGenerator().generate(
        run_dir=args.run_dir or None,
        table_json=args.table_json or None,
        output_dir=str(report_out) if report_out else None,
        title=args.title,
    )
    table_json = Path(args.table_json).expanduser().resolve() if args.table_json else None
    run_dir = Path(args.run_dir).expanduser().resolve() if args.run_dir else None
    config = ControlRunConfig(
        benchmark="unknown",
        block=ControlBlock.REPORT,
        stage=ExperimentStage(args.stage),
        report=ReportConfig(generate=True, report_dir=str(report_out or ""), report_title=args.title),
    )
    gate_report = evaluate_control_gates(config, run_dir=run_dir, table_json=table_json)
    summary = create_control_run_summary(
        control_run_id=f"report_{local_timestamp()}",
        benchmark="unknown",
        block=ControlBlock.REPORT,
        stage=ExperimentStage(args.stage),
        paths={**paths, "run_dir": str(run_dir or ""), "table_json": str(table_json or "")},
    )
    summary.status = ControlRunStatus.SUCCESS if gate_report.passed else ControlRunStatus.BLOCKED
    for gate in gate_report.results:
        summary.add_gate(GateSummary.from_gate_result(gate))
    summary_path = Path(paths.get("validation", str(report_out / "report_summary.json" if report_out else "report_summary.json"))).with_name("report_summary.json")
    save_summary(summary, summary_path)
    print(json.dumps({"paths": paths, "summary": str(summary_path), "gates_passed": gate_report.passed}, indent=2, ensure_ascii=False))
    return 1 if args.strict and not gate_report.passed else 0


def _ablation_spec(args: argparse.Namespace) -> int:
    _setup_logging(args.log_level)
    layout, spec_payload, variants = _write_ablation_spec(args)
    print(json.dumps({
        "ablation_spec": str(layout.ablation_spec_path),
        "variants": str(layout.ablation_variants_path),
        "variant_count": len(variants),
        "note": "Use `run_paper_experiment.py ablation` or `run_benchmark.py ablation` to enqueue/run these variants.",
    }, indent=2, ensure_ascii=False))
    return 0


async def _ablation(args: argparse.Namespace) -> int:
    _setup_logging(args.log_level)
    if args.sample_ids_file:
        os.environ["HOTPOT_SAMPLE_IDS_FILE"] = str(Path(args.sample_ids_file).expanduser().resolve())
    layout, spec_payload, variants = _write_ablation_spec(args)
    queue_path = Path(args.queue_path).expanduser().resolve() if args.queue_path else layout.queue_dir / "ablation_queue.json"
    registry_path = Path(args.registry_path).expanduser().resolve() if args.registry_path else layout.queue_dir / "ablation_registry.jsonl"
    if args.dry_run:
        print(json.dumps({
            "dry_run": True,
            "ablation_spec": str(layout.ablation_spec_path),
            "variants": [variant.to_dict() for variant in variants],
            "queue_path": str(queue_path),
            "registry_path": str(registry_path),
        }, indent=2, ensure_ascii=False))
        return 0

    from framework.experiment_queue import ExperimentQueue, QueueExecutor, QueueTask
    from framework.experiment_registry import ExperimentRegistry

    queue = ExperimentQueue(queue_path)
    for index, variant in enumerate(variants):
        overrides = dict(variant.config_overrides)
        overrides.update({
            "cache_mode": args.cache_mode,
            "stage": ExperimentStage.FROZEN.value,
            "frozen_evaluation": True,
            "ablation_variant_id": variant.variant_id,
            "ablation_label": variant.label,
        })
        if args.sample_ids_file:
            overrides["HOTPOT_SAMPLE_IDS_FILE"] = str(Path(args.sample_ids_file).expanduser().resolve())
        if args.sampling_protocol:
            overrides["sampling_protocol"] = str(Path(args.sampling_protocol).expanduser().resolve())
        task = QueueTask(
            task_id=f"ablation_{args.benchmark}_{variant.variant_id}",
            benchmark=args.benchmark,
            env_file=str(Path(args.env).expanduser().resolve()),
            system="sirchmunk",
            seed=args.seed,
            cache_mode=args.cache_mode,
            stage=ExperimentStage.FROZEN.value,
            limit=args.limit,
            priority=args.priority + index,
            run_id=f"ablation_{args.benchmark}_{variant.variant_id}",
            kind="sirchmunk",
            config_overrides=overrides,
            max_attempts=max(int(args.task_max_attempts or 1), 1),
            meta={"variant": variant.to_dict(), "ablation_spec": spec_payload},
        )
        queue.add_task(task, replace=args.replace)
    run_results = []
    if args.run:
        registry = ExperimentRegistry(registry_path)
        executor = QueueExecutor(queue, registry=registry, max_concurrent=args.max_concurrent, resume=True)
        run_results = await executor.run_pending(max_tasks=args.max_tasks)
    summary = {
        "ablation_spec": str(layout.ablation_spec_path),
        "variants": str(layout.ablation_variants_path),
        "queue_path": str(queue_path),
        "registry_path": str(registry_path),
        "variant_count": len(variants),
        "queue_summary": queue.summary(),
        "run_results": [task.to_dict() for task in run_results],
    }
    summary_path = layout.ablation_dir / "ablation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({**summary, "summary": str(summary_path)}, indent=2, ensure_ascii=False))
    return 0


def _write_ablation_spec(args: argparse.Namespace):
    if args.output_dir:
        output_base = Path(args.output_dir).expanduser().resolve()
    else:
        output_base = (_SCRIPT_DIR / args.benchmark / "output").resolve()
    layout = for_benchmark_output_dir(output_base)
    layout.ensure(blocks=(ControlBlock.ABLATION,))
    layout.queue_dir.mkdir(parents=True, exist_ok=True)
    layout.ablation_dir.mkdir(parents=True, exist_ok=True)

    from framework.ablation_matrix import default_lens_ablation_spec

    spec = default_lens_ablation_spec(benchmark_prefix=args.benchmark_prefix)
    spec.design = args.design
    spec.max_combinations = args.max_combinations
    variants = spec.generate()
    spec_payload = {
        "name": spec.name,
        "design": spec.design,
        "axes": [axis.to_dict() for axis in spec.axes],
        "baseline": spec.baseline,
        "max_combinations": spec.max_combinations,
        "metadata": spec.metadata,
    }
    layout.ablation_spec_path.write_text(json.dumps(spec_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    layout.ablation_variants_path.write_text(json.dumps([v.to_dict() for v in variants], indent=2, ensure_ascii=False), encoding="utf-8")
    return layout, spec_payload, variants


def _run_cmd(cmd: List[str]) -> int:
    logger.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, text=True, check=False)
    return int(result.returncode)


def _write_sample_ids_from_results(results_path: Path, output_path: Path, *, run_id: str) -> str:
    sample_ids: List[str] = []
    with Path(results_path).open(encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            sample_id = row.get("sample_id") or row.get("hotpot_id") or row.get("id")
            if sample_id:
                sample_ids.append(str(sample_id))
    if not sample_ids:
        raise ValueError(f"No sample IDs found in {results_path}")
    write_sample_ids(
        output_path,
        sample_ids,
        metadata={"source_results": str(Path(results_path).resolve()), "run_id": run_id},
    )
    logger.info("Persisted %d actual sample IDs for evaluation: %s", len(sample_ids), output_path)
    return str(output_path.resolve())


def _infer_run_dir_from_results(results_path: Path) -> Optional[Path]:
    parent = results_path.parent
    if parent.name == "results" and parent.parent.exists():
        return parent.parent
    candidate = parent.parent if parent.name == "results" else parent
    return candidate if (candidate / "manifest.json").exists() else None


def _read_json(path: Optional[Path]) -> Dict[str, Any]:
    if not path or not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _split_csv(raw: str) -> List[str]:
    return [part.strip() for part in str(raw or "").split(",") if part.strip()]


def main() -> None:
    args = _parse_args()
    if args.command == "main":
        raise SystemExit(asyncio.run(_main_experiment(args)))
    if args.command == "report":
        raise SystemExit(_report(args))
    if args.command == "ablation-spec":
        raise SystemExit(_ablation_spec(args))
    if args.command == "ablation":
        raise SystemExit(asyncio.run(_ablation(args)))
    raise SystemExit(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
