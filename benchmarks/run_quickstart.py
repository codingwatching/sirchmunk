#!/usr/bin/env python3
"""One-command benchmark quickstart.

The default path runs a small benchmark smoke test with the configured runtime,
then generates a ResearchOps report from the produced run artifact.  Optional
``--run-evaluation`` turns the same run into a reproducible baseline-comparison
smoke by freezing the produced sample IDs and calling ``run_evaluation.py``.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional

_SCRIPT_DIR = Path(__file__).parent.resolve()
_PROJECT_ROOT = _SCRIPT_DIR.parent
_SRC = _PROJECT_ROOT / "src"
for _p in (str(_SCRIPT_DIR), str(_SRC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from evaluation.sampling_protocol import write_sample_ids  # noqa: E402
from framework.registry import load_benchmark_adapter, supported_benchmarks  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a one-command benchmark smoke test, report, and optional baseline evaluation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--benchmark", "-b", default="hotpotqa", choices=supported_benchmarks())
    parser.add_argument("--env", "-e", default="", help="Benchmark env file. Defaults to the benchmark quickstart profile when available.")
    parser.add_argument("--limit", "-l", type=int, default=None, help="Smoke sample limit. Omit to use benchmark profile env; fallback is 10.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-iter", "-n", type=int, default=1)
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--title", default="", help="Report title. Defaults to '<BENCHMARK> Quickstart Smoke Test Report'.")
    parser.add_argument("--context-corpus-mode", default="sample", choices=["sample", "wiki", "hybrid"], help="HotpotQA-only quickstart corpus mode.")
    parser.add_argument("--skip-smoke", action="store_true", help="Skip run_research_loop.py and use --sirchmunk-results for evaluation/reporting.")
    parser.add_argument("--skip-report", action="store_true", help="Only run smoke test; do not generate run report.")
    parser.add_argument("--run-evaluation", action="store_true", help="Generate a baseline comparison table from the smoke run or --sirchmunk-results.")
    parser.add_argument("--sirchmunk-results", default="", help="Existing Sirchmunk predictions JSONL for --skip-smoke or evaluation mode.")
    parser.add_argument("--evaluation-output-dir", default="", help="Output directory for run_evaluation.py artifacts.")
    parser.add_argument("--evaluation-golden-n", type=int, default=None, help="GoldenSet size for evaluation when no fixed sample IDs are used.")
    parser.add_argument("--sampling-method", default="fixed_ids", choices=["fixed_ids", "simple_random", "stratified", "full", "diagnostic_rare"])
    parser.add_argument("--sampling-protocol", default="", help="Frozen sampling protocol JSON for run_evaluation.py.")
    parser.add_argument("--sample-ids-file", default="", help="Existing sample IDs JSON. If omitted, quickstart writes one from the smoke results.")
    parser.add_argument("--strata", default="type,supporting_fact_bucket")
    parser.add_argument("--baselines", default="", help="Comma-separated smoke baselines passed to run_evaluation.py. bm25/naive_rag are quickstart-local baselines; use bm25_rag,hybrid_rag,react for paper-oriented checks. Long-context is intentionally unsupported in the current scope.")
    parser.add_argument("--import-baseline", action="append", dest="import_baseline", metavar="NAME=PATH")
    parser.add_argument("--import-baseline-setup", action="append", dest="import_baseline_setup", metavar="NAME=PATH")
    parser.add_argument("--import-published", action="append", dest="import_published", metavar="'Name:acc=XX,cov=XX,lat=XX'")
    parser.add_argument("--lightrag-predictions", default="")
    parser.add_argument("--lightrag-setup-metrics", default="")
    parser.add_argument("--graphrag-predictions", default="")
    parser.add_argument("--graphrag-setup-metrics", default="")
    parser.add_argument("--baseline-sample-timeout", type=float, default=0.0)
    parser.add_argument("--baseline-max-runtime", type=float, default=0.0)
    parser.add_argument("--generate-evaluation-report", action="store_true", help="Ask run_evaluation.py to generate a metric-first comparison report.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    env_path = _resolve_env_path(args)
    _preflight_dependencies(args.benchmark)

    adapter = load_benchmark_adapter(args.benchmark, str(env_path))
    limit = _resolve_limit(args.limit, adapter, default=10)
    output_dir = Path(adapter.get_output_dir()).resolve()
    before = _latest_run_dir(output_dir)
    title = args.title or f"{args.benchmark.upper()} Quickstart Smoke Test Report"

    print("== Benchmarks Quickstart ==")
    print(f"benchmark : {args.benchmark}")
    print(f"env       : {env_path}")
    print(f"limit     : {limit}")
    print(f"seed      : {args.seed}")
    if args.benchmark == "hotpotqa":
        print(f"corpus    : {args.context_corpus_mode}")
        if args.context_corpus_mode == "sample" and args.run_evaluation:
            print("warning   : sample-context smoke uses answerable parquet context; baseline scores are health checks, not raw-corpus retrieval claims")
    print()

    child_env = _child_env(args, limit)
    run_dir: Optional[Path] = None
    predictions_path = Path(args.sirchmunk_results).expanduser().resolve() if args.sirchmunk_results else None

    if not args.skip_smoke:
        loop_cmd = [
            sys.executable,
            str(_SCRIPT_DIR / "run_research_loop.py"),
            "--benchmark",
            args.benchmark,
            "--env",
            str(env_path),
            "--limit",
            str(limit),
            "--seed",
            str(args.seed),
            "--max-iter",
            str(args.max_iter),
            "--dry-run",
            "--log-level",
            args.log_level,
        ]
        print("[1/3] Running smoke test...")
        _run_step(loop_cmd, input_text="skip\nn\n", step_name="smoke test", env=child_env)
        run_dir = _latest_run_dir(output_dir)
        if run_dir is None or run_dir == before:
            raise SystemExit(f"No new run artifact found under {output_dir / 'runs'}")
        predictions_path = run_dir / "results" / "predictions.jsonl"
    else:
        if not predictions_path:
            raise SystemExit("--skip-smoke requires --sirchmunk-results when --run-evaluation is needed.")
        run_dir = _infer_run_dir_from_results(predictions_path)

    report_paths: Dict[str, str] = {}
    if run_dir and not args.skip_report:
        print("[2/3] Generating run report...")
        report_dir = run_dir / "reports"
        report_cmd = [
            sys.executable,
            str(_SCRIPT_DIR / "run_report.py"),
            "--run-dir",
            str(run_dir),
            "--output-dir",
            str(report_dir),
            "--title",
            title,
        ]
        _run_step(report_cmd, step_name="report generation", env=child_env)
        report_paths = {
            "report_md": str(report_dir / "report.md"),
            "report_tex": str(report_dir / "report.tex"),
            "validation": str(report_dir / "validation.json"),
        }

    evaluation_paths: Dict[str, str] = {}
    if args.run_evaluation:
        if not predictions_path or not predictions_path.exists():
            raise SystemExit("run_evaluation requires a Sirchmunk predictions JSONL from smoke run or --sirchmunk-results.")
        print("[3/3] Generating baseline comparison table...")
        evaluation_paths = _run_evaluation(args, env_path, predictions_path, run_dir, limit, child_env)

    metrics = _read_json(run_dir / "results" / "metrics.json") if run_dir else {}
    validation = _read_json(Path(report_paths.get("validation", ""))) if report_paths else {}
    _print_summary(run_dir, metrics, report_paths, validation, evaluation_paths)


def _resolve_env_path(args: argparse.Namespace) -> Path:
    if args.env:
        env_path = Path(args.env).expanduser().resolve()
        if env_path.exists() or args.benchmark != "hotpotqa":
            return env_path
        _raise_missing_env(env_path, args.benchmark)
    if args.benchmark == "hotpotqa":
        env_path = _SCRIPT_DIR / "hotpotqa" / ".env.hotpotqa.exploration"
        if env_path.exists():
            return env_path.resolve()
        _raise_missing_env(env_path, args.benchmark)
    return (_SCRIPT_DIR / args.benchmark / f".env.{args.benchmark}").resolve()


def _child_env(args: argparse.Namespace, limit: int) -> dict[str, str]:
    child_env = os.environ.copy()
    if args.sample_ids_file:
        child_env["HOTPOT_SAMPLE_IDS_FILE"] = str(Path(args.sample_ids_file).expanduser().resolve())
    if args.benchmark == "hotpotqa":
        child_env["HOTPOT_LIMIT"] = str(limit)
        child_env["HOTPOT_CONTEXT_CORPUS_MODE"] = args.context_corpus_mode
        child_env["HOTPOT_CONTEXT_CORPUS_PROVENANCE"] = args.context_corpus_mode
        child_env["HOTPOT_CONTEXT_CORPUS_RISK"] = _context_corpus_risk(args)
        if args.context_corpus_mode == "sample":
            child_env["HOTPOT_REQUIRE_CONTEXT_ANSWERABLE"] = "true"
    return child_env


def _context_corpus_risk(args: argparse.Namespace) -> str:
    if args.benchmark != "hotpotqa":
        return ""
    if args.context_corpus_mode == "sample":
        return "oracle_sample_context,evaluation_set_context_index"
    if args.context_corpus_mode == "hybrid":
        return "sample_context_plus_raw_wiki"
    return "raw_wiki"


def _run_evaluation(
    args: argparse.Namespace,
    env_path: Path,
    predictions_path: Path,
    run_dir: Optional[Path],
    limit: int,
    child_env: dict[str, str],
) -> Dict[str, str]:
    eval_out = Path(args.evaluation_output_dir).expanduser().resolve() if args.evaluation_output_dir else Path(load_benchmark_adapter(args.benchmark, str(env_path)).get_output_dir()).resolve() / "quickstart_eval"
    eval_out.mkdir(parents=True, exist_ok=True)
    sample_ids_file = Path(args.sample_ids_file).expanduser().resolve() if args.sample_ids_file else eval_out / "quickstart_sample_ids.json"
    if not args.sample_ids_file:
        _write_sample_ids_from_results(predictions_path, sample_ids_file)

    cmd = [
        sys.executable,
        str(_SCRIPT_DIR / "run_evaluation.py"),
        "--benchmark",
        args.benchmark,
        "--env",
        str(env_path),
        "--sirchmunk-results",
        str(predictions_path),
        "--output-dir",
        str(eval_out),
        "--golden-seed",
        str(args.seed),
        "--baseline-sample-timeout",
        str(args.baseline_sample_timeout),
        "--baseline-max-runtime",
        str(args.baseline_max_runtime),
        "--context-corpus-provenance",
        args.context_corpus_mode,
        "--context-corpus-risk",
        _context_corpus_risk(args),
    ]
    if args.sampling_protocol:
        cmd.extend(["--sampling-protocol", args.sampling_protocol])
    elif args.sampling_method == "fixed_ids":
        cmd.extend(["--sample-ids-file", str(sample_ids_file), "--sampling-method", "fixed_ids"])
    else:
        golden_n = args.evaluation_golden_n if args.evaluation_golden_n is not None else limit
        cmd.extend(["--sampling-method", args.sampling_method, "--golden-n", str(golden_n), "--strata", args.strata])
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
    if args.generate_evaluation_report:
        cmd.append("--generate-report")
        if run_dir:
            cmd.extend(["--run-artifact-dir", str(run_dir)])
    _run_step(cmd, step_name="baseline evaluation", env=child_env)
    out = {
        "paper_table_json": str(eval_out / "paper_table.json"),
        "paper_table_md": str(eval_out / "paper_table.md"),
        "paper_table_tex": str(eval_out / "paper_table.tex"),
        "sample_ids": str(sample_ids_file),
    }
    if args.generate_evaluation_report:
        out["evaluation_report"] = str(eval_out / "report" / "report.md")
    return out


def _write_sample_ids_from_results(results_path: Path, output_path: Path) -> None:
    sample_ids = []
    with results_path.open(encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            sample_id = row.get("sample_id") or row.get("hotpot_id") or row.get("id")
            if sample_id:
                sample_ids.append(str(sample_id))
    if not sample_ids:
        raise SystemExit(f"No sample IDs found in {results_path}")
    write_sample_ids(output_path, sample_ids, metadata={"source_results": str(results_path)})


def _resolve_limit(cli_limit: Optional[int], adapter: Any, *, default: int) -> int:
    if cli_limit is not None:
        return max(int(cli_limit), 0)
    getter = getattr(adapter, "get_profile_limit", None)
    if callable(getter):
        return max(int(getter(default)), 0)
    return default


def _run_step(
    command: list[str],
    *,
    step_name: str,
    input_text: str | None = None,
    env: dict[str, str] | None = None,
) -> None:
    try:
        subprocess.run(
            command,
            input=input_text,
            text=input_text is not None,
            check=True,
            env=env,
        )
    except subprocess.CalledProcessError as exc:
        raise SystemExit(
            f"Quickstart {step_name} failed with exit code {exc.returncode}. "
            "See the logs above for the root cause."
        ) from exc


def _preflight_dependencies(benchmark: str) -> None:
    if benchmark != "hotpotqa":
        return
    required = {
        "pandas": "pandas",
        "pyarrow": "pyarrow",
    }
    missing = [
        package
        for module, package in required.items()
        if importlib.util.find_spec(module) is None
    ]
    if not missing:
        return
    missing_text = ", ".join(missing)
    raise SystemExit(
        "Missing dependencies for HotpotQA parquet loading: "
        f"{missing_text}\n"
        "Install benchmark dependencies with:\n"
        "  pip install -r requirements/core.txt -r requirements/benchmarks.txt\n"
        "or, minimally:\n"
        f"  pip install {' '.join(missing)}"
    )


def _raise_missing_env(env_path: Path, benchmark: str) -> None:
    example = env_path.with_name(env_path.name.lstrip(".") + ".example")
    if benchmark == "hotpotqa" and env_path.name.startswith(".env.hotpotqa."):
        profile = env_path.name.replace(".env.hotpotqa.", "")
        example = env_path.with_name(f"env.hotpotqa.{profile}.example")
    raise SystemExit(
        f"Env file not found: {env_path}\n"
        f"Create it first, e.g.: cp {example} {env_path}"
    )


def _latest_run_dir(output_dir: Path) -> Optional[Path]:
    runs_dir = output_dir / "runs"
    if not runs_dir.exists():
        return None
    candidates = [p for p in runs_dir.iterdir() if p.is_dir() and (p / "manifest.json").exists()]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _infer_run_dir_from_results(results_path: Path) -> Optional[Path]:
    try:
        if results_path.parent.name == "results" and (results_path.parents[1] / "manifest.json").exists():
            return results_path.parents[1]
    except IndexError:
        return None
    return None


def _read_json(path: Path) -> Dict:
    if not path or not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _print_summary(
    run_dir: Optional[Path],
    metrics: Dict,
    report_paths: Dict[str, str],
    validation: Dict,
    evaluation_paths: Dict[str, str],
) -> None:
    print()
    print("== Quickstart Summary ==")
    print(f"run_dir   : {run_dir or 'N/A'}")
    print(f"samples   : {metrics.get('n', 'N/A')}")
    print(f"accuracy  : {metrics.get('accuracy', 'N/A')}")
    print(f"llm_acc   : {metrics.get('llm_assisted_accuracy', 'N/A')}")
    print(f"official_em: {metrics.get('official_exact_match', 'N/A')}")
    print(f"official_f1_correct: {metrics.get('official_f1_correct', 'N/A')}")
    print(f"official_f1: {metrics.get('f1', 'N/A')}")
    print(f"coverage  : {metrics.get('coverage', 'N/A')}")
    print(f"evidence_recall: {metrics.get('evidence_recall', 'N/A')}")
    if "target_slot_verification_rate" in metrics:
        print(f"target_slot_verify: {metrics.get('target_slot_verification_rate')}")
    print(f"source_grounding: {metrics.get('source_grounding_accuracy', 'N/A')}")
    failure = metrics.get("failure_classification", {}) if isinstance(metrics, dict) else {}
    system_failures = failure.get("system_failures", 0)
    try:
        coverage_value = float(metrics.get("coverage", 0) or 0)
    except (TypeError, ValueError):
        coverage_value = 0.0
    quickstart_ok = (
        metrics.get("n", 0) not in (0, "N/A")
        and system_failures == 0
        and coverage_value > 0
    ) if metrics else "N/A"
    print(f"sys_fail  : {system_failures}")
    qgate = metrics.get("quality_gate") if isinstance(metrics.get("quality_gate"), dict) else {}
    if qgate:
        print(f"pipeline_ok: {qgate.get('pipeline_ok', qgate.get('quality_ok'))}")
        print(f"quality_ok: {qgate.get('quality_ok')}")
        if qgate.get("failed_pipeline_checks"):
            print(f"pipeline_failed: {qgate.get('failed_pipeline_checks')}")
        if qgate.get("failed_quality_checks"):
            print(f"quality_failed: {qgate.get('failed_quality_checks')}")
        elif qgate.get("failed_checks"):
            print(f"quality_failed: {qgate.get('failed_checks')}")
    print(f"quickstart_ok: {quickstart_ok}")
    if report_paths:
        print(f"report.md : {report_paths['report_md']}")
        print(f"report.tex: {report_paths['report_tex']}")
        print(f"validation: {report_paths['validation']}")
        print(f"paper_ready: {validation.get('passed', 'N/A')}")
    if evaluation_paths:
        print(f"table.json: {evaluation_paths['paper_table_json']}")
        print(f"sample_ids: {evaluation_paths['sample_ids']}")
        if evaluation_paths.get("evaluation_report"):
            print(f"eval_report: {evaluation_paths['evaluation_report']}")


if __name__ == "__main__":
    main()
