#!/usr/bin/env python3
"""One-command benchmark quickstart.

Default behavior runs a small HotpotQA exploration smoke test with the configured
real LLM provider, then generates a ResearchOps report from the produced run
artifact.  It is intended for setup validation, not for paper claims.
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

from framework.registry import load_benchmark_adapter, supported_benchmarks  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a one-command benchmark smoke test and report.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--benchmark", "-b", default="hotpotqa", choices=supported_benchmarks())
    parser.add_argument("--env", "-e", default="benchmarks/hotpotqa/.env.hotpotqa.exploration")
    parser.add_argument("--limit", "-l", type=int, default=None, help="Sample limit. Omit to use benchmark profile env, e.g. HOTPOT_LIMIT; fallback is 10.")
    parser.add_argument("--max-iter", "-n", type=int, default=1)
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--title", default="HotpotQA Quickstart Smoke Test Report")
    parser.add_argument("--context-corpus-mode", default="sample", choices=["sample", "wiki", "hybrid"], help="HotpotQA search corpus mode for quickstart.")
    parser.add_argument("--skip-report", action="store_true", help="Only run smoke test; do not generate report.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    env_path = Path(args.env).expanduser().resolve()
    if not env_path.exists():
        _raise_missing_env(env_path)
    _preflight_dependencies(args.benchmark)

    adapter = load_benchmark_adapter(args.benchmark, str(env_path))
    limit = _resolve_limit(args.limit, adapter, default=10)
    output_dir = Path(adapter.get_output_dir()).resolve()
    before = _latest_run_dir(output_dir)

    print("== Benchmarks Quickstart ==")
    print(f"benchmark : {args.benchmark}")
    print(f"env       : {env_path}")
    print(f"limit     : {limit}")
    if args.benchmark == "hotpotqa":
        print(f"corpus    : {args.context_corpus_mode}")
    print()

    child_env = os.environ.copy()
    if args.benchmark == "hotpotqa":
        child_env["HOTPOT_LIMIT"] = str(limit)
        child_env["HOTPOT_CONTEXT_CORPUS_MODE"] = args.context_corpus_mode
        if args.context_corpus_mode == "sample":
            child_env["HOTPOT_REQUIRE_CONTEXT_ANSWERABLE"] = "true"

    loop_cmd = [
        sys.executable,
        str(_SCRIPT_DIR / "run_research_loop.py"),
        "--benchmark",
        args.benchmark,
        "--env",
        str(env_path),
        "--limit",
        str(limit),
        "--max-iter",
        str(args.max_iter),
        "--dry-run",
        "--log-level",
        args.log_level,
    ]
    print("[1/2] Running smoke test...")
    _run_step(loop_cmd, input_text="skip\nn\n", step_name="smoke test", env=child_env)

    run_dir = _latest_run_dir(output_dir)
    if run_dir is None or run_dir == before:
        raise SystemExit(f"No new run artifact found under {output_dir / 'runs'}")

    report_paths: Dict[str, str] = {}
    if not args.skip_report:
        print("[2/2] Generating report...")
        report_dir = run_dir / "reports"
        report_cmd = [
            sys.executable,
            str(_SCRIPT_DIR / "run_report.py"),
            "--run-dir",
            str(run_dir),
            "--output-dir",
            str(report_dir),
            "--title",
            args.title,
        ]
        _run_step(report_cmd, step_name="report generation", env=child_env)
        report_paths = {
            "report_md": str(report_dir / "report.md"),
            "report_tex": str(report_dir / "report.tex"),
            "validation": str(report_dir / "validation.json"),
        }

    metrics = _read_json(run_dir / "results" / "metrics.json")
    validation = _read_json(Path(report_paths.get("validation", ""))) if report_paths else {}
    _print_summary(run_dir, metrics, report_paths, validation)


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


def _raise_missing_env(env_path: Path) -> None:
    rel = env_path
    example = env_path.with_name(env_path.name.lstrip(".") + ".example")
    if env_path.name.startswith(".env.hotpotqa."):
        profile = env_path.name.replace(".env.hotpotqa.", "")
        example = env_path.with_name(f"env.hotpotqa.{profile}.example")
    raise SystemExit(
        f"Env file not found: {rel}\n"
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


def _read_json(path: Path) -> Dict:
    if not path or not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _print_summary(run_dir: Path, metrics: Dict, report_paths: Dict[str, str], validation: Dict) -> None:
    print()
    print("== Quickstart Summary ==")
    print(f"run_dir   : {run_dir}")
    print(f"samples   : {metrics.get('n', 'N/A')}")
    print(f"accuracy  : {metrics.get('accuracy', 'N/A')}")
    print(f"coverage  : {metrics.get('coverage', 'N/A')}")
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
    )
    print(f"sys_fail  : {system_failures}")
    print(f"quickstart_ok: {quickstart_ok}")
    if report_paths:
        print(f"report.md : {report_paths['report_md']}")
        print(f"report.tex: {report_paths['report_tex']}")
        print(f"validation: {report_paths['validation']}")
        print(f"paper_ready: {validation.get('passed', 'N/A')}")


if __name__ == "__main__":
    main()
