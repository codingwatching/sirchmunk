#!/usr/bin/env python3
"""Unified ResearchOps benchmark control facade for P1.

The facade intentionally delegates to focused entry points instead of becoming a
monolithic experiment engine:

- assets      -> run_baseline_assets.py prepare/validate/status
- smoke-tune  -> run_quickstart.py
- main        -> run_paper_experiment.py main
- ablation    -> run_paper_experiment.py ablation
- queue       -> run_queue.py
- report      -> run_paper_experiment.py report
- status      -> local summary/registry inspection
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import List

_SCRIPT_DIR = Path(__file__).parent.resolve()

_BLOCKS = {"assets", "smoke-tune", "smoke", "tune", "main", "ablation", "report", "status", "queue"}
_ASSET_SUBCOMMANDS = {"prepare", "validate", "status", "scaling", "update-readiness"}


def _usage() -> str:
    return """Usage:
  python benchmarks/run_benchmark.py <block> [options]
  python benchmarks/run_benchmark.py --block <block> [options]

Blocks:
  assets      Build/validate baseline assets and asset_registry.jsonl
  smoke-tune  Run quickstart smoke/tuning flow
  main        Run/assemble formal main experiment artifacts
  ablation    Queue/run frozen ablation variants
  queue       Access the lower-level experiment queue
  report      Generate report/table artifacts from existing outputs
  status      Inspect a run_summary.json or asset_registry.jsonl

Examples:
  python benchmarks/run_benchmark.py assets --benchmark hotpotqa --env benchmarks/hotpotqa/.env.hotpotqa.frozen --methods bm25_rag
  python benchmarks/run_benchmark.py smoke-tune --benchmark hotpotqa --env benchmarks/hotpotqa/.env.hotpotqa.exploration --limit 20
  python benchmarks/run_benchmark.py main --benchmark hotpotqa --env benchmarks/hotpotqa/.env.hotpotqa.frozen --sirchmunk-results output/results.jsonl --generate-report
"""


def _parse_block(argv: List[str]) -> tuple[str, List[str]]:
    if not argv or argv[0] in {"-h", "--help"}:
        print(_usage())
        raise SystemExit(0)
    if argv[0] == "--block":
        if len(argv) < 2:
            raise SystemExit("--block requires a value")
        block = argv[1]
        rest = argv[2:]
    elif argv[0].startswith("--block="):
        block = argv[0].split("=", 1)[1]
        rest = argv[1:]
    else:
        block = argv[0]
        rest = argv[1:]
    block = block.strip().lower()
    if block not in _BLOCKS:
        raise SystemExit(f"Unknown block: {block}\n\n{_usage()}")
    if block in {"smoke", "tune"}:
        block = "smoke-tune"
    return block, rest


def _status(rest: List[str]) -> int:
    parser = argparse.ArgumentParser(description="Inspect ResearchOps control status")
    parser.add_argument("--summary", default="", help="ControlRunSummary JSON path")
    parser.add_argument("--asset-registry", default="", help="Asset registry JSONL path")
    parser.add_argument("--benchmark", default="")
    parser.add_argument("--method", default="")
    parser.add_argument("--methods", default="")
    parser.add_argument("--stage", default="")
    parser.add_argument("--status", default="")
    parser.add_argument("--type", default="")
    parser.add_argument("--corpus-hash", default="")
    parser.add_argument("--config-hash", default="")
    parser.add_argument("--reusable-only", action="store_true")
    args = parser.parse_args(rest)
    if args.asset_registry:
        cmd = [
            sys.executable,
            str(_SCRIPT_DIR / "run_baseline_assets.py"),
            "status",
            "--asset-registry",
            args.asset_registry,
        ]
        for key in ("benchmark", "method", "methods", "stage", "status", "type", "corpus_hash", "config_hash"):
            value = getattr(args, key)
            if value:
                cmd.extend(["--" + key.replace("_", "-"), value])
        if args.reusable_only:
            cmd.append("--reusable-only")
        return _run(cmd)
    if args.summary:
        path = Path(args.summary).expanduser().resolve()
        if not path.exists():
            raise SystemExit(f"summary does not exist: {path}")
        data = json.loads(path.read_text(encoding="utf-8"))
        compact = {
            "summary": str(path),
            "control_run_id": data.get("control_run_id"),
            "benchmark": data.get("benchmark"),
            "block": data.get("block"),
            "stage": data.get("stage"),
            "status": data.get("status"),
            "blocked": data.get("blocked"),
            "paper_ready": data.get("paper_ready"),
            "failed_gates": [
                gate.get("name")
                for gate in data.get("gates", [])
                if gate.get("blocking") and not gate.get("passed")
            ],
            "paths": data.get("paths", {}),
        }
        print(json.dumps(compact, indent=2, ensure_ascii=False))
        return 0
    print(_usage())
    return 0


def _command_for(block: str, rest: List[str]) -> List[str]:
    if block == "assets":
        if rest and rest[0] in _ASSET_SUBCOMMANDS:
            return [sys.executable, str(_SCRIPT_DIR / "run_baseline_assets.py"), rest[0], *rest[1:]]
        return [sys.executable, str(_SCRIPT_DIR / "run_baseline_assets.py"), "prepare", *rest]
    if block == "smoke-tune":
        return [sys.executable, str(_SCRIPT_DIR / "run_quickstart.py"), *rest]
    if block == "main":
        return [sys.executable, str(_SCRIPT_DIR / "run_paper_experiment.py"), "main", *rest]
    if block == "ablation":
        if rest and rest[0] in {"ablation", "ablation-spec"}:
            return [sys.executable, str(_SCRIPT_DIR / "run_paper_experiment.py"), *rest]
        return [sys.executable, str(_SCRIPT_DIR / "run_paper_experiment.py"), "ablation", *rest]
    if block == "queue":
        return [sys.executable, str(_SCRIPT_DIR / "run_queue.py"), *rest]
    if block == "report":
        return [sys.executable, str(_SCRIPT_DIR / "run_paper_experiment.py"), "report", *rest]
    raise ValueError(block)


def _run(cmd: List[str]) -> int:
    print("$ " + " ".join(cmd))
    return int(subprocess.run(cmd, check=False).returncode)


def main() -> None:
    block, rest = _parse_block(sys.argv[1:])
    if block == "status":
        raise SystemExit(_status(rest))
    raise SystemExit(_run(_command_for(block, rest)))


if __name__ == "__main__":
    main()
