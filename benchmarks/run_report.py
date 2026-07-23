#!/usr/bin/env python3
"""Generate a metric-first ResearchOps report from existing artifacts."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).parent.resolve()
_PROJECT_ROOT = _SCRIPT_DIR.parent
_SRC = _PROJECT_ROOT / "src"
for _p in (str(_SCRIPT_DIR), str(_SRC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from evaluation.report_generator import ReportGenerator  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate ResearchOps academic report")
    parser.add_argument("--run-dir", default="", help="Run artifact directory")
    parser.add_argument("--table-json", default="", help="Paper table JSON path")
    parser.add_argument("--output-dir", default="", help="Report output directory")
    parser.add_argument("--title", default="Sirchmunk ResearchOps Report", help="Report title")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not args.run_dir and not args.table_json:
        raise SystemExit("--run-dir or --table-json is required")
    paths = ReportGenerator().generate(
        run_dir=args.run_dir or None,
        table_json=args.table_json or None,
        output_dir=args.output_dir or None,
        title=args.title,
    )
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
