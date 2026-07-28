#!/usr/bin/env python3
"""benchmarks/run_sampling.py — paper-grade sampling protocol CLI.

Examples::

    python benchmarks/run_sampling.py describe \
      --benchmark hotpotqa \
      --env benchmarks/hotpotqa/.env.hotpotqa.frozen

    python benchmarks/run_sampling.py create \
      --benchmark hotpotqa \
      --env benchmarks/hotpotqa/.env.hotpotqa.frozen \
      --method stratified \
      --target-n 500 \
      --seed 42 \
      --strata type,supporting_fact_bucket

    python benchmarks/run_sampling.py validate \
      --manifest benchmarks/hotpotqa/sampling_manifest_stratified_42_500.json
"""
from __future__ import annotations

import argparse
import json
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
    describe_population,
    validate_sampling_manifest,
    write_sample_ids,
)
from framework.registry import load_benchmark_adapter, supported_benchmarks  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create and validate auditable benchmark sampling protocols.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    describe = sub.add_parser("describe", help="Describe benchmark population distribution")
    _add_common_benchmark_args(describe)

    create = sub.add_parser("create", help="Create a sampled GoldenSet and sampling manifest")
    _add_common_benchmark_args(create)
    create.add_argument("--method", default="stratified", choices=["simple_random", "stratified", "full", "diagnostic_rare", "fixed_ids"])
    create.add_argument("--target-n", type=int, default=500, dest="target_n")
    create.add_argument("--seed", type=int, default=42)
    create.add_argument("--strata", default=",".join(DEFAULT_HOTPOTQA_STRATA))
    create.add_argument("--allocation", default="proportional", choices=["proportional", "equal", "uniform"])
    create.add_argument("--min-per-stratum", type=int, default=1, dest="min_per_stratum")
    create.add_argument("--expected-population-size", type=int, default=0, dest="expected_population_size")
    create.add_argument("--output-dir", default="", dest="output_dir")
    create.add_argument("--sample-ids-file", default="", dest="sample_ids_file", help="Existing sample IDs JSON for fixed_ids sampling")
    create.add_argument("--force", action="store_true", help="Regenerate even when GoldenSet already exists")

    validate = sub.add_parser("validate", help="Validate a sampling manifest or GoldenSet JSON")
    validate.add_argument("--manifest", required=True, help="Sampling manifest or GoldenSet JSON path")
    return parser.parse_args()


def _add_common_benchmark_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--benchmark", "-b", required=True, choices=supported_benchmarks())
    parser.add_argument("--env", "-e", required=True, help="benchmark .env profile path")


def main() -> int:
    args = _parse_args()
    if args.command == "validate":
        return _validate(args)

    adapter = load_benchmark_adapter(args.benchmark, str(Path(args.env).resolve()))
    if args.command == "describe":
        return _describe(adapter)
    if args.command == "create":
        return _create(args, adapter)
    raise ValueError(args.command)


def _describe(adapter) -> int:
    if hasattr(adapter, "describe_split"):
        description = adapter.describe_split()
    else:
        description = describe_population(_load_sampling_population(adapter, seed=42))
    print(json.dumps(description, indent=2, ensure_ascii=False))
    return 0


def _create(args: argparse.Namespace, adapter) -> int:
    samples = _load_sampling_population(adapter, seed=args.seed)
    population_size = len(samples)
    expected_population_size = args.expected_population_size
    if not expected_population_size and args.benchmark == "hotpotqa":
        expected_population_size = DEFAULT_HOTPOTQA_POPULATION_SIZE
    target_n = 0 if args.method in {"full", "diagnostic_rare", "fixed_ids"} else args.target_n
    if args.method == "fixed_ids" and not args.sample_ids_file:
        raise ValueError("--method fixed_ids requires --sample-ids-file")
    protocol = create_sampling_protocol(
        benchmark=args.benchmark,
        split=_split_from_adapter(adapter),
        population_size=population_size,
        method=args.method,
        seed=args.seed,
        target_n=target_n,
        strata=args.strata if args.method == "stratified" else "",
        allocation=args.allocation,
        min_per_stratum=args.min_per_stratum,
        expected_population_size=expected_population_size,
        sample_ids_file=str(Path(args.sample_ids_file).expanduser().resolve()) if args.sample_ids_file else "",
    )
    out_dir = Path(args.output_dir).resolve() if args.output_dir else _SCRIPT_DIR / args.benchmark
    manager = GoldenSetManager(str(out_dir))
    golden = manager.get_or_create(
        adapter=adapter,
        seed=args.seed,
        n=target_n,
        force_recreate=args.force,
        sampling_protocol=protocol,
    )

    stem = f"sampling_{protocol.method}_{protocol.seed}_{protocol.target_n or 'full'}"
    protocol_path = out_dir / f"{stem}_protocol.json"
    manifest_path = out_dir / f"{stem}_manifest.json"
    sample_ids_path = out_dir / f"{stem}_sample_ids.json"
    protocol_path.write_text(json.dumps(golden.sampling_protocol, indent=2, ensure_ascii=False), encoding="utf-8")
    manifest_path.write_text(json.dumps(golden.sampling_manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    write_sample_ids(
        sample_ids_path,
        golden.sample_ids(),
        metadata={
            "benchmark": args.benchmark,
            "method": protocol.method,
            "seed": protocol.seed,
            "target_n": protocol.target_n,
            "golden_set_checksum": golden.checksum,
        },
    )

    validation = validate_sampling_manifest(golden.sampling_manifest)
    summary = {
        "golden_set_path": manager.get_path(args.seed, target_n, sampling_protocol=protocol),
        "protocol_path": str(protocol_path),
        "manifest_path": str(manifest_path),
        "sample_ids_path": str(sample_ids_path),
        "population_size": golden.population_size,
        "n_questions": golden.n_questions,
        "sample_id_checksum": golden.sample_id_checksum(),
        "validation": validation,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if validation["passed"] else 1


def _validate(args: argparse.Namespace) -> int:
    data = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    manifest = data.get("sampling_manifest", data) if isinstance(data, dict) else data
    result = validate_sampling_manifest(manifest)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["passed"] else 1


def _split_from_adapter(adapter) -> str:
    try:
        config = adapter.get_run_config()
        return str(config.get("split") or "validation")
    except Exception:
        return "validation"


def _load_sampling_population(adapter, *, seed: int):
    loader = getattr(adapter, "load_sampling_population", None)
    if callable(loader):
        return loader(seed=seed)
    return adapter.load_samples(limit=0, seed=seed)


if __name__ == "__main__":
    raise SystemExit(main())
