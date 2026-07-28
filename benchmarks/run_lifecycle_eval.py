#!/usr/bin/env python3
"""Run baseline lifecycle feasibility evaluation.

This CLI is intentionally separate from ``run_research_loop.py``: it evaluates
full-corpus preprocessing/indexing feasibility for baselines under a declared
resource budget, and produces feasibility tables with timeout/OOM/N/A reasons.
"""
from __future__ import annotations

import argparse
import asyncio
import importlib
import logging
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import List

_SCRIPT_DIR = Path(__file__).parent.resolve()   # benchmarks/
_PROJECT_ROOT = _SCRIPT_DIR.parent              # project root
_SRC = _PROJECT_ROOT / "src"
for _p in (str(_SCRIPT_DIR), str(_SRC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from baselines import BM25RAGBaseline, HybridRAGBaseline, LocalBM25Baseline, NaiveRAGBaseline  # noqa: E402
from baselines.base_adapter import BaselineAdapter  # noqa: E402
from evaluation.table_generator import PaperTableGenerator  # noqa: E402
from framework.baseline_lifecycle import BaselineLifecycleManager  # noqa: E402
from framework.lifecycle_schema import (
    BaselineLifecycleRecord,
    BaselinePhase,
    ResourceBudget,
)  # noqa: E402
from framework.registry import load_benchmark_adapter, supported_benchmarks  # noqa: E402
from framework.time_utils import local_timestamp  # noqa: E402

logger = logging.getLogger("run_lifecycle_eval")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Baseline lifecycle feasibility evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--benchmark", "-b", choices=supported_benchmarks(), required=True)
    parser.add_argument("--env", "-e", required=True, help="Benchmark env file path")
    parser.add_argument(
        "--baselines",
        default="bm25,naive_rag",
        help=(
            "Comma-separated baseline names or module:factory specs. "
            "Built-ins: bm25, bm25_rag, hybrid_rag, naive_rag."
        ),
    )
    parser.add_argument("--limit", type=int, default=20, help="Samples used to infer corpus paths")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--corpus-scale", default="fullwiki")
    parser.add_argument("--output-dir", default="", help="Override lifecycle output directory")
    parser.add_argument("--build-timeout", type=float, default=0.0, help="Build wall-clock budget in seconds")
    parser.add_argument("--max-ram-bytes", type=int, default=0)
    parser.add_argument("--max-disk-bytes", type=int, default=0)
    parser.add_argument("--max-llm-tokens", type=int, default=0)
    parser.add_argument("--max-api-cost-usd", type=float, default=0.0)
    parser.add_argument("--retry-count", type=int, default=0)
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s  %(name)-25s  %(levelname)-7s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _build_golden_like(adapter, *, limit: int, seed: int):
    samples = adapter.load_samples(limit=limit, seed=seed)
    rows = []
    for sample in samples:
        rows.append({
            "sample_id": sample.sample_id,
            "question": sample.question,
            "gold_answer": sample.gold_answer,
            "metadata": sample.metadata,
        })
    return SimpleNamespace(samples=rows)


def _load_baselines(spec: str) -> List[BaselineAdapter]:
    baselines: List[BaselineAdapter] = []
    for raw_name in [s.strip() for s in spec.split(",") if s.strip()]:
        lower = raw_name.lower()
        if lower in {"bm25", "bm25_local"}:
            baselines.append(LocalBM25Baseline())
        elif lower in {"bm25_rag", "rag_bm25"}:
            baselines.append(BM25RAGBaseline())
        elif lower in {"hybrid_rag", "rag_hybrid"}:
            baselines.append(HybridRAGBaseline())
        elif lower in {"naive_rag", "naive_rag_local"}:
            baselines.append(NaiveRAGBaseline())
        elif ":" in raw_name:
            module_name, _, factory_name = raw_name.partition(":")
            module = importlib.import_module(module_name)
            factory = getattr(module, factory_name)
            baseline = factory()
            if not isinstance(baseline, BaselineAdapter):
                raise TypeError(f"Factory {raw_name} did not return BaselineAdapter")
            baselines.append(baseline)
        else:
            raise ValueError(f"Unknown baseline '{raw_name}'. Use bm25, bm25_rag, hybrid_rag, naive_rag, or module:factory.")
    return baselines


def _sirchmunk_no_index_record(run_id: str, benchmark: str, corpus_manifest: dict, corpus_scale: str) -> BaselineLifecycleRecord:
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


async def _main() -> int:
    args = _parse_args()
    _setup_logging(args.log_level)

    env_file = str(Path(args.env).expanduser().resolve())
    adapter = load_benchmark_adapter(args.benchmark, env_file)
    output_dir = Path(args.output_dir or adapter.get_output_dir()).resolve()
    lifecycle_output = output_dir / "lifecycle_eval"
    lifecycle_output.mkdir(parents=True, exist_ok=True)

    run_id = f"lifecycle_{args.benchmark}_{local_timestamp()}"
    budget = ResourceBudget(
        wall_clock_seconds=args.build_timeout,
        max_ram_bytes=args.max_ram_bytes,
        max_disk_bytes=args.max_disk_bytes,
        max_llm_tokens=args.max_llm_tokens,
        max_api_cost_usd=args.max_api_cost_usd,
        retry_count=args.retry_count,
    )

    logger.info("Lifecycle eval: benchmark=%s run_id=%s", args.benchmark, run_id)
    corpus_manifest = adapter.get_dataset_manifest()
    golden_like = _build_golden_like(adapter, limit=args.limit, seed=args.seed)
    baselines = _load_baselines(args.baselines)

    manager = BaselineLifecycleManager(lifecycle_output, resource_budget=budget)
    records: List[BaselineLifecycleRecord] = []

    # Always include Sirchmunk as no-index feasibility row for the paper table.
    sirchmunk_record = _sirchmunk_no_index_record(
        run_id, args.benchmark, corpus_manifest, args.corpus_scale
    )
    manager.save_record(sirchmunk_record)
    records.append(sirchmunk_record)

    for baseline in baselines:
        logger.info("Building baseline lifecycle: %s", baseline.citation_name)
        record = await manager.run_build(
            baseline,
            run_id=run_id,
            benchmark=args.benchmark,
            corpus_manifest=corpus_manifest,
            golden_set=golden_like,
            bm_adapter=adapter,
            corpus_scale=args.corpus_scale,
        )
        records.append(record)
        logger.info(
            "Baseline %s phase=%s ready=%s failure=%s",
            baseline.name,
            record.phase.value,
            record.index_ready,
            record.failure_reason.value,
        )

    table = PaperTableGenerator(benchmark_name=f"{args.benchmark} lifecycle")
    paths = table.generate_feasibility_table(records, str(lifecycle_output / "tables"))

    summary_path = lifecycle_output / "lifecycle_summary.json"
    summary_path.write_text(
        __import__("json").dumps(
            {
                "run_id": run_id,
                "benchmark": args.benchmark,
                "env_file": env_file,
                "resource_budget": budget.to_dict(),
                "records": [r.to_dict() for r in records],
                "table_paths": paths,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    logger.info("Lifecycle summary: %s", summary_path)
    return 0


def main() -> None:
    sys.exit(asyncio.run(_main()))


if __name__ == "__main__":
    main()
