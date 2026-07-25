#!/usr/bin/env python3
"""Run multi-scale baseline lifecycle feasibility study."""
from __future__ import annotations

import argparse
import asyncio
import importlib
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, List

_SCRIPT_DIR = Path(__file__).parent.resolve()
_PROJECT_ROOT = _SCRIPT_DIR.parent
_SRC = _PROJECT_ROOT / "src"
for _p in (str(_SCRIPT_DIR), str(_SRC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from baselines import LocalBM25Baseline, NaiveRAGBaseline  # noqa: E402
from baselines.base_adapter import BaselineAdapter  # noqa: E402
from framework.lifecycle_schema import ResourceBudget  # noqa: E402
from framework.registry import load_benchmark_adapter, supported_benchmarks  # noqa: E402
from framework.scaling_study import CorpusScaleSpec, ScalingStudyManager  # noqa: E402
from hotpotqa.subset_sampler import create_corpus_subset  # noqa: E402

logger = logging.getLogger("run_scaling_study")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multi-scale baseline lifecycle feasibility study",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--benchmark", "-b", choices=supported_benchmarks(), required=True)
    parser.add_argument("--env", "-e", required=True)
    parser.add_argument("--baselines", default="bm25,naive_rag")
    parser.add_argument(
        "--scales",
        default="10k:10000,100k:100000,fullwiki:0",
        help="Comma-separated NAME:DOCS pairs. DOCS=0 means full corpus.",
    )
    parser.add_argument("--limit", type=int, default=20, help="Questions for golden-like path inference")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--subset-output-dir", default="")
    parser.add_argument("--subset-strategy", default="random_shard", choices=["random_shard", "prefix"])
    parser.add_argument("--materialize", default="symlink", choices=["manifest", "symlink", "copy"])
    parser.add_argument("--build-timeout", type=float, default=0.0)
    parser.add_argument("--max-ram-bytes", type=int, default=0)
    parser.add_argument("--max-disk-bytes", type=int, default=0)
    parser.add_argument("--max-llm-tokens", type=int, default=0)
    parser.add_argument("--max-api-cost-usd", type=float, default=0.0)
    parser.add_argument("--q-values", default="1,10,100,1000")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s  %(name)-25s  %(levelname)-7s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _parse_scales(raw: str) -> List[tuple[str, int]]:
    scales: List[tuple[str, int]] = []
    for chunk in [c.strip() for c in raw.split(",") if c.strip()]:
        if ":" not in chunk:
            raise ValueError(f"Invalid scale '{chunk}', expected NAME:DOCS")
        name, _, docs = chunk.partition(":")
        scales.append((name.strip(), int(docs.strip())))
    return scales


def _parse_q_values(raw: str) -> List[int]:
    return [max(int(x.strip()), 1) for x in raw.split(",") if x.strip()]


def _baseline_factories(spec: str) -> List[Callable[[], BaselineAdapter]]:
    factories: List[Callable[[], BaselineAdapter]] = []
    for raw_name in [s.strip() for s in spec.split(",") if s.strip()]:
        lower = raw_name.lower()
        if lower in {"bm25", "bm25_local"}:
            factories.append(lambda: LocalBM25Baseline())
        elif lower in {"naive_rag", "naive_rag_local"}:
            factories.append(lambda: NaiveRAGBaseline())
        elif ":" in raw_name:
            module_name, _, factory_name = raw_name.partition(":")

            def _factory(m=module_name, f=factory_name):
                module = importlib.import_module(m)
                baseline = getattr(module, f)()
                if not isinstance(baseline, BaselineAdapter):
                    raise TypeError(f"Factory {m}:{f} did not return BaselineAdapter")
                return baseline

            factories.append(_factory)
        else:
            raise ValueError(f"Unknown baseline '{raw_name}'. Use bm25, naive_rag, or module:factory.")
    return factories


def _golden_like(adapter, *, limit: int, seed: int):
    from types import SimpleNamespace

    samples = adapter.load_samples(limit=limit, seed=seed)
    return SimpleNamespace(samples=[{
        "sample_id": s.sample_id,
        "question": s.question,
        "gold_answer": s.gold_answer,
        "metadata": s.metadata,
    } for s in samples])


def _wiki_dir_from_manifest(adapter) -> str:
    manifest = adapter.get_dataset_manifest()
    wiki_dir = str(manifest.get("wiki_dir") or "")
    if not wiki_dir:
        raise ValueError("Adapter manifest does not provide wiki_dir; scaling study requires corpus directory.")
    return wiki_dir


def _make_scale_specs(args: argparse.Namespace, adapter, output_dir: Path) -> List[CorpusScaleSpec]:
    wiki_dir = _wiki_dir_from_manifest(adapter)
    subset_root = Path(args.subset_output_dir or (output_dir / "corpus_subsets")).resolve()
    specs: List[CorpusScaleSpec] = []
    for name, max_docs in _parse_scales(args.scales):
        if max_docs <= 0:
            base_manifest = dict(adapter.get_dataset_manifest())
            base_manifest.update({"corpus_scale": name, "doc_count": base_manifest.get("doc_count", 0) or 0})
            specs.append(CorpusScaleSpec(
                name=name,
                max_docs=0,
                corpus_dir=wiki_dir,
                manifest=base_manifest,
                materialized=False,
            ))
            continue
        manifest = create_corpus_subset(
            wiki_dir,
            output_dir=subset_root,
            scale_name=name,
            max_docs=max_docs,
            seed=args.seed,
            strategy=args.subset_strategy,
            materialize=args.materialize,
        )
        specs.append(CorpusScaleSpec(
            name=name,
            max_docs=max_docs,
            corpus_dir=manifest.subset_dir,
            manifest=manifest.to_dict(),
            materialized=manifest.materialized,
        ))
    return specs


async def _main() -> int:
    args = _parse_args()
    _setup_logging(args.log_level)
    adapter = load_benchmark_adapter(args.benchmark, str(Path(args.env).resolve()))
    output_dir = Path(args.output_dir or adapter.get_output_dir()).resolve() / "scaling_study"
    output_dir.mkdir(parents=True, exist_ok=True)

    budget = ResourceBudget(
        wall_clock_seconds=args.build_timeout,
        max_ram_bytes=args.max_ram_bytes,
        max_disk_bytes=args.max_disk_bytes,
        max_llm_tokens=args.max_llm_tokens,
        max_api_cost_usd=args.max_api_cost_usd,
    )
    q_values = _parse_q_values(args.q_values)
    scale_specs = _make_scale_specs(args, adapter, output_dir)
    manager = ScalingStudyManager(
        adapter,
        output_dir,
        resource_budget=budget,
        q_values=q_values,
    )
    run_id = f"scaling_{args.benchmark}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    result = await manager.run(
        baseline_factories=_baseline_factories(args.baselines),
        scales=scale_specs,
        golden_set=_golden_like(adapter, limit=args.limit, seed=args.seed),
        run_id=run_id,
    )
    logger.info("Scaling study complete: %s", output_dir / "scaling_study_summary.json")
    logger.info("Records: %d", len(result.records))
    return 0


def main() -> None:
    sys.exit(asyncio.run(_main()))


if __name__ == "__main__":
    main()
