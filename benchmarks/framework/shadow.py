"""framework/shadow.py — ShadowEvaluator

Lightweight cross-benchmark pre-evaluation, dedicated to Layer 0/1 hypotheses of type
CONFIG_CHANGE.

Core idea:
  Before the full experiment, estimate the effect of a change using about 10% of the
  samples of each benchmark.
  Cost: ~10-15 questions x N benchmarks, versus the full 150 questions x N benchmarks.
  Only CONFIG_CHANGE is supported, because it can be realized by overriding search_kwargs.
  PIPELINE_PATCH / PROMPT_FIX cannot be estimated this way; they are skipped with a hint
  for manual review.

How it works:
  1. extract the keys of hypothesis.config_changes that map to search_kwargs
  2. run a small batch per adapter with base_kwargs + overrides
  3. compare against the baseline and return an ImpactMatrix
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .schema import ChangeType, ConfigLayer, ImprovementHypothesis, PredictionResult

logger = logging.getLogger(__name__)

# env key -> search kwarg rule: strip the benchmark prefix, then lowercase
# Supported search kwargs, mirroring AgenticSearch.search() parameters
_SEARCH_KWARG_NAMES = frozenset({
    "mode", "top_k_files", "max_token_budget", "enable_dir_scan",
})

# Benchmark-specific prefixes; add a prefix here when adding a benchmark
_BM_PREFIXES = ("HOTPOT_", "MECHANISM_")


@dataclass
class BmShadowResult:
    """Shadow evaluation result of a single benchmark."""
    benchmark: str
    n_samples: int
    accuracy: float
    coverage: float
    avg_latency: float
    applicable: bool = True    # False means this hypothesis has no applicable kwarg for the benchmark
    error: Optional[str] = None

    def accuracy_delta(self, baseline: float) -> float:
        return self.accuracy - baseline

    def coverage_delta(self, baseline: float) -> float:
        return self.coverage - baseline


@dataclass
class ShadowImpactMatrix:
    """Estimated impact matrix of a hypothesis across all benchmarks."""
    hypothesis_id: str
    results: Dict[str, BmShadowResult] = field(default_factory=dict)
    # {bm: {accuracy_delta, coverage_delta}}（vs baseline_vector）
    deltas: Dict[str, Dict[str, float]] = field(default_factory=dict)

    @property
    def pareto_status(self) -> str:
        """Derive the Pareto status from the accuracy delta."""
        applicable = {bm: r for bm, r in self.results.items() if r.applicable}
        if not applicable:
            return "unknown"
        deltas = [self.deltas.get(bm, {}).get("accuracy_delta", 0)
                  for bm in applicable]
        if all(d >= 0 for d in deltas) and any(d > 0 for d in deltas):
            return "dominant"
        if all(d < -2.0 for d in deltas):
            return "harmful"
        if any(d > 0 for d in deltas) and any(d < -2.0 for d in deltas):
            return "trade_off"
        return "neutral"

    def print_summary(self, hypothesis_title: str = "") -> None:
        title = hypothesis_title or self.hypothesis_id
        icon = {"dominant": "✅", "harmful": "❌", "trade_off": "⚖️ ", "neutral": "➖",
                "unknown": "❓"}.get(self.pareto_status, "❓")
        print(f"\n  Shadow Eval [{self.hypothesis_id}] {title[:50]}")
        print(f"  Pareto预估: {icon} {self.pareto_status.upper()}")
        print(f"  {'Benchmark':<25} {'n':>4} {'Δacc%':>7} {'Δcov%':>7}")
        print("  " + "─" * 46)
        for bm, r in sorted(self.results.items()):
            if not r.applicable:
                print(f"  {bm:<25} {'N/A':>4} {'n/a':>7} {'n/a':>7}  (not applicable)")
                continue
            if r.error:
                print(f"  {bm:<25} {'err':>4}  (error: {r.error[:40]})")
                continue
            d = self.deltas.get(bm, {})
            delta_acc = d.get("accuracy_delta", 0)
            delta_cov = d.get("coverage_delta", 0)
            s = lambda v: f"+{v:.1f}" if v >= 0 else f"{v:.1f}"
            print(f"  {bm:<25} {r.n_samples:>4} {s(delta_acc):>7} {s(delta_cov):>7}")
        print()


def _env_key_to_search_kwarg(key: str) -> Optional[str]:
    """Map an env key to a search kwarg name.

    Rule: strip the benchmark prefix (HOTPOT_, MECHANISM_, ...) and lowercase the rest.
    Keys outside the _SEARCH_KWARG_NAMES allowlist return None.
    """
    upper_key = key.upper()
    for prefix in _BM_PREFIXES:
        if upper_key.startswith(prefix):
            remainder = upper_key[len(prefix):].lower()
            if remainder in _SEARCH_KWARG_NAMES:
                return remainder
    return None


def _coerce_kwarg(kwarg_name: str, value: str) -> Any:
    """Coerce a string value into the type expected by the search kwarg."""
    if kwarg_name == "top_k_files":
        return int(value)
    if kwarg_name == "max_token_budget":
        return int(value)
    if kwarg_name == "enable_dir_scan":
        return str(value).lower() in ("true", "1", "yes")
    # String-typed values such as mode are returned as-is
    return value


class ShadowEvaluator:
    """Lightweight cross-benchmark pre-evaluator.

    Supports only CONFIG_CHANGE hypotheses whose config_changes map onto search_kwargs.
    PIPELINE_PATCH / PROMPT_FIX cannot be pre-evaluated: an empty result is returned and
    the reason is logged.

    Usage::

        evaluator = ShadowEvaluator()
        matrix = await evaluator.evaluate(
            hypothesis, adapters,
            baseline_vector={"financebench": {"accuracy": 24.0, "coverage": 40.0}},
            sample_fraction=0.10,
        )
        matrix.print_summary(hypothesis.title)
    """

    async def evaluate(
        self,
        hypothesis: ImprovementHypothesis,
        adapters: List[Any],          # List[BenchmarkAdapter]
        baseline_vector: Dict[str, Dict[str, float]],
        sample_fraction: float = 0.10,
        seed: int = 99,
    ) -> ShadowImpactMatrix:
        """Run a small batch on every adapter concurrently and return the impact matrix.

        Args:
            hypothesis:       improvement hypothesis to evaluate.
            adapters:         list of every registered BenchmarkAdapter.
            baseline_vector:  current baseline metric vector {bm: {accuracy, coverage, ...}}.
            sample_fraction:  sampling fraction (default 10%).
            seed:             random seed used for reproducibility.

        Returns:
            A ShadowImpactMatrix holding the delta estimate of each benchmark.
        """
        matrix = ShadowImpactMatrix(hypothesis_id=hypothesis.hypothesis_id)

        # Only CONFIG_CHANGE is supported
        if hypothesis.change_type != ChangeType.CONFIG_CHANGE:
            logger.info(
                "[Shadow] %s: skip non-CONFIG_CHANGE (%s)",
                hypothesis.hypothesis_id, hypothesis.change_type.value,
            )
            for adapter in adapters:
                matrix.results[adapter.name] = BmShadowResult(
                    benchmark=adapter.name, n_samples=0,
                    accuracy=0, coverage=0, avg_latency=0, applicable=False,
                )
            return matrix

        # Run the shadow eval of each adapter concurrently
        tasks = {
            adapter.name: asyncio.create_task(
                self._eval_single_adapter(
                    hypothesis, adapter, baseline_vector.get(adapter.name, {}),
                    sample_fraction, seed,
                )
            )
            for adapter in adapters
        }

        for bm_name, task in tasks.items():
            try:
                result = await task
            except Exception as exc:
                logger.warning("[Shadow] %s on %s failed: %s",
                               hypothesis.hypothesis_id, bm_name, exc)
                result = BmShadowResult(
                    benchmark=bm_name, n_samples=0,
                    accuracy=0, coverage=0, avg_latency=0,
                    applicable=True, error=str(exc)[:80],
                )
            matrix.results[bm_name] = result

        # Compute the delta against the baseline
        for bm, r in matrix.results.items():
            if not r.applicable or r.error:
                continue
            baseline_acc = baseline_vector.get(bm, {}).get("accuracy", 0)
            baseline_cov = baseline_vector.get(bm, {}).get("coverage", 0)
            matrix.deltas[bm] = {
                "accuracy_delta": round(r.accuracy - baseline_acc, 2),
                "coverage_delta": round(r.coverage - baseline_cov, 2),
            }

        return matrix

    async def _eval_single_adapter(
        self,
        hypothesis: ImprovementHypothesis,
        adapter: Any,
        baseline: Dict[str, float],
        sample_fraction: float,
        seed: int,
    ) -> BmShadowResult:
        """Run the shadow evaluation of a single adapter."""
        # Build the search_kwargs override
        overrides: Dict[str, Any] = {}
        for env_key, new_val in hypothesis.config_changes.items():
            kwarg = _env_key_to_search_kwarg(env_key)
            if kwarg is not None:
                overrides[kwarg] = _coerce_kwarg(kwarg, new_val)

        if not overrides:
            return BmShadowResult(
                benchmark=adapter.name, n_samples=0,
                accuracy=0, coverage=0, avg_latency=0, applicable=False,
            )

        # Resolve the sample count (at least 5, at most 30 questions)
        full_samples = adapter.load_samples(limit=0, seed=seed)
        n = max(5, min(30, int(len(full_samples) * sample_fraction)))
        samples = adapter.load_samples(limit=n, seed=seed)

        # Reuse the original searcher and override search_kwargs
        searcher = adapter.build_searcher()
        judge = adapter.build_judge()
        base_kwargs = dict(adapter.get_search_kwargs())
        base_kwargs.update(overrides)

        # Run sequentially: shadow eval needs no concurrency and this lowers API pressure
        correct = 0
        coverage_count = 0
        elapsed_list = []
        errors = 0

        for sample in samples:
            try:
                search_paths = adapter.get_search_paths(sample)
                result = await searcher.search(
                    query=sample.question,
                    paths=search_paths,
                    return_context=True,
                    **base_kwargs,
                )
                prediction = getattr(result, "answer", "") or str(result)

                jc, cov = False, False
                if judge:
                    try:
                        jr = await judge.judge(
                            prediction=prediction,
                            gold_answer=sample.gold_answer,
                            question=sample.question,
                        )
                        jc = jr.get("equivalent", False)
                        cr = await judge.judge_coverage(
                            prediction=prediction, question=sample.question
                        )
                        cov = cr.get("has_coverage", False)
                    except Exception:
                        pass

                if jc:
                    correct += 1
                if cov:
                    coverage_count += 1

                elapsed = getattr(result, "elapsed", 0)
                if elapsed:
                    elapsed_list.append(elapsed)

            except Exception as exc:
                errors += 1
                logger.debug("[Shadow] sample error on %s: %s", adapter.name, exc)

        valid = len(samples) - errors
        if valid == 0:
            return BmShadowResult(
                benchmark=adapter.name, n_samples=len(samples),
                accuracy=0, coverage=0, avg_latency=0,
                applicable=True, error="all samples failed",
            )

        return BmShadowResult(
            benchmark=adapter.name,
            n_samples=valid,
            accuracy=round(correct / valid * 100, 1),
            coverage=round(coverage_count / valid * 100, 1),
            avg_latency=round(sum(elapsed_list) / len(elapsed_list), 1) if elapsed_list else 0,
            applicable=True,
        )
