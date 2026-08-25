# Copyright (c) ModelScope Contributors. All rights reserved.
"""Statistical stop decision strategies for LENS sampling loop.

This module provides two StopStrategy implementations:
- ThresholdStop: Hard-threshold baseline (backward compatible)
- StatisticalStopDecider: Bayesian confidence-bound based stop decision

Both conform to the ``StopStrategy`` protocol defined in ``lens_protocols.py``.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, List, Tuple

from sirchmunk.learnings.lens_config import LensConfig
from sirchmunk.utils import LogCallback, create_logger


# z critical value for 95% confidence (two-sided)
_Z_CRIT_95 = 1.96


@dataclass
class ConfidenceEstimate:
    """Bayesian posterior confidence estimate.

    Attributes:
        mean: Posterior mean of the score distribution.
        std: Posterior standard deviation.
        ci_lower: Lower bound of the credible interval.
        ci_upper: Upper bound of the credible interval.
        n_observations: Number of observations used in the fit.
    """

    mean: float
    std: float
    ci_lower: float
    ci_upper: float
    n_observations: int


class ThresholdStop:
    """Hard-threshold stop strategy (backward compatible baseline).

    When ``enable_statistical_stop=False`` this strategy is used,
    providing behavior equivalent to the original
    ``top_seeds[0].score >= confidence_threshold`` check.

    Implements the ``StopStrategy`` protocol.
    """

    async def should_stop(
        self,
        all_candidates: List[Any],
        query: str,
        round_num: int,
        max_rounds: int,
        confidence_threshold: float = 8.5,
        **kwargs: Any,
    ) -> Tuple[bool, str]:
        """Decide whether to stop based on a simple score threshold.

        Returns:
            Tuple of (should_stop, reason).
        """
        if round_num >= max_rounds:
            return True, "max_rounds_reached"

        if not all_candidates:
            return False, "no_candidates"

        sorted_candidates = sorted(
            all_candidates, key=lambda x: x.score, reverse=True
        )

        if sorted_candidates[0].score >= confidence_threshold:
            return (
                True,
                f"threshold_met: {sorted_candidates[0].score:.1f} >= {confidence_threshold}",
            )

        return False, "continue"


class StatisticalStopDecider:
    """Bayesian confidence-bound based statistical stop decider.

    Four-condition stop rules (by priority):
        1. Statistical confidence: posterior CI_lower > threshold
           (Normal-Normal conjugate update)
        2. Multi-source completeness: for multi-hop queries, ensure all
           dimensions are covered
        3. Information value decay: expected gain from continued sampling
           falls below threshold
        4. Budget exhaustion: remaining token budget is insufficient

    Implements the ``StopStrategy`` protocol.
    """

    def __init__(
        self,
        config: LensConfig,
        log_callback: LogCallback = None,
    ) -> None:
        self.config = config
        self._log = create_logger(log_callback=log_callback)

        # Normal-Normal conjugate hyperparameters
        self._prior_mean: float = 6.0
        self._prior_var: float = 10.0
        self._obs_var: float = 2.25  # σ_obs = 1.5

    async def should_stop(
        self,
        all_candidates: List[Any],
        query: str,
        round_num: int,
        max_rounds: int,
        confidence_threshold: float = 8.5,
        **kwargs: Any,
    ) -> Tuple[bool, str]:
        """Multi-condition stop decision.

        kwargs may contain:
            - tokens_remaining: float — remaining token budget
            - multi_source_intent: float (0-1) — multi-hop intent probability

        Returns:
            Tuple of (should_stop, reason).
        """
        # 1. Max rounds hard stop
        if round_num >= max_rounds:
            return True, "max_rounds_reached"

        if not all_candidates:
            return False, "no_candidates"

        # Use the passed confidence_threshold (protocol contract) rather than
        # config value to maintain consistency with ThresholdStop behavior.
        effective_threshold = confidence_threshold

        # 2. Statistical confidence via Bayesian posterior
        conf = self._fit_posterior(all_candidates)

        if conf.ci_lower >= effective_threshold:
            # 2b. Multi-source completeness gate
            multi_source_intent = kwargs.get("multi_source_intent", 0.0)
            if multi_source_intent > 0.5:
                if not self._check_multi_source(all_candidates, query):
                    return False, "multi_source_incomplete"
            return (
                True,
                f"confident: CI=[{conf.ci_lower:.2f}, {conf.ci_upper:.2f}]",
            )

        # 3. Information value decay
        info_value = self._estimate_info_value(all_candidates, round_num)
        if info_value < self.config.info_value_threshold:
            return (
                True,
                f"diminishing_returns: info_value={info_value:.2f}",
            )

        # 4. Budget exhaustion
        tokens_remaining = kwargs.get("tokens_remaining", float("inf"))
        if tokens_remaining < self.config.min_tokens_remaining:
            return (
                True,
                f"budget_exhausted: {tokens_remaining:.0f} remaining",
            )

        return False, "continue_exploring"

    def _fit_posterior(self, candidates: List[Any]) -> ConfidenceEstimate:
        """Normal-Normal conjugate Bayesian posterior fit.

        Assumptions:
            - Prior: μ ~ Normal(prior_mean=6.0, prior_var=10.0)
            - Observations: score_i ~ Normal(μ, obs_var=2.25) (σ_obs=1.5)

        Posterior update:
            - precision_post = precision_prior + n * precision_obs
            - mean_post = (precision_prior * prior_mean
                          + precision_obs * sum(scores)) / precision_post
            - var_post = 1 / precision_post

        Credible interval: mean_post ± z_crit * sqrt(var_post)
        """
        scores = [c.score for c in candidates if c.score > 0]
        n = len(scores)

        if n == 0:
            return ConfidenceEstimate(
                mean=self._prior_mean,
                std=math.sqrt(self._prior_var),
                ci_lower=0.0,
                ci_upper=self._prior_mean + _Z_CRIT_95 * math.sqrt(self._prior_var),
                n_observations=0,
            )

        precision_prior = 1.0 / self._prior_var
        precision_obs = 1.0 / self._obs_var

        precision_post = precision_prior + n * precision_obs
        mean_post = (
            precision_prior * self._prior_mean + precision_obs * sum(scores)
        ) / precision_post
        var_post = 1.0 / precision_post
        std_post = math.sqrt(var_post)

        ci_lower = mean_post - _Z_CRIT_95 * std_post
        ci_upper = mean_post + _Z_CRIT_95 * std_post

        return ConfidenceEstimate(
            mean=mean_post,
            std=std_post,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            n_observations=n,
        )

    def _check_multi_source(
        self, candidates: List[Any], query: str
    ) -> bool:
        """Multi-source completeness check.

        Heuristics:
            - Extract years from query (e.g., "FY2021-2023" → [2021, 2022, 2023])
            - Detect covered years from high-score candidates' reasoning
            - Require at least N-1 years covered

        Also checks positional diversity:
            - If all high-score candidates come from same region (distance < 5000),
              and multi_source_intent > 0.7, consider incomplete
        """
        # Extract year mentions from query
        year_pattern = re.compile(r"(?:20\d{2}|19\d{2})")
        query_years = set(year_pattern.findall(query))

        # Also look for ranges like "2021-2023"
        range_pattern = re.compile(r"(20\d{2})\s*[-–—to]+\s*(20\d{2})")
        for m in range_pattern.finditer(query):
            start_year, end_year = int(m.group(1)), int(m.group(2))
            for y in range(start_year, end_year + 1):
                query_years.add(str(y))

        if not query_years:
            # No year requirements detected — assume completeness
            return True

        # Check coverage in high-score candidates' reasoning
        high_score_candidates = [
            c for c in candidates if c.score >= 6.0
        ]
        if not high_score_candidates:
            return False

        covered_years: set = set()
        for c in high_score_candidates:
            reasoning_text = f"{c.reasoning} {c.content}"
            found = year_pattern.findall(reasoning_text)
            covered_years.update(found)

        # Require at least N-1 coverage
        required = len(query_years)
        covered_count = len(query_years & covered_years)
        if covered_count < max(1, required - 1):
            return False

        # Positional diversity check
        if len(high_score_candidates) >= 2:
            positions = [c.start_idx for c in high_score_candidates]
            span = max(positions) - min(positions)
            if span < self.config.multi_source_span_threshold:
                # All candidates clustered in same region
                return False

        return True

    def _estimate_info_value(
        self, candidates: List[Any], round_num: int
    ) -> float:
        """Estimate the information value of continued sampling.

        Simplified model:
            - Lower variance among top-5 scores → more stable ranking
              → lower information value
            - More rounds → diminishing marginal returns

        Formula: info_value = var(top_5_scores) + 1.0 / (round_num + 1)
        """
        sorted_candidates = sorted(
            candidates, key=lambda x: x.score, reverse=True
        )
        top_scores = [c.score for c in sorted_candidates[:5]]

        if len(top_scores) < 2:
            # Insufficient data — assume high information value
            return float("inf")

        # Compute sample variance
        mean_score = sum(top_scores) / len(top_scores)
        variance = sum((s - mean_score) ** 2 for s in top_scores) / (
            len(top_scores) - 1
        )

        # Marginal return decay
        round_decay = 1.0 / (round_num + 1)

        return variance + round_decay
