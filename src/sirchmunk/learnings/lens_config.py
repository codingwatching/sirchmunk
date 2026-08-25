# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Tuple


@dataclass
class LensConfig:
    """Centralized configuration for LENS (Latent Evidence Navigation & Sampling).

    All LENS-related parameters are managed here. Use ``LensConfig.from_env()``
    to load values from environment variables (prefix ``LENS_``).

    Design principle:
        - All ``enable_*`` flags default to False for backward compatibility.
        - Numeric parameters carry sensible defaults matching the existing
          MonteCarloEvidenceSampling behavior.
    """

    # ------------------------------------------------------------------
    # Feature flags (default all False for backward compatibility)
    # ------------------------------------------------------------------
    enable_batch_ranking: bool = False
    """LENS_ENABLE_BATCH_RANKING — activate BatchRankingEvaluator."""

    enable_multi_arm: bool = False
    """LENS_ENABLE_MULTI_ARM — activate MultiArmNavigator."""

    enable_adaptive_mix: bool = False
    """LENS_ENABLE_ADAPTIVE_MIX — activate AdaptiveProposalMixer."""

    enable_reasoning_exploit: bool = False
    """LENS_ENABLE_REASONING_EXPLOIT — activate ReasoningChainExploiter."""

    enable_statistical_stop: bool = False
    """LENS_ENABLE_STATISTICAL_STOP — activate StatisticalStopDecider."""

    # ------------------------------------------------------------------
    # Sampling parameters (shared with existing + new modules)
    # ------------------------------------------------------------------
    max_rounds: int = 3
    """LENS_MAX_ROUNDS — maximum number of sampling rounds."""

    probe_window: int = 500
    """LENS_PROBE_WINDOW — character size of each probe sampling window."""

    roi_window: int = 2000
    """LENS_ROI_WINDOW — character size of the expanded ROI context."""

    samples_per_round: int = 5
    """LENS_SAMPLES_PER_ROUND — number of samples generated per round."""

    top_k_seeds: int = 2
    """LENS_TOP_K_SEEDS — top K high-value seeds kept for next round."""

    # ------------------------------------------------------------------
    # BatchRankingEvaluator params
    # ------------------------------------------------------------------
    batch_size: int = 5
    """LENS_BATCH_SIZE — number of samples evaluated in one LLM call."""

    rank_score_map: Tuple[float, ...] = (9.5, 8.5, 7.5, 6.5, 5.5)
    """LENS_RANK_SCORE_MAP — rank-to-score mapping (1st place → highest)."""

    # ------------------------------------------------------------------
    # MultiArmNavigator params
    # ------------------------------------------------------------------
    k_arms: int = 4
    """LENS_K_ARMS — number of bandit arms (document segments)."""

    min_arm_allocation: int = 1
    """LENS_MIN_ARM_ALLOCATION — minimum samples allocated to each arm."""

    arm_merge_ci_overlap: float = 0.5
    """LENS_ARM_MERGE_CI_OVERLAP — CI overlap ratio to merge arms."""

    # ------------------------------------------------------------------
    # AdaptiveProposalMixer params
    # ------------------------------------------------------------------
    min_exploration_ratio: float = 0.15
    """LENS_MIN_EXPLORATION_RATIO — epsilon floor for exploration."""

    decay_factor: float = 0.8
    """LENS_DECAY_FACTOR — exponential decay factor for exploration ratio."""

    # ------------------------------------------------------------------
    # StatisticalStopDecider params
    # ------------------------------------------------------------------
    confidence_level: float = 0.95
    """LENS_CONFIDENCE_LEVEL — statistical confidence level for stop decision."""

    confidence_threshold: float = 8.0
    """LENS_CONFIDENCE_THRESHOLD — score threshold for early stopping."""

    info_value_threshold: float = 2.0
    """LENS_INFO_VALUE_THRESHOLD — expected information value threshold."""

    min_tokens_remaining: int = 2000
    """LENS_MIN_TOKENS_REMAINING — minimum tokens before budget-exhaustion stop."""

    relevance_floor: float = 4.0
    """LENS_RELEVANCE_FLOOR — minimum score for candidate inclusion in final results."""

    multi_source_span_threshold: int = 5000
    """LENS_MULTI_SOURCE_SPAN_THRESHOLD — positional diversity span for multi-source check."""

    # ------------------------------------------------------------------
    # ReasoningChainExploiter params
    # ------------------------------------------------------------------
    max_targeted_samples: int = 2
    """LENS_MAX_TARGETED_SAMPLES — max targeted samples from reasoning."""

    signal_score_range: Tuple[float, float] = (4.0, 8.0)
    """LENS_SIGNAL_SCORE_RANGE — only extract signals from partially relevant samples."""

    @classmethod
    def from_env(cls) -> LensConfig:
        """Create a LensConfig instance from environment variables.

        All environment variables use the ``LENS_`` prefix. Values fall back
        to dataclass defaults when the corresponding env var is absent.

        Returns:
            LensConfig: Configuration populated from environment.
        """

        def _bool(key: str, default: bool) -> bool:
            v = os.environ.get(key, str(default)).strip().lower()
            return v in ("true", "1", "yes")

        def _int(key: str, default: int) -> int:
            try:
                return int(os.environ.get(key, str(default)))
            except (ValueError, TypeError):
                return default

        def _float(key: str, default: float) -> float:
            try:
                return float(os.environ.get(key, str(default)))
            except (ValueError, TypeError):
                return default

        def _float_tuple(key: str, default: Tuple[float, ...]) -> Tuple[float, ...]:
            raw = os.environ.get(key, "")
            if not raw.strip():
                return default
            try:
                return tuple(float(x.strip()) for x in raw.split(","))
            except (ValueError, TypeError):
                return default

        return cls(
            # Feature flags
            enable_batch_ranking=_bool("LENS_ENABLE_BATCH_RANKING", False),
            enable_multi_arm=_bool("LENS_ENABLE_MULTI_ARM", False),
            enable_adaptive_mix=_bool("LENS_ENABLE_ADAPTIVE_MIX", False),
            enable_reasoning_exploit=_bool("LENS_ENABLE_REASONING_EXPLOIT", False),
            enable_statistical_stop=_bool("LENS_ENABLE_STATISTICAL_STOP", False),
            # Sampling parameters
            max_rounds=_int("LENS_MAX_ROUNDS", 3),
            probe_window=_int("LENS_PROBE_WINDOW", 500),
            roi_window=_int("LENS_ROI_WINDOW", 2000),
            samples_per_round=_int("LENS_SAMPLES_PER_ROUND", 5),
            top_k_seeds=_int("LENS_TOP_K_SEEDS", 2),
            # BatchRankingEvaluator
            batch_size=_int("LENS_BATCH_SIZE", 5),
            rank_score_map=_float_tuple(
                "LENS_RANK_SCORE_MAP", (9.5, 8.5, 7.5, 6.5, 5.5)
            ),
            # MultiArmNavigator
            k_arms=_int("LENS_K_ARMS", 4),
            min_arm_allocation=_int("LENS_MIN_ARM_ALLOCATION", 1),
            arm_merge_ci_overlap=_float("LENS_ARM_MERGE_CI_OVERLAP", 0.5),
            # AdaptiveProposalMixer
            min_exploration_ratio=_float("LENS_MIN_EXPLORATION_RATIO", 0.15),
            decay_factor=_float("LENS_DECAY_FACTOR", 0.8),
            # StatisticalStopDecider
            confidence_level=_float("LENS_CONFIDENCE_LEVEL", 0.95),
            confidence_threshold=_float("LENS_CONFIDENCE_THRESHOLD", 8.0),
            info_value_threshold=_float("LENS_INFO_VALUE_THRESHOLD", 2.0),
            min_tokens_remaining=_int("LENS_MIN_TOKENS_REMAINING", 2000),
            relevance_floor=_float("LENS_RELEVANCE_FLOOR", 4.0),
            multi_source_span_threshold=_int("LENS_MULTI_SOURCE_SPAN_THRESHOLD", 5000),
            # ReasoningChainExploiter
            max_targeted_samples=_int("LENS_MAX_TARGETED_SAMPLES", 2),
            signal_score_range=_float_tuple(
                "LENS_SIGNAL_SCORE_RANGE", (4.0, 8.0)
            ),
        )
