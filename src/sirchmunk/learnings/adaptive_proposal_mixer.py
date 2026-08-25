# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations

import math
from typing import Tuple

from loguru import logger

from sirchmunk.learnings.lens_config import LensConfig


class AdaptiveProposalMixer:
    """自适应提议分布混合器。

    动态计算 exploration vs exploitation 的比例 λ(t)，
    确保全局探索下界 ε 始终存在（防止陷入局部最优）。

    公式: λ(t) = max(ε, decay^t × entropy_factor × budget_factor)

    The mixer balances two competing objectives:
    - Exploration: discovering new high-value regions in the document space.
    - Exploitation: refining known high-value regions for better evidence.

    λ(t) ∈ [ε, 1.0]:
    - λ → 1.0 means mostly exploration (high uncertainty, early rounds).
    - λ → ε means mostly exploitation (low uncertainty, late rounds).

    Attributes:
        config: LensConfig providing decay_factor and min_exploration_ratio.
        epsilon: Minimum exploration ratio floor (prevents local optima traps).
        decay: Exponential decay factor per round.
    """

    def __init__(self, config: LensConfig) -> None:
        """Initialize the AdaptiveProposalMixer.

        Args:
            config: LensConfig providing min_exploration_ratio (ε) and
                decay_factor for the exponential schedule.
        """
        self.config = config
        self.epsilon: float = config.min_exploration_ratio
        self.decay: float = config.decay_factor

    def compute_lambda(
        self,
        round_num: int,
        posterior_entropy: float,
        budget_remaining: float,
    ) -> float:
        """计算混合权重 λ(t)。

        Combines three signals into a single exploration weight:
        1. Temporal decay: decay^t — reduces exploration over rounds.
        2. Entropy factor: normalized posterior uncertainty from bandit arms.
        3. Budget factor: remaining budget proportion encourages exploration
           when budget is ample and exploitation when scarce.

        Formula:
            λ(t) = max(ε, decay^t × entropy_factor × budget_factor)

        Args:
            round_num: Current round number (1-indexed).
            posterior_entropy: Total posterior entropy from MultiArmNavigator.
                Higher entropy → more uncertainty → more exploration.
            budget_remaining: Fraction of total token budget remaining,
                in [0.0, 1.0]. 1.0 means full budget available.

        Returns:
            λ ∈ [ε, 1.0], where higher values favor exploration.
        """
        # 1. Temporal decay: exponentially decreasing exploration over rounds
        temporal = math.pow(self.decay, round_num - 1)

        # 2. Entropy factor: sigmoid normalization of posterior entropy
        # Maps entropy from [0, +inf) to (0, 1] — high entropy → factor ≈ 1
        # Using tanh for smooth saturation
        entropy_factor = math.tanh(posterior_entropy) if posterior_entropy > 0 else 0.0

        # 3. Budget factor: square root scaling for smoother transition
        # When budget is full (1.0): factor = 1.0 (explore freely)
        # When budget is low (0.1): factor ≈ 0.32 (conserve, exploit)
        budget_factor = math.sqrt(max(0.0, min(1.0, budget_remaining)))

        # Combine signals
        raw_lambda = temporal * max(entropy_factor, 0.1) * budget_factor

        # Clamp to [ε, 1.0]
        lambda_t = max(self.epsilon, min(1.0, raw_lambda))

        logger.debug(
            f"[AdaptiveProposalMixer] λ(t={round_num})={lambda_t:.3f} "
            f"(temporal={temporal:.3f}, entropy={entropy_factor:.3f}, "
            f"budget={budget_factor:.3f})"
        )

        return lambda_t

    def split_budget(
        self, total_samples: int, lambda_t: float
    ) -> Tuple[int, int]:
        """将总采样数分为 exploration 和 exploitation。

        Exploration samples are distributed to high-uncertainty arms or
        random positions. Exploitation samples are focused around known
        high-value regions.

        Guarantees:
        - At least 1 sample for exploration (when total >= 2).
        - At least 1 sample for exploitation (when total >= 2).

        Args:
            total_samples: Total number of samples to generate this round.
            lambda_t: Exploration weight from compute_lambda(), in [ε, 1.0].

        Returns:
            Tuple of (num_exploration, num_exploitation) summing to total_samples.
        """
        if total_samples <= 0:
            return (0, 0)

        if total_samples == 1:
            # Single sample: decide by lambda threshold
            if lambda_t >= 0.5:
                return (1, 0)
            else:
                return (0, 1)

        # Compute exploration count
        num_exploration = max(1, round(total_samples * lambda_t))
        # Ensure at least 1 for exploitation
        num_exploration = min(num_exploration, total_samples - 1)
        num_exploitation = total_samples - num_exploration

        logger.debug(
            f"[AdaptiveProposalMixer] Budget split: "
            f"explore={num_exploration}, exploit={num_exploitation} "
            f"(total={total_samples}, λ={lambda_t:.3f})"
        )

        return (num_exploration, num_exploitation)
