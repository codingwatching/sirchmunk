# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Tuple

from sirchmunk.learnings.evidence_processor import SampleWindow


@dataclass
class ArmState:
    """Tracks the state of a single bandit arm (document segment).

    Attributes:
        arm_id: Unique identifier for this arm.
        segment_start: Start character index in the document.
        segment_end: End character index in the document.
        pull_count: Number of times this arm has been sampled.
        total_reward: Cumulative reward (sum of scores).
        mean_reward: Running mean reward.
        ci_lower: Lower bound of the confidence interval.
        ci_upper: Upper bound of the confidence interval.
    """

    arm_id: int
    segment_start: int
    segment_end: int
    pull_count: int = 0
    total_reward: float = 0.0
    mean_reward: float = 0.0
    ci_lower: float = 0.0
    ci_upper: float = 0.0


class EvidenceEvaluator(Protocol):
    """Protocol for evaluating candidate evidence fragments.

    Implementations score and rank a batch of candidate sample windows
    against the user query, returning samples annotated with scores and
    reasoning.
    """

    async def evaluate(
        self,
        samples: List[SampleWindow],
        query: str,
        keywords: Optional[Dict[str, float]] = None,
    ) -> List[SampleWindow]:
        """Evaluate and rank candidate samples.

        Args:
            samples: Candidate sample windows to evaluate.
            query: The user query string.
            keywords: Optional keyword-to-IDF-weight mapping for enhanced
                relevance scoring.

        Returns:
            The input samples annotated with ``score`` and ``reasoning``
            fields, sorted by relevance (highest first).
        """
        ...


class SamplingStrategy(Protocol):
    """Protocol for evidence sampling strategies.

    Implementations generate candidate sampling windows for each round,
    balancing exploration and exploitation based on prior results.
    """

    async def next_round(
        self,
        round_num: int,
        query: str,
        keywords: Optional[Dict[str, float]] = None,
        prev_candidates: Optional[List[SampleWindow]] = None,
        doc_content: str = "",
        doc_len: int = 0,
    ) -> List[SampleWindow]:
        """Generate candidate samples for the next round.

        Args:
            round_num: Current round number (1-indexed).
            query: The user query string.
            keywords: Optional keyword-to-IDF-weight mapping.
            prev_candidates: Evaluated candidates from previous rounds,
                sorted by score descending. None on the first round.
            doc_content: Full document content for position-based sampling.
            doc_len: Length of the document in characters.

        Returns:
            A list of new candidate sample windows to evaluate.
        """
        ...


class StopStrategy(Protocol):
    """Protocol for stop decision logic.

    Implementations determine whether the sampling loop should terminate
    based on statistical confidence, information gain, or score thresholds.
    """

    async def should_stop(
        self,
        all_candidates: List[SampleWindow],
        query: str,
        round_num: int,
        max_rounds: int,
        confidence_threshold: float = 8.5,
        **kwargs: Any,
    ) -> Tuple[bool, str]:
        """Decide whether to stop the sampling loop.

        Args:
            all_candidates: All evaluated candidates so far, sorted by
                score descending.
            query: The user query string.
            round_num: Current round number (1-indexed).
            max_rounds: Maximum allowed rounds.
            confidence_threshold: Score threshold for high-confidence stop.
            **kwargs: Additional strategy-specific parameters.

        Returns:
            A tuple of (should_stop, reason) where reason is a human-readable
            explanation of the decision.
        """
        ...
