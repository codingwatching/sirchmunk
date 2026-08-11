# Copyright (c) ModelScope Contributors. All rights reserved.
"""Answer policies for benchmark evaluation.

Benchmarks score a refusal exactly like a wrong answer: abstaining earns no
credit, so an evaluated system should report its best supported span rather
than decline. That is an evaluation-protocol decision, which belongs here
rather than in the retrieval pipeline.

The pipeline holds up its end by always emitting a candidate answer alongside
its own rating of the supporting evidence, so a policy here decides whether a
weakly supported candidate is shown or replaced by a refusal. Nothing is
reconstructed after the fact, because the candidate is never discarded.

The pipeline exposes ``sirchmunk.answer_policy.AnswerPolicy`` for this purpose;
adapters construct :class:`NoAbstentionAnswerPolicy` and pass it to
``AgenticSearch``, so ``search.py`` no longer needs to read benchmark
environment flags to decide how an answer should be reported.
"""
from __future__ import annotations

import os

_TRUTHY = {"1", "true", "yes", "on"}

# Legacy flag that gated this behaviour while it lived inside the pipeline.
# Reading it here preserves existing benchmark configurations.
REFUSAL_FALLBACK_ENV = "SIRCHMUNK_REFUSAL_FALLBACK"


class NoAbstentionAnswerPolicy:
    """Report the best supported span instead of abstaining.

    Args:
        prefer_span: Report a valid short span backed by evidence instead of a
            refusal.
        forced_guess: Additionally allow one extra best-effort synthesis call
            when the pipeline would otherwise refuse.
    """

    __slots__ = ("_prefer_span", "_forced_guess")

    def __init__(self, prefer_span: bool = True, forced_guess: bool = True) -> None:
        self._prefer_span = bool(prefer_span)
        self._forced_guess = bool(forced_guess)

    def prefers_span_over_refusal(self) -> bool:
        return self._prefer_span

    def allows_forced_guess(self) -> bool:
        return self._forced_guess

    def renders_refusal_for(self, sufficiency: str) -> bool:
        """Never render a refusal: an unanswered question scores as wrong."""
        return False

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return (
            f"NoAbstentionAnswerPolicy(prefer_span={self._prefer_span}, "
            f"forced_guess={self._forced_guess})"
        )


class HonestRefusalAnswerPolicy:
    """Keep refusals intact; used by arms that measure abstention behaviour."""

    __slots__ = ()

    def prefers_span_over_refusal(self) -> bool:
        return False

    def allows_forced_guess(self) -> bool:
        return False

    def renders_refusal_for(self, sufficiency: str) -> bool:
        """Refuse only when the agent rated its own evidence as unrelated.

        The pipeline now always produces a candidate answer plus a rating of the
        evidence behind it, so abstention is a presentation choice made here.
        ``partial`` still answers: the candidate is inference-backed, which is
        worth showing, whereas ``absent`` marks a guess this arm suppresses.
        """
        return str(sufficiency or "").strip().lower() == "absent"


def policy_from_env(env: dict | None = None) -> object:
    """Build the answer policy implied by the benchmark environment.

    Returns a no-abstention policy when the legacy refusal-fallback flag is
    truthy, otherwise the honest-refusal policy. Keeping the flag lookup in the
    benchmarks layer means the pipeline receives an explicit decision instead of
    inspecting evaluation configuration itself.
    """
    source = env if env is not None else os.environ
    enabled = str(source.get(REFUSAL_FALLBACK_ENV, "")).strip().lower() in _TRUTHY
    return NoAbstentionAnswerPolicy() if enabled else HonestRefusalAnswerPolicy()
