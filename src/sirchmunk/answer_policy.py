# Copyright (c) ModelScope Contributors. All rights reserved.
"""Answer policy seam for the search pipeline.

The retrieval chain decides *what the evidence says*. Whether a low-confidence
outcome should be reported as a refusal or as a best-supported guess is a
consumer decision, not a retrieval one: an interactive product wants an honest
"no results", while an offline evaluation that awards no credit for abstention
wants the best candidate span instead.

Keeping that decision behind a policy object lets the pipeline stay neutral.
The default policy implements product semantics; evaluation harnesses supply
their own policy rather than having the pipeline read their environment flags.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class AnswerPolicy(Protocol):
    """Consumer-supplied decisions about how to report weak answers."""

    def prefers_span_over_refusal(self) -> bool:
        """Return True to report a supported span instead of refusing.

        When True, a refusal outcome that nonetheless carries a valid short
        answer span backed by non-empty evidence is reported as that span.
        Such answers are still never persisted as knowledge.
        """

    def allows_forced_guess(self) -> bool:
        """Return True to allow one extra best-effort synthesis on refusal.

        This spends an additional LLM call to ask for the most plausible span
        supported by partial evidence. Product callers leave this off so that
        an honest refusal stays a refusal.
        """


class DefaultAnswerPolicy:
    """Product semantics: an honest refusal is preserved as a refusal."""

    __slots__ = ()

    def prefers_span_over_refusal(self) -> bool:
        return False

    def allows_forced_guess(self) -> bool:
        return False


DEFAULT_ANSWER_POLICY = DefaultAnswerPolicy()
"""Shared default instance; the policy is stateless."""
