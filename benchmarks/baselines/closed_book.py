"""baselines/closed_book.py — Closed-book (no-retrieval) reference baseline.

Why this baseline exists
------------------------
On a benchmark built from a public corpus, an open-book system can score without
retrieving anything: the answer may already be in the model's parameters.
Measured on HotpotQA G_500, the ReAct baseline answered 106 of the 160 questions
where it retrieved none of the gold supporting titles — 62% even after excluding
yes/no questions, which are guessable. Those points say nothing about retrieval.

Closed-book answers from the model alone, with no corpus access, so its score is
the share of the benchmark reachable by memorisation. That number is the floor
any open-book result has to be read against: the portion of an open-book score
at or below the closed-book score is not evidence of retrieval capability.

It is a reference point, not a competitor. It deliberately has no index, no
setup cost, and no evidence, so evidence-based metrics report zero for it — that
is the correct reading, not a defect.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List

from .base_adapter import BaselineAdapter, BaselinePrediction, BaselineSetupResult

_CLOSED_BOOK_PROMPT = """Answer the question from your own knowledge. No documents are provided.

Question:
{question}

Instructions:
- Reply with the minimal answer span only: a name, date, number, or yes/no.
- No explanation, no sentence, no reasoning, no citations.
- If you are unsure, still give your single most likely answer rather than
  declining, so the score reflects what you know rather than how cautious you
  are.
"""


class ClosedBookBaseline(BaselineAdapter):
    """Answer without retrieval, to measure the memorisation floor."""

    def __init__(
        self,
        *,
        name: str = "closed_book",
        citation_name: str = "Closed-Book (no retrieval)",
        llm: Any = None,
    ) -> None:
        self._name = name
        self._citation = citation_name
        self._llm = llm
        self._setup = BaselineSetupResult()

    @property
    def name(self) -> str:
        return self._name

    @property
    def citation_name(self) -> str:
        return self._citation

    async def prepare(self, golden_set: Any = None, bm_adapter: Any = None) -> BaselineSetupResult:
        """Acquire the generation LLM. Nothing is indexed, by definition.

        Setup cost is genuinely zero rather than merely unmeasured: there is no
        corpus pass, so the lifecycle table should show zero for this row.
        """
        start = time.monotonic()
        if self._llm is None and bm_adapter is not None:
            try:
                self._llm = bm_adapter.build_searcher().llm
            except Exception:
                self._llm = None
        self._setup = BaselineSetupResult(
            setup_seconds=round(time.monotonic() - start, 4),
            preprocessing_seconds=0.0,
            index_build_seconds=0.0,
            storage_bytes=0,
            indexed_documents=0,
            expected_documents=0,
            index_required=False,
            query_ready_immediately=True,
            metadata={"index_required": False, "reason": "closed-book reference"},
        )
        return self._setup

    async def predict(self, question: str, context_paths: List[str]) -> BaselinePrediction:
        """Answer from parameters alone; ``context_paths`` is ignored on purpose."""
        start = time.monotonic()
        if self._llm is None:
            return BaselinePrediction(
                answer="",
                elapsed=round(time.monotonic() - start, 4),
                tokens_used=0,
                metadata={
                    "baseline_type": "closed_book",
                    "error": "llm_unavailable",
                    "setup_metrics": self.collect_setup_metrics(),
                },
            )

        prompt = _CLOSED_BOOK_PROMPT.format(question=question)
        resp = await self._llm.achat(
            messages=[{"role": "user", "content": prompt}],
            stream=False,
        )
        answer = resp.content or ""
        usage = getattr(resp, "usage", {}) or {}
        tokens = int(usage.get("total_tokens", 0) or 0)

        return BaselinePrediction(
            answer=answer,
            elapsed=round(time.monotonic() - start, 4),
            tokens_used=tokens,
            metadata={
                "baseline_type": "closed_book",
                # No read_file_ids and no evidence_sources: this system reads
                # nothing, so evidence recall and source grounding must come out
                # at zero. Emitting empty lists here keeps that explicit rather
                # than letting a downstream default imply the fields were lost.
                "read_file_ids": [],
                "evidence_sources": [],
                "retrieval_free": True,
                "setup_metrics": self.collect_setup_metrics(),
            },
        )

    def collect_setup_metrics(self) -> Dict[str, Any]:
        return {
            "setup_seconds": self._setup.setup_seconds,
            "preprocessing_seconds": self._setup.preprocessing_seconds,
            "index_build_seconds": self._setup.index_build_seconds,
            "storage_bytes": self._setup.storage_bytes,
            "indexed_documents": self._setup.indexed_documents,
            "expected_documents": self._setup.expected_documents,
            "index_required": False,
            "query_ready_immediately": True,
        }

    def index_status(self) -> Dict[str, Any]:
        return {
            "index_ready": True,
            "indexed_documents": 0,
            "expected_documents": 0,
            "index_required": False,
        }

    def extra_metadata(self) -> Dict[str, Any]:
        return {
            "baseline_type": "closed_book",
            "index_required": False,
            "rebuild_required": False,
            "query_ready_immediately": True,
            "retrieval_free": True,
        }
