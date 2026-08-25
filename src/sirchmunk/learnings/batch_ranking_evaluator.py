# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations

import asyncio
import json
import re
from typing import Dict, List, Optional, Tuple

from loguru import logger

from sirchmunk.learnings.evidence_processor import SampleWindow
from sirchmunk.learnings.lens_config import LensConfig
from sirchmunk.llm.openai_chat import OpenAIChat
from sirchmunk.llm.prompts import EVALUATE_EVIDENCE_SAMPLE, LISTWISE_RANKING_PROMPT


class PointwiseEvaluator:
    """Wraps the existing per-sample LLM evaluation logic as the default/fallback evaluator.

    Implements the ``EvidenceEvaluator`` protocol by scoring each sample
    independently and concurrently via asyncio.gather.
    """

    def __init__(self, llm: OpenAIChat, log_callback=None) -> None:
        """Initialize the PointwiseEvaluator.

        Args:
            llm: An OpenAIChat instance for LLM calls.
            log_callback: Optional logging callback (unused, kept for interface consistency).
        """
        self.llm = llm

    async def evaluate(
        self,
        samples: List[SampleWindow],
        query: str,
        keywords: Optional[Dict[str, float]] = None,
    ) -> List[SampleWindow]:
        """Evaluate samples individually and concurrently.

        Each sample is scored independently using the EVALUATE_EVIDENCE_SAMPLE
        prompt. Results are sorted by score descending.

        Args:
            samples: Candidate sample windows to evaluate.
            query: The user query string.
            keywords: Optional keyword-to-IDF-weight mapping (unused in pointwise mode).

        Returns:
            The input samples annotated with ``score`` and ``reasoning``,
            sorted by relevance (highest first).
        """
        if not samples:
            return []

        tasks = [self._evaluate_single(s, query) for s in samples]
        evaluated = await asyncio.gather(*tasks)
        result = sorted(evaluated, key=lambda s: s.score, reverse=True)
        return result

    async def _evaluate_single(self, sample: SampleWindow, query: str) -> SampleWindow:
        """Evaluate a single sample using the pointwise prompt.

        Args:
            sample: The candidate sample window.
            query: The user query string.

        Returns:
            The sample annotated with score and reasoning.
        """
        prompt = EVALUATE_EVIDENCE_SAMPLE.format(
            query=query,
            sample_source=sample.source,
            sample_content=sample.content,
        )
        try:
            resp_obj = await self.llm.achat([{"role": "user", "content": prompt}])
            resp: str = resp_obj.content

            data = self._parse_evaluation_json(resp)
            if data is not None:
                sample.score = float(data.get("score", 0))
                sample.reasoning = data.get("reasoning", "")
            else:
                logger.warning(
                    f"Unparseable LLM response for sample at {sample.start_idx}, "
                    f"response (first 200 chars): {resp[:200]!r}"
                )
                sample.score = 0.0
        except Exception as e:
            logger.warning(f"Error evaluating sample at {sample.start_idx}: {e}")
            sample.score = 0.0

        return sample

    @staticmethod
    def _parse_evaluation_json(text: str) -> Optional[dict]:
        """Extract {"score": ..., "reasoning": ...} from LLM output.

        Tries direct parse, markdown-fence stripping, outermost {…} extraction,
        and regex score fallback.

        Args:
            text: Raw LLM response text.

        Returns:
            Parsed dict or None on failure.
        """
        if not text:
            return None
        text = text.strip()

        # 1. Direct parse
        try:
            return json.loads(text)
        except (json.JSONDecodeError, TypeError):
            pass

        # 2. Strip markdown code fences
        cleaned = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
        cleaned = re.sub(r"```\s*$", "", cleaned, flags=re.MULTILINE).strip()
        try:
            return json.loads(cleaned)
        except (json.JSONDecodeError, TypeError):
            pass

        # 3. Extract first {...} block
        m = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except (json.JSONDecodeError, TypeError):
                pass
            m2 = re.search(r"\{[^{}]+\}", cleaned)
            if m2:
                try:
                    return json.loads(m2.group())
                except (json.JSONDecodeError, TypeError):
                    pass

        # 4. Regex fallback for score
        score_m = re.search(r'"score"\s*:\s*(\d+(?:\.\d+)?)', text)
        if score_m:
            return {"score": float(score_m.group(1)), "reasoning": ""}

        return None


class BatchRankingEvaluator:
    """Listwise ranking evaluator implementing the EvidenceEvaluator protocol.

    Evaluates K candidate fragments in a single LLM call by constructing a
    listwise ranking prompt. Supports chunked ranking when samples exceed
    ``batch_size``, and falls back to ``PointwiseEvaluator`` on parse failure.
    """

    # Maximum character length for each candidate snippet in the prompt.
    _SNIPPET_TRUNCATE_LEN = 300

    def __init__(self, llm: OpenAIChat, config: LensConfig, log_callback=None) -> None:
        """Initialize the BatchRankingEvaluator.

        Args:
            llm: An OpenAIChat instance for LLM calls.
            config: LensConfig providing batch_size and rank_score_map.
            log_callback: Optional logging callback (unused, kept for interface consistency).
        """
        self.llm = llm
        self.config = config
        self._fallback = PointwiseEvaluator(llm, log_callback=log_callback)

    async def evaluate(
        self,
        samples: List[SampleWindow],
        query: str,
        keywords: Optional[Dict[str, float]] = None,
    ) -> List[SampleWindow]:
        """Evaluate and rank candidate samples via listwise ranking.

        Implements the ``EvidenceEvaluator`` protocol. Falls back to pointwise
        evaluation if ranking parse fails.

        Args:
            samples: Candidate sample windows to evaluate.
            query: The user query string.
            keywords: Optional keyword-to-IDF-weight mapping.

        Returns:
            The input samples annotated with ``score`` and ``reasoning``,
            sorted by relevance (highest first).
        """
        if not samples:
            return []

        if len(samples) == 1:
            # Single sample — pointwise is more efficient
            return await self._fallback.evaluate(samples, query, keywords)

        try:
            ranked = await self.rank_batch(samples, query, keywords)
            return ranked
        except Exception as e:
            logger.warning(f"BatchRankingEvaluator failed, falling back to pointwise: {e}")
            return await self._fallback.evaluate(samples, query, keywords)

    async def rank_batch(
        self,
        samples: List[SampleWindow],
        query: str,
        keywords: Optional[Dict[str, float]] = None,
    ) -> List[SampleWindow]:
        """Rank all samples via chunked listwise ranking.

        When the number of samples exceeds ``config.batch_size``, splits into
        chunks, ranks each independently, then merges by assigned score.

        Args:
            samples: Candidate sample windows to rank.
            query: The user query string.
            keywords: Optional keyword-to-IDF-weight mapping.

        Returns:
            Samples annotated with score and reasoning, sorted descending.

        Raises:
            ValueError: If all chunks fail to parse.
        """
        batch_size = self.config.batch_size

        if len(samples) <= batch_size:
            # Single chunk — rank directly
            ranked_indices = await self._rank_chunk(samples, query)
            if ranked_indices is None:
                raise ValueError("Failed to parse ranking response")
            return self._assign_scores(samples, ranked_indices)

        # Multi-chunk: split, rank independently, merge by score
        chunks: List[List[SampleWindow]] = [
            samples[i : i + batch_size]
            for i in range(0, len(samples), batch_size)
        ]

        all_scored: List[SampleWindow] = []
        for chunk in chunks:
            ranked_indices = await self._rank_chunk(chunk, query)
            if ranked_indices is None:
                # Fallback individual chunk to pointwise
                logger.warning("Chunk ranking failed, falling back to pointwise for this chunk")
                chunk_result = await self._fallback.evaluate(chunk, query, keywords)
                all_scored.extend(chunk_result)
            else:
                scored_chunk = self._assign_scores(chunk, ranked_indices)
                all_scored.extend(scored_chunk)

        # Sort all scored samples by score descending
        all_scored.sort(key=lambda s: s.score, reverse=True)
        return all_scored

    async def _rank_chunk(
        self, chunk_samples: List[SampleWindow], query: str
    ) -> Optional[List[Tuple[int, str]]]:
        """Execute listwise ranking for a single chunk (<=batch_size).

        Args:
            chunk_samples: Samples in this chunk.
            query: The user query string.

        Returns:
            List of (original_index, reasoning) tuples ordered by rank
            (best first), or None if parse fails.
        """
        prompt = self._build_prompt(chunk_samples, query)

        try:
            resp_obj = await self.llm.achat([{"role": "user", "content": prompt}])
            resp_text: str = resp_obj.content
        except Exception as e:
            logger.warning(f"LLM call failed in _rank_chunk: {e}")
            return None

        return self._parse_response(resp_text, len(chunk_samples))

    def _build_prompt(self, samples: List[SampleWindow], query: str) -> str:
        """Construct the listwise ranking prompt.

        Each candidate is labeled [A], [B], [C], ... and truncated to
        ``_SNIPPET_TRUNCATE_LEN`` characters.

        Args:
            samples: Samples to include as candidates.
            query: The user query string.

        Returns:
            The formatted prompt string.
        """
        candidates_block = []
        for i, sample in enumerate(samples):
            label = chr(ord("A") + i)
            snippet = sample.content[: self._SNIPPET_TRUNCATE_LEN]
            if len(sample.content) > self._SNIPPET_TRUNCATE_LEN:
                snippet += "..."
            candidates_block.append(f"[{label}] {snippet}")

        candidates_text = "\n\n".join(candidates_block)
        return LISTWISE_RANKING_PROMPT.format(
            query=query,
            num_candidates=len(samples),
            candidates=candidates_text,
        )

    def _parse_response(
        self, response_text: str, num_samples: int
    ) -> Optional[List[Tuple[int, str]]]:
        """Parse the JSON ranking response from the LLM.

        Expected format: {"ranking": [0, 2, 1, ...], "reasons": ["...", ...]}

        Args:
            response_text: Raw LLM response.
            num_samples: Expected number of candidates.

        Returns:
            List of (original_index, reasoning) in rank order, or None on failure.
        """
        if not response_text:
            return None

        text = response_text.strip()

        # Try parsing strategies
        data = self._try_parse_json(text)
        if data is None:
            return None

        # Validate required fields
        ranking = data.get("ranking")
        reasons = data.get("reasons", [])

        if not isinstance(ranking, list):
            logger.warning("Ranking response missing 'ranking' array")
            return None

        if len(ranking) != num_samples:
            # Tolerate partial rankings — use what we have
            if len(ranking) < 1:
                logger.warning("Empty ranking array")
                return None
            logger.debug(
                f"Ranking length mismatch: expected {num_samples}, got {len(ranking)}"
            )

        # Validate indices are within range
        valid_indices = set(range(num_samples))
        for idx in ranking:
            if not isinstance(idx, int) or idx not in valid_indices:
                logger.warning(f"Invalid index {idx} in ranking (num_samples={num_samples})")
                return None

        # Check for duplicate indices
        if len(set(ranking)) != len(ranking):
            logger.warning("Duplicate indices in ranking response")
            return None

        # Pad reasons if shorter than ranking
        while len(reasons) < len(ranking):
            reasons.append("")

        return [(ranking[i], reasons[i]) for i in range(len(ranking))]

    def _assign_scores(
        self,
        samples: List[SampleWindow],
        ranked_indices: List[Tuple[int, str]],
    ) -> List[SampleWindow]:
        """Map rank positions to scores and annotate samples.

        Uses ``config.rank_score_map`` for score assignment. Ranks beyond the
        map length receive the minimum score with linear decay.

        Args:
            samples: The original sample list.
            ranked_indices: (original_index, reasoning) pairs ordered by rank.

        Returns:
            Samples sorted by score descending with score/reasoning populated.
        """
        score_map = self.config.rank_score_map
        min_score = score_map[-1] if score_map else 5.0

        for rank_pos, (orig_idx, reasoning) in enumerate(ranked_indices):
            if rank_pos < len(score_map):
                score = score_map[rank_pos]
            else:
                # Linear decay below the minimum mapped score
                decay = (rank_pos - len(score_map) + 1) * 0.5
                score = max(min_score - decay, 1.0)

            samples[orig_idx].score = score
            samples[orig_idx].reasoning = reasoning

        # Sort by score descending
        return sorted(samples, key=lambda s: s.score, reverse=True)

    @staticmethod
    def _try_parse_json(text: str) -> Optional[dict]:
        """Attempt multiple strategies to extract a JSON object from text.

        Args:
            text: Raw text potentially containing JSON.

        Returns:
            Parsed dict or None.
        """
        # 1. Direct parse
        try:
            return json.loads(text)
        except (json.JSONDecodeError, TypeError):
            pass

        # 2. Strip markdown fences
        cleaned = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
        cleaned = re.sub(r"```\s*$", "", cleaned, flags=re.MULTILINE).strip()
        try:
            return json.loads(cleaned)
        except (json.JSONDecodeError, TypeError):
            pass

        # 3. Extract first {...} block
        m = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except (json.JSONDecodeError, TypeError):
                pass

        return None
