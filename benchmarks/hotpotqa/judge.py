"""HotpotQA answer judge.

Implements the official-style HotpotQA normalization, exact match, and token
F1 fast paths, with an optional LLM semantic equivalence fallback for cases
where lexical overlap is low but the answer may still be semantically correct.
"""
from __future__ import annotations

import json
import logging
import re
import string
from collections import Counter
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_ARTICLES_RE = re.compile(r"\b(a|an|the)\b", flags=re.IGNORECASE)
_REFUSAL_RE = re.compile(
    r"\b(i cannot|i can't|unable to|not able to|i don't know|unknown|"
    r"no results found|cannot determine|insufficient data|not found)\b",
    flags=re.IGNORECASE,
)

_JUDGE_PROMPT = """\
You are judging a HotpotQA answer. Decide whether the prediction answers the
question equivalently to the gold answer. Be strict about entity identity, but
allow harmless aliases, abbreviations, and wording differences.

Question: {question}
Gold answer: {gold}
Prediction: {prediction}

Return ONLY JSON with keys:
{{
  "equivalent": true/false,
  "confidence": 0.0-1.0,
  "reasoning": "short reason"
}}
"""


class HotpotQAJudge:
    """Judge HotpotQA answer correctness and lightweight coverage."""

    def __init__(
        self,
        llm: Optional[Any] = None,
        *,
        enable_llm_judge: bool = True,
        llm_fallback_f1_threshold: float = 0.3,
        equivalence_f1_threshold: float = 0.8,
        confidence_threshold: float = 0.65,
    ) -> None:
        self._llm = llm
        self._enable_llm_judge = enable_llm_judge
        self._llm_fallback_f1_threshold = llm_fallback_f1_threshold
        self._equivalence_f1_threshold = equivalence_f1_threshold
        self._confidence_threshold = confidence_threshold
        self._cache: Dict[tuple[str, str, str], Dict[str, Any]] = {}

    async def judge(
        self,
        prediction: str,
        gold_answer: str,
        question: str = "",
    ) -> Dict[str, Any]:
        """Return HotpotQA answer equivalence metrics.

        The result includes official lexical metrics regardless of whether the
        final boolean decision comes from the lexical fast path or the optional
        LLM fallback.
        """
        short_prediction = extract_short_answer(prediction)
        em = exact_match_score(short_prediction, gold_answer)
        f1 = f1_score(short_prediction, gold_answer)

        base = {
            "em": em,
            "f1": f1,
            "official_em": em,
            "official_f1": f1,
            "official_exact_match": em >= 1.0,
            "official_f1_correct": f1 >= self._equivalence_f1_threshold,
            "normalized_prediction": normalize_answer(short_prediction),
            "normalized_gold": normalize_answer(gold_answer),
            "short_prediction": short_prediction,
            "tokens_used": 0,
            "cached": False,
            "error": None,
        }

        if _is_refusal(short_prediction):
            return {
                **base,
                "equivalent": False,
                "confidence": 1.0,
                "reasoning": "Prediction is a refusal or empty answer.",
                "llm_judge_used": False,
                "llm_equivalent": None,
            }

        if em >= 1.0:
            return {
                **base,
                "equivalent": True,
                "confidence": 1.0,
                "reasoning": "Normalized exact match.",
                "llm_judge_used": False,
                "llm_equivalent": None,
            }

        if f1 >= self._equivalence_f1_threshold:
            return {
                **base,
                "equivalent": True,
                "confidence": min(0.99, max(0.8, f1)),
                "reasoning": "Token F1 exceeds equivalence threshold.",
                "llm_judge_used": False,
                "llm_equivalent": None,
            }

        should_call_llm = (
            self._enable_llm_judge
            and self._llm is not None
            and f1 < max(self._llm_fallback_f1_threshold, self._equivalence_f1_threshold)
        )
        if not should_call_llm:
            return {
                **base,
                "equivalent": False,
                "confidence": 1.0 - f1,
                "reasoning": "Lexical metrics below equivalence threshold; LLM fallback not available or disabled.",
                "llm_judge_used": False,
                "llm_equivalent": None,
            }

        cache_key = (
            normalize_answer(question),
            normalize_answer(short_prediction),
            normalize_answer(gold_answer),
        )
        if cache_key in self._cache:
            cached = dict(self._cache[cache_key])
            cached.update(base)
            cached["cached"] = True
            return cached

        prompt = _JUDGE_PROMPT.format(
            question=question or "N/A",
            gold=gold_answer,
            prediction=short_prediction,
        )
        tokens_used = 0
        try:
            resp = await self._llm.achat(
                messages=[{"role": "user", "content": prompt}],
                stream=False,
            )
            tokens_used = _extract_tokens(resp)
            parsed = _parse_json_response(resp.content or "")
            equivalent = bool(parsed.get("equivalent", False))
            confidence = _clamp_float(parsed.get("confidence", 0.0))
            reasoning = str(parsed.get("reasoning", ""))[:500]
            if equivalent and confidence < self._confidence_threshold:
                equivalent = False
                reasoning = (
                    f"LLM confidence {confidence:.2f} below threshold "
                    f"{self._confidence_threshold:.2f}. " + reasoning
                )
            result = {
                **base,
                "equivalent": equivalent,
                "confidence": confidence,
                "reasoning": reasoning or "LLM semantic judge.",
                "tokens_used": tokens_used,
                "llm_judge_used": True,
                "llm_equivalent": equivalent,
            }
            self._cache[cache_key] = {
                k: v for k, v in result.items() if k not in ("cached", "tokens_used")
            }
            return result
        except Exception as exc:
            logger.warning("HotpotQA LLM judge failed: %s", exc)
            return {
                **base,
                "equivalent": False,
                "confidence": 0.0,
                "reasoning": f"LLM judge failed: {exc}",
                "tokens_used": tokens_used,
                "llm_judge_used": True,
                "llm_equivalent": False,
                "error": str(exc),
            }

    async def judge_coverage(
        self,
        prediction: str,
        question: str,
    ) -> Dict[str, Any]:
        """Lightweight answer coverage check.

        Coverage is intentionally conservative and LLM-free for P0: any
        non-refusal prediction with a plausible answer span is considered to
        have answer coverage. Source grounding is handled separately by the
        HotpotQAEvidenceEvaluator.
        """
        short_prediction = extract_short_answer(prediction)
        has_coverage = bool(short_prediction.strip()) and not _is_refusal(short_prediction)
        return {
            "has_coverage": has_coverage,
            "confidence": 0.8 if has_coverage else 1.0,
            "reasoning": "Non-refusal answer span detected." if has_coverage else "No usable answer span.",
            "tokens_used": 0,
            "error": None,
        }


def normalize_answer(text: str) -> str:
    """Official HotpotQA/SQuAD-style answer normalization."""
    if text is None:
        return ""
    text = str(text).lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = _ARTICLES_RE.sub(" ", text)
    return " ".join(text.split())


def exact_match_score(prediction: str, gold_answer: str) -> float:
    return 1.0 if normalize_answer(prediction) == normalize_answer(gold_answer) else 0.0


def f1_score(prediction: str, gold_answer: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold_answer).split()
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def extract_short_answer(text: str) -> str:
    """Extract a concise answer span from verbose model output when possible."""
    if not text:
        return ""
    s = str(text).strip()
    patterns = [
        r"\*\*Answer\s*:\s*(.+?)(?:\*\*|\n|$)",
        r"(?:^|\n)Answer\s*:\s*(.+?)(?:\n|$)",
        r"(?:^|\n)Final answer\s*:\s*(.+?)(?:\n|$)",
        r"(?:^|\n)The answer is\s+(.+?)(?:\.|\n|$)",
    ]
    for pattern in patterns:
        match = re.search(pattern, s, flags=re.IGNORECASE | re.DOTALL)
        if match:
            candidate = match.group(1).strip()
            candidate = re.sub(r"^[-*\s]+", "", candidate).strip()
            if candidate:
                return candidate[:500]
    lines = [line.strip() for line in s.splitlines() if line.strip()]
    if not lines:
        return s[:500]
    return lines[-1][:500] if len(lines[-1]) <= 500 else s[:500]


def _is_refusal(text: str) -> bool:
    if not text or not text.strip():
        return True
    return bool(_REFUSAL_RE.search(text[:300]))


def _extract_tokens(resp: Any) -> int:
    usage = getattr(resp, "usage", None)
    if isinstance(usage, dict):
        return int(usage.get("total_tokens") or usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0))
    return 0


def _parse_json_response(raw: str) -> Dict[str, Any]:
    cleaned = re.sub(r"```(?:json)?\s*", "", raw or "").strip().rstrip("`").strip()
    candidates = [cleaned]
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if match:
        candidates.append(match.group(0))
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except (json.JSONDecodeError, TypeError, ValueError):
            continue
    lower = cleaned.lower()
    return {
        "equivalent": "true" in lower and "false" not in lower,
        "confidence": 0.5,
        "reasoning": cleaned[:300],
    }


def _clamp_float(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0
