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

# ---------------------------------------------------------------------------
# Deterministic canonicalization (no LLM): surface forms that cannot change
# which entity or quantity an answer refers to.
#
# Official HotpotQA normalization only lowercases, strips punctuation and drops
# articles, so it scores "three" against "3" as a miss. Those misses are not
# retrieval or reasoning failures, and counting them as such misdirects tuning.
# What follows is deliberately narrow: every rule here is reference-preserving.
#
# Explicitly NOT handled, because these do change the referent and must be left
# to a semantic judge: administrative or organizational levels ("Albany" vs
# "Albany County", "BBC" vs "BBC Radio 1"), and brand versus category
# ("Plymouth Gin" vs "gin").
# ---------------------------------------------------------------------------

_NUMBER_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13",
    "fourteen": "14", "fifteen": "15", "sixteen": "16", "seventeen": "17",
    "eighteen": "18", "nineteen": "19", "twenty": "20",
}

_ORDINAL_WORDS = {
    "first": "1", "second": "2", "third": "3", "fourth": "4", "fifth": "5",
    "sixth": "6", "seventh": "7", "eighth": "8", "ninth": "9", "tenth": "10",
    "eleventh": "11", "twelfth": "12",
}

# Units and counted nouns. Dropped only when they trail a number, so that
# "78.5 mi long" reduces to "78.5" while "Long Island" keeps both words.
_MEASURE_WORDS = frozenset({
    "mi", "mile", "miles", "km", "kilometre", "kilometres", "kilometer",
    "kilometers", "m", "metre", "metres", "meter", "meters", "ft", "foot",
    "feet", "long", "tall", "high", "wide", "deep",
    "member", "members", "award", "awards", "year", "years", "time", "times",
    "season", "seasons", "album", "albums", "episode", "episodes",
    "employee", "employees", "vote", "votes", "people", "person",
    "goal", "goals", "point", "points", "win", "wins", "day", "days",
})

# Legal-entity suffixes, dropped only from the tail of the answer.
_CORPORATE_SUFFIXES = frozenset({
    "inc", "incorporated", "ltd", "limited", "llc", "plc", "corp",
    "corporation", "holdings", "holding", "co", "company", "group", "sa", "ag",
})

_NUMERIC_RE = re.compile(r"^\d+(?:[.,]\d+)?$")


def _is_numeric_token(token: str) -> bool:
    return bool(_NUMERIC_RE.match(token))


def canonicalize_answer(text: str) -> str:
    """Reduce reference-preserving surface variation on top of normalization.

    Applied to gold and prediction alike, so it can only ever turn a pair that
    means the same thing into a match; it never relaxes what an answer refers
    to. See the rule commentary above for what is intentionally left alone.
    """
    base = normalize_answer(text)
    if not base:
        return ""
    # "5th" -> "5": an ordinal suffix on a digit never changes the referent.
    base = re.sub(r"\b(\d+)(?:st|nd|rd|th)\b", r"\1", base)
    tokens = base.split()

    # Ordinal words are unconditionally interchangeable with their digits
    # ("Fifth Avenue" and "5th Avenue" name one street).
    tokens = [_ORDINAL_WORDS.get(t, t) for t in tokens]

    # Number words convert only as a lone answer or when a unit follows, so a
    # name that happens to contain one ("One Direction") is left intact.
    converted: list[str] = []
    for idx, token in enumerate(tokens):
        spelled = _NUMBER_WORDS.get(token)
        if spelled is not None:
            lone = len(tokens) == 1
            unit_follows = idx + 1 < len(tokens) and tokens[idx + 1] in _MEASURE_WORDS
            converted.append(spelled if (lone or unit_follows) else token)
        else:
            converted.append(token)

    # Drop units/counted nouns that trail a number.
    kept: list[str] = []
    for token in converted:
        if token in _MEASURE_WORDS and kept and _is_numeric_token(kept[-1]):
            continue
        kept.append(token)

    # Drop legal-entity suffixes from the tail only, never mid-name.
    while len(kept) > 1 and kept[-1] in _CORPORATE_SUFFIXES:
        kept.pop()

    return " ".join(kept)


def normalized_exact_match_score(prediction: str, gold_answer: str) -> float:
    """Exact match after deterministic canonicalization.

    Reported alongside ``official_em`` rather than replacing it: the official
    number stays comparable with published results, while this one measures the
    same answers without penalising surface form.
    """
    pred = canonicalize_answer(prediction)
    gold = canonicalize_answer(gold_answer)
    return 1.0 if pred and gold and pred == gold else 0.0


_REFUSAL_RE = re.compile(
    r"\b(i cannot|i can't|unable to|not able to|i don't know|unknown|"
    r"no results found|cannot determine|insufficient data|not found)\b",
    flags=re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Answer form: is the prediction a minimal span, or a sentence that explains
# itself?
#
# This is scored separately from semantic equivalence because the two are
# independent properties, and conflating them makes the metric unreadable. It
# matters for cross-system fairness: official EM penalises a system that answers
# in prose, while a purely semantic judge does not, so a verbose system collects
# semantic credit that a span-constrained system is denied. Reporting form
# alongside equivalence lets a comparison hold both systems to one standard.
# ---------------------------------------------------------------------------

_ANSWER_FORM_MAX_TOKENS = 12

_EXPLANATORY_RE = re.compile(
    r"\b(based on|according to|as stated|the evidence|the retrieved|"
    r"therefore|because|which means|this means|in summary|note that|"
    r"the answer is|source|passage)\b",
    flags=re.IGNORECASE,
)

_SENTENCE_SPLIT_RE = re.compile(r"[.!?](?:\s|$)")


def answer_form_report(text: str) -> Dict[str, Any]:
    """Describe whether *text* reads as a minimal answer span.

    Reports the reason rather than a bare boolean so a failing comparison can be
    attributed instead of merely observed.
    """
    raw = (text or "").strip()
    if not raw:
        return {"form_compliant": False, "form_reason": "empty", "form_tokens": 0}
    tokens = normalize_answer(raw).split()
    n = len(tokens)
    # Trailing punctuation on a single clause is fine; two or more sentences
    # means the answer is narrating rather than answering.
    sentences = [s for s in _SENTENCE_SPLIT_RE.split(raw) if s.strip()]
    if len(sentences) > 1:
        return {"form_compliant": False, "form_reason": "multi_sentence", "form_tokens": n}
    if _EXPLANATORY_RE.search(raw):
        return {"form_compliant": False, "form_reason": "explanatory", "form_tokens": n}
    if n > _ANSWER_FORM_MAX_TOKENS:
        return {"form_compliant": False, "form_reason": "too_long", "form_tokens": n}
    return {"form_compliant": True, "form_reason": "", "form_tokens": n}


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
        normalized_em = normalized_exact_match_score(short_prediction, gold_answer)

        base = {
            "em": em,
            "f1": f1,
            "official_em": em,
            "official_f1": f1,
            "official_exact_match": em >= 1.0,
            "official_f1_correct": f1 >= self._equivalence_f1_threshold,
            "normalized_em": normalized_em,
            "normalized_exact_match": normalized_em >= 1.0,
            "canonical_prediction": canonicalize_answer(short_prediction),
            "canonical_gold": canonicalize_answer(gold_answer),
            "normalized_prediction": normalize_answer(short_prediction),
            "normalized_gold": normalize_answer(gold_answer),
            "short_prediction": short_prediction,
            "tokens_used": 0,
            "cached": False,
            "error": None,
            # Whether the verdict is a real judgement. False for every decided
            # outcome; True only when the judge could not reach one, so an
            # infrastructure failure is never silently counted as a wrong
            # answer by consumers that read ``equivalent`` alone.
            "indeterminate": False,
            "judge_status": "",
            **answer_form_report(short_prediction),
        }

        # Lexical agreement is checked before the refusal test on purpose. The
        # refusal vocabulary overlaps with legitimate answers ("unknown" is
        # itself a gold answer for some questions) and with narration inside a
        # long but correct answer, so an answer that already matches the gold
        # must not be discarded for containing one of those words.
        if em >= 1.0:
            return {
                **base,
                "equivalent": True,
                "confidence": 1.0,
                "reasoning": "Normalized exact match.",
                "llm_judge_used": False,
                "llm_equivalent": None,
                "judge_status": "official_exact",
            }

        if normalized_em >= 1.0:
            # Same referent, different surface form ("three"/"3",
            # "Tumi Holdings, Inc."/"Tumi"). Decided deterministically, so this
            # holds even with the LLM judge disabled, where such pairs were
            # previously scored as wrong answers.
            return {
                **base,
                "equivalent": True,
                "confidence": 1.0,
                "reasoning": "Exact match after deterministic canonicalization.",
                "llm_judge_used": False,
                "llm_equivalent": None,
                "judge_status": "canonical_exact",
            }

        if _is_refusal(short_prediction):
            return {
                **base,
                "equivalent": False,
                "confidence": 1.0,
                "reasoning": "Prediction is a refusal or empty answer.",
                "llm_judge_used": False,
                "llm_equivalent": None,
                "judge_status": "refusal",
            }

        if f1 >= self._equivalence_f1_threshold:
            return {
                **base,
                "equivalent": True,
                "confidence": min(0.99, max(0.8, f1)),
                "reasoning": "Token F1 exceeds equivalence threshold.",
                "llm_judge_used": False,
                "llm_equivalent": None,
                "judge_status": "lexical_f1",
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
                "judge_status": "lexical_only",
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
            if not parsed.get("_parsed", False):
                # The judge answered but not in the agreed shape, so there is no
                # verdict to record. Guessing from loose keyword presence would
                # manufacture a decision, and the old fallback confidence of 0.5
                # silently became "wrong answer" under the confidence gate.
                return {
                    **base,
                    "equivalent": False,
                    "confidence": 0.0,
                    "reasoning": "LLM judge response was not parseable as a verdict.",
                    "tokens_used": tokens_used,
                    "llm_judge_used": True,
                    "llm_equivalent": None,
                    "indeterminate": True,
                    "judge_status": "indeterminate_unparseable",
                    "error": "judge_response_unparseable",
                }
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
                "judge_status": "llm_semantic",
            }
            self._cache[cache_key] = {
                k: v for k, v in result.items() if k not in ("cached", "tokens_used")
            }
            return result
        except Exception as exc:
            # A transport or provider failure says nothing about the answer.
            # Recording it as a wrong answer would depress the metric exactly
            # when the infrastructure is unhealthy, so it is surfaced as
            # indeterminate for the caller to account for or fail on.
            logger.warning("HotpotQA LLM judge failed: %s", exc)
            return {
                **base,
                "equivalent": False,
                "confidence": 0.0,
                "reasoning": f"LLM judge failed: {exc}",
                "tokens_used": tokens_used,
                "llm_judge_used": True,
                "llm_equivalent": None,
                "indeterminate": True,
                "judge_status": "indeterminate_error",
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
    """Extract a concise answer span from verbose model output when possible.

    This is the *scoring* caliber: it decides which characters of a prediction
    EM and F1 are computed over, so changing it moves published metrics for
    every system, including already-recorded runs.

    It is intentionally not shared with the pipeline-side extractor in
    ``sirchmunk.search.AgenticSearch._extract_answer_span``, which shapes the
    product's returned answer instead. The two agree on the overwhelming
    majority of responses but diverge on a few shapes, so unifying them is a
    metric-affecting change that requires a re-scored comparison rather than a
    refactor.
    """
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
    """Parse a judge verdict, reporting whether a real verdict was found.

    ``_parsed`` lets the caller distinguish a parsed verdict from an
    unparseable response. Inferring the answer from stray "true"/"false" text
    used to turn malformed output into a confident-looking decision.
    """
    cleaned = re.sub(r"```(?:json)?\s*", "", raw or "").strip().rstrip("`").strip()
    candidates = [cleaned]
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if match:
        candidates.append(match.group(0))
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict) and "equivalent" in parsed:
                return {**parsed, "_parsed": True}
        except (json.JSONDecodeError, TypeError, ValueError):
            continue
    return {
        "equivalent": False,
        "confidence": 0.0,
        "reasoning": cleaned[:300],
        "_parsed": False,
    }


def _clamp_float(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0
