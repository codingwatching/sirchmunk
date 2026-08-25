"""Structured badcase and answer-type diagnostics for ResearchOps reports."""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

try:
    from hotpotqa.judge import extract_short_answer, normalize_answer
except Exception:  # pragma: no cover - keeps this module benchmark-agnostic
    extract_short_answer = None  # type: ignore[assignment]
    normalize_answer = None  # type: ignore[assignment]


_MONTHS = (
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
)
_YES_NO = {"yes", "no"}


def load_prediction_rows(path: str | Path) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with p.open(encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def build_badcase_outputs(rows: Iterable[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows = list(rows)
    badcases = [row for row in rows if not bool(row.get("judge_correct"))]
    records = [_badcase_record(row) for row in badcases]

    failure_counts = Counter(record["failure_type"] for record in records)
    root_counts = Counter(record["root_cause"] for record in records)
    fixability_counts = Counter(record["fixability"] for record in records)
    by_type: Dict[str, Counter] = defaultdict(Counter)
    by_expected_type: Dict[str, Counter] = defaultdict(Counter)
    by_fixability: Dict[str, Counter] = defaultdict(Counter)
    for record in records:
        by_type[str(record.get("question_type") or "unknown")][record["failure_type"]] += 1
        by_expected_type[str(record.get("expected_answer_type") or "unknown")][record["failure_type"]] += 1
        by_fixability[str(record.get("fixability") or "unknown")][record["failure_type"]] += 1

    taxonomy = {
        "total_samples": len(rows),
        "badcase_count": len(records),
        "badcase_rate": round(len(records) / max(len(rows), 1) * 100, 2),
        "failure_type_breakdown": dict(sorted(failure_counts.items())),
        "root_cause_breakdown": dict(sorted(root_counts.items())),
        "fixability_breakdown": dict(sorted(fixability_counts.items())),
        "by_question_type": {key: dict(value) for key, value in sorted(by_type.items())},
        "by_expected_answer_type": {key: dict(value) for key, value in sorted(by_expected_type.items())},
        "by_fixability": {key: dict(value) for key, value in sorted(by_fixability.items())},
        "cases": records[:50],
    }
    return records, taxonomy


def build_answer_type_consistency(rows: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    rows = list(rows)
    records = [_answer_type_record(row) for row in rows]
    mismatch_records = [record for record in records if record["has_type_mismatch"]]
    gold_conflicts = [record for record in records if record["gold_question_type_conflict"]]
    prediction_conflicts = [record for record in records if record["prediction_type_mismatch"]]

    return {
        "total_samples": len(rows),
        "type_mismatch_count": len(mismatch_records),
        "type_mismatch_rate": round(len(mismatch_records) / max(len(rows), 1) * 100, 2),
        "gold_question_type_conflicts": len(gold_conflicts),
        "prediction_type_mismatches": len(prediction_conflicts),
        "expected_answer_type_distribution": dict(Counter(record["expected_answer_type"] for record in records)),
        "gold_answer_type_distribution": dict(Counter(record["gold_answer_type"] for record in records)),
        "prediction_answer_type_distribution": dict(Counter(record["prediction_answer_type"] for record in records)),
        "records": records,
    }


def quality_gate(metrics: Dict[str, Any], thresholds: Dict[str, float] | None = None) -> Dict[str, Any]:
    thresholds = thresholds or {
        "coverage": 80.0,
        "source_grounding_accuracy": 80.0,
        "official_em": 70.0,
        "official_f1_correct": 75.0,
        "evidence_recall": 60.0,
        "llm_official_em_gap": 20.0,
    }
    failure_info = metrics.get("failure_classification", {}) or {}
    pipeline_checks = {
        "system_failures": int(failure_info.get("system_failures", 0) or 0) == 0,
        "coverage": _metric_at_least(metrics, "coverage", thresholds["coverage"]),
        "source_grounding_accuracy": _metric_at_least(metrics, "source_grounding_accuracy", thresholds["source_grounding_accuracy"]),
    }
    quality_checks = {
        "official_em_or_f1_correct": (
            _metric_at_least(metrics, "official_exact_match", thresholds["official_em"])
            or _metric_at_least(metrics, "official_f1_correct", thresholds["official_f1_correct"])
        ),
        "evidence_recall": _metric_at_least(metrics, "evidence_recall", thresholds["evidence_recall"]),
        "llm_official_em_gap": _metric_gap_at_most(
            metrics,
            "llm_assisted_accuracy",
            "official_exact_match",
            thresholds["llm_official_em_gap"],
        ),
    }
    failed_pipeline = [name for name, ok in pipeline_checks.items() if not ok]
    failed_quality = [name for name, ok in quality_checks.items() if not ok]
    checks = {**pipeline_checks, **quality_checks}
    return {
        "pipeline_ok": not failed_pipeline,
        "quality_ok": not failed_pipeline and not failed_quality,
        "checks": checks,
        "pipeline_checks": pipeline_checks,
        "quality_checks": quality_checks,
        "failed_checks": failed_pipeline + failed_quality,
        "failed_pipeline_checks": failed_pipeline,
        "failed_quality_checks": failed_quality,
        "thresholds": thresholds,
    }


def _metric_gap_at_most(metrics: Dict[str, Any], high_key: str, low_key: str, threshold: float) -> bool:
    if high_key not in metrics or low_key not in metrics:
        return True
    try:
        return float(metrics.get(high_key) or 0.0) - float(metrics.get(low_key) or 0.0) <= threshold
    except (TypeError, ValueError):
        return False


def _metric_at_least(metrics: Dict[str, Any], key: str, threshold: float) -> bool:
    if key not in metrics:
        return True
    try:
        return float(metrics.get(key) or 0.0) >= threshold
    except (TypeError, ValueError):
        return False


def _badcase_record(row: Dict[str, Any]) -> Dict[str, Any]:
    telemetry = row.get("telemetry", {}) or {}
    expected = _infer_expected_answer_type(str(row.get("question") or ""))
    gold_type = _infer_answer_type(str(row.get("gold_answer") or ""))
    prediction_text = str(row.get("prediction") or "")
    short_prediction = _short_answer(prediction_text)
    prediction_type = _infer_answer_type(short_prediction)
    failure_type, root_cause = _classify_failure(row, expected, gold_type, prediction_type)
    fixability = _classify_fixability(row, failure_type, root_cause)
    return {
        "sample_id": row.get("sample_id", ""),
        "question": row.get("question", ""),
        "gold_answer": row.get("gold_answer", ""),
        "prediction": prediction_text,
        "short_prediction": short_prediction,
        "question_type": row.get("type", telemetry.get("question_type", "")),
        "level": row.get("level", ""),
        "failure_type": failure_type,
        "root_cause": root_cause,
        "fixability": fixability,
        "expected_answer_type": expected,
        "gold_answer_type": gold_type,
        "prediction_answer_type": prediction_type,
        "gold_question_type_conflict": _type_conflict(expected, gold_type),
        "prediction_type_mismatch": _type_conflict(expected, prediction_type),
        "coverage": bool(row.get("coverage")),
        "judge_correct": bool(row.get("judge_correct")),
        "official_exact_match": bool(telemetry.get("official_exact_match", False)),
        "official_f1_correct": bool(telemetry.get("official_f1_correct", False)),
        "llm_judge_used": bool(telemetry.get("llm_judge_used", False)),
        "llm_equivalent": telemetry.get("llm_equivalent"),
        "evidence_recall": telemetry.get("evidence_recall", 0.0),
        "supporting_sentence_recall": telemetry.get("supporting_sentence_recall"),
        "num_files_read": telemetry.get("num_files_read", 0),
        "target_slot": telemetry.get("target_slot", ""),
        "target_slot_verified": telemetry.get("target_slot_verified"),
        "target_slot_failure_reason": telemetry.get("target_slot_failure_reason", ""),
        "reasoning": telemetry.get("reasoning", ""),
    }


def _answer_type_record(row: Dict[str, Any]) -> Dict[str, Any]:
    telemetry = row.get("telemetry", {}) or {}
    expected = _infer_expected_answer_type(str(row.get("question") or ""))
    gold_type = _infer_answer_type(str(row.get("gold_answer") or ""))
    short_prediction = _short_answer(str(row.get("prediction") or ""))
    prediction_type = _infer_answer_type(short_prediction)
    gold_conflict = _type_conflict(expected, gold_type)
    pred_conflict = _type_conflict(expected, prediction_type)
    return {
        "sample_id": row.get("sample_id", ""),
        "expected_answer_type": expected,
        "gold_answer_type": gold_type,
        "prediction_answer_type": prediction_type,
        "gold_question_type_conflict": gold_conflict,
        "prediction_type_mismatch": pred_conflict,
        "has_type_mismatch": gold_conflict or pred_conflict,
        "official_exact_match": bool(telemetry.get("official_exact_match", False)),
        "official_f1_correct": bool(telemetry.get("official_f1_correct", False)),
        "judge_correct": bool(row.get("judge_correct")),
        "target_slot": telemetry.get("target_slot", ""),
        "target_slot_verified": telemetry.get("target_slot_verified"),
        "target_slot_failure_reason": telemetry.get("target_slot_failure_reason", ""),
    }


def _classify_failure(row: Dict[str, Any], expected: str, gold_type: str, prediction_type: str) -> Tuple[str, str]:
    telemetry = row.get("telemetry", {}) or {}
    prediction = str(row.get("prediction") or "")
    short_prediction = _short_answer(prediction)
    if row.get("error"):
        return "system_error", "system_failure"
    if int(telemetry.get("num_files_read", 0) or 0) == 0:
        return "retrieval_failure", "retrieval_failure"
    if _type_conflict(expected, gold_type):
        return "gold_question_type_conflict", "dataset_or_prompt_ambiguity"
    if float(telemetry.get("evidence_recall", 0.0) or 0.0) == 0.0:
        return "evidence_miss", "retrieval_failure"
    if telemetry.get("target_slot_verified") is False and not _is_refusal(short_prediction):
        return "target_slot_mismatch", "predicate_or_span_error"
    if _is_refusal(short_prediction):
        return "refusal", "synthesis_error"
    if _is_overbroad(short_prediction):
        return "overbroad_answer", "answer_finalization_error"
    if _type_conflict(expected, prediction_type):
        return "answer_type_mismatch", "answer_finalization_error"
    return "answer_mismatch", "predicate_or_span_error"


def _classify_fixability(row: Dict[str, Any], failure_type: str, root_cause: str) -> str:
    telemetry = row.get("telemetry", {}) or {}
    if failure_type == "gold_question_type_conflict" or root_cause == "dataset_or_prompt_ambiguity":
        return "dataset_ambiguous"
    if bool(telemetry.get("official_f1_correct", False)) and not bool(row.get("judge_correct")):
        return "evaluator_sensitive"
    if failure_type in {"retrieval_failure", "evidence_miss"} or root_cause == "retrieval_failure":
        return "retrieval_fixable"
    return "model_fixable"


def _infer_expected_answer_type(question: str) -> str:
    q = question.lower().strip()
    if re.match(r"^(are|is|was|were|do|does|did|has|have|had|can|could|will|would|should)\b", q):
        return "yes_no"
    if "what year" in q or "which year" in q or "in what year" in q:
        return "year"
    if "what date" in q or "which date" in q or "when" in q:
        return "date"
    if q.startswith("where") or " where " in q:
        return "location"
    if q.startswith("who") or " which person" in q:
        return "person"
    if q.startswith("how many") or q.startswith("how much"):
        return "number"
    if q.startswith("name a ") or q.startswith("name an "):
        return "single_entity"
    if q.startswith("which") or q.startswith("what"):
        return "single_entity"
    return "phrase"


def _infer_answer_type(text: str) -> str:
    value = _clean_answer_text(text)
    lower = value.lower().strip()
    if not lower:
        return "empty"
    if lower in _YES_NO or lower.startswith("yes ") or lower.startswith("no "):
        return "yes_no"
    if re.fullmatch(r"\d{4}", lower):
        return "year"
    if any(month in lower for month in _MONTHS) and re.search(r"\d{4}", lower):
        return "date"
    if re.fullmatch(r"[\d,]+(?:\.\d+)?\s*(?:%|percent|episodes?|season)?", lower):
        return "number"
    if _is_overbroad(value):
        return "list"
    if "," in value and not any(ch.isdigit() for ch in value):
        return "location"
    if len(value.split()) <= 6:
        return "single_entity"
    return "phrase"


def _short_answer(text: str) -> str:
    raw = str(text or "")
    if extract_short_answer is not None:
        try:
            candidate = str(extract_short_answer(raw) or "")
            if candidate and not _is_reserved_label(candidate):
                return candidate
        except Exception:
            pass
    match = re.search(r"\*\*Answer\s*:\s*(.+?)(?:\*\*|\n|$)", raw, flags=re.I | re.S)
    if match:
        candidate = _clean_answer_text(match.group(1))
        if candidate and not _is_reserved_label(candidate):
            return candidate
    extracted = _extract_marked_value(raw, "Extracted value")
    if extracted:
        return extracted
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    return _clean_answer_text(lines[0]) if lines else ""


def _extract_marked_value(text: str, label: str) -> str:
    match = re.search(
        rf"(?:^|\n)\s*\**{re.escape(label)}\**\s*:\s*(.+?)(?=\n\s*\n|\n\s*\**[A-Z][^:\n]{{0,80}}\**\s*:|\Z)",
        str(text or ""),
        flags=re.I | re.S,
    )
    return _clean_answer_text(match.group(1)) if match else ""


def _is_reserved_label(text: str) -> bool:
    cleaned = re.sub(r"\s+", " ", str(text or "")).strip().strip("*:： ").lower()
    cleaned = re.sub(r"[^a-z0-9 ]+", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    labels = {"answer", "source", "source passage", "evidence", "extracted value", "precise answer", "summary"}
    return cleaned in labels or any(cleaned.startswith(f"{label} ") for label in labels)


def _clean_answer_text(text: str) -> str:
    value = str(text or "").strip().strip('"').strip("'")
    value = re.sub(r"^\*\*Answer\s*:\s*", "", value, flags=re.I).strip()
    value = value.strip("* ")
    return value


def _type_conflict(expected: str, actual: str) -> bool:
    if not expected or expected in {"phrase", "list"} or not actual or actual == "empty":
        return False
    compatible = {
        "single_entity": {"single_entity", "person", "organization", "location", "work_title", "phrase"},
        "person": {"single_entity", "person"},
        "organization": {"single_entity", "organization"},
        "location": {"single_entity", "location"},
        "work_title": {"single_entity", "work_title"},
        "date": {"date", "year"},
        "year": {"year"},
        "number": {"number", "year"},
        "yes_no": {"yes_no"},
    }
    return actual not in compatible.get(expected, {expected})


def _is_refusal(text: str) -> bool:
    lower = str(text or "").lower()
    refusal_markers = (
        "should_answer=false",
        "not found",
        "cannot determine",
        "insufficient",
        "does not provide",
        "does not mention",
        "does not state",
        "not provided",
        "not stated",
    )
    return not lower.strip() or any(marker in lower for marker in refusal_markers)


def _is_overbroad(text: str) -> bool:
    cleaned = _clean_answer_text(text)
    if ";" in cleaned:
        return True
    if re.search(r"\b(?:and|or)\b", cleaned, flags=re.I) and "," in cleaned:
        return True
    return len([part for part in cleaned.split(",") if part.strip()]) >= 3
