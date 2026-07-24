"""HotpotQA metric aggregation helpers."""
from __future__ import annotations

from collections import defaultdict
from statistics import median
from typing import Any, Dict, List


def compute_hotpotqa_metrics(results: List[Any]) -> Dict[str, Any]:
    """Aggregate HotpotQA-specific metrics from PredictionResult-like rows."""
    n = len(results)
    if n == 0:
        return {"n": 0}

    correct = sum(1 for r in results if bool(getattr(r, "judge_correct", False)))
    coverage = sum(1 for r in results if bool(getattr(r, "coverage", False)))
    latencies = sorted(float(getattr(r, "elapsed", 0.0) or 0.0) for r in results)

    telemetry = [getattr(r, "telemetry", {}) or {} for r in results]
    em_values = [float(t.get("official_em", t.get("em", 0.0)) or 0.0) for t in telemetry]
    f1_values = [float(t.get("official_f1", t.get("f1", 0.0)) or 0.0) for t in telemetry]
    official_exact = [bool(t.get("official_exact_match", False)) for t in telemetry]
    official_f1_correct = [bool(t.get("official_f1_correct", False)) for t in telemetry]
    llm_used = [bool(t.get("llm_judge_used", False)) for t in telemetry]
    llm_equivalent = [bool(t.get("llm_equivalent", False)) for t in telemetry if t.get("llm_judge_used", False)]
    evidence_values = [float(t.get("evidence_recall", 0.0) or 0.0) for t in telemetry]
    grounded = [bool(t.get("answer_source_grounded", False)) for t in telemetry]
    judge_tokens = sum(int(t.get("judge_tokens", 0) or 0) for t in telemetry)
    total_tokens = sum(int(t.get("total_tokens", 0) or 0) for t in telemetry) + judge_tokens

    by_type = _breakdown(results, "type")
    by_level = _breakdown(results, "level")

    return {
        "n": n,
        "accuracy": round(correct / n * 100, 2),
        "llm_assisted_accuracy": round(correct / n * 100, 2),
        "official_exact_match": round(sum(official_exact) / n * 100, 2),
        "official_f1_correct": round(sum(official_f1_correct) / n * 100, 2),
        "llm_judge_usage_rate": round(sum(llm_used) / n * 100, 2),
        "llm_judge_accuracy_on_judged": round(sum(llm_equivalent) / max(len(llm_equivalent), 1) * 100, 2) if llm_equivalent else 0.0,
        "coverage": round(coverage / n * 100, 2),
        "em": round(sum(em_values) / n * 100, 2),
        "f1": round(sum(f1_values) / n * 100, 2),
        "evidence_recall": round(sum(evidence_values) / n * 100, 2),
        "source_grounding_accuracy": round(sum(grounded) / n * 100, 2),
        "latency": {
            "avg": round(sum(latencies) / n, 2),
            "p50": round(_percentile(latencies, 50), 2),
            "p95": round(_percentile(latencies, 95), 2),
            "p99": round(_percentile(latencies, 99), 2),
            "median": round(median(latencies), 2),
        },
        "token_usage": {
            "total_tokens": total_tokens,
            "judge_tokens": judge_tokens,
            "avg_tokens_per_question": round(total_tokens / n, 1),
        },
        "by_type": by_type,
        "by_level": by_level,
    }


def _breakdown(results: List[Any], metadata_key: str) -> Dict[str, Dict[str, Any]]:
    groups: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {"n": 0, "correct": 0, "coverage": 0, "em": 0.0, "f1": 0.0, "official_exact": 0, "evidence_recall": 0.0}
    )
    for r in results:
        raw = getattr(r, "raw", {}) or {}
        telemetry = getattr(r, "telemetry", {}) or {}
        group = str(raw.get(metadata_key) or raw.get("metadata", {}).get(metadata_key) or "unknown")
        g = groups[group]
        g["n"] += 1
        g["correct"] += 1 if getattr(r, "judge_correct", False) else 0
        g["coverage"] += 1 if getattr(r, "coverage", False) else 0
        g["em"] += float(telemetry.get("official_em", telemetry.get("em", 0.0)) or 0.0)
        g["f1"] += float(telemetry.get("official_f1", telemetry.get("f1", 0.0)) or 0.0)
        g["official_exact"] += 1 if telemetry.get("official_exact_match", False) else 0
        g["evidence_recall"] += float(telemetry.get("evidence_recall", 0.0) or 0.0)

    out: Dict[str, Dict[str, Any]] = {}
    for key, g in groups.items():
        n = max(g["n"], 1)
        out[key] = {
            "n": g["n"],
            "accuracy": round(g["correct"] / n * 100, 2),
            "coverage": round(g["coverage"] / n * 100, 2),
            "official_exact_match": round(g["official_exact"] / n * 100, 2),
            "em": round(g["em"] / n * 100, 2),
            "f1": round(g["f1"] / n * 100, 2),
            "evidence_recall": round(g["evidence_recall"] / n * 100, 2),
        }
    return dict(sorted(out.items()))


def _percentile(values: List[float], percentile: int) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    idx = (len(values) - 1) * percentile / 100
    lower = int(idx)
    upper = min(lower + 1, len(values) - 1)
    weight = idx - lower
    return values[lower] * (1 - weight) + values[upper] * weight
