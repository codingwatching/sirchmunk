"""Unified metric engine for benchmark and baseline results."""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, List


class MetricEngine:
    """Aggregate common ResearchOps metrics from result-like objects."""

    @staticmethod
    def aggregate(results: Iterable[Any], *, group_key: str = "question_type") -> Dict[str, Any]:
        rows = list(results)
        n = len(rows)
        if n == 0:
            return {"n": 0}

        correct = [bool(getattr(r, "judge_correct", False)) for r in rows]
        coverage = [bool(getattr(r, "coverage", False)) for r in rows]
        latencies = sorted(float(getattr(r, "elapsed", 0.0) or 0.0) for r in rows)
        telemetry = [getattr(r, "telemetry", {}) or getattr(r, "metadata", {}) or {} for r in rows]

        em_values = [_float(t.get("em", 0.0)) for t in telemetry]
        f1_values = [_float(t.get("f1", 0.0)) for t in telemetry]
        evidence_values = [_float(t.get("evidence_recall", 0.0)) for t in telemetry]
        total_tokens = sum(_int(t.get("total_tokens", 0)) for t in telemetry)
        judge_tokens = sum(_int(t.get("judge_tokens", 0)) for t in telemetry)
        if total_tokens == 0:
            total_tokens = sum(_int(getattr(r, "tokens_used", 0)) for r in rows)
        if judge_tokens == 0:
            judge_tokens = sum(_int(getattr(r, "judge_tokens", 0)) for r in rows)

        return {
            "n": n,
            "accuracy": round(sum(correct) / n * 100, 2),
            "coverage": round(sum(coverage) / n * 100, 2),
            "em": round(sum(em_values) / n * 100, 2),
            "f1": round(sum(f1_values) / n * 100, 2),
            "evidence_recall": round(sum(evidence_values) / n * 100, 2),
            "latency": {
                "avg": round(sum(latencies) / n, 2),
                "p50": round(percentile(latencies, 50), 2),
                "p95": round(percentile(latencies, 95), 2),
                "p99": round(percentile(latencies, 99), 2),
            },
            "token_usage": {
                "total_tokens": total_tokens + judge_tokens,
                "search_tokens": total_tokens,
                "judge_tokens": judge_tokens,
                "avg_tokens_per_question": round((total_tokens + judge_tokens) / n, 1),
            },
            "by_group": MetricEngine.breakdown(rows, group_key=group_key),
            "setup_metrics": collect_setup_metrics(rows),
        }

    @staticmethod
    def breakdown(results: Iterable[Any], *, group_key: str = "question_type") -> Dict[str, Any]:
        groups = defaultdict(lambda: {"n": 0, "correct": 0, "coverage": 0})
        for r in results:
            metadata = getattr(r, "metadata", None) or getattr(r, "raw", {}) or {}
            if "metadata" in metadata and isinstance(metadata["metadata"], dict):
                metadata = metadata["metadata"]
            group = metadata.get(group_key) or getattr(r, "question_type", "") or "unknown"
            groups[str(group)]["n"] += 1
            groups[str(group)]["correct"] += 1 if getattr(r, "judge_correct", False) else 0
            groups[str(group)]["coverage"] += 1 if getattr(r, "coverage", False) else 0
        out = {}
        for key, data in groups.items():
            n = max(data["n"], 1)
            out[key] = {
                "n": data["n"],
                "accuracy": round(data["correct"] / n * 100, 2),
                "coverage": round(data["coverage"] / n * 100, 2),
            }
        return dict(sorted(out.items()))


def collect_setup_metrics(results: Iterable[Any]) -> Dict[str, Any]:
    setup_values: List[Dict[str, Any]] = []
    for r in results:
        metadata = getattr(r, "metadata", {}) or {}
        setup = metadata.get("setup_metrics") if isinstance(metadata, dict) else None
        if isinstance(setup, dict) and setup:
            setup_values.append(setup)
    if not setup_values:
        return {}
    first = setup_values[0]
    return {
        "setup_seconds": _float(first.get("setup_seconds", 0.0)),
        "preprocessing_seconds": _float(first.get("preprocessing_seconds", 0.0)),
        "index_build_seconds": _float(first.get("index_build_seconds", 0.0)),
        "storage_bytes": _int(first.get("storage_bytes", 0)),
        "indexed_documents": _int(first.get("indexed_documents", 0)),
        "metadata": first.get("metadata", {}),
    }


def amortized_cost(build_cost: float, update_cost: float, query_cost: float, q: int) -> float:
    """Compute lifecycle amortized cost per query.

    Formula: C_avg(Q) = (C_build + C_update) / Q + C_query.
    """
    q = max(_int(q), 1)
    return (_float(build_cost) + _float(update_cost)) / q + _float(query_cost)


def amortized_cost_curve(
    *,
    build_cost: float,
    query_cost: float,
    update_cost: float = 0.0,
    q_values: Iterable[int] = (1, 10, 100, 1000),
) -> Dict[str, float]:
    """Return {Q: C_avg(Q)} for standard query-volume points."""
    return {
        str(max(_int(q), 1)): round(
            amortized_cost(build_cost, update_cost, query_cost, max(_int(q), 1)),
            6,
        )
        for q in q_values
    }


def lifecycle_cost_curve(
    record: Any,
    *,
    query_cost: float = 0.0,
    update_cost: float = 0.0,
    q_values: Iterable[int] = (1, 10, 100, 1000),
) -> Dict[str, Any]:
    """Build a machine-readable amortized cost curve from a lifecycle record."""
    data = record.to_dict() if hasattr(record, "to_dict") else dict(record)
    build_cost = (
        _float(data.get("api_cost_usd", 0.0))
        or _float(data.get("build_time_seconds", 0.0))
    )
    return {
        "baseline_name": data.get("baseline_name", ""),
        "citation_name": data.get("citation_name", ""),
        "corpus_scale": data.get("corpus_scale", ""),
        "build_cost": build_cost,
        "update_cost": _float(update_cost),
        "query_cost": _float(query_cost),
        "curve": amortized_cost_curve(
            build_cost=build_cost,
            update_cost=update_cost,
            query_cost=query_cost,
            q_values=q_values,
        ),
    }


def scaling_efficiency(records: Iterable[Any]) -> Dict[str, Any]:
    """Aggregate lifecycle records into per-scale build/storage metrics."""
    rows = []
    for record in records:
        data = record.to_dict() if hasattr(record, "to_dict") else dict(record)
        docs = _int(data.get("corpus_size_docs", 0))
        indexed_docs = _int(data.get("indexed_documents", 0))
        build_seconds = _float(data.get("build_time_seconds", 0.0))
        disk_bytes = _int(data.get("disk_bytes", 0))
        rows.append({
            "baseline_name": data.get("baseline_name", ""),
            "corpus_scale": data.get("corpus_scale", ""),
            "corpus_size_docs": docs,
            "indexed_documents": indexed_docs,
            "index_required": bool(data.get("index_required", True)),
            "build_time_seconds": build_seconds,
            "disk_bytes": disk_bytes,
            "seconds_per_doc": round(build_seconds / docs, 8) if docs else 0.0,
            "bytes_per_doc": round(disk_bytes / docs, 2) if docs else 0.0,
            "failure_reason": data.get("failure_reason", "none"),
            "query_eligible": bool(data.get("query_eligible", False)),
        })
    return {"scales": rows}


def percentile(values: List[float], pct: int) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    idx = (len(values) - 1) * pct / 100
    lo = int(idx)
    hi = min(lo + 1, len(values) - 1)
    weight = idx - lo
    return values[lo] * (1 - weight) + values[hi] * weight


def _float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0
