"""Stale-index analysis for dynamic G_n/D_n raw-corpus evaluation.

The update-readiness table records whether a system *declares* that it needs a
rebuild after a corpus change. It cannot show what that rebuild requirement
costs in answer quality. This module supports the missing measurement: when the
raw corpus grows from ``D_{n-1}`` to ``D_n``, how much answer and evidence
quality does a system lose while its index is still the one built on
``D_{n-1}``?

Two arms are compared on exactly the same newly added questions
(``delta = G_n \\ G_{n-1}``), whose supporting evidence articles only exist in
``D_n``:

- fresh arm: system prepared on ``D_n``, queried on the delta questions
- stale arm: system prepared on ``D_{n-1}``, queried on the same delta
  questions while the corpus path already points at ``D_n``

Index-heavy systems answer from their internal index, so a stale index cannot
reach newly added evidence and the gap is expected to be positive. Index-free
systems read the corpus at query time, so their gap is expected to stay near
zero. That near-zero gap is a measured control result rather than an
assumption, which is why the arm is executed for every system.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence

# Metrics reported per arm, in percentage points.
ARM_METRIC_KEYS = ("official_em", "official_f1", "evidence_recall", "judge_accuracy", "coverage")


def compute_delta_sample_ids(previous_ids: Sequence[str], current_ids: Sequence[str]) -> List[str]:
    """Return question ids that exist in the current stage but not the previous one.

    Nested ``G_n`` sets are prefixes of one frozen parent order, so the delta is
    the newly appended suffix. Set difference is used instead of slicing so a
    non-nested stage pair degrades to an explicit, still-correct delta.
    """
    previous = {str(sample_id) for sample_id in previous_ids or []}
    seen: set[str] = set()
    delta: List[str] = []
    for sample_id in current_ids or []:
        key = str(sample_id)
        if key in previous or key in seen:
            continue
        seen.add(key)
        delta.append(key)
    return delta


def summarize_arm(results: Iterable[Any]) -> Dict[str, Any]:
    """Aggregate one arm's per-sample results into percentage-point metrics."""
    rows = list(results or [])
    return {
        "n": len(rows),
        "official_em": _avg([_telemetry_metric(row, "official_em") for row in rows]) * 100.0,
        "official_f1": _avg([_telemetry_metric(row, "official_f1") for row in rows]) * 100.0,
        "evidence_recall": _avg([_evidence_recall(row) for row in rows]) * 100.0,
        "judge_accuracy": _avg([1.0 if getattr(row, "judge_correct", False) else 0.0 for row in rows]) * 100.0,
        "coverage": _avg([1.0 if getattr(row, "coverage", False) else 0.0 for row in rows]) * 100.0,
        "failure_count": sum(1 for row in rows if str(getattr(row, "failure_reason", "") or "")),
    }


def staleness_gap(fresh: Dict[str, Any], stale: Dict[str, Any]) -> Dict[str, float]:
    """Return fresh-minus-stale gaps; positive means the stale index lost quality."""
    gaps: Dict[str, float] = {}
    for key in ARM_METRIC_KEYS:
        gaps[f"{key}_gap"] = round(_num(fresh.get(key)) - _num(stale.get(key)), 4)
    return gaps


def classify_staleness_expectation(index_required: bool, query_ready_immediately: bool) -> str:
    """Label why a system is or is not exposed to index staleness."""
    if index_required and not query_ready_immediately:
        return "index_dependent"
    if not index_required and query_ready_immediately:
        return "index_free"
    return "mixed"


def build_staleness_row(
    *,
    system_name: str,
    baseline_name: str,
    from_stage: str,
    to_stage: str,
    delta_sample_ids: Sequence[str],
    fresh_results: Iterable[Any],
    stale_results: Iterable[Any],
    index_required: bool,
    query_ready_immediately: bool,
    stale_index_setup_metrics: Dict[str, Any] | None = None,
    from_corpus_checksum: str = "",
    to_corpus_checksum: str = "",
    delta_sample_id_checksum: str = "",
    stale_arm_mode: str = "measured",
    failure_message: str = "",
) -> Dict[str, Any]:
    """Assemble one machine-readable staleness row for tables and audit records."""
    fresh = summarize_arm(fresh_results)
    stale = summarize_arm(stale_results)
    setup = dict(stale_index_setup_metrics or {})
    row: Dict[str, Any] = {
        "system_name": system_name,
        "baseline_name": baseline_name,
        "transition": f"{from_stage}->{to_stage}",
        "from_stage": from_stage,
        "to_stage": to_stage,
        "delta_sample_count": len(list(delta_sample_ids or [])),
        "fresh_arm_sample_count": fresh["n"],
        "stale_arm_sample_count": stale["n"],
        "index_required": bool(index_required),
        "query_ready_immediately": bool(query_ready_immediately),
        "staleness_expectation": classify_staleness_expectation(index_required, query_ready_immediately),
        "stale_arm_mode": stale_arm_mode,
        "failure_message": failure_message,
        "stale_index_indexed_documents": _num(setup.get("indexed_documents")),
        "stale_index_build_seconds": _num(setup.get("index_build_seconds")),
        "from_corpus_checksum": from_corpus_checksum,
        "to_corpus_checksum": to_corpus_checksum,
        "delta_sample_id_checksum": delta_sample_id_checksum,
        "fresh_failure_count": fresh["failure_count"],
        "stale_failure_count": stale["failure_count"],
    }
    for key in ARM_METRIC_KEYS:
        row[f"fresh_{key}"] = round(_num(fresh.get(key)), 4)
        row[f"stale_{key}"] = round(_num(stale.get(key)), 4)
    row.update(staleness_gap(fresh, stale))
    row["stale_failure_rate"] = round(
        (stale["failure_count"] / stale["n"] * 100.0) if stale["n"] else 0.0,
        4,
    )
    return row


def summarize_staleness_rows(rows: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate staleness rows by expectation class for report-level claims."""
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    measured = 0
    for row in rows or []:
        if str(row.get("stale_arm_mode")) == "measured":
            measured += 1
        grouped.setdefault(str(row.get("staleness_expectation") or "unknown"), []).append(row)
    summary: Dict[str, Any] = {"row_count": 0, "measured_row_count": measured, "by_expectation": {}}
    total = 0
    for expectation, group in sorted(grouped.items()):
        total += len(group)
        summary["by_expectation"][expectation] = {
            "row_count": len(group),
            "avg_official_em_gap": round(_avg([_num(item.get("official_em_gap")) for item in group]), 4),
            "avg_evidence_recall_gap": round(_avg([_num(item.get("evidence_recall_gap")) for item in group]), 4),
            "max_evidence_recall_gap": round(max((_num(item.get("evidence_recall_gap")) for item in group), default=0.0), 4),
            "systems": sorted({str(item.get("system_name") or "") for item in group}),
        }
    summary["row_count"] = total
    return summary


def _telemetry_metric(result: Any, key: str) -> float:
    telemetry = getattr(result, "telemetry", {}) or {}
    if isinstance(telemetry, dict) and key in telemetry:
        return _num(telemetry.get(key))
    return _num(getattr(result, key, 0.0))


def _evidence_recall(result: Any) -> float:
    telemetry = getattr(result, "telemetry", {}) or {}
    if isinstance(telemetry, dict) and telemetry.get("evidence_recall") is not None:
        return _num(telemetry.get("evidence_recall"))
    return _num(getattr(result, "evidence_recall", 0.0))


def _avg(values: Sequence[float]) -> float:
    items = list(values or [])
    if not items:
        return 0.0
    return sum(items) / len(items)


def _num(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return default


__all__ = [
    "ARM_METRIC_KEYS",
    "build_staleness_row",
    "classify_staleness_expectation",
    "compute_delta_sample_ids",
    "staleness_gap",
    "summarize_arm",
    "summarize_staleness_rows",
]
