"""Paper table helpers for dynamic G_n/D_n evaluation artifacts."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


class DynamicPaperTableGenerator:
    """Generate dynamic result, update-readiness, and snapshot audit tables.

    This helper consumes machine-readable rows produced by the dynamic evaluation
    pipeline. It complements the generic PaperTableGenerator with stage-aware
    views required by the dynamic raw-corpus protocol.
    """

    def generate_dynamic_main_table(self, rows: Iterable[Dict[str, Any]], output_dir: str | Path) -> Dict[str, str]:
        headers = ["System", "G/D Stage", "EM", "F1", "Evidence Recall", "Evidence Trace", "Latency", "Tokens", "Oracle Calls", "Setup/Update"]
        normalized = [_dynamic_row(row) for row in rows]
        return _write_table_set(output_dir, "dynamic_main_results", headers, normalized)

    def generate_update_readiness_table(self, rows: Iterable[Dict[str, Any]], output_dir: str | Path) -> Dict[str, str]:
        headers = ["System", "Transition", "Update (s)", "Rebuild Required", "Query-ready"]
        normalized = [_update_row(row) for row in rows]
        return _write_table_set(output_dir, "update_readiness", headers, normalized)

    def generate_lifecycle_main_table(self, rows: Iterable[Dict[str, Any]], output_dir: str | Path) -> Dict[str, str]:
        headers = ["System", "G/D Stage", "Setup", "Index", "Storage", "Rebuild", "Query-ready", "C_avg@100"]
        normalized = [_lifecycle_row(row) for row in rows]
        return _write_table_set(output_dir, "lifecycle_main", headers, normalized)

    def generate_budget_quality_table(self, rows: Iterable[Dict[str, Any]], output_dir: str | Path) -> Dict[str, str]:
        headers = ["System", "G/D Stage", "EM", "F1", "Evidence Recall", "Oracle Calls", "Tokens", "Latency"]
        normalized = [_budget_quality_row(row) for row in rows]
        return _write_table_set(output_dir, "budget_quality", headers, normalized)

    def generate_snapshot_audit_table(self, snapshots: Iterable[Dict[str, Any]], output_dir: str | Path) -> Dict[str, str]:
        headers = ["Snapshot", "Samples", "Articles", "Evidence", "Distractors", "Background", "Checksum"]
        normalized = [_snapshot_row(row) for row in snapshots]
        return _write_table_set(output_dir, "snapshot_audit", headers, normalized)


def _dynamic_row(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "System": row.get("system_name") or row.get("system") or "",
        "G/D Stage": row.get("stage_name") or row.get("stage") or "",
        "EM": row.get("official_em", row.get("em", "")),
        "F1": row.get("official_f1", row.get("f1", "")),
        "Evidence Recall": row.get("evidence_recall", ""),
        "Evidence Trace": row.get("evidence_trace_coverage", ""),
        "Latency": row.get("avg_latency", row.get("latency", "")),
        "Tokens": row.get("avg_tokens", row.get("tokens", "")),
        "Oracle Calls": row.get("avg_oracle_calls", ""),
        "Setup/Update": row.get("setup_update", row.get("setup_seconds", "")),
        "query_budget_summary": row.get("query_budget_summary", {}),
        "sample_id_checksum": row.get("sample_id_checksum", ""),
        "frozen_order_checksum": row.get("frozen_order_checksum", ""),
        "corpus_checksum": row.get("corpus_checksum", ""),
    }


def _update_row(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "System": row.get("system_name") or row.get("baseline_name") or "",
        "Transition": row.get("transition") or row.get("mutation_id") or "",
        "Update (s)": row.get("update_time_seconds", ""),
        "Rebuild Required": row.get("rebuild_required", ""),
        "Query-ready": row.get("query_ready_immediately", row.get("query_ready", "")),
        "corpus_checksum": row.get("corpus_checksum") or row.get("to_corpus_checksum", ""),
    }


def _lifecycle_row(row: Dict[str, Any]) -> Dict[str, Any]:
    setup = row.get("setup_metrics") if isinstance(row.get("setup_metrics"), dict) else {}
    setup_seconds = _num(setup.get("setup_seconds", row.get("setup_seconds", row.get("setup_update", 0.0))))
    index_seconds = _num(setup.get("index_build_seconds", row.get("index_build_seconds", 0.0)))
    storage_bytes = _num(setup.get("storage_bytes", row.get("storage_bytes", 0.0)))
    avg_latency = _num(row.get("avg_latency", 0.0))
    c_avg_100 = setup_seconds / 100.0 + avg_latency
    return {
        "System": row.get("system_name") or row.get("system") or "",
        "G/D Stage": row.get("stage_name") or row.get("stage") or "",
        "Setup": setup_seconds,
        "Index": index_seconds,
        "Storage": storage_bytes,
        "Rebuild": bool(setup.get("rebuild_required", row.get("rebuild_required", False))),
        "Query-ready": bool(setup.get("query_ready_immediately", row.get("query_ready_immediately", False))),
        "C_avg@100": c_avg_100,
        "sample_id_checksum": row.get("sample_id_checksum", ""),
        "frozen_order_checksum": row.get("frozen_order_checksum", ""),
        "corpus_checksum": row.get("corpus_checksum", ""),
    }


def _budget_quality_row(row: Dict[str, Any]) -> Dict[str, Any]:
    budget = row.get("query_budget_summary") if isinstance(row.get("query_budget_summary"), dict) else {}
    return {
        "System": row.get("system_name") or row.get("system") or "",
        "G/D Stage": row.get("stage_name") or row.get("stage") or "",
        "EM": row.get("official_em", row.get("em", "")),
        "F1": row.get("official_f1", row.get("f1", "")),
        "Evidence Recall": row.get("evidence_recall", ""),
        "Oracle Calls": budget.get("avg_oracle_calls", row.get("avg_oracle_calls", "")),
        "Tokens": budget.get("avg_total_tokens", row.get("avg_tokens", "")),
        "Latency": budget.get("avg_latency_seconds", row.get("avg_latency", "")),
        "sample_id_checksum": row.get("sample_id_checksum", ""),
        "frozen_order_checksum": row.get("frozen_order_checksum", ""),
        "corpus_checksum": row.get("corpus_checksum", ""),
        "query_budget_summary": budget,
    }


def _snapshot_row(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "Snapshot": row.get("stage_name") or row.get("snapshot") or "",
        "Samples": row.get("sample_count", ""),
        "Articles": row.get("article_count", ""),
        "Evidence": row.get("evidence_article_count", ""),
        "Distractors": row.get("context_distractor_count", ""),
        "Background": row.get("background_article_count", ""),
        "Checksum": row.get("corpus_checksum", ""),
    }


def _write_table_set(output_dir: str | Path, stem: str, headers: List[str], rows: List[Dict[str, Any]]) -> Dict[str, str]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    md_path = out / f"{stem}.md"
    tex_path = out / f"{stem}.tex"
    json_path = out / f"{stem}.json"
    md_path.write_text(_to_markdown(headers, rows), encoding="utf-8")
    tex_path.write_text(_to_latex(headers, rows, stem), encoding="utf-8")
    json_path.write_text(json.dumps({"headers": headers, "rows": rows}, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"markdown": str(md_path), "latex": str(tex_path), "json": str(json_path)}


def _to_markdown(headers: List[str], rows: List[Dict[str, Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(_fmt(row.get(header, "")) for header in headers) + " |")
    return "\n".join(lines) + "\n"


def _to_latex(headers: List[str], rows: List[Dict[str, Any]], stem: str) -> str:
    lines = [
        "\\begin{table}[t]",
        f"\\caption{{{stem.replace('_', ' ').title()}}}\\label{{tab:{stem}}}",
        "\\centering",
        "\\begin{tabular}{" + "l" * len(headers) + "}",
        "\\hline",
        " & ".join(_escape(header) for header in headers) + r" \\",
        "\\hline",
    ]
    for row in rows:
        lines.append(" & ".join(_escape(_fmt(row.get(header, ""))) for header in headers) + r" \\")
    lines += ["\\hline", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines) + "\n"


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _num(value: Any, default: float = 0.0) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return default


def _escape(value: Any) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in text)


__all__ = ["DynamicPaperTableGenerator"]
