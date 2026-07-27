"""V4 paper table helpers for dynamic G_n/D_n evaluation artifacts."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


class V4PaperTableGenerator:
    """Generate v4 dynamic result/update/snapshot audit tables.

    This helper is intentionally lightweight and consumes machine-readable rows
    produced by the dynamic evaluation pipeline. It does not replace the generic
    PaperTableGenerator; it adds stage-aware views required by the v4 plan.
    """

    def generate_dynamic_main_table(self, rows: Iterable[Dict[str, Any]], output_dir: str | Path) -> Dict[str, str]:
        headers = ["System", "G/D Stage", "EM", "F1", "Evidence Recall", "Latency", "Tokens", "Setup/Update"]
        normalized = [_dynamic_row(row) for row in rows]
        return _write_table_set(output_dir, "dynamic_main_results", headers, normalized)

    def generate_update_readiness_table(self, rows: Iterable[Dict[str, Any]], output_dir: str | Path) -> Dict[str, str]:
        headers = ["System", "Transition", "Update (s)", "Rebuild Required", "Query-ready"]
        normalized = [_update_row(row) for row in rows]
        return _write_table_set(output_dir, "update_readiness", headers, normalized)

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
        "Latency": row.get("avg_latency", row.get("latency", "")),
        "Tokens": row.get("avg_tokens", row.get("tokens", "")),
        "Setup/Update": row.get("setup_update", row.get("setup_seconds", "")),
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
        "corpus_checksum": row.get("corpus_checksum", ""),
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


__all__ = ["V4PaperTableGenerator"]
