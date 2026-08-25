# Copyright (c) ModelScope Contributors. All rights reserved.
"""Human-readable rendering for AgenticSearch results.

The retrieval pipeline keeps a short answer register for benchmarks and
programmatic consumers. This module renders the same SearchContext into a richer
Markdown report for users without changing the minimal answer itself or issuing
additional LLM calls.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable


def _display_path(path: Any) -> str:
    text = str(path or "").strip()
    if not text:
        return "unknown source"
    try:
        return Path(text).name or text
    except Exception:
        return text


def _truncate(text: Any, limit: int) -> str:
    value = str(text or "").strip()
    value = " ".join(value.split())
    if len(value) <= limit:
        return value
    return value[: max(0, limit - 1)].rstrip() + "…"


def _iter_cluster_evidence(cluster: Any) -> Iterable[dict[str, Any]]:
    for ev in getattr(cluster, "evidences", []) or []:
        source = _display_path(getattr(ev, "file_or_url", ""))
        summary = _truncate(getattr(ev, "summary", ""), 500)
        snippets = getattr(ev, "snippets", []) or []
        quote = ""
        if snippets:
            first = snippets[0]
            if isinstance(first, dict):
                quote = first.get("snippet") or first.get("text") or first.get("content") or ""
            else:
                quote = str(first)
        yield {
            "source": source,
            "summary": summary,
            "quote": _truncate(quote, 700),
        }


def _iter_telemetry_snippets(context: Any) -> Iterable[dict[str, Any]]:
    telemetry = getattr(context, "telemetry", None)
    if not isinstance(telemetry, dict):
        return []
    snippets = telemetry.get("evidence_snippets") or []
    sources = list(getattr(context, "read_file_ids", []) or [])
    rows = []
    for idx, snippet in enumerate(snippets[:5], 1):
        source = _display_path(sources[idx - 1]) if idx - 1 < len(sources) else f"snippet {idx}"
        rows.append({"source": source, "summary": "", "quote": _truncate(snippet, 700)})
    return rows


def _dedupe_evidence(rows: Iterable[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    seen: set[tuple[str, str]] = set()
    result: list[dict[str, Any]] = []
    for row in rows:
        key = (str(row.get("source", "")), str(row.get("quote") or row.get("summary") or ""))
        if key in seen:
            continue
        seen.add(key)
        result.append(row)
        if len(result) >= limit:
            break
    return result


def _decision_points(context: Any) -> list[str]:
    points: list[str] = []
    telemetry = getattr(context, "telemetry", None)
    telemetry = telemetry if isinstance(telemetry, dict) else {}

    if telemetry.get("agentic_loop_used"):
        points.append("Used the prior-warmed agentic loop for answer synthesis.")
    if telemetry.get("evidence_reranking_applied"):
        selected = telemetry.get("evidence_reranking_selected_files") or []
        points.append(f"Applied evidence-coverage reranking and selected {len(selected)} file(s).")
    if telemetry.get("bridge_research_used"):
        points.append("Triggered bridge re-search to recover a second-hop entity.")
    if telemetry.get("forced_guess_used"):
        points.append("Used a last-resort evidence-grounded synthesis fallback.")
    if telemetry.get("react_explore_fallback_used"):
        points.append("Used bounded ReAct exploration after weaker retrieval paths were insufficient.")

    read_files = list(getattr(context, "read_file_ids", []) or [])
    if read_files:
        points.append(f"Read {len(read_files)} source file(s): " + ", ".join(_display_path(p) for p in read_files[:5]) + ("…" if len(read_files) > 5 else ""))

    searches = list(getattr(context, "search_history", []) or [])
    if searches:
        points.append("Issued search query/queries: " + "; ".join(_truncate(q, 120) for q in searches[:5]))

    if telemetry.get("evidence_sufficiency"):
        points.append(f"Evidence sufficiency rated as `{telemetry.get('evidence_sufficiency')}` by the answering loop.")

    return points


def render_search_response(
    context: Any,
    *,
    max_evidence: int = 5,
) -> str:
    """Render a SearchContext into user-facing Markdown.

    This function is intentionally deterministic and LLM-free. It formats the
    short answer, available source snippets, retrieval decisions, and cost
    telemetry already collected by the pipeline.
    """
    answer = str(getattr(context, "answer", "") or "").strip() or "No results found."
    lines: list[str] = ["## Answer", "", answer]

    cluster = getattr(context, "cluster", None)
    cluster_content = getattr(cluster, "content", None)
    if cluster_content and str(cluster_content).strip() and str(cluster_content).strip() != answer:
        lines.extend(["", "## Summary", "", _truncate(cluster_content, 1200)])

    evidence_rows = _dedupe_evidence(
        list(_iter_cluster_evidence(cluster)) + list(_iter_telemetry_snippets(context)),
        max_evidence,
    )
    if evidence_rows:
        lines.extend(["", "## Evidence", ""])
        for idx, row in enumerate(evidence_rows, 1):
            lines.append(f"{idx}. **Source:** `{row.get('source')}`")
            if row.get("summary"):
                lines.append(f"   - Summary: {row['summary']}")
            if row.get("quote"):
                lines.append(f"   - Quote: {row['quote']}")
    else:
        lines.extend(["", "## Evidence", "", "No source snippet was recorded in the search context."])

    decisions = _decision_points(context)
    if decisions:
        lines.extend(["", "## Retrieval Decisions", ""])
        lines.extend(f"- {item}" for item in decisions)

    lines.extend([
        "",
        "## Search Telemetry",
        "",
        f"- LLM calls: {len(getattr(context, 'llm_usages', []) or [])}",
        f"- LLM tokens: {getattr(context, 'total_llm_tokens', 0)} / {getattr(context, 'max_token_budget', 0)}",
        f"- Loop count: {getattr(context, 'loop_count', 0)} / {getattr(context, 'max_loops', 0)}",
        f"- Files read: {len(getattr(context, 'read_file_ids', []) or [])}",
    ])

    return "\n".join(lines).strip()
