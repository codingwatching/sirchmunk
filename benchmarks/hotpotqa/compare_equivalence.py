#!/usr/bin/env python
# Copyright (c) ModelScope Contributors. All rights reserved.
"""Per-sample equivalence comparison between two HotpotQA result sets.

Equivalence-preserving refactors must leave the evaluated behaviour intact.
Aggregate metrics alone cannot show that: two runs can post identical EM while
disagreeing on individual questions. This tool pairs runs by sample id and
reports both the metric deltas and the per-sample answer diff, so a refactor
can be accepted only when the questions that changed are enumerable and
explainable.

Usage:
    python benchmarks/hotpotqa/compare_equivalence.py REFERENCE.jsonl CANDIDATE.jsonl
    python benchmarks/hotpotqa/compare_equivalence.py REF.jsonl CAND.jsonl --show 20
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Per-metric acceptance bands, measured rather than assumed.
#
# The pipeline calls an LLM, so two runs of identical code do not reproduce each
# other: three D_150 runs of behaviour-equivalent code spanned 1.33pp on EM,
# 1.19pp on F1 and 4.27pp on evidence recall, with roughly one answer in nine
# differing. A refactor therefore cannot be validated against an exact match;
# it is accepted when every metric stays inside that observed spread, with a
# small margin. Re-measure these bands if the model, corpus or concurrency
# changes, since they are properties of the run, not of the code.
_NOISE_BAND: Dict[str, float] = {
    "EM": 2.0,
    "F1": 2.0,
    "Ev.Rec": 5.0,
    "Sent.Rec": 5.0,
}

# Metrics compared between the two runs. Answer-quality metrics first, then
# the retrieval and cost metrics that must also stay flat.
_METRIC_KEYS: List[Tuple[str, str]] = [
    ("official_em", "EM"),
    ("official_f1", "F1"),
    ("evidence_recall", "Ev.Rec"),
    ("supporting_sentence_recall", "Sent.Rec"),
    ("answer_source_grounded", "SrcGnd"),
]


def _load(path: Path) -> Dict[str, Dict[str, Any]]:
    """Index one results JSONL by sample id."""
    rows: Dict[str, Dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            sid = record.get("sample_id") or record.get("id") or record.get("_id")
            if sid is not None:
                rows[str(sid)] = record
    return rows


def _num(value: Any) -> float:
    return float(value) if isinstance(value, (int, float)) else 0.0


def _telemetry(record: Dict[str, Any]) -> Dict[str, Any]:
    """Return the telemetry mapping regardless of nesting shape."""
    for candidate in (
        record.get("telemetry"),
        (record.get("metadata") or {}).get("telemetry"),
    ):
        if isinstance(candidate, dict):
            return candidate
    return {}


def _metric(record: Dict[str, Any], key: str) -> float:
    telemetry = _telemetry(record)
    if key in telemetry:
        return _num(telemetry.get(key))
    return _num(record.get(key))


def _prediction(record: Dict[str, Any]) -> str:
    telemetry = _telemetry(record)
    for key in ("normalized_prediction", "prediction", "answer"):
        value = telemetry.get(key) if key in telemetry else record.get(key)
        if value:
            return str(value)
    return ""


def _gold(record: Dict[str, Any]) -> str:
    telemetry = _telemetry(record)
    for key in ("normalized_gold", "gold_answer", "answer_gold"):
        value = telemetry.get(key) if key in telemetry else record.get(key)
        if value:
            return str(value)
    return ""


def _norm(text: str) -> str:
    return re.sub(r"[^\w\u4e00-\u9fff]+", " ", str(text).lower()).strip()


def _cost(record: Dict[str, Any]) -> Tuple[float, float]:
    """Return ``(tokens, latency)`` for the record.

    Token accounting lives under telemetry in the quickstart result schema and
    at the top level in the baseline schema, so both spellings are accepted.
    """
    telemetry = _telemetry(record)
    tokens = (
        _num(telemetry.get("total_tokens"))
        or _num(telemetry.get("total_llm_tokens"))
        or _num(record.get("tokens_used"))
    )
    latency = _num(record.get("elapsed")) or _num(record.get("latency"))
    return tokens, latency


def _aggregate(rows: Dict[str, Dict[str, Any]], ids: List[str]) -> Dict[str, float]:
    n = max(len(ids), 1)
    summary = {
        label: sum(_metric(rows[i], key) for i in ids) / n * 100
        for key, label in _METRIC_KEYS
    }
    # Cost is reported in native units, not percentages.
    summary["Tok"] = sum(_cost(rows[i])[0] for i in ids) / n
    summary["Lat"] = sum(_cost(rows[i])[1] for i in ids) / n
    return summary


def compare(ref_path: Path, cand_path: Path, show: int) -> int:
    """Print the equivalence report; return a process exit code."""
    reference, candidate = _load(ref_path), _load(cand_path)
    shared = sorted(set(reference) & set(candidate))
    if not shared:
        print("no overlapping sample ids — cannot compare", file=sys.stderr)
        return 2

    print(f"reference : {ref_path}  (n={len(reference)})")
    print(f"candidate : {cand_path}  (n={len(candidate)})")
    print(f"paired    : {len(shared)}")
    only_ref = sorted(set(reference) - set(candidate))
    only_cand = sorted(set(candidate) - set(reference))
    if only_ref or only_cand:
        print(f"  ! unpaired: reference-only={len(only_ref)} candidate-only={len(only_cand)}")

    ref_agg, cand_agg = _aggregate(reference, shared), _aggregate(candidate, shared)
    print("\n=== aggregate metrics (paired subset) ===")
    print(f"  {'metric':10s} {'reference':>10s} {'candidate':>10s} {'delta':>9s}")
    for label in list(dict(_METRIC_KEYS).values()) + ["Tok", "Lat"]:
        delta = cand_agg[label] - ref_agg[label]
        print(f"  {label:10s} {ref_agg[label]:10.2f} {cand_agg[label]:10.2f} {delta:+9.2f}")

    # Per-sample divergence: answer text changes and EM flips are reported
    # separately, since a changed answer that keeps the same score still means
    # the refactor was not behaviour-preserving.
    text_changed: List[str] = []
    em_gained: List[str] = []
    em_lost: List[str] = []
    for sid in shared:
        ref_row, cand_row = reference[sid], candidate[sid]
        if _norm(_prediction(ref_row)) != _norm(_prediction(cand_row)):
            text_changed.append(sid)
        ref_em, cand_em = _metric(ref_row, "official_em"), _metric(cand_row, "official_em")
        if ref_em <= 0 < cand_em:
            em_gained.append(sid)
        elif cand_em <= 0 < ref_em:
            em_lost.append(sid)

    identical_rate = (len(shared) - len(text_changed)) / len(shared) * 100
    print("\n=== per-sample divergence ===")
    print(f"  identical answers : {len(shared) - len(text_changed)}/{len(shared)}  ({identical_rate:.1f}%)")
    print(f"  answer text changed: {len(text_changed)}")
    print(f"  EM gained / lost   : {len(em_gained)} / {len(em_lost)}   (net {len(em_gained) - len(em_lost):+d})")

    if text_changed and show:
        print(f"\n  first {min(show, len(text_changed))} changed answers:")
        for sid in text_changed[:show]:
            ref_row, cand_row = reference[sid], candidate[sid]
            flag = "GAIN" if sid in em_gained else ("LOST" if sid in em_lost else "same")
            print(f"    [{flag}] {sid}")
            print(f"        ref : {_prediction(ref_row)[:90]!r}")
            print(f"        cand: {_prediction(cand_row)[:90]!r}")
            print(f"        gold: {_gold(cand_row)[:70]!r}")

    em_delta = cand_agg["EM"] - ref_agg["EM"]
    f1_delta = cand_agg["F1"] - ref_agg["F1"]
    strict = not text_changed
    breaches = [
        f"{label} {cand_agg[label] - ref_agg[label]:+.2f} (band +-{band:.1f})"
        for label, band in _NOISE_BAND.items()
        if abs(cand_agg[label] - ref_agg[label]) > band
    ]

    print("\n=== verdict ===")
    if strict:
        print("  STRICT EQUIVALENT — every paired answer is byte-identical after normalization")
        return 0
    if not breaches:
        print(
            f"  WITHIN NOISE — {len(text_changed)} answers differ, but every metric "
            f"stays inside the measured run-to-run band"
        )
        print(f"  EM {em_delta:+.2f} / F1 {f1_delta:+.2f}; review the changed answers above")
        return 0
    print("  OUT OF BAND — the following metrics moved beyond run-to-run noise:")
    for breach in breaches:
        print(f"    {breach}")
    return 1


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare two HotpotQA result sets for behavioural equivalence.",
    )
    parser.add_argument("reference", type=Path, help="results JSONL captured before the change")
    parser.add_argument("candidate", type=Path, help="results JSONL captured after the change")
    parser.add_argument("--show", type=int, default=10, help="how many changed answers to print")
    args = parser.parse_args()
    for path in (args.reference, args.candidate):
        if not path.is_file():
            parser.error(f"not a file: {path}")
    raise SystemExit(compare(args.reference, args.candidate, args.show))


if __name__ == "__main__":
    main()
