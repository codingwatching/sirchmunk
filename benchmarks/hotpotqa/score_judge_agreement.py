"""Score the semantic judge against the labelled calibration worksheet.

Run only after labels are fixed: the key holding each item's system and judge
verdict is opened here for the first time, so labels cannot be adjusted once the
answers are visible.

Reports Cohen's kappa plus precision/recall, and repeats the same figures per
system. The per-system view is the one that matters for cross-system claims: a
judge that is lenient toward one system's answer style inflates that system
specifically, and a single pooled kappa hides exactly that.

AMBIGUOUS items are excluded from agreement, not silently folded into one class.
Forcing a binary label on a genuinely undetermined pair would inflate agreement
by construction. Their share is reported instead, since a large ambiguous
fraction is itself a finding about the benchmark's gold answers.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple


def cohens_kappa(pairs: List[Tuple[bool, bool]]) -> Dict[str, Any]:
    """Cohen's kappa for two binary raters over the same items."""
    n = len(pairs)
    if n == 0:
        return {"n": 0, "kappa": None, "observed_agreement": None}
    both_yes = sum(1 for a, b in pairs if a and b)
    both_no = sum(1 for a, b in pairs if not a and not b)
    a_yes = sum(1 for a, _ in pairs if a)
    b_yes = sum(1 for _, b in pairs if b)

    po = (both_yes + both_no) / n
    pe = (a_yes / n) * (b_yes / n) + ((n - a_yes) / n) * ((n - b_yes) / n)
    kappa = None if pe >= 1.0 else (po - pe) / (1.0 - pe)

    # Reference label is the rubric label; judge is the system under test.
    tp = both_yes
    fp = sum(1 for ref, jd in pairs if jd and not ref)
    fn = sum(1 for ref, jd in pairs if ref and not jd)
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    return {
        "n": n,
        "kappa": round(kappa, 4) if kappa is not None else None,
        "observed_agreement": round(po, 4),
        "expected_agreement": round(pe, 4),
        "judge_precision": round(precision, 4) if precision is not None else None,
        "judge_recall": round(recall, 4) if recall is not None else None,
        "judge_false_positive": fp,
        "judge_false_negative": fn,
    }


def interpret(kappa: float | None) -> str:
    if kappa is None:
        return "undefined"
    if kappa < 0.20:
        return "poor"
    if kappa < 0.40:
        return "fair"
    if kappa < 0.60:
        return "moderate"
    if kappa < 0.80:
        return "substantial — below the 0.80 bar for decision use"
    return "almost perfect"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worksheet", required=True)
    parser.add_argument("--key", required=True)
    args = parser.parse_args()

    ws = json.loads(Path(args.worksheet).read_text(encoding="utf-8"))
    key = json.loads(Path(args.key).read_text(encoding="utf-8"))

    overall: List[Tuple[bool, bool]] = []
    by_system: Dict[str, List[Tuple[bool, bool]]] = defaultdict(list)
    ambiguous = 0
    unlabelled = 0
    disagreements: List[Dict[str, Any]] = []

    for item in ws["worksheet"]:
        label = item.get("label")
        meta = key.get(item["item_id"])
        if meta is None:
            continue
        if not label:
            unlabelled += 1
            continue
        if label == "AMBIGUOUS":
            ambiguous += 1
            continue
        ref = label == "EQUIVALENT"
        jd = bool(meta["judge_equivalent"])
        overall.append((ref, jd))
        by_system[str(meta.get("system") or "?")].append((ref, jd))
        if ref != jd:
            disagreements.append({
                "system": meta.get("system"),
                "rubric_label": label,
                "rubric_rule": item.get("label_rule"),
                "judge_equivalent": jd,
                "question": item.get("question", "")[:110],
                "gold": item.get("gold"),
                "prediction": item.get("prediction", "")[:110],
            })

    report = {
        "provenance": ws["meta"].get("label_provenance"),
        "n_items": len(ws["worksheet"]),
        "n_scored": len(overall),
        "n_ambiguous_excluded": ambiguous,
        "n_unlabelled": unlabelled,
        "overall": cohens_kappa(overall),
        "by_system": {s: cohens_kappa(p) for s, p in sorted(by_system.items())},
    }
    report["overall"]["interpretation"] = interpret(report["overall"]["kappa"])

    print(json.dumps(report, indent=2, ensure_ascii=False))

    out = Path(args.worksheet).parent / "judge_agreement_report.json"
    out.write_text(json.dumps(
        {"report": report, "disagreements": disagreements},
        indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nDisagreements: {len(disagreements)} (written to {out})")

    kappas = [v["kappa"] for v in report["by_system"].values() if v["kappa"] is not None]
    if len(kappas) > 1:
        print(
            f"Per-system kappa spread: {min(kappas):.3f}..{max(kappas):.3f} "
            f"({max(kappas) - min(kappas):.3f}). A wide spread means the judge "
            f"treats systems unequally and cross-system claims need care."
        )


if __name__ == "__main__":
    main()
