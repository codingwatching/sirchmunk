"""Build a blind, stratified labelling set for judge calibration.

Why this exists
---------------
An uncalibrated semantic judge is an unknown ruler. Calibration needs reference
labels, and the honest constraint is that the labels available here come from a
model, not a human. That makes the resulting agreement an *upper bound* on judge
quality — two models sharing training priors agree more readily than a model and
a human — and it must be reported as such.

What this file does to keep the exercise defensible:

* **Blind.** The emitted worksheet carries only question, gold and prediction.
  The judge's own verdict is withheld, so a label cannot be anchored to it.
* **Anonymous.** The producing system is withheld, so no label can favour the
  system under development.
* **Pre-registered rubric.** ``RUBRIC`` below is fixed before any pair is read,
  so a verdict is an application of stated criteria rather than a rationalisation
  of whatever the data happens to show.
* **Stratified and balanced.** Equal quota per (system, judge verdict) stratum,
  so a system contributing more rows cannot dominate the statistic — and since
  strata are keyed on the withheld verdict, both verdicts are represented
  equally without the labeller knowing which is which.
* **Deterministic first.** Rules decide what rules can decide, and those
  decisions are reproducible and auditable. Only the remainder is left to
  judgement, and that remainder is written out in full for inspection.

Usage:
    python benchmarks/hotpotqa/build_calibration_set.py \\
        --review .../judge_calibration_review.json --per-stratum 20
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR.parent))

from hotpotqa.judge import (  # noqa: E402
    canonicalize_answer,
    normalize_answer,
    normalized_exact_match_score,
)

# ---------------------------------------------------------------------------
# Pre-registered rubric. Fixed before reading any pair.
# ---------------------------------------------------------------------------
RUBRIC = {
    "EQUIVALENT": [
        "E1 canonical match: differs only in spelled numbers, ordinals, units "
        "trailing a number, or legal-entity suffix.",
        "E2 qualifier-only difference: one string contains the other and the "
        "extra words are an appositive, title, or scope qualifier that does not "
        "select a different entity (\"Francis Egerton\" / \"Francis Egerton, 3rd "
        "Duke of Bridgewater\").",
        "E3 alias or transliteration of the same referent (\"Big Ben\" / \"Great "
        "Bell of the clock\", \"Tranquebar\" / \"Tharangambadi\").",
        "E4 same quantity or date expressed differently (\"190,000 employees\" / "
        "\"almost 190,000\").",
    ],
    "NOT_EQUIVALENT": [
        "N1 different entities of the same type (different person, team, place).",
        "N2 different level in a containment hierarchy where the question asks "
        "for one level (\"Albany\" / \"Albany County\", \"BBC\" / \"BBC Radio 1\").",
        "N3 category answered where an instance is asked, or the reverse "
        "(\"gin\" / \"Plymouth Gin\").",
        "N4 answers a different dimension of the question (asked an address, "
        "answered a name).",
    ],
    "AMBIGUOUS": [
        "A1 gold carries narrative or redundant wording that leaves the required "
        "granularity undetermined.",
        "A2 the question admits more than one defensible answer.",
    ],
}


def _rule_verdict(gold: str, pred: str) -> Dict[str, Any]:
    """Apply the deterministic part of the rubric.

    Returns a verdict only for cases rules can settle. Everything else is
    explicitly deferred, so judgement calls stay visible instead of hiding
    inside a heuristic.
    """
    if normalized_exact_match_score(pred, gold) >= 1.0:
        return {"label": "EQUIVALENT", "rule": "E1", "deterministic": True}

    cg, cp = canonicalize_answer(gold), canonicalize_answer(pred)
    if not cg or not cp:
        return {"label": None, "rule": "", "deterministic": False}

    # Containment is necessary for E2 but not sufficient: a contained string can
    # equally be a different hierarchy level (N2) or a bare category (N3), which
    # rules cannot separate without world knowledge.
    if cg != cp and (cg in cp or cp in cg):
        return {"label": None, "rule": "E2?/N2?/N3?", "deterministic": False}

    return {"label": None, "rule": "", "deterministic": False}


def build(review_path: Path, per_stratum: int, seed: int,
         exclude_sample_ids: set[str] | None = None) -> Dict[str, Any]:
    payload = json.loads(review_path.read_text(encoding="utf-8"))
    review: List[Dict[str, Any]] = payload.get("review", [])
    exclude = exclude_sample_ids or set()

    strata: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
    for row in review:
        gold, pred = str(row.get("gold") or ""), str(row.get("prediction") or "")
        if not gold or not pred:
            continue
        # A held-out split must not reuse any pair already labelled for tuning,
        # or the agreement it reports is measured on the data the prompt was
        # shaped against.
        if str(row.get("sample_id")) in exclude:
            continue
        strata[(row.get("system", "?"), bool(row.get("judge_equivalent")))].append(row)

    rng = random.Random(seed)
    picked: List[Dict[str, Any]] = []
    stratum_report = {}
    for key in sorted(strata, key=lambda k: (str(k[0]), k[1])):
        rows = sorted(strata[key], key=lambda r: str(r.get("sample_id")))
        take = rows if len(rows) <= per_stratum else rng.sample(rows, per_stratum)
        stratum_report[f"{key[0]}|judge={key[1]}"] = {
            "available": len(rows), "picked": len(take),
        }
        picked.extend(take)

    worksheet: List[Dict[str, Any]] = []
    key: Dict[str, Any] = {}
    for row in picked:
        gold, pred = str(row["gold"]), str(row["prediction"])
        rule = _rule_verdict(gold, pred)
        item_id = hashlib.sha256(
            f"{row.get('sample_id')}|{row.get('system')}|{normalize_answer(pred)}".encode()
        ).hexdigest()[:12]
        worksheet.append({
            "item_id": item_id,
            # Blind fields only. system / judge verdict deliberately absent.
            "question": str(row.get("question") or ""),
            "gold": gold,
            "prediction": pred,
            "rule_label": rule["label"],
            "rule_applied": rule["rule"],
            "deterministic": rule["deterministic"],
            "label": rule["label"],  # to be completed where deterministic is false
            "label_rule": rule["rule"] if rule["deterministic"] else "",
        })
        # Built here, in the same pass, so an item can never be paired with
        # another row's identity. Written to a separate file so that inspecting
        # the worksheet cannot reveal the verdict being calibrated against.
        key[item_id] = {
            "sample_id": row.get("sample_id"),
            "system": row.get("system"),
            "judge_equivalent": bool(row.get("judge_equivalent")),
            "slice": row.get("slice"),
        }
    rng.shuffle(worksheet)

    det = sum(1 for w in worksheet if w["deterministic"])
    return {
        "meta": {
            "source_review": str(review_path),
            "seed": seed,
            "per_stratum": per_stratum,
            "n_items": len(worksheet),
            "deterministic_prelabelled": det,
            "needs_judgement": len(worksheet) - det,
            "strata": stratum_report,
            "label_provenance": "model-assisted, not human ground truth; "
                                "agreement derived from it is an upper bound",
            "rubric": RUBRIC,
        },
        "worksheet": worksheet,
        "key": key,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review", required=True)
    parser.add_argument("--per-stratum", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--tag", default="", help="filename suffix, e.g. 'heldout'")
    parser.add_argument("--exclude-key", default="",
                        help="a calibration key whose sample_ids are held out")
    args = parser.parse_args()

    review_path = Path(args.review).resolve()
    exclude: set[str] = set()
    if args.exclude_key:
        prior = json.loads(Path(args.exclude_key).read_text(encoding="utf-8"))
        exclude = {str(v.get("sample_id")) for v in prior.values()}
    built = build(review_path, args.per_stratum, args.seed, exclude)

    out_dir = Path(args.out_dir).resolve() if args.out_dir else review_path.parent
    suffix = f"_{args.tag}" if args.tag else ""
    ws_path = out_dir / f"judge_calibration_worksheet{suffix}.json"
    key_path = out_dir / f"judge_calibration_key{suffix}.json"
    ws_path.write_text(json.dumps(
        {"meta": built["meta"], "worksheet": built["worksheet"]},
        indent=2, ensure_ascii=False), encoding="utf-8")
    key_path.write_text(json.dumps(built["key"], indent=2, ensure_ascii=False),
                        encoding="utf-8")

    m = built["meta"]
    print(json.dumps({k: v for k, v in m.items() if k != "rubric"}, indent=2, ensure_ascii=False))
    print(f"\nWorksheet (blind): {ws_path}")
    print(f"Key (withheld until labels are fixed): {key_path}")


if __name__ == "__main__":
    main()
