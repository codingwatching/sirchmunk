"""Judge calibration: measure the semantic judge against checkable ground truth.

A semantic judge that has never been calibrated is not a better ruler than a
broken one — it is an unknown ruler. This builds two ground-truth slices that
need no human labelling, then reports where the judge disagrees with them:

  POSITIVE  canonical_em == 1  -> the pair provably means the same thing
            (spelled numbers, ordinals, trailing units, entity suffixes).
            A judge that says "not equivalent" here is too strict, and this
            slice is sound: canonicalization only ever removes surface form.

  DISJOINT  gold and prediction share no content token, neither contains the
            other, and no number is involved. This slice is a *screen*, not
            ground truth: aliases and transliterations legitimately land in it
            ("Big Ben"/"Great Bell of the clock", "claymation"/"clay animation",
            "Tranquebar"/"Tharangambadi"), and for those the judge is right and
            the screen is wrong. So disagreements here are *candidates for
            review*, not counted errors.

Only the POSITIVE slice yields a defensible error count. Everything else, the
disjoint screen included, is written out for human labelling. Reporting the
screen's disagreements as judge errors would be the same mistake as calling a
surface-form miss a reasoning failure.

Usage:
    python benchmarks/hotpotqa/calibrate_judge.py \\
        --runs benchmarks/hotpotqa/output/dynamic_eval_r1/runs \\
        --stage G_500_D_500
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR.parent))

from hotpotqa.judge import (  # noqa: E402
    answer_form_report,
    canonicalize_answer,
    exact_match_score,
    normalize_answer,
    normalized_exact_match_score,
)

SYSTEMS = {
    "bm25_rag": "baseline_bm25_rag.jsonl",
    "hybrid_rag": "baseline_hybrid_rag.jsonl",
    "react": "baseline_react.jsonl",
    "lens_full": "baseline_ablation_lens_full.jsonl",
}

_STOP = frozenset("of in on at to for and or a an the is was were".split())


def _short_prediction(row: Dict[str, Any]) -> str:
    jr = (row.get("metadata") or {}).get("judge_result") or {}
    return str(jr.get("short_prediction") or "")


def _content_tokens(text: str) -> set[str]:
    return {t for t in normalize_answer(text).split() if t and t not in _STOP}


def _provably_equivalent(gold: str, pred: str) -> bool:
    return normalized_exact_match_score(pred, gold) >= 1.0


def _lexically_disjoint(gold: str, pred: str) -> bool:
    """Screen for pairs with no lexical overlap at all.

    Not a claim that the referents differ: aliases and transliterations share no
    tokens either. Used only to route pairs to review.
    """
    cg, cp = canonicalize_answer(gold), canonicalize_answer(pred)
    if not cg or not cp:
        return False
    if cg in cp or cp in cg:
        return False
    tg, tp = _content_tokens(gold), _content_tokens(pred)
    if not tg or not tp:
        return False
    if tg & tp:
        return False
    # Numbers are excluded: "3" vs "4" share no token but a judge disagreeing
    # there is a different question from entity confusion.
    if any(re.fullmatch(r"\d+", t) for t in tg | tp):
        return False
    return True


def analyse(runs_dir: Path, stage: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    base = runs_dir / stage / "baselines"
    summary: Dict[str, Any] = {"stage": stage, "systems": {}}
    review: List[Dict[str, Any]] = []

    for name, fname in SYSTEMS.items():
        run_path = base / fname
        judge_path = base / fname.replace(".jsonl", ".judge.json")
        if not run_path.exists() or not judge_path.exists():
            continue
        rows = {json.loads(line)["sample_id"]: json.loads(line) for line in run_path.open()}
        verdicts = {
            r["sample_id"]: r
            for r in json.load(judge_path.open()).get("per_sample", [])
        }

        pos_total = pos_judge_agrees = 0
        disjoint_total = disjoint_judge_says_different = 0
        boundary = 0

        for sid, verdict in verdicts.items():
            row = rows.get(sid)
            if not row:
                continue
            gold = str(row.get("gold_answer") or "")
            pred = _short_prediction(row)
            if not gold or not pred:
                continue
            said_equivalent = bool(verdict.get("equivalent"))

            if _provably_equivalent(gold, pred):
                pos_total += 1
                pos_judge_agrees += int(said_equivalent)
                if not said_equivalent:
                    review.append({
                        "slice": "false_negative", "system": name, "sample_id": sid,
                        "gold": gold, "prediction": pred,
                        "judge_equivalent": said_equivalent,
                    })
            elif _lexically_disjoint(gold, pred):
                disjoint_total += 1
                disjoint_judge_says_different += int(not said_equivalent)
                if said_equivalent:
                    review.append({
                        "slice": "disjoint_judge_said_equivalent", "system": name,
                        "sample_id": sid, "gold": gold, "prediction": pred,
                        "judge_equivalent": said_equivalent,
                        "form": answer_form_report(pred),
                        "note": "no lexical overlap; may still be a valid alias",
                        "human_label": None,
                    })
            elif not exact_match_score(pred, gold):
                boundary += 1
                review.append({
                    "slice": "needs_human_label", "system": name, "sample_id": sid,
                    "gold": gold, "prediction": pred,
                    "judge_equivalent": said_equivalent,
                    "form": answer_form_report(pred),
                    "human_label": None,
                })

        summary["systems"][name] = {
            "provable_positive_n": pos_total,
            "judge_recall_on_positive": round(pos_judge_agrees / pos_total, 4) if pos_total else None,
            "judge_false_negative_n": pos_total - pos_judge_agrees,
            "lexically_disjoint_n": disjoint_total,
            "disjoint_judge_said_equivalent_n": disjoint_total - disjoint_judge_says_different,
            "boundary_needing_labels": boundary,
        }
    return summary, review


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", required=True, help="Directory holding stage subdirectories")
    parser.add_argument("--stage", required=True, help="Stage name, e.g. G_500_D_500")
    parser.add_argument("--out", default="", help="Where to write the review file")
    args = parser.parse_args()

    runs_dir = Path(args.runs).resolve()
    summary, review = analyse(runs_dir, args.stage)

    print(json.dumps(summary, indent=2, ensure_ascii=False))

    out = Path(args.out) if args.out else runs_dir / args.stage / "judge_calibration_review.json"
    out.write_text(
        json.dumps({"summary": summary, "review": review}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    fn = sum(s.get("judge_false_negative_n") or 0 for s in summary["systems"].values())
    disj = sum(s.get("disjoint_judge_said_equivalent_n") or 0 for s in summary["systems"].values())
    need = sum(s.get("boundary_needing_labels") or 0 for s in summary["systems"].values())
    print(
        f"\nDefensible judge errors: {fn} false negative on the provable-positive "
        f"slice.\n{disj} lexically disjoint pairs the judge called equivalent — "
        f"candidates for review, not counted errors, since aliases live here.\n"
        f"{need + disj} pairs need human labels before the semantic metric can "
        f"carry an external claim.\nReview file: {out}"
    )


if __name__ == "__main__":
    main()
