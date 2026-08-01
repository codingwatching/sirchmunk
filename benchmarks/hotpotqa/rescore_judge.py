#!/usr/bin/env python3
"""Post-hoc judge-based semantic-equivalence scoring over existing run JSONLs.

Official EM penalizes surface-form mismatches that a human would accept
("IRA" vs "Provisional Irish Republican Army", "Panama City" vs "Panama City,
Panama"). This script re-scores existing predictions with the LLM judge
(semantic equivalence), without re-running any retrieval. EM hits take the
lexical fast path (zero LLM cost); only non-EM answered predictions go to the
judge. Results are written next to each input as *.judge.json summaries.

Usage: python benchmarks/hotpotqa/rescore_judge.py --stage G_250_D_250 \
    --runs benchmarks/hotpotqa/output/dynamic_eval_r1/runs
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR.parent))

SYSTEMS = {
    "bm25_rag": "baseline_bm25_rag.jsonl",
    "hybrid_rag": "baseline_hybrid_rag.jsonl",
    "react": "baseline_react.jsonl",
    "lens_full": "baseline_ablation_lens_full.jsonl",
}


async def rescore(path: Path, judge, concurrency: int = 8) -> dict:
    rows = [json.loads(line) for line in path.open()]
    sem = asyncio.Semaphore(concurrency)

    async def one(row):
        pred = row.get("prediction") or ""
        gold = row.get("gold_answer") or ""
        question = row.get("question") or ""
        async with sem:
            verdict = await judge.judge(pred, gold, question)
        return {
            "sample_id": row.get("sample_id"),
            "official_em": verdict.get("official_em", 0),
            "equivalent": bool(verdict.get("equivalent")),
            "llm_judge_used": bool(verdict.get("llm_judge_used")),
            "confidence": verdict.get("confidence"),
        }

    results = await asyncio.gather(*[one(r) for r in rows])
    n = len(results)
    em = sum(r["official_em"] for r in results) / max(n, 1) * 100
    acc = sum(1 for r in results if r["equivalent"]) / max(n, 1) * 100
    used = sum(1 for r in results if r["llm_judge_used"])
    return {
        "n": n,
        "official_em": round(em, 1),
        "judge_semantic_acc": round(acc, 1),
        "llm_judge_calls": used,
        "per_sample": results,
    }


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="benchmarks/hotpotqa/output/dynamic_eval_r1/runs")
    ap.add_argument("--stage", action="append", default=[], help="stage dir name, repeatable; default all three")
    ap.add_argument("--env", default="benchmarks/hotpotqa/.env.hotpotqa.base")
    ap.add_argument("--concurrency", type=int, default=8)
    args = ap.parse_args()

    from hotpotqa.adapter import HotpotQAAdapter
    from hotpotqa.judge import HotpotQAJudge

    os.environ.setdefault("HOTPOT_ENABLE_LLM_JUDGE", "true")
    adapter = HotpotQAAdapter(args.env)
    llm = adapter.build_searcher().llm
    judge = HotpotQAJudge(llm=llm, enable_llm_judge=True)

    stages = args.stage or ["G_125_D_125", "G_250_D_250", "G_500_D_500"]
    runs = Path(args.runs)
    summary: dict = {}
    for stage in stages:
        for system, fname in SYSTEMS.items():
            path = runs / stage / "baselines" / fname
            if not path.exists():
                continue
            result = await rescore(path, judge, args.concurrency)
            out = path.with_suffix(".judge.json")
            out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            summary.setdefault(stage, {})[system] = {
                k: result[k] for k in ("n", "official_em", "judge_semantic_acc", "llm_judge_calls")
            }
            print(f"[{stage}] {system:11s} n={result['n']:3d} EM={result['official_em']:5.1f} "
                  f"judge_acc={result['judge_semantic_acc']:5.1f} (llm_calls={result['llm_judge_calls']})",
                  flush=True)
    (runs / "judge_rescore_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
