#!/usr/bin/env python3
"""Build a G_125-distribution-aligned development set for LENS tuning.

The dev set must match the frozen G_125 stage on the (type x
supporting_fact_bucket) strata so that improvements measured on it transfer to
G_125, and it must not overlap the frozen 500-question parent so that tuning on
it never touches held-out evaluation questions. The parent golden set carries
per-question strata metadata; G_125's realized stratum proportions are the
allocation target, and questions are drawn from the fullwiki validation pool
that remains after removing the entire frozen parent.

Selection is deterministic under ``--seed``. The output is written next to the
other golden sets and its per-stratum deviation from G_125 is reported.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

HERE = Path(__file__).resolve().parent


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _samples(obj: Any) -> List[dict]:
    return obj.get("samples") if isinstance(obj, dict) else obj


def _stratum(sample: dict) -> tuple:
    md = sample.get("metadata", {}) or {}
    return (md.get("type"), md.get("supporting_fact_bucket"))


def _g125_ids(sampling_dir: Path) -> List[str]:
    raw = _load_json(sampling_dir / "G_125_sample_ids.json")
    if isinstance(raw, dict):
        for key in ("sample_ids", "ids"):
            if key in raw:
                return [str(x) for x in raw[key]]
        return [str(x) for x in next(iter(raw.values()))]
    return [str(x) for x in raw]


def _sf_bucket(count: int) -> str:
    """Match the parent's supporting_fact_bucket convention (2, 3, 4, 5_plus)."""
    if count <= 2:
        return "2"
    if count == 3:
        return "3"
    if count == 4:
        return "4"
    return "5_plus"


def _load_parquet_pool(parquet_dir: Path, exclude_ids: set) -> List[dict]:
    """Load the fullwiki validation split and shape it like golden samples.

    Strata metadata (type, supporting_fact_bucket) is derived with the same
    convention the sampler uses, so proportions are comparable to G_125. The
    entire frozen parent is removed so the resulting dev pool is fully
    held out.
    """
    import pyarrow.parquet as pq

    files = sorted(parquet_dir.glob("validation*.parquet"))
    if not files:
        raise FileNotFoundError(f"no validation*.parquet in {parquet_dir}")
    out: List[dict] = []
    for f in files:
        table = pq.read_table(f)
        cols = table.to_pydict()
        n = len(cols.get("id", cols.get("_id", [])))
        ids = cols.get("id") or cols.get("_id")
        qs = cols.get("question")
        ans = cols.get("answer")
        types = cols.get("type")
        sfs = cols.get("supporting_facts")
        for i in range(n):
            sid = str(ids[i])
            if sid in exclude_ids:
                continue
            sf = sfs[i] if sfs else None
            sf_count = len(sf.get("title", [])) if isinstance(sf, dict) else (len(sf) if sf else 0)
            out.append({
                "sample_id": sid,
                "question": qs[i] if qs else "",
                "gold_answer": ans[i] if ans else "",
                "metadata": {
                    "type": types[i] if types else None,
                    "supporting_fact_count": sf_count,
                    "supporting_fact_bucket": _sf_bucket(sf_count),
                },
            })
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parent", default=str(HERE / "golden_set_stratified_42_500_ba54f9ad.json"))
    ap.add_argument("--sampling-dir", default=str(HERE / "output/dynamic_eval/sampling"))
    ap.add_argument("--pool", default="", help="fullwiki validation golden pool; defaults to parent-only exclusion source")
    ap.add_argument("--parquet-pool", default="", help="fullwiki validation parquet dir; questions bucketed with the same strata as the parent")
    ap.add_argument("-n", type=int, default=125)
    ap.add_argument("--seed", type=int, default=207)
    ap.add_argument("--out", default=str(HERE / "golden_set_dev_g125aligned.json"))
    args = ap.parse_args()

    import random

    parent = _samples(_load_json(Path(args.parent)))
    parent_ids = {str(s["sample_id"]) for s in parent}
    g125_ids = set(_g125_ids(Path(args.sampling_dir)))

    # Target allocation: G_125 realized proportions over the strata.
    g125_samples = [s for s in parent if str(s["sample_id"]) in g125_ids]
    if len(g125_samples) != len(g125_ids):
        print(f"WARN: matched {len(g125_samples)}/{len(g125_ids)} G_125 ids in parent", file=sys.stderr)
    target = Counter(_stratum(s) for s in g125_samples)
    total = sum(target.values())

    # Candidate pool: prefer an explicit fullwiki pool; otherwise use the parent
    # remainder (parent minus G_125). The parent remainder is already disjoint
    # from G_125 but not from the frozen G_250 tail, so an explicit pool is
    # preferred for a fully held-out dev set.
    if args.pool:
        pool = _samples(_load_json(Path(args.pool)))
        pool = [s for s in pool if str(s["sample_id"]) not in parent_ids]
        pool_note = f"external pool minus frozen parent (n={len(pool)})"
    elif args.parquet_pool:
        pool = _load_parquet_pool(Path(args.parquet_pool), parent_ids)
        pool_note = f"fullwiki validation parquet minus frozen parent (n={len(pool)})"
    else:
        pool = [s for s in parent if str(s["sample_id"]) not in g125_ids]
        pool_note = f"parent remainder minus G_125 (n={len(pool)})"

    by_stratum: Dict[tuple, List[dict]] = defaultdict(list)
    for s in pool:
        by_stratum[_stratum(s)].append(s)

    rng = random.Random(args.seed)
    chosen: List[dict] = []
    shortfall: Dict[tuple, int] = {}
    for stratum, target_count in target.items():
        want = round(target_count / total * args.n)
        avail = by_stratum.get(stratum, [])
        rng.shuffle(avail)
        take = avail[:want]
        chosen.extend(take)
        if len(take) < want:
            shortfall[stratum] = want - len(take)

    # Top up to exactly n from the largest remaining strata, keeping determinism.
    if len(chosen) < args.n:
        chosen_ids = {str(s["sample_id"]) for s in chosen}
        rest = [s for s in pool if str(s["sample_id"]) not in chosen_ids]
        rng.shuffle(rest)
        chosen.extend(rest[: args.n - len(chosen)])
    chosen = chosen[: args.n]

    # Report deviation from G_125 proportions.
    dev = Counter(_stratum(s) for s in chosen)
    max_dev = 0.0
    lines = []
    for stratum in sorted(set(target) | set(dev), key=lambda x: str(x)):
        p_t = target.get(stratum, 0) / total * 100
        p_d = dev.get(stratum, 0) / max(len(chosen), 1) * 100
        max_dev = max(max_dev, abs(p_t - p_d))
        lines.append(f"  {str(stratum):40s} G125={p_t:5.1f}%  dev={p_d:5.1f}%  d={p_t-p_d:+.1f}")

    ids = [str(s["sample_id"]) for s in chosen]
    checksum = hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()[:8]
    overlap = len(set(ids) & (parent_ids if not args.pool else set()))

    out = {
        "samples": chosen,
        "metadata": {
            "purpose": "LENS tuning dev set, G_125-distribution-aligned",
            "n": len(chosen),
            "seed": args.seed,
            "strata": ["type", "supporting_fact_bucket"],
            "aligned_to": "G_125",
            "candidate_pool": pool_note,
            "max_abs_proportion_delta_pct": round(max_dev, 3),
            "shortfall_strata": {str(k): v for k, v in shortfall.items()},
            "sample_id_checksum": checksum,
            "overlap_with_frozen_parent": overlap,
        },
    }
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"dev set: n={len(chosen)} seed={args.seed} checksum={checksum}")
    print(f"pool: {pool_note}")
    print(f"overlap with frozen parent: {overlap} (must be 0 for held-out)")
    print(f"max abs stratum deviation vs G_125: {max_dev:.2f} pp")
    if shortfall:
        print(f"shortfall strata (pool exhausted): {shortfall}")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
