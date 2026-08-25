"""framework/pareto.py — ParetoTracker

Pareto tracker for multi-benchmark joint optimization.

Core concepts:
  metrics_vector = {benchmark_name: {accuracy, coverage, avg_latency}}
  A dominates B := A has accuracy >= B on every benchmark and > B on at least one.

Persistence: multi_experiments.jsonl, kept separate from the single-benchmark
experiments.jsonl

Use cases:
  - MultiAdapterOrchestrator records the metric vector of every joint experiment
  - Check Pareto dominance before committing any Layer 0/1 change
  - Track whether the Pareto frontier stopped expanding (convergence signal)
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .time_utils import now_local_iso

logger = logging.getLogger(__name__)

# An accuracy drop beyond this value (percentage points) counts as a regression
_REGRESSION_THRESHOLD = 2.0


@dataclass
class MultiMetricsPoint:
    """Multi-dimensional metric snapshot of one joint experiment."""
    run_id: str
    timestamp: str                                    # ISO 8601
    git_commit: str
    config_hash: str                                  # Global config hash
    metrics_vector: Dict[str, Dict[str, float]]       # {bm: {accuracy, coverage, avg_latency}}
    is_pareto_optimal: bool = True                    # Whether the point lies on the current Pareto frontier
    notes: str = ""


@dataclass
class MultiDelta:
    """Difference between two joint experiments."""
    run_id_a: str
    run_id_b: str
    # {bm_name: {accuracy_delta, coverage_delta, latency_delta}}
    per_bm_delta: Dict[str, Dict[str, float]] = field(default_factory=dict)
    pareto_status: str = ""   # "dominant" | "trade_off" | "harmful" | "neutral"

    def print_summary(self) -> None:
        status_icon = {
            "dominant":  "✅",
            "trade_off": "⚖️ ",
            "harmful":   "❌",
            "neutral":   "➖",
        }.get(self.pareto_status, "❓")
        print(f"\n── Multi-Benchmark Delta: {self.run_id_a} → {self.run_id_b} ──")
        print(f"  Pareto Status: {status_icon}  {self.pareto_status.upper()}")
        print(f"  {'Benchmark':<25} {'Δacc%':>7} {'Δcov%':>7} {'Δlat':>7}")
        print("  " + "─" * 50)

        def s(v: float) -> str:
            return f"+{v:.2f}" if v >= 0 else f"{v:.2f}"

        for bm, d in sorted(self.per_bm_delta.items()):
            print(f"  {bm:<25} {s(d.get('accuracy_delta', 0)):>7} "
                  f"{s(d.get('coverage_delta', 0)):>7} "
                  f"{s(d.get('latency_delta', 0)):>7}")
        print()


class ParetoTracker:
    """Pareto tracker for joint experiments.

    Usage::

        tracker = ParetoTracker("benchmarks/multi_experiments.jsonl")
        pt = tracker.record_multi(run_id, metrics_vector, git_commit, config_hash)
        delta = tracker.compare_runs("run_001", "run_002")
        delta.print_summary()
        converged, msg = tracker.convergence_check(window=3)
    """

    def __init__(self, path: str = "benchmarks/multi_experiments.jsonl") -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_multi(
        self,
        run_id: str,
        metrics_vector: Dict[str, Dict[str, float]],
        git_commit: str = "unknown",
        config_hash: str = "unknown",
        notes: str = "",
    ) -> MultiMetricsPoint:
        """Record one joint experiment, computing Pareto optimality and flagging regressions.

        Appends first, then recomputes the global Pareto ranking (O(N^2), but N is usually
        small).
        """
        point = MultiMetricsPoint(
            run_id=run_id,
            timestamp=now_local_iso(),
            git_commit=git_commit,
            config_hash=config_hash,
            metrics_vector=metrics_vector,
            is_pareto_optimal=True,
            notes=notes,
        )
        self._append(point)
        self._recompute_pareto_flags()

        # Regression detection against the previous record
        history = self._load_all()
        if len(history) >= 2:
            prev = history[-2]
            for bm, vec in metrics_vector.items():
                prev_acc = prev.metrics_vector.get(bm, {}).get("accuracy", 0)
                curr_acc = vec.get("accuracy", 0)
                if curr_acc < prev_acc - _REGRESSION_THRESHOLD:
                    logger.warning(
                        "[ParetoTracker] ⚠️  REGRESSION on %s: %.1f%% → %.1f%%  (run=%s)",
                        bm, prev_acc, curr_acc, run_id
                    )

        return point

    def compare_runs(self, run_id_a: str, run_id_b: str) -> Optional[MultiDelta]:
        """Compare two joint experiments and return per-benchmark deltas plus Pareto status."""
        all_pts = {p.run_id: p for p in self._load_all()}
        a, b = all_pts.get(run_id_a), all_pts.get(run_id_b)
        if not a or not b:
            logger.warning("[ParetoTracker] compare: run not found (%s, %s)", run_id_a, run_id_b)
            return None

        per_bm: Dict[str, Dict[str, float]] = {}
        benchmarks = set(a.metrics_vector) | set(b.metrics_vector)

        for bm in benchmarks:
            va = a.metrics_vector.get(bm, {})
            vb = b.metrics_vector.get(bm, {})
            per_bm[bm] = {
                "accuracy_delta":  vb.get("accuracy", 0)  - va.get("accuracy", 0),
                "coverage_delta":  vb.get("coverage", 0)  - va.get("coverage", 0),
                "latency_delta":   vb.get("avg_latency", 0) - va.get("avg_latency", 0),
            }

        # Pareto status classification
        acc_deltas = [v["accuracy_delta"] for v in per_bm.values()]
        if all(d >= 0 for d in acc_deltas) and any(d > 0 for d in acc_deltas):
            status = "dominant"
        elif all(d < -_REGRESSION_THRESHOLD for d in acc_deltas):
            status = "harmful"
        elif any(d > 0 for d in acc_deltas) and any(d < -_REGRESSION_THRESHOLD for d in acc_deltas):
            status = "trade_off"
        else:
            status = "neutral"

        return MultiDelta(
            run_id_a=run_id_a,
            run_id_b=run_id_b,
            per_bm_delta=per_bm,
            pareto_status=status,
        )

    def latest_n(self, n: int = 5) -> List[MultiMetricsPoint]:
        """Return the last N joint experiment records."""
        return self._load_all()[-n:]

    def pareto_frontier(self) -> List[MultiMetricsPoint]:
        """Return the current set of Pareto-optimal points."""
        return [p for p in self._load_all() if p.is_pareto_optimal]

    def convergence_check(
        self, window: int = 3, threshold: float = 1.0
    ) -> Tuple[bool, str]:
        """Check whether the Pareto frontier stopped expanding over the last `window` runs.

        Criterion: for `window` consecutive runs, the average accuracy delta across all
        benchmarks stays below threshold.
        """
        history = self._load_all()
        if len(history) < window + 1:
            return False, f"Not enough records ({len(history)}) for convergence check"

        recent = history[-(window + 1):]
        deltas = []
        for i in range(1, len(recent)):
            prev, curr = recent[i - 1], recent[i]
            bms = set(prev.metrics_vector) & set(curr.metrics_vector)
            if not bms:
                continue
            avg_delta = sum(
                abs(curr.metrics_vector[bm].get("accuracy", 0)
                    - prev.metrics_vector[bm].get("accuracy", 0))
                for bm in bms
            ) / len(bms)
            deltas.append(avg_delta)

        if all(d < threshold for d in deltas[-window:]):
            return True, (
                f"Pareto frontier converged: avg accuracy deltas "
                f"{[f'{d:.2f}%' for d in deltas[-window:]]} all < {threshold}%"
            )
        return False, f"Not converged: recent avg deltas = {[f'{d:.2f}%' for d in deltas]}"

    def print_history(self, n: int = 8) -> None:
        """Print the joint experiment history table."""
        points = self.latest_n(n)
        if not points:
            print("  (no multi-benchmark experiments recorded yet)")
            return

        # Collect every benchmark name
        all_bms = sorted({bm for p in points for bm in p.metrics_vector})
        header_bms = "  ".join(f"{bm[:12]:>12}" for bm in all_bms)
        print(f"\n{'Run ID':<35}  {header_bms}  {'Pareto':>6}")
        print("─" * (35 + 15 * len(all_bms) + 10))

        for p in points:
            bm_cols = "  ".join(
                f"{p.metrics_vector.get(bm, {}).get('accuracy', 0):>11.1f}%"
                for bm in all_bms
            )
            pareto_tag = " ✅" if p.is_pareto_optimal else " ──"
            print(f"  {p.run_id:<33}  {bm_cols}{pareto_tag}")
        print()

    # ------------------------------------------------------------------
    # Static helpers — Pareto dominance
    # ------------------------------------------------------------------

    @staticmethod
    def dominates(
        vec_a: Dict[str, Dict],
        vec_b: Dict[str, Dict],
        metric: str = "accuracy",
    ) -> bool:
        """Return True if vec_a Pareto-dominates vec_b on `metric`.

        Dominance rule:
          - For ALL shared benchmarks: a[bm][metric] >= b[bm][metric]
          - For AT LEAST ONE benchmark: a[bm][metric] > b[bm][metric]
        """
        shared = set(vec_a) & set(vec_b)
        if not shared:
            return False
        all_ge = all(
            vec_a[bm].get(metric, 0) >= vec_b[bm].get(metric, 0)
            for bm in shared
        )
        any_gt = any(
            vec_a[bm].get(metric, 0) > vec_b[bm].get(metric, 0)
            for bm in shared
        )
        return all_ge and any_gt

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _append(self, point: MultiMetricsPoint) -> None:
        row = {
            "run_id":          point.run_id,
            "timestamp":       point.timestamp,
            "git_commit":      point.git_commit,
            "config_hash":     point.config_hash,
            "metrics_vector":  point.metrics_vector,
            "is_pareto_optimal": point.is_pareto_optimal,
            "notes":           point.notes,
        }
        with open(self._path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _load_all(self) -> List[MultiMetricsPoint]:
        if not self._path.exists():
            return []
        points: List[MultiMetricsPoint] = []
        with open(self._path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    points.append(MultiMetricsPoint(
                        run_id=d["run_id"],
                        timestamp=d.get("timestamp", ""),
                        git_commit=d.get("git_commit", "unknown"),
                        config_hash=d.get("config_hash", "unknown"),
                        metrics_vector=d.get("metrics_vector", {}),
                        is_pareto_optimal=d.get("is_pareto_optimal", True),
                        notes=d.get("notes", ""),
                    ))
                except (json.JSONDecodeError, KeyError):
                    pass
        return points

    def _recompute_pareto_flags(self) -> None:
        """Recompute Pareto optimality for every recorded point and update the file."""
        points = self._load_all()
        if not points:
            return

        # Use accuracy as the primary optimization dimension
        dominated = set()
        for i, a in enumerate(points):
            for j, b in enumerate(points):
                if i != j and j not in dominated:
                    if self.dominates(a.metrics_vector, b.metrics_vector):
                        dominated.add(j)

        for i, p in enumerate(points):
            p.is_pareto_optimal = (i not in dominated)

        # Overwrite the file
        with open(self._path, "w", encoding="utf-8") as f:
            for p in points:
                row = {
                    "run_id":          p.run_id,
                    "timestamp":       p.timestamp,
                    "git_commit":      p.git_commit,
                    "config_hash":     p.config_hash,
                    "metrics_vector":  p.metrics_vector,
                    "is_pareto_optimal": p.is_pareto_optimal,
                    "notes":           p.notes,
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
