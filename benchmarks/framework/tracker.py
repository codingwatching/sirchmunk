"""framework/tracker.py — ExperimentTracker

维护 benchmarks/experiments.jsonl（每行一个实验快照）。

功能：
- record()    保存实验记录
- compare()   计算两次实验的 delta（含回退检测）
- latest_n()  查看最近 N 次实验趋势
- convergence_check()  连续 delta < threshold 时建议停止
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .schema import ExperimentRecord
from .time_utils import now_local_iso

logger = logging.getLogger(__name__)

# 若 accuracy 降幅超过此值，标记为回退
_REGRESSION_THRESHOLD = 2.0   # 百分点
# 连续多少次 delta < convergence_threshold 后认为收敛
_CONVERGENCE_WINDOW = 3


class ExperimentDelta:
    """两次实验之间的差异。"""

    def __init__(
        self,
        run_id_a: str,
        run_id_b: str,
        accuracy_delta: float,
        coverage_delta: float,
        latency_delta: float,
        token_delta: float,
        is_regression: bool,
        notes: str = "",
    ) -> None:
        self.run_id_a = run_id_a
        self.run_id_b = run_id_b
        self.accuracy_delta = accuracy_delta
        self.coverage_delta = coverage_delta
        self.latency_delta = latency_delta
        self.token_delta = token_delta
        self.is_regression = is_regression
        self.notes = notes

    def print_summary(self) -> None:
        sign = lambda v: f"+{v:.2f}" if v >= 0 else f"{v:.2f}"
        regression_tag = "  ⚠️  REGRESSION DETECTED" if self.is_regression else ""
        print(f"\n── Experiment Delta: {self.run_id_a} → {self.run_id_b} ──")
        print(f"  Accuracy:  {sign(self.accuracy_delta)}%{regression_tag}")
        print(f"  Coverage:  {sign(self.coverage_delta)}%")
        print(f"  Latency:   {sign(self.latency_delta)}s")
        print(f"  Tokens:    {sign(self.token_delta)}")
        if self.notes:
            print(f"  Notes:     {self.notes}")
        if self.accuracy_delta < -_REGRESSION_THRESHOLD:
            print("  ❌ Accuracy dropped significantly. Consider reverting the last change.")
        print()


class ExperimentTracker:
    """实验追踪器。

    Usage::

        tracker = ExperimentTracker("benchmarks/experiments.jsonl")
        tracker.record(run_id, benchmark, metrics, config, git_commit, config_hash)
        delta = tracker.compare("run_a", "run_b")
        delta.print_summary()
        records = tracker.latest_n(5)
    """

    def __init__(self, experiments_path: str = "benchmarks/experiments.jsonl") -> None:
        self._path = Path(experiments_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record(
        self,
        run_id: str,
        benchmark: str,
        metrics: Dict,
        config: Dict,
        git_commit: str = "unknown",
        config_hash: str = "unknown",
        results_path: str = "",
        notes: str = "",
    ) -> ExperimentRecord:
        """保存一次实验记录。

        自动检测是否为回退（与上一条同 benchmark 的记录相比）。
        """
        prev = self._latest_for_benchmark(benchmark)
        is_regression = False
        if prev:
            prev_acc = prev.metrics.get("accuracy", 0)
            cur_acc = metrics.get("accuracy", 0)
            if cur_acc < prev_acc - _REGRESSION_THRESHOLD:
                is_regression = True
                logger.warning(
                    "[Tracker] ⚠️  REGRESSION: %s accuracy %.1f%% → %.1f%%  (run_id=%s)",
                    benchmark, prev_acc, cur_acc, run_id,
                )

        record = ExperimentRecord(
            run_id=run_id,
            benchmark=benchmark,
            timestamp=now_local_iso(),
            git_commit=git_commit,
            config_hash=config_hash,
            metrics=metrics,
            results_path=results_path,
            notes=notes,
            is_regression=is_regression,
        )
        self._append(record)
        logger.info(
            "[Tracker] Recorded %s  acc=%.1f%%  cov=%.1f%%  regression=%s",
            run_id,
            metrics.get("accuracy", 0),
            metrics.get("coverage", 0),
            is_regression,
        )
        return record

    def compare(self, run_id_a: str, run_id_b: str) -> Optional[ExperimentDelta]:
        """计算两个实验之间的 delta。

        Args:
            run_id_a: 基线实验 ID（较早）。
            run_id_b: 待比较实验 ID（较新）。

        Returns:
            ExperimentDelta；若任一 run_id 不存在则返回 None。
        """
        all_records = self._load_all()
        rec_map = {r.run_id: r for r in all_records}

        a = rec_map.get(run_id_a)
        b = rec_map.get(run_id_b)
        if not a or not b:
            logger.warning(
                "[Tracker] compare: run_id not found (a=%s, b=%s)", run_id_a, run_id_b
            )
            return None

        def _get(rec: ExperimentRecord, key: str, default=0.0) -> float:
            return float(rec.metrics.get(key, default))

        acc_delta = _get(b, "accuracy") - _get(a, "accuracy")
        cov_delta = _get(b, "coverage") - _get(a, "coverage")
        lat_delta = _get(b, "avg_latency") - _get(a, "avg_latency")
        tok_delta = (
            _get(b, "token_usage", {})
            if isinstance(b.metrics.get("token_usage"), (int, float))
            else float(b.metrics.get("token_usage", {}).get("avg_tokens_per_question", 0))
               - float(a.metrics.get("token_usage", {}).get("avg_tokens_per_question", 0))
        )

        is_regression = acc_delta < -_REGRESSION_THRESHOLD

        notes_parts = []
        if a.config_hash != b.config_hash:
            notes_parts.append(f"config changed ({a.config_hash[:8]} → {b.config_hash[:8]})")
        if a.git_commit != b.git_commit:
            notes_parts.append(f"code changed ({a.git_commit} → {b.git_commit})")

        return ExperimentDelta(
            run_id_a=run_id_a,
            run_id_b=run_id_b,
            accuracy_delta=acc_delta,
            coverage_delta=cov_delta,
            latency_delta=lat_delta,
            token_delta=tok_delta if isinstance(tok_delta, float) else 0.0,
            is_regression=is_regression,
            notes="; ".join(notes_parts),
        )

    def latest_n(self, n: int = 5, benchmark: Optional[str] = None) -> List[ExperimentRecord]:
        """返回最近 N 条实验记录（可按 benchmark 过滤）。"""
        all_records = self._load_all()
        if benchmark:
            all_records = [r for r in all_records if r.benchmark == benchmark]
        return all_records[-n:]

    def convergence_check(
        self,
        benchmark: str,
        threshold: float = 1.0,
        window: int = _CONVERGENCE_WINDOW,
    ) -> Tuple[bool, str]:
        """检查最近 window 次实验 accuracy delta 是否均 < threshold。

        Returns:
            (is_converged, message)
        """
        records = self.latest_n(window + 1, benchmark=benchmark)
        if len(records) < window + 1:
            return False, f"Not enough records ({len(records)}) for convergence check"

        deltas = []
        for i in range(1, len(records)):
            prev_acc = records[i - 1].metrics.get("accuracy", 0)
            curr_acc = records[i].metrics.get("accuracy", 0)
            deltas.append(abs(curr_acc - prev_acc))

        recent_deltas = deltas[-window:]
        if all(d < threshold for d in recent_deltas):
            msg = (
                f"Converged: last {window} accuracy deltas "
                f"{[f'{d:.2f}%' for d in recent_deltas]} all < {threshold}%. "
                "Consider changing optimization direction."
            )
            return True, msg

        return False, f"Not converged: recent deltas = {[f'{d:.2f}%' for d in recent_deltas]}"

    def print_history(self, benchmark: Optional[str] = None, n: int = 10) -> None:
        """打印实验历史表格。"""
        records = self.latest_n(n, benchmark=benchmark)
        if not records:
            print("  (no experiments recorded yet)")
            return
        print(f"\n{'Run ID':<35} {'Acc%':>6} {'Cov%':>6} {'Latency':>8} {'Git':>12}  Notes")
        print("─" * 90)
        for r in records:
            acc = r.metrics.get("accuracy", 0)
            cov = r.metrics.get("coverage", 0)
            lat = r.metrics.get("avg_latency", 0)
            reg_tag = " ⚠️" if r.is_regression else ""
            note = r.notes[:30] if r.notes else ""
            print(
                f"  {r.run_id:<33} {acc:>5.1f} {cov:>5.1f} {lat:>7.1f}s "
                f"{r.git_commit:>12}  {note}{reg_tag}"
            )
        print()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _append(self, record: ExperimentRecord) -> None:
        """追加一行到 JSONL 文件。"""
        row = asdict(record)
        with open(self._path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _load_all(self) -> List[ExperimentRecord]:
        """加载全部实验记录。"""
        if not self._path.exists():
            return []
        records: List[ExperimentRecord] = []
        with open(self._path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    records.append(ExperimentRecord(
                        run_id=d.get("run_id", ""),
                        benchmark=d.get("benchmark", ""),
                        timestamp=d.get("timestamp", ""),
                        git_commit=d.get("git_commit", "unknown"),
                        config_hash=d.get("config_hash", "unknown"),
                        metrics=d.get("metrics", {}),
                        results_path=d.get("results_path", ""),
                        notes=d.get("notes", ""),
                        is_regression=d.get("is_regression", False),
                    ))
                except (json.JSONDecodeError, KeyError) as exc:
                    logger.warning("[Tracker] Skipping malformed record: %s", exc)
        return records

    def _latest_for_benchmark(self, benchmark: str) -> Optional[ExperimentRecord]:
        """返回指定 benchmark 的最新一条记录。"""
        records = [r for r in self._load_all() if r.benchmark == benchmark]
        return records[-1] if records else None
