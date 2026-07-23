"""framework/orchestrator.py — ResearchOrchestrator

外层研究循环：
  benchmark运行 → 记录 → badcase分析 → 打印delta → 生成建议 → 人工确认 → 应用改动 → 重复

收敛条件：
  1. 达到 max_iterations
  2. 用户选择 quit
  3. 连续 convergence_window 次 accuracy delta < convergence_threshold
"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Optional

from .adapter import BenchmarkAdapter
from .analyzer import BadCaseAnalyzer
from .advisor import ImprovementAdvisor
from .confirm import HumanConfirmLoop
from .runner import UnifiedExperimentRunner
from .schema import BenchmarkSample, PredictionResult
from .tracker import ExperimentTracker

logger = logging.getLogger(__name__)


class ResearchOrchestrator:
    """研究编排器 — 将流水线各阶段组合为单一可调用对象。

    Usage::

        adapter = FinanceBenchAdapter(env_file=...)
        orchestrator = ResearchOrchestrator(
            adapter=adapter,
            experiments_path="benchmarks/experiments.jsonl",
        )
        await orchestrator.run(max_iterations=5, limit=50)
    """

    def __init__(
        self,
        adapter: BenchmarkAdapter,
        experiments_path: str = "benchmarks/experiments.jsonl",
        dry_run: bool = False,
    ) -> None:
        """
        Args:
            adapter:           Benchmark 适配器实例。
            experiments_path:  experiments.jsonl 路径。
            dry_run:           若 True，CONFIG_CHANGE 不实际写文件。
        """
        self._adapter = adapter
        self._runner = UnifiedExperimentRunner(adapter)
        self._tracker = ExperimentTracker(experiments_path)
        self._confirm = HumanConfirmLoop(dry_run=dry_run)

        # LLM 实例从 adapter 懒获取（避免重复构建）
        self._llm = None

    async def run(
        self,
        max_iterations: int = 5,
        limit: int = 0,
        seed: int = 42,
        convergence_threshold: float = 1.0,
        convergence_window: int = 3,
        skip_run_path: Optional[str] = None,
    ) -> None:
        """启动研究循环。

        Args:
            max_iterations:       最大迭代次数。
            limit:                每次实验的样本数（0=全量）。
            seed:                 随机种子。
            convergence_threshold:连续 delta < 此值时认为收敛（百分点）。
            convergence_window:   收敛判断所需的连续轮数。
            skip_run_path:        若提供，跳过运行直接从此 JSONL 加载结果做分析。
        """
        adapter = self._adapter

        # 延迟构建 LLM（借用 adapter 的搜索器里的 LLM）
        if self._llm is None:
            try:
                searcher = adapter.build_searcher()
                self._llm = getattr(searcher, "llm", None)
            except Exception:
                self._llm = None

        analyzer = BadCaseAnalyzer(llm=self._llm)
        advisor = ImprovementAdvisor(llm=self._llm)

        prev_run_id: Optional[str] = None

        print(f"\n{'='*64}")
        print(f"  Research Loop: {adapter.name.upper()}")
        print(f"  Max iterations: {max_iterations}  |  Limit: {limit or 'ALL'}")
        print(f"{'='*64}\n")

        # 打印历史
        self._tracker.print_history(benchmark=adapter.name)

        for iteration in range(1, max_iterations + 1):
            print(f"\n{'─'*64}")
            print(f"  Iteration {iteration}/{max_iterations}")
            print(f"{'─'*64}\n")

            # ── Step 1: 运行实验 ──────────────────────────────────────
            if skip_run_path and iteration == 1:
                print(f"  [Orch] --skip-run: loading from {skip_run_path}")
                results = UnifiedExperimentRunner.load_results_from_jsonl(skip_run_path)
                run_id = Path(skip_run_path).stem
                meta = {
                    "run_id": run_id,
                    "benchmark": adapter.name,
                    "timestamp": "",
                    "git_commit": "unknown",
                    "config_hash": "unknown",
                    "results_path": skip_run_path,
                }
            else:
                results, meta = await self._runner.run(limit=limit, seed=seed)
                run_id = meta["run_id"]

            # ── Step 2: 计算 metrics ──────────────────────────────────
            metrics = _compute_basic_metrics(results)
            metrics["config"] = adapter.get_run_config()

            # ── Step 3: 记录实验 ──────────────────────────────────────
            self._tracker.record(
                run_id=run_id,
                benchmark=adapter.name,
                metrics=metrics,
                config=adapter.get_run_config(),
                git_commit=meta.get("git_commit", "unknown"),
                config_hash=meta.get("config_hash", "unknown"),
                results_path=meta.get("results_path", ""),
            )

            # ── Step 4: 打印 delta ────────────────────────────────────
            if prev_run_id:
                delta = self._tracker.compare(prev_run_id, run_id)
                if delta:
                    delta.print_summary()

            prev_run_id = run_id

            # ── Step 5: BadCase 分析 ──────────────────────────────────
            samples_map = self._build_samples_map(results)
            analysis_schema = adapter.get_analysis_schema()
            report = await analyzer.analyze(
                results=results,
                samples_map=samples_map,
                question_type_key=analysis_schema.get("primary_group_key", "question_type"),
            )
            BadCaseAnalyzer.print_report(report)

            # ── Step 6: 生成改进建议 ──────────────────────────────────
            hypotheses = await advisor.suggest(
                report=report,
                config=adapter.get_run_config(),
                env_file=adapter.env_file,
            )

            # ── Step 7: 人工确认 ──────────────────────────────────────
            chosen, applied = self._confirm.review(hypotheses)

            if chosen is None or (
                not chosen and _user_quit_signal(hypotheses, chosen)
            ):
                print("\n  [Orch] 用户退出研究循环。")
                break

            if not chosen:
                print("  [Orch] 用户跳过本轮，继续下一次迭代。")

            # ── Step 8: 收敛检测 ──────────────────────────────────────
            converged, conv_msg = self._tracker.convergence_check(
                benchmark=adapter.name,
                threshold=convergence_threshold,
                window=convergence_window,
            )
            if converged:
                print(f"\n  ✅ {conv_msg}")
                print("  建议更换优化方向或结束本轮研究。")
                try:
                    stop = input("  是否继续? [y/N] > ").strip().lower()
                except (EOFError, KeyboardInterrupt):
                    stop = "n"
                if stop not in ("y", "yes"):
                    break

        # 最终历史
        print(f"\n{'='*64}")
        print("  研究循环结束，最终实验历史：")
        self._tracker.print_history(benchmark=adapter.name)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_samples_map(results: List[PredictionResult]) -> Dict[str, BenchmarkSample]:
        """从 results.raw 重建 {sample_id: BenchmarkSample}，供 BadCaseAnalyzer 使用。"""
        from .schema import BenchmarkSample
        samples_map: Dict[str, BenchmarkSample] = {}
        for r in results:
            raw = r.raw or {}
            sid = r.sample_id
            question = raw.get("question", sid)
            gold = raw.get("gold_answer", "")
            metadata = {k: v for k, v in raw.items()
                        if k not in ("question", "gold_answer", "prediction",
                                     "judge_correct", "coverage", "elapsed",
                                     "telemetry", "error", "sample_id")}
            samples_map[sid] = BenchmarkSample(
                sample_id=sid,
                question=question,
                gold_answer=gold,
                metadata=metadata,
            )
        return samples_map


def _compute_basic_metrics(results: List[PredictionResult]) -> dict:
    """计算基本指标 dict（供 tracker 记录）。"""
    n = len(results)
    if n == 0:
        return {"n": 0, "accuracy": 0.0, "coverage": 0.0}

    correct = sum(1 for r in results if r.judge_correct)
    coverage = sum(1 for r in results if r.coverage)
    latencies = sorted([r.elapsed for r in results if r.elapsed])
    telemetry = [r.telemetry or {} for r in results]
    search_tokens = sum(t.get("total_tokens", 0) for t in telemetry)
    judge_tokens = sum(t.get("judge_tokens", 0) for t in telemetry)
    em_values = [float(t.get("em", 0.0) or 0.0) for t in telemetry]
    f1_values = [float(t.get("f1", 0.0) or 0.0) for t in telemetry]
    evidence_values = [float(t.get("evidence_recall", 0.0) or 0.0) for t in telemetry]

    return {
        "n": n,
        "accuracy": round(correct / n * 100, 2),
        "coverage": round(coverage / n * 100, 2),
        "em": round(sum(em_values) / n * 100, 2),
        "f1": round(sum(f1_values) / n * 100, 2),
        "evidence_recall": round(sum(evidence_values) / n * 100, 2),
        "avg_latency": round(sum(latencies) / len(latencies), 2) if latencies else 0.0,
        "latency_p50": round(_percentile(latencies, 50), 2) if latencies else 0.0,
        "latency_p95": round(_percentile(latencies, 95), 2) if latencies else 0.0,
        "total_time_seconds": round(sum(latencies), 2),
        "token_usage": {
            "total_tokens": search_tokens + judge_tokens,
            "search_tokens": search_tokens,
            "judge_tokens": judge_tokens,
            "avg_tokens_per_question": round((search_tokens + judge_tokens) / n, 1),
        },
    }


def _percentile(values: List[float], percentile: int) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    idx = (len(values) - 1) * percentile / 100
    lo = int(idx)
    hi = min(lo + 1, len(values) - 1)
    weight = idx - lo
    return values[lo] * (1 - weight) + values[hi] * weight


def _user_quit_signal(hypotheses, chosen) -> bool:
    """chosen 为 None（quit）时返回 True；chosen 为空列表（skip）时返回 False。"""
    return chosen is None
