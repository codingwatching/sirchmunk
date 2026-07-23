"""evaluation/suite.py — BaselineEvaluationSuite

竞品横向评估的执行引擎。

核心设计：
  1. 数据公平：所有系统使用完全相同的 GoldenSet（同 seed + 同问题集）
  2. Judge 公平：全部预测通过同一个 BenchmarkAdapter.build_judge() 评分
  3. 执行隔离：竞品评估与自改进循环完全独立，不共享任何状态
  4. 断点续算：每个系统的结果实时写 JSONL，崩溃后可从已完成系统继续

支持三类竞品输入：
  a. 实时 predict   : BaselineAdapter.predict() 在线调用
  b. predict_by_id  : GoldCopyMock / FixedAccuracyMock / ManualImportAdapter
  c. 已有 JSONL     : 通过 ManualImportAdapter 预加载后走路径 b

评估流程（单个竞品系统）：
  for sample in golden_set.samples:
    1. 调用 baseline.predict(question, context_paths) 获取 prediction
    2. 调用 judge.judge(prediction, gold, question) → judge_correct
    3. 调用 judge.judge_coverage(prediction, question) → coverage
    4. 组装 BaselineResult 并写入 JSONL
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# 导入 baselines 接口（不依赖 framework/ 自改进模块）
import sys as _sys
_HERE = Path(__file__).parent.resolve()       # evaluation/
_BENCHMARKS = _HERE.parent                    # benchmarks/
if str(_BENCHMARKS) not in _sys.path:
    _sys.path.insert(0, str(_BENCHMARKS))

from baselines.base_adapter import BaselineAdapter, BaselineResult


class BaselineEvaluationSuite:
    """竞品横向评估套件。

    Usage::

        from evaluation.suite import BaselineEvaluationSuite
        from evaluation.golden_set import GoldenSetManager
        from baselines import ConstantMockBaseline, RandomAnswerMockBaseline

        # 准备 golden set
        manager = GoldenSetManager("benchmarks/hotpotqa")
        gs = manager.get_or_create(adapter=hotpot_adapter, seed=42, n=50)

        # 定义竞品
        baselines = [
            ConstantMockBaseline(),
            RandomAnswerMockBaseline(seed=42),
        ]

        # 运行
        suite = BaselineEvaluationSuite(
            bm_adapter=hotpot_adapter,
            baselines=baselines,
            output_dir="benchmarks/hotpotqa/output/baselines/",
        )
        results = await suite.run(gs)
        # results: {"constant_mock": [...], "random_mock": [...]}
    """

    def __init__(
        self,
        bm_adapter,                          # BenchmarkAdapter（提供数据 + judge）
        baselines: List[BaselineAdapter],
        output_dir: str,
        max_concurrent: int = 3,             # 系统级并发（同时评估几个竞品）
    ) -> None:
        """
        Args:
            bm_adapter:     BenchmarkAdapter 实例（提供 get_search_paths + build_judge）。
            baselines:      BaselineAdapter 列表。
            output_dir:     竞品结果 JSONL 的输出目录。
            max_concurrent: 同时运行的竞品系统数（一般保持默认，避免 API 限流）。
        """
        self._bm_adapter = bm_adapter
        self._baselines = baselines
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._max_concurrent = max_concurrent

    async def run(
        self,
        golden_set,                          # GoldenSet
        skip_existing: bool = True,
    ) -> Dict[str, List[BaselineResult]]:
        """对 golden_set 中的所有样本，逐个系统进行评估。

        Args:
            golden_set:    GoldenSet 实例，含所有要评估的样本。
            skip_existing: 若某系统的结果 JSONL 已存在，跳过该系统（断点续算）。

        Returns:
            {system_name: [BaselineResult, ...]}
        """
        # 检查竞品可用性
        available = []
        for b in self._baselines:
            if b.is_available():
                available.append(b)
            else:
                logger.warning("[Suite] Baseline '%s' unavailable, skipping.", b.name)

        if not available:
            logger.warning("[Suite] No available baselines.")
            return {}

        # 构建 judge（所有系统共用同一 judge 实例，保证公平性）
        judge = self._bm_adapter.build_judge()
        if judge is None:
            raise ValueError(
                "BenchmarkAdapter.build_judge() returned None. "
                "Judge is required for fair baseline comparison."
            )

        # 系统级信号量控制并发
        semaphore = asyncio.Semaphore(self._max_concurrent)
        all_results: Dict[str, List[BaselineResult]] = {}

        async def _run_one_system(baseline: BaselineAdapter):
            async with semaphore:
                out_path = self._output_dir / f"baseline_{baseline.name}.jsonl"
                if skip_existing and out_path.exists():
                    logger.info("[Suite] '%s' already done, loading from %s",
                                baseline.name, out_path)
                    all_results[baseline.name] = self._load_results(str(out_path))
                    return

                logger.info("[Suite] Evaluating '%s' (%d samples)...",
                            baseline.citation_name, len(golden_set.samples))
                setup = await baseline.prepare(golden_set=golden_set, bm_adapter=self._bm_adapter)
                logger.info(
                    "[Suite] '%s' setup: %.2fs, docs=%d, storage=%d bytes",
                    baseline.citation_name,
                    setup.setup_seconds,
                    setup.indexed_documents,
                    setup.storage_bytes,
                )
                try:
                    results = await self._eval_baseline(baseline, golden_set, judge, str(out_path))
                finally:
                    await baseline.cleanup()
                all_results[baseline.name] = results
                logger.info("[Suite] '%s' done: acc=%.1f%%",
                            baseline.citation_name,
                            sum(r.judge_correct for r in results) / max(len(results), 1) * 100)

        tasks = [asyncio.create_task(_run_one_system(b)) for b in available]
        await asyncio.gather(*tasks)
        return all_results

    async def _eval_baseline(
        self,
        baseline: BaselineAdapter,
        golden_set,
        judge: Any,
        out_path: str,
    ) -> List[BaselineResult]:
        """评估单个竞品系统的所有样本。"""
        results: List[BaselineResult] = []
        request_delay = baseline.get_request_delay()
        max_conc = baseline.get_max_concurrent()
        sample_semaphore = asyncio.Semaphore(max_conc)

        async def _eval_one(sample_dict: dict) -> BaselineResult:
            async with sample_semaphore:
                sid = sample_dict["sample_id"]
                question = sample_dict["question"]
                gold = sample_dict["gold_answer"]
                qt_key = "question_type"
                try:
                    qt_key = self._bm_adapter.get_analysis_schema().get("primary_group_key", "question_type")
                except Exception:
                    pass
                qt = sample_dict.get("metadata", {}).get(qt_key, "")

                # 获取 search_paths（与 Sirchmunk 使用相同路径，保证公平）
                try:
                    from framework.schema import BenchmarkSample
                    sample_obj = BenchmarkSample(
                        sample_id=sid,
                        question=question,
                        gold_answer=gold,
                        metadata=sample_dict.get("metadata", {}),
                    )
                    context_paths = self._bm_adapter.get_search_paths(sample_obj)
                except Exception:
                    context_paths = []

                # 调用竞品预测（支持 predict_by_id 的系统优先使用）
                prediction_obj = None
                error: Optional[str] = None
                try:
                    if hasattr(baseline, "predict_by_id"):
                        prediction_obj = baseline.predict_by_id(sid)
                    if prediction_obj is None:
                        prediction_obj = await baseline.run(question, context_paths)
                except Exception as exc:
                    error = str(exc)
                    logger.warning("[Suite] %s predict failed on %s: %s",
                                   baseline.name, sid, exc)

                prediction_text = prediction_obj.answer if prediction_obj else ""
                pred_elapsed = prediction_obj.elapsed if prediction_obj else 0.0
                pred_tokens = prediction_obj.tokens_used if prediction_obj else 0

                # Judge 评分（与 Sirchmunk 完全相同的 judge 实例）
                judge_correct = False
                coverage = False
                judge_tokens = 0
                judge_payload: Dict[str, Any] = {}
                try:
                    eval_result = await baseline.evaluate(
                        prediction=prediction_text,
                        gold_answer=gold,
                        question=question,
                        judge=judge,
                    )
                    judge_correct = bool(eval_result.get("judge_correct", False))
                    coverage = bool(eval_result.get("coverage", False))
                    judge_tokens = int(eval_result.get("judge_tokens", 0) or 0)
                    judge_payload = {
                        "judge_result": eval_result.get("judge_result", {}),
                        "coverage_result": eval_result.get("coverage_result", {}),
                    }
                except Exception as exc:
                    logger.warning("[Suite] evaluate failed on %s/%s: %s",
                                   baseline.name, sid, exc)

                result = BaselineResult(
                    sample_id=sid,
                    system_name=baseline.citation_name,
                    question=question,
                    gold_answer=gold,
                    prediction=prediction_text,
                    judge_correct=judge_correct,
                    coverage=coverage,
                    elapsed=pred_elapsed,
                    tokens_used=pred_tokens,
                    judge_tokens=judge_tokens,
                    question_type=qt,
                    error=error,
                    metadata={
                        **baseline.extra_metadata(),
                        **(prediction_obj.metadata if prediction_obj else {}),
                        "setup_metrics": baseline.collect_setup_metrics(),
                        **judge_payload,
                    },
                )

                # 实时写入（防崩溃）
                with open(out_path, "a", encoding="utf-8") as fp:
                    fp.write(json.dumps(self._result_to_dict(result), ensure_ascii=False) + "\n")

                if request_delay > 0:
                    await asyncio.sleep(request_delay)

                return result

        tasks = [asyncio.create_task(_eval_one(s)) for s in golden_set.samples]
        results = list(await asyncio.gather(*tasks))
        return results

    @staticmethod
    def _result_to_dict(r: BaselineResult) -> dict:
        return {
            "sample_id":     r.sample_id,
            "system_name":   r.system_name,
            "question":      r.question,
            "gold_answer":   r.gold_answer,
            "prediction":    r.prediction,
            "judge_correct": r.judge_correct,
            "coverage":      r.coverage,
            "elapsed":       r.elapsed,
            "tokens_used":   r.tokens_used,
            "judge_tokens":  r.judge_tokens,
            "question_type": r.question_type,
            "error":         r.error,
            "metadata":      r.metadata,
            "setup_metrics": r.metadata.get("setup_metrics", {}) if isinstance(r.metadata, dict) else {},
        }

    @staticmethod
    def _load_results(path: str) -> List[BaselineResult]:
        """从已有 JSONL 加载 BaselineResult 列表（断点续算用）。"""
        results = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    results.append(BaselineResult(
                        sample_id=d.get("sample_id", ""),
                        system_name=d.get("system_name", ""),
                        question=d.get("question", ""),
                        gold_answer=d.get("gold_answer", ""),
                        prediction=d.get("prediction", ""),
                        judge_correct=bool(d.get("judge_correct", False)),
                        coverage=bool(d.get("coverage", False)),
                        elapsed=float(d.get("elapsed", 0)),
                        tokens_used=int(d.get("tokens_used", 0)),
                        judge_tokens=int(d.get("judge_tokens", 0)),
                        question_type=d.get("question_type", ""),
                        error=d.get("error"),
                        metadata={
                            **(d.get("metadata", {}) if isinstance(d.get("metadata", {}), dict) else {}),
                            "setup_metrics": d.get("setup_metrics", d.get("metadata", {}).get("setup_metrics", {}) if isinstance(d.get("metadata", {}), dict) else {}),
                        },
                    ))
                except (json.JSONDecodeError, KeyError, TypeError):
                    pass
        return results
