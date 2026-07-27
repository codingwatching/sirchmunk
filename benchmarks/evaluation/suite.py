"""evaluation/suite.py — BaselineEvaluationSuite

竞品横向评估的执行引擎。

核心设计：
  1. 数据公平：所有系统使用完全相同的 GoldenSet（同 seed + 同问题集）
  2. Judge 公平：全部预测通过同一个 BenchmarkAdapter.build_judge() 评分
  3. 执行隔离：竞品评估与自改进循环完全独立，不共享任何状态
  4. 断点续算：每个系统的结果实时写 JSONL，崩溃后可从已完成系统继续

支持三类竞品输入：
  a. 实时 predict   : BaselineAdapter.predict() 在线调用
  b. predict_by_id  : ManualImportAdapter 等按样本 ID 返回预计算预测
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
from framework.guards import BudgetExceeded, BudgetGuard, GuardConfig, SampleTimeout, TimeoutGuard


class BaselineEvaluationSuite:
    """竞品横向评估套件。

    Usage::

        from evaluation.suite import BaselineEvaluationSuite
        from evaluation.golden_set import GoldenSetManager
        from baselines import LocalBM25Baseline, NaiveRAGBaseline

        # 准备 golden set
        manager = GoldenSetManager("benchmarks/hotpotqa")
        gs = manager.get_or_create(adapter=hotpot_adapter, seed=42, n=50)

        # 定义竞品
        baselines = [
            LocalBM25Baseline(max_files=20000),
            NaiveRAGBaseline(max_files=5000),
        ]

        # 运行
        suite = BaselineEvaluationSuite(
            bm_adapter=hotpot_adapter,
            baselines=baselines,
            output_dir="benchmarks/hotpotqa/output/baselines/",
        )
        results = await suite.run(gs)
    """

    def __init__(
        self,
        bm_adapter,                          # BenchmarkAdapter（提供数据 + judge）
        baselines: List[BaselineAdapter],
        output_dir: str,
        max_concurrent: int = 3,             # 系统级并发（同时评估几个竞品）
        guard_config: Optional[GuardConfig | Dict[str, Any]] = None,
    ) -> None:
        """
        Args:
            bm_adapter:     BenchmarkAdapter 实例（提供 get_search_paths + build_judge）。
            baselines:      BaselineAdapter 列表。
            output_dir:     竞品结果 JSONL 的输出目录。
            max_concurrent: 同时运行的竞品系统数（一般保持默认，避免 API 限流）。
            guard_config:   可选预算/超时守卫配置，用于将 baseline 失败精确分类。
        """
        self._bm_adapter = bm_adapter
        self._baselines = baselines
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._max_concurrent = max_concurrent
        self._guard_config = _coerce_guard_config(guard_config)

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
        """评估单个竞品系统的所有样本，并分类预算/超时/导入缺失失败。"""
        results: List[BaselineResult] = []
        results_lock = asyncio.Lock()
        request_delay = baseline.get_request_delay()
        max_conc = baseline.get_max_concurrent()
        sample_semaphore = asyncio.Semaphore(max_conc)
        budget_guard = BudgetGuard(self._guard_config, output_dir=self._output_dir)
        timeout_guard = TimeoutGuard(self._guard_config.sample_timeout_seconds)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text("", encoding="utf-8")

        async def _finalize(result: BaselineResult) -> BaselineResult:
            with open(out_path, "a", encoding="utf-8") as fp:
                fp.write(json.dumps(self._result_to_dict(result), ensure_ascii=False) + "\n")
            async with results_lock:
                results.append(result)
            if request_delay > 0 and result.failure_reason not in {"budget_exceeded", "import_missing"}:
                await asyncio.sleep(request_delay)
            return result

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

                sample_obj = None
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

                setup_metrics = baseline.collect_setup_metrics()
                base_metadata: Dict[str, Any] = {
                    **baseline.extra_metadata(),
                    "setup_metrics": setup_metrics,
                }
                telemetry: Dict[str, Any] = {
                    "baseline_name": baseline.name,
                    "system_name": baseline.citation_name,
                }
                prediction_obj = None
                prediction_text = ""
                pred_elapsed = 0.0
                pred_tokens = 0
                judge_correct = False
                coverage = False
                judge_tokens = 0
                judge_payload: Dict[str, Any] = {}
                error: Optional[str] = None
                failure_reason = ""
                error_type = ""
                failure_phase = ""

                try:
                    async with results_lock:
                        telemetry["budget_before_sample"] = budget_guard.check_before_sample(results)
                except BudgetExceeded as exc:
                    error = str(exc)
                    failure_reason = "budget_exceeded"
                    error_type = exc.__class__.__name__
                    failure_phase = "budget"

                import_required = bool(getattr(baseline, "requires_import_coverage", lambda: False)())
                has_predict_by_id = hasattr(baseline, "predict_by_id")
                if not error:
                    try:
                        if has_predict_by_id:
                            prediction_obj = baseline.predict_by_id(sid)
                        if prediction_obj is None and import_required:
                            error = f"Imported prediction missing for sample_id={sid}"
                            failure_reason = "import_missing"
                            error_type = "ImportMissing"
                            failure_phase = "import"
                        elif prediction_obj is None:
                            prediction_obj = await timeout_guard.run_sample(baseline.run(question, context_paths))
                    except SampleTimeout as exc:
                        error = str(exc)
                        failure_reason = "timeout"
                        error_type = exc.__class__.__name__
                        failure_phase = "prediction"
                    except Exception as exc:
                        error = str(exc)
                        failure_reason = "prediction_error"
                        error_type = exc.__class__.__name__
                        failure_phase = "prediction"

                if prediction_obj is not None:
                    prediction_text = prediction_obj.answer
                    pred_elapsed = prediction_obj.elapsed
                    pred_tokens = prediction_obj.tokens_used
                    if isinstance(prediction_obj.metadata, dict):
                        base_metadata.update(prediction_obj.metadata)
                        _merge_prediction_telemetry(telemetry, prediction_obj.metadata)
                elif not error:
                    error = "Baseline returned no prediction."
                    failure_reason = "prediction_error"
                    error_type = "NoPrediction"
                    failure_phase = "prediction"

                if import_required:
                    base_metadata["imported_baseline"] = True
                    base_metadata.setdefault("import_status", "missing" if failure_reason == "import_missing" else "imported")
                    if failure_reason == "import_missing":
                        base_metadata["missing_sample_id"] = sid

                if prediction_obj is not None and not error:
                    try:
                        eval_result = await timeout_guard.run_sample(baseline.evaluate(
                            prediction=prediction_text,
                            gold_answer=gold,
                            question=question,
                            judge=judge,
                        ))
                        judge_correct = bool(eval_result.get("judge_correct", False))
                        coverage = bool(eval_result.get("coverage", False))
                        judge_tokens = int(eval_result.get("judge_tokens", 0) or 0)
                        judge_result = eval_result.get("judge_result", {})
                        coverage_result = eval_result.get("coverage_result", {})
                        judge_payload = {
                            "judge_result": judge_result,
                            "coverage_result": coverage_result,
                        }
                        if isinstance(judge_result, dict):
                            telemetry.update(judge_result)
                        if isinstance(coverage_result, dict):
                            telemetry["coverage_result"] = coverage_result
                    except SampleTimeout as exc:
                        error = str(exc)
                        failure_reason = "timeout"
                        error_type = exc.__class__.__name__
                        failure_phase = "judge"
                    except Exception as exc:
                        error = str(exc)
                        failure_reason = "judge_error"
                        error_type = exc.__class__.__name__
                        failure_phase = "judge"

                telemetry.update({
                    "total_tokens": pred_tokens,
                    "judge_tokens": judge_tokens,
                    "failure_reason": failure_reason,
                    "error_type": error_type,
                    "failure_phase": failure_phase,
                })
                base_metadata.update(judge_payload)
                if failure_reason:
                    base_metadata["failure_reason"] = failure_reason
                    base_metadata["failure_phase"] = failure_phase

                if sample_obj is not None:
                    enrich = getattr(self._bm_adapter, "enrich_telemetry", None)
                    if callable(enrich):
                        try:
                            extra = enrich(
                                sample=sample_obj,
                                prediction=prediction_text,
                                telemetry=telemetry,
                                elapsed=pred_elapsed,
                                judge_correct=judge_correct,
                                coverage=coverage,
                            )
                            if isinstance(extra, dict):
                                telemetry.update(extra)
                        except Exception as exc:
                            telemetry["enrich_telemetry_error"] = str(exc)
                try:
                    evidence_recall = float(telemetry.get("evidence_recall", 0.0) or 0.0)
                except (TypeError, ValueError):
                    evidence_recall = 0.0

                result = BaselineResult(
                    sample_id=sid,
                    system_name=baseline.citation_name,
                    question=question,
                    gold_answer=gold,
                    prediction=prediction_text,
                    judge_correct=judge_correct,
                    coverage=coverage,
                    evidence_recall=evidence_recall,
                    elapsed=pred_elapsed,
                    tokens_used=pred_tokens,
                    judge_tokens=judge_tokens,
                    question_type=qt,
                    error=error,
                    failure_reason=failure_reason,
                    telemetry=telemetry,
                    metadata=base_metadata,
                )
                return await _finalize(result)

        tasks = [asyncio.create_task(_eval_one(s)) for s in golden_set.samples]
        return list(await asyncio.gather(*tasks))

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
            "evidence_recall": r.evidence_recall,
            "elapsed":       r.elapsed,
            "tokens_used":   r.tokens_used,
            "judge_tokens":  r.judge_tokens,
            "question_type": r.question_type,
            "error":         r.error,
            "failure_reason": r.failure_reason,
            "telemetry":     r.telemetry,
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
                    metadata = d.get("metadata", {}) if isinstance(d.get("metadata", {}), dict) else {}
                    telemetry = d.get("telemetry", {}) if isinstance(d.get("telemetry", {}), dict) else {}
                    setup_metrics = d.get("setup_metrics", metadata.get("setup_metrics", {}))
                    failure_reason = d.get("failure_reason") or metadata.get("failure_reason") or telemetry.get("failure_reason") or ""
                    results.append(BaselineResult(
                        sample_id=d.get("sample_id", ""),
                        system_name=d.get("system_name", ""),
                        question=d.get("question", ""),
                        gold_answer=d.get("gold_answer", ""),
                        prediction=d.get("prediction", ""),
                        judge_correct=bool(d.get("judge_correct", False)),
                        coverage=bool(d.get("coverage", False)),
                        evidence_recall=float(d.get("evidence_recall", telemetry.get("evidence_recall", 0.0)) or 0.0),
                        elapsed=float(d.get("elapsed", 0)),
                        tokens_used=int(d.get("tokens_used", 0)),
                        judge_tokens=int(d.get("judge_tokens", 0)),
                        question_type=d.get("question_type", ""),
                        error=d.get("error"),
                        failure_reason=str(failure_reason),
                        telemetry=telemetry,
                        metadata={
                            **metadata,
                            "setup_metrics": setup_metrics,
                        },
                    ))
                except (json.JSONDecodeError, KeyError, TypeError):
                    pass
        return results

def _merge_prediction_telemetry(telemetry: Dict[str, Any], metadata: Dict[str, Any]) -> None:
    """Promote baseline-specific retrieval metadata into common telemetry keys."""
    nested = metadata.get("telemetry") if isinstance(metadata.get("telemetry"), dict) else {}
    for source in (nested, metadata):
        for key in ("read_file_ids", "retrieval_logs", "evidence_sources", "evidence_snippets", "search_history"):
            value = source.get(key) if isinstance(source, dict) else None
            if value and not telemetry.get(key):
                telemetry[key] = value
    top_items = []
    for key in ("top_chunks", "top_docs"):
        value = metadata.get(key)
        if isinstance(value, list):
            top_items.extend(value)
    if top_items:
        paths = []
        for item in top_items:
            if isinstance(item, dict) and item.get("path"):
                paths.append(str(item["path"]))
        if paths:
            telemetry.setdefault("read_file_ids", paths)
            telemetry.setdefault("evidence_sources", paths)

def _coerce_guard_config(value: Optional[GuardConfig | Dict[str, Any]]) -> GuardConfig:
    if value is None:
        return GuardConfig()
    if isinstance(value, GuardConfig):
        return value
    return GuardConfig.from_run_config(dict(value))
