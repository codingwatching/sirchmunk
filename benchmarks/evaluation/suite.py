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
        max_concurrent: Optional[int] = None,  # 系统级并发（同时评估几个竞品）
        guard_config: Optional[GuardConfig | Dict[str, Any]] = None,
        corpus_metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Args:
            bm_adapter:     BenchmarkAdapter 实例（提供 get_search_paths + build_judge）。
            baselines:      BaselineAdapter 列表。
            output_dir:     竞品结果 JSONL 的输出目录。
            max_concurrent: 同时运行的竞品系统数。留空时取 benchmark adapter 的并发
                            配置（HotpotQA 为 HOTPOT_MAX_CONCURRENT），使该配置在系统
                            维度真正生效，而不是停留在一个硬编码常量上。注意这只控制
                            "同时跑几个竞品"；每个竞品内部的样本级并发由该竞品自己的
                            get_max_concurrent() 决定，默认串行以避免 API 限流并保持
                            延迟指标可比。
            guard_config:   可选预算/超时守卫配置，用于将 baseline 失败精确分类。
        """
        self._bm_adapter = bm_adapter
        self._baselines = baselines
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._max_concurrent = _resolve_system_concurrency(bm_adapter, max_concurrent)
        self._guard_config = _coerce_guard_config(guard_config)
        self._corpus_metadata = dict(corpus_metadata or {})

    async def run(
        self,
        golden_set,                          # GoldenSet
        skip_existing: bool = True,
        *,
        skip_prepare: bool = False,
        skip_cleanup: bool = False,
    ) -> Dict[str, List[BaselineResult]]:
        """对 golden_set 中的所有样本，逐个系统进行评估。

        Args:
            golden_set:    GoldenSet 实例，含所有要评估的样本。
            skip_existing: 若某系统的结果 JSONL 已存在，跳过该系统（断点续算）。
            skip_prepare:  复用 baseline 当前已有的索引状态，不再调用 prepare()。
                           stale-index 对照臂依赖此选项保留上一个语料快照的索引。
            skip_cleanup:  评估结束后不释放 baseline 资源，便于后续 stage 复用其
                           索引状态（持久化索引的竞品在 cleanup 后可能无法再查询）。

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
        logger.info(
            "[Suite] system-level concurrency=%d over %d baseline(s); per-sample concurrency: %s",
            self._max_concurrent,
            len(self._baselines),
            ", ".join(
                f"{b.name}={_resolve_sample_concurrency(self._bm_adapter, b)}"
                for b in self._baselines
            ) or "n/a",
        )
        all_results: Dict[str, List[BaselineResult]] = {}

        async def _run_one_system(baseline: BaselineAdapter):
            async with semaphore:
                out_path = self._output_dir / f"baseline_{baseline.name}.jsonl"
                if skip_existing and out_path.exists():
                    can_reuse, reason = _can_reuse_baseline_cache(out_path, baseline, golden_set)
                    if can_reuse:
                        logger.info("[Suite] '%s' already done, loading from %s",
                                    baseline.name, out_path)
                        all_results[baseline.name] = self._load_results(str(out_path))
                        return
                    logger.info(
                        "[Suite] Stale cache for '%s' ignored: %s; rebuilding %s",
                        baseline.name,
                        reason,
                        out_path,
                    )

                logger.info("[Suite] Evaluating '%s' (%d samples)...",
                            baseline.citation_name, len(golden_set.samples))
                if skip_prepare:
                    logger.info(
                        "[Suite] '%s' reusing existing index state; prepare() skipped",
                        baseline.citation_name,
                    )
                else:
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
                    if not skip_cleanup:
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
        max_conc = _resolve_sample_concurrency(self._bm_adapter, baseline)
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
                corpus_metadata = _baseline_corpus_metadata(self._corpus_metadata, setup_metrics)
                baseline_identity = _baseline_cache_identity(baseline)
                base_metadata: Dict[str, Any] = {
                    **baseline.extra_metadata(),
                    **corpus_metadata,
                    "baseline_cache_identity": baseline_identity,
                    "setup_metrics": setup_metrics,
                }
                telemetry: Dict[str, Any] = {
                    "baseline_name": baseline.name,
                    "system_name": baseline.citation_name,
                    "result_schema_version": baseline_identity["result_schema_version"],
                    "adapter_class": baseline_identity["adapter_class"],
                    "config_hash": baseline_identity["config_hash"],
                    **corpus_metadata,
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
                        base_metadata["baseline_cache_identity"] = baseline_identity
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
                evidence_traces = _normalize_evidence_traces(telemetry, base_metadata, context_paths)
                query_budget = _normalize_query_budget(
                    telemetry,
                    elapsed=pred_elapsed,
                    tokens=pred_tokens,
                    judge_tokens=judge_tokens,
                    guard_config=self._guard_config,
                    measured_sample_concurrency=max_conc,
                )
                telemetry["evidence_traces"] = evidence_traces
                telemetry["query_budget"] = query_budget
                base_metadata["evidence_traces"] = evidence_traces
                base_metadata["query_budget"] = query_budget
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
        metadata = r.metadata if isinstance(r.metadata, dict) else {}
        identity = metadata.get("baseline_cache_identity") if isinstance(metadata.get("baseline_cache_identity"), dict) else {}
        return {
            "result_schema_version": identity.get("result_schema_version", ""),
            "baseline_name": identity.get("baseline_name", ""),
            "citation_name": identity.get("citation_name", r.system_name),
            "adapter_class": identity.get("adapter_class", ""),
            "config_hash": identity.get("config_hash", ""),
            "baseline_cache_identity": identity,
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
            "metadata":      metadata,
            "setup_metrics": metadata.get("setup_metrics", {}),
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


def _baseline_corpus_metadata(corpus_metadata: Dict[str, Any], setup_metrics: Dict[str, Any]) -> Dict[str, Any]:
    metadata = dict(corpus_metadata or {})
    provenance = str(metadata.get("corpus_provenance") or "")
    risk = str(metadata.get("corpus_risk") or "")
    try:
        indexed_documents = int(setup_metrics.get("indexed_documents") or 0)
    except (TypeError, ValueError):
        indexed_documents = 0
    index_required = bool(setup_metrics.get("index_required")) or indexed_documents > 0
    if provenance == "sample" and index_required:
        extra = "evaluation_set_context_index"
        risks = {part.strip() for part in risk.split(",") if part.strip()}
        risks.add(extra)
        metadata["corpus_risk"] = ",".join(sorted(risks))
        metadata["baseline_index_scope"] = "evaluation_set_sample_context"
    elif provenance == "sample":
        metadata.setdefault("baseline_index_scope", "per_sample_oracle_context")
    return metadata


def _baseline_cache_identity(baseline: BaselineAdapter) -> Dict[str, Any]:
    identity_fn = getattr(baseline, "cache_identity", None)
    if callable(identity_fn):
        value = identity_fn()
        if isinstance(value, dict):
            return {
                "result_schema_version": str(value.get("result_schema_version", "")),
                "baseline_name": str(value.get("baseline_name", baseline.name)),
                "citation_name": str(value.get("citation_name", baseline.citation_name)),
                "adapter_class": str(value.get("adapter_class", baseline.__class__.__name__)),
                "config_hash": str(value.get("config_hash", "")),
            }
    return {
        "result_schema_version": str(getattr(baseline, "result_schema_version", "baseline_result_v2")),
        "baseline_name": baseline.name,
        "citation_name": baseline.citation_name,
        "adapter_class": f"{baseline.__class__.__module__}.{baseline.__class__.__qualname__}",
        "config_hash": "",
    }


def _can_reuse_baseline_cache(path: Path, baseline: BaselineAdapter, golden_set: Any = None) -> tuple[bool, str]:
    first = _read_first_jsonl_record(path)
    if not first:
        return False, "empty_or_unreadable_jsonl"
    expected = _baseline_cache_identity(baseline)
    actual = first.get("baseline_cache_identity") if isinstance(first.get("baseline_cache_identity"), dict) else {}
    if not actual:
        actual = {
            "result_schema_version": first.get("result_schema_version", ""),
            "baseline_name": first.get("baseline_name", ""),
            "citation_name": first.get("citation_name", first.get("system_name", "")),
            "adapter_class": first.get("adapter_class", ""),
            "config_hash": first.get("config_hash", ""),
        }
    for key, expected_value in expected.items():
        actual_value = str(actual.get(key, ""))
        if actual_value != str(expected_value):
            return False, f"{key}_mismatch(actual={actual_value!r}, expected={expected_value!r})"
    expected_ids = _expected_sample_ids(golden_set)
    if expected_ids:
        observed_ids, malformed = _read_jsonl_sample_ids(path)
        if malformed:
            return False, f"malformed_jsonl_records={malformed}"
        if len(observed_ids) != len(expected_ids):
            return False, f"row_count_mismatch(actual={len(observed_ids)}, expected={len(expected_ids)})"
        duplicate_count = len(observed_ids) - len(set(observed_ids))
        if duplicate_count:
            return False, f"duplicate_sample_ids={duplicate_count}"
        expected_set = set(expected_ids)
        observed_set = set(observed_ids)
        if expected_set != observed_set:
            missing = sorted(expected_set - observed_set)[:5]
            extra = sorted(observed_set - expected_set)[:5]
            return False, f"sample_ids_mismatch(missing={missing}, extra={extra})"
    return True, "ok"



def _expected_sample_ids(golden_set: Any = None) -> List[str]:
    if golden_set is None:
        return []
    sample_ids_fn = getattr(golden_set, "sample_ids", None)
    if callable(sample_ids_fn):
        try:
            return [str(sample_id) for sample_id in sample_ids_fn()]
        except Exception:
            return []
    ids: List[str] = []
    for sample in getattr(golden_set, "samples", []) or []:
        if isinstance(sample, dict) and sample.get("sample_id"):
            ids.append(str(sample["sample_id"]))
    return ids


def _read_jsonl_sample_ids(path: Path) -> tuple[List[str], int]:
    ids: List[str] = []
    malformed = 0
    try:
        with path.open(encoding="utf-8") as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    malformed += 1
                    continue
                if isinstance(row, dict) and row.get("sample_id"):
                    ids.append(str(row["sample_id"]))
                else:
                    malformed += 1
    except OSError:
        return ids, malformed + 1
    return ids, malformed


def _read_first_jsonl_record(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with path.open(encoding="utf-8") as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                return row if isinstance(row, dict) else None
    except (OSError, json.JSONDecodeError):
        return None
    return None


def _merge_prediction_telemetry(telemetry: Dict[str, Any], metadata: Dict[str, Any]) -> None:
    """Promote baseline-specific retrieval metadata into common telemetry keys."""
    nested = metadata.get("telemetry") if isinstance(metadata.get("telemetry"), dict) else {}
    for source in (nested, metadata):
        for key in ("read_file_ids", "retrieval_logs", "evidence_sources", "evidence_snippets", "search_history"):
            value = source.get(key) if isinstance(source, dict) else None
            if value and not telemetry.get(key):
                telemetry[key] = value
    # Promote scalar query-budget signals so query-budget normalization can see
    # them. Agentic baselines (ReAct, LENS ablation adapter) report loop_count
    # and llm_calls under metadata or metadata['telemetry'] rather than at the
    # top telemetry level; without this promotion oracle_calls/llm_calls collapse
    # to zero in the budget tables.
    for source in (nested, metadata):
        if not isinstance(source, dict):
            continue
        for key in ("loop_count", "llm_calls", "total_llm_calls", "oracle_calls",
                    "num_files_read", "search_history_count", "total_tokens"):
            value = source.get(key)
            if value in (None, ""):
                continue
            if telemetry.get(key) in (None, "", 0, 0.0):
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


def _normalize_evidence_traces(
    telemetry: Dict[str, Any],
    metadata: Dict[str, Any],
    context_paths: List[str],
) -> List[Dict[str, Any]]:
    """Return a compact, JSON-safe evidence trace list shared by all baselines."""
    traces: List[Dict[str, Any]] = []
    seen = set()

    def _append(trace: Dict[str, Any]) -> None:
        source_path = str(trace.get("source_path") or trace.get("path") or trace.get("file") or "")
        title = str(trace.get("title") or trace.get("source_title") or "")
        snippet = str(trace.get("snippet") or trace.get("text") or trace.get("evidence") or "")
        key = (source_path, title, snippet[:120])
        if not any(key) or key in seen:
            return
        seen.add(key)
        traces.append({
            "source_path": source_path,
            "title": title,
            "span_start": _safe_int(trace.get("span_start", trace.get("start", -1)), -1),
            "span_end": _safe_int(trace.get("span_end", trace.get("end", -1)), -1),
            "snippet": snippet[:2000],
            "score": _safe_float(trace.get("score", trace.get("fused_score", 0.0))),
            "metadata": trace.get("metadata", {}) if isinstance(trace.get("metadata"), dict) else {},
        })

    for source in (telemetry, metadata):
        raw = source.get("evidence_traces") if isinstance(source, dict) else None
        if isinstance(raw, list):
            for item in raw:
                if isinstance(item, dict):
                    _append(item)
                elif item:
                    _append({"source_path": str(item)})

    for source in (telemetry, metadata):
        if not isinstance(source, dict):
            continue
        snippets = source.get("evidence_snippets") if isinstance(source.get("evidence_snippets"), list) else []
        for key in ("evidence_sources", "read_file_ids"):
            values = source.get(key)
            if not isinstance(values, list):
                continue
            for index, value in enumerate(values):
                if isinstance(value, dict):
                    _append(value)
                elif value:
                    snippet = snippets[index] if index < len(snippets) else ""
                    _append({"source_path": str(value), "snippet": str(snippet)})
        for key in ("top_chunks", "top_docs"):
            values = source.get(key)
            if isinstance(values, list):
                for value in values:
                    if isinstance(value, dict):
                        _append(value)

    if not traces:
        for path in context_paths or []:
            _append({"source_path": str(path), "metadata": {"role": "context_path"}})
    return traces


def _normalize_query_budget(
    telemetry: Dict[str, Any],
    *,
    elapsed: float,
    tokens: int,
    judge_tokens: int,
    guard_config: GuardConfig,
    measured_sample_concurrency: int = 1,
) -> Dict[str, Any]:
    """Normalize query-time budget telemetry across baseline implementations."""
    existing = telemetry.get("query_budget") if isinstance(telemetry.get("query_budget"), dict) else {}
    read_items = telemetry.get("read_file_ids") if isinstance(telemetry.get("read_file_ids"), list) else []
    search_history = telemetry.get("search_history") if isinstance(telemetry.get("search_history"), list) else []
    retrieval_logs = telemetry.get("retrieval_logs") if isinstance(telemetry.get("retrieval_logs"), list) else []
    llm_calls = _first_number(existing, telemetry, "llm_calls", "oracle_calls", "loop_count")
    return {
        "latency_seconds": _safe_float(existing.get("latency_seconds", elapsed)),
        "total_tokens": _safe_int(existing.get("total_tokens", tokens + judge_tokens)),
        "search_tokens": _safe_int(existing.get("search_tokens", tokens)),
        "judge_tokens": _safe_int(existing.get("judge_tokens", judge_tokens)),
        "oracle_calls": _safe_int(existing.get("oracle_calls", llm_calls)),
        "llm_calls": _safe_int(existing.get("llm_calls", llm_calls)),
        "search_calls": _safe_int(existing.get("search_calls", len(search_history) or len(retrieval_logs))),
        "read_calls": _safe_int(existing.get("read_calls", len(read_items))),
        "sample_timeout_seconds": _safe_float(existing.get("sample_timeout_seconds", guard_config.sample_timeout_seconds)),
        "max_runtime_seconds": _safe_float(existing.get("max_runtime_seconds", guard_config.max_runtime_seconds)),
        "max_total_tokens": _safe_int(existing.get("max_total_tokens", guard_config.max_total_tokens)),
        "budget_exceeded": str(telemetry.get("failure_reason") or "") == "budget_exceeded",
        # Latency is only comparable between systems measured at the same
        # concurrency, so pin the value this sample was actually measured under.
        "measured_sample_concurrency": _safe_int(
            existing.get("measured_sample_concurrency", measured_sample_concurrency), 1
        ),
    }


def _first_number(*sources: Dict[str, Any], default: float = 0.0) -> float:
    for source in sources:
        if not isinstance(source, dict):
            continue
        for key in ("oracle_calls", "llm_calls", "total_llm_calls", "loop_count"):
            value = source.get(key)
            if value not in (None, ""):
                return _safe_float(value, default)
    return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _baseline_sample_concurrency(baseline: BaselineAdapter) -> int:
    """Per-sample concurrency a baseline declares for itself, for logging only."""
    try:
        return max(int(baseline.get_max_concurrent()), 1)
    except (AttributeError, TypeError, ValueError):
        return 1


def _resolve_sample_concurrency(bm_adapter: Any, baseline: BaselineAdapter) -> int:
    """Resolve how many samples of one baseline may be in flight at once.

    The baseline's own declaration is the default. A positive benchmark-level
    override raises it, which is what makes multi-turn LLM baselines finish in
    hours instead of days, but only for baselines that allow concurrent queries.
    The resolved value is recorded per result so the reported latency stays
    interpretable: latency measured under concurrency is a property of the
    measurement setup, not of the system alone.
    """
    declared = _baseline_sample_concurrency(baseline)
    getter = getattr(bm_adapter, "get_baseline_sample_concurrency", None)
    if not callable(getter):
        return declared
    try:
        override = int(getter())
    except (TypeError, ValueError):
        return declared
    if override <= 0:
        return declared
    supports = getattr(baseline, "supports_query_concurrency", None)
    if callable(supports) and not supports():
        return declared
    return max(override, 1)


def _resolve_system_concurrency(bm_adapter: Any, max_concurrent: Optional[int]) -> int:
    """Resolve how many baseline systems may be evaluated at once.

    An explicit value always wins. Otherwise the benchmark adapter's concurrency
    configuration is used, so a profile-level setting reaches the suite instead of
    being shadowed by a hardcoded constant. Falls back to 1 when the adapter
    exposes nothing usable, since serial evaluation is always safe.
    """
    if max_concurrent is not None:
        return max(int(max_concurrent), 1)
    getter = getattr(bm_adapter, "get_max_concurrent", None)
    if callable(getter):
        try:
            return max(int(getter()), 1)
        except (TypeError, ValueError):
            pass
    return 1


def _coerce_guard_config(value: Optional[GuardConfig | Dict[str, Any]]) -> GuardConfig:
    if value is None:
        return GuardConfig()
    if isinstance(value, GuardConfig):
        return value
    return GuardConfig.from_run_config(dict(value))
