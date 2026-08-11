"""baselines/base_adapter.py — BaselineAdapter 抽象基类

竞品系统适配器接口，完全独立于 framework/ 自改进循环。

设计原则：
- 比 BenchmarkAdapter 更薄：只需实现 predict() 和若干属性
- 不涉及 build_searcher / work_path / Config 三层隔离 等自改进相关概念
- 数据结构自成体系，通过 evaluation/suite.py 与 framework/schema.py 桥接

竞品接入方式对应关系：
    有 Python SDK      → SdkBaseline (baselines/sdk_baseline.py)
    有 HTTP API        → 继承 BaselineAdapter，在 predict() 中发 HTTP 请求
    只有发表数字        → ManualImportAdapter (baselines/sdk_baseline.py)
"""
from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

try:
    from framework.lifecycle_schema import BaselineIndexValidation, FailureReason
    _HAS_LIFECYCLE_SCHEMA = True
except ImportError:  # pragma: no cover - allows direct module loading in isolation
    BaselineIndexValidation = None  # type: ignore
    FailureReason = None  # type: ignore
    _HAS_LIFECYCLE_SCHEMA = False


# ---------------------------------------------------------------------------
# 核心数据结构
# ---------------------------------------------------------------------------


@dataclass
class BaselineSetupResult:
    """Baseline preparation metrics used for fair setup-cost accounting."""
    setup_seconds: float = 0.0
    preprocessing_seconds: float = 0.0
    index_build_seconds: float = 0.0
    storage_bytes: int = 0
    indexed_documents: int = 0
    expected_documents: int = 0
    build_completed: bool = True
    index_ready: bool = True
    failure_reason: str = "none"
    failure_message: str = ""
    peak_ram_bytes: int = 0
    preprocess_llm_tokens: int = 0
    api_cost_usd: float = 0.0
    artifact_dir: str = ""
    index_required: bool = True
    rebuild_required: bool = False
    query_ready_immediately: bool = False
    partial_index: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BaselinePrediction:
    """单次预测结果，由 BaselineAdapter.predict() 返回。"""
    answer: str
    elapsed: float              # 耗时（秒）
    tokens_used: int = 0        # 消耗 token 数（若可获取）
    metadata: Dict[str, Any] = field(default_factory=dict)
    """metadata 可存放系统特有信息（如检索到的文档列表等）。"""


@dataclass
class BaselineResult:
    """经过 Judge 评估后的单样本完整结果。

    由 BaselineEvaluationSuite 生成，用于 PaperTableGenerator 汇聚。
    字段命名有意对齐 framework/schema.py 的 PredictionResult，
    以便 PaperTableGenerator 统一处理。
    """
    sample_id: str
    system_name: str                    # 对应 BaselineAdapter.citation_name
    question: str
    gold_answer: str
    prediction: str
    judge_correct: bool
    coverage: bool
    evidence_recall: float = 0.0
    elapsed: float = 0.0
    tokens_used: int = 0
    judge_tokens: int = 0
    question_type: str = ""             # 从 sample metadata 提取，用于分类统计
    error: Optional[str] = None
    failure_reason: str = ""
    telemetry: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# 抽象基类
# ---------------------------------------------------------------------------


class BaselineAdapter(ABC):
    """竞品系统适配器抽象基类。

    实现者只需覆盖 predict()、name 和 citation_name。
    is_available() 可选覆盖（用于在运行前检查 API key / SDK 等依赖）。

    Usage::

        class MySystemAdapter(BaselineAdapter):
            @property
            def name(self) -> str:
                return "my_system_v1"

            @property
            def citation_name(self) -> str:
                return "MySystem v1.0 (Author et al., 2024)"

            async def predict(self, question, context_paths) -> BaselinePrediction:
                answer = call_my_system(question, context_paths)
                return BaselinePrediction(answer=answer, elapsed=0.5)
    """

    result_schema_version = "baseline_result_v2"

    # ------------------------------------------------------------------
    # 必须实现
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def name(self) -> str:
        """系统内部标识符（用于文件命名、日志），不含空格。
        
        例如: "gpt4o_zeroshot", "naive_rag_v2"
        """

    @property
    @abstractmethod
    def citation_name(self) -> str:
        """论文表格中的展示名称（允许空格、括号）。
        
        例如: "GPT-4o (zero-shot)", "Naive RAG", "LENS (ours)"
        """

    @abstractmethod
    async def predict(
        self,
        question: str,
        context_paths: List[str],
    ) -> BaselinePrediction:
        """给定问题和文档路径，返回答案。

        Args:
            question:      原始问题文本。
            context_paths: 与本题相关的文档路径列表（由 BenchmarkAdapter
                           提供，与 Sirchmunk 使用相同路径，保证对比公平）。
                           竞品可以使用这些路径，也可以忽略（如纯参数化模型）。

        Returns:
            BaselinePrediction，包含答案文本和耗时。
        """

    # ------------------------------------------------------------------
    # 检索契约（评估口径的前提）
    # ------------------------------------------------------------------

    # How this system obtains its answers. Every baseline is compared on the
    # same terms, and those terms depend on this being declared honestly:
    #
    #   "retrieval_based" — reads the corpus. Must report what it read via
    #       read_file_ids / evidence_sources, otherwise evidence recall and
    #       source grounding silently read as zero and the system looks like it
    #       retrieved nothing.
    #   "retrieval_free"  — answers from model parameters alone. Zero evidence
    #       is the correct result, not missing data.
    #
    # Defaulting to retrieval_based is deliberate: a new adapter that forgets to
    # declare gets held to the stricter obligation and fails the check below,
    # rather than passing quietly with unreported evidence.
    retrieval_mode: str = "retrieval_based"

    RETRIEVAL_MODES = ("retrieval_based", "retrieval_free")

    def validate_prediction_contract(self, prediction: "BaselinePrediction") -> List[str]:
        """Check one prediction against the declared retrieval mode.

        Returns a list of problems, empty when the prediction is consistent with
        what the adapter claims to be. This exists because a missing evidence
        field is not visible in the score: the run completes, the metric reads
        zero, and the system is recorded as having retrieved nothing. Comparing
        such a row against systems that do report evidence is not meaningful, so
        the mismatch has to surface as an error rather than as a low number.
        """
        problems: List[str] = []
        mode = str(getattr(self, "retrieval_mode", "") or "")
        if mode not in self.RETRIEVAL_MODES:
            problems.append(
                f"{self.name}: retrieval_mode={mode!r} is not one of {self.RETRIEVAL_MODES}"
            )
            return problems

        meta = prediction.metadata or {}
        read_ids = meta.get("read_file_ids")
        sources = meta.get("evidence_sources")

        if mode == "retrieval_free":
            if read_ids or sources:
                problems.append(
                    f"{self.name}: declared retrieval_free but reported evidence "
                    f"({len(read_ids or [])} read_file_ids, {len(sources or [])} sources)"
                )
            return problems

        # retrieval_based: the fields must be present, so that an empty result
        # means "retrieved nothing on this question" rather than "never wired up".
        if read_ids is None and sources is None:
            problems.append(
                f"{self.name}: declared retrieval_based but reported neither "
                "read_file_ids nor evidence_sources, so evidence metrics cannot "
                "distinguish a retrieval miss from missing instrumentation"
            )
        return problems

    # ------------------------------------------------------------------
    # 可选覆盖
    # ------------------------------------------------------------------

    async def prepare(self, golden_set: Any = None, bm_adapter: Any = None) -> BaselineSetupResult:
        """Prepare the baseline before prediction and return setup metrics.

        Implementations that build an index, graph, embeddings, or cache must
        account for that work here so setup cost is included in reports.
        """
        return BaselineSetupResult()

    async def run(self, question: str, context_paths: List[str]) -> BaselinePrediction:
        """Run one prediction through the adapter implementation."""
        return await self.predict(question, context_paths)

    async def evaluate(self, prediction: str, gold_answer: str, question: str, judge: Any) -> Dict[str, Any]:
        """Evaluate one prediction with the benchmark judge.

        ``judge_correct`` stays the primary boolean, but the surface-form-tolerant
        match and the indeterminate flag are lifted to the top level too: both
        need to be aggregated per system, and reaching into the nested judge
        payload for them is easy to forget, which is how an unusable metric goes
        unnoticed.
        """
        jr = await judge.judge(prediction=prediction, gold_answer=gold_answer, question=question)
        cr = await judge.judge_coverage(prediction=prediction, question=question)
        return {
            "judge_correct": bool(jr.get("equivalent", False)),
            "normalized_em": float(jr.get("normalized_em", 0.0) or 0.0),
            "judge_indeterminate": bool(jr.get("indeterminate", False)),
            "judge_status": str(jr.get("judge_status", "") or ""),
            "answer_form_compliant": bool(jr.get("form_compliant", True)),
            "answer_form_reason": str(jr.get("form_reason", "") or ""),
            "coverage": bool(cr.get("has_coverage", False)),
            "judge_tokens": int(jr.get("tokens_used", 0) or 0) + int(cr.get("tokens_used", 0) or 0),
            "judge_result": jr,
            "coverage_result": cr,
        }

    async def cleanup(self) -> None:
        """Release resources after evaluation."""
        return None

    async def update_index(self, mutation: Any, bm_adapter: Any = None) -> Dict[str, Any]:
        """Update an existing baseline index after corpus mutation.

        The default implementation declares that incremental update is not
        supported. Dynamic update studies will record this as requiring a full
        rebuild rather than treating it as a runtime error.
        """
        return {
            "update_supported": False,
            "rebuild_required": True,
            "failure_reason": "update_not_supported",
        }

    def estimate_update_cost(self, mutation: Any) -> Dict[str, Any]:
        """Return optional update-cost estimate without mutating the baseline."""
        return {"rebuild_required": True}

    def collect_setup_metrics(self) -> Dict[str, Any]:
        """Return setup metrics after prepare()."""
        return {}

    def adapter_class_path(self) -> str:
        """Return a stable adapter implementation identity for cache reuse checks."""
        return f"{self.__class__.__module__}.{self.__class__.__qualname__}"

    def baseline_config(self) -> Dict[str, Any]:
        """Return lightweight constructor/runtime config used to validate cached JSONL reuse."""
        ignored = {
            "llm", "setup", "docs", "chunks", "df", "avgdl", "rag", "predictions",
            "last_paths", "dependency_error", "cache", "index", "searcher",
        }
        config: Dict[str, Any] = {}
        for raw_key, value in sorted(getattr(self, "__dict__", {}).items()):
            key = str(raw_key).lstrip("_")
            if key in ignored:
                continue
            safe_value = _json_safe_baseline_value(value)
            if safe_value is not None:
                config[key] = safe_value
        return config

    def config_hash(self) -> str:
        """Stable short hash of adapter identity and reusable configuration."""
        payload = {
            "baseline_name": self.name,
            "citation_name": self.citation_name,
            "adapter_class": self.adapter_class_path(),
            "schema_version": self.result_schema_version,
            "config": self.baseline_config(),
        }
        raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    def cache_identity(self) -> Dict[str, Any]:
        """Identity fields that must match before an existing baseline JSONL can be reused."""
        return {
            "result_schema_version": self.result_schema_version,
            "baseline_name": self.name,
            "citation_name": self.citation_name,
            "adapter_class": self.adapter_class_path(),
            "config_hash": self.config_hash(),
        }

    def is_index_ready(self) -> bool:
        """Return whether a full-corpus index is ready for query evaluation.

        Index-free and manual-import baselines can keep the default ``True``.
        Index-heavy baselines should override this to prevent partial indexes
        from entering warm-query quality tables.
        """
        return True

    def is_index_required(self) -> bool:
        """Return whether this baseline requires a built index before queries."""
        return True

    def is_query_ready_immediately(self) -> bool:
        """Return whether corpus changes are query-ready without index rebuild."""
        return False

    def validate_index(self, corpus_manifest: Optional[Dict[str, Any]] = None) -> Any:
        """Validate that the built index covers the declared corpus.

        The default implementation is suitable for index-free and manual-import baselines.
        Indexing baselines should return ``BaselineIndexValidation`` or a
        dict with ``index_ready`` and coverage metadata.
        """
        if not _HAS_LIFECYCLE_SCHEMA or BaselineIndexValidation is None:
            return {"index_ready": True}
        return BaselineIndexValidation(index_ready=True)

    def get_lifecycle_metadata(self) -> Dict[str, Any]:
        """Return baseline-specific lifecycle metadata for artifact records."""
        return {}

    def classify_failure(self, exc: Exception) -> str:
        """Map an exception to a structured lifecycle failure reason."""
        text = str(exc).lower()
        if "timeout" in text or "timed out" in text:
            return "timeout"
        if "out of memory" in text or "oom" in text:
            return "oom"
        if "no space" in text or "disk" in text:
            return "disk_exceeded"
        if "api" in text and "budget" in text:
            return "api_budget_exceeded"
        if "import" in text or "dependency" in text or "module" in text:
            return "dependency_missing"
        return "unknown"

    def is_available(self) -> bool:
        """检查系统依赖是否满足（API key、SDK import 等）。

        运行前调用，返回 False 时该系统被跳过并记录警告。
        默认返回 True（乐观假设）。
        """
        return True

    def requires_import_coverage(self) -> bool:
        """Return True for baselines whose predictions must be fully pre-imported."""
        return False

    def get_max_concurrent(self) -> int:
        """最大并发请求数。默认 1（串行），避免 API 限流。"""
        return 1

    def supports_query_concurrency(self) -> bool:
        """本竞品是否允许多个样本同时查询。

        默认 True。声明 False 的竞品不会被 benchmark 级的并发覆盖影响，用于
        查询路径共享可变实例、并发安全性未经验证的情形。
        """
        return True

    def get_request_delay(self) -> float:
        """每次请求间延迟（秒）。默认 1.0。"""
        return 1.0

    def extra_metadata(self) -> Dict[str, Any]:
        """附加到 BaselineResult.metadata 的系统级元数据。
        
        如: {"model": "gpt-4o-2024-05-13", "temperature": 0}
        """
        return {}


def _json_safe_baseline_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        safe_list = [_json_safe_baseline_value(item) for item in value]
        return [item for item in safe_list if item is not None]
    if isinstance(value, dict):
        safe_dict = {}
        for key, item in sorted(value.items(), key=lambda kv: str(kv[0])):
            safe_item = _json_safe_baseline_value(item)
            if safe_item is not None:
                safe_dict[str(key)] = safe_item
        return safe_dict
    return None
