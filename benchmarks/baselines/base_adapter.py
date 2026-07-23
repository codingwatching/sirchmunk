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
    Mock 测试          → MockBaseline (baselines/mock.py)
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


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
    elapsed: float
    tokens_used: int = 0
    judge_tokens: int = 0
    question_type: str = ""             # 从 sample metadata 提取，用于分类统计
    error: Optional[str] = None
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
    # 可选覆盖
    # ------------------------------------------------------------------

    async def prepare(self, golden_set: Any = None, bm_adapter: Any = None) -> BaselineSetupResult:
        """Prepare the baseline before prediction and return setup metrics.

        Implementations that build an index, graph, embeddings, or cache must
        account for that work here so setup cost is included in reports.
        """
        return BaselineSetupResult()

    async def run(self, question: str, context_paths: List[str]) -> BaselinePrediction:
        """Run one prediction. Defaults to predict() for backward compatibility."""
        return await self.predict(question, context_paths)

    async def evaluate(self, prediction: str, gold_answer: str, question: str, judge: Any) -> Dict[str, Any]:
        """Evaluate one prediction with the benchmark judge."""
        jr = await judge.judge(prediction=prediction, gold_answer=gold_answer, question=question)
        cr = await judge.judge_coverage(prediction=prediction, question=question)
        return {
            "judge_correct": bool(jr.get("equivalent", False)),
            "coverage": bool(cr.get("has_coverage", False)),
            "judge_tokens": int(jr.get("tokens_used", 0) or 0) + int(cr.get("tokens_used", 0) or 0),
            "judge_result": jr,
            "coverage_result": cr,
        }

    async def cleanup(self) -> None:
        """Release resources after evaluation."""
        return None

    def collect_setup_metrics(self) -> Dict[str, Any]:
        """Return setup metrics after prepare()."""
        return {}

    def is_available(self) -> bool:
        """检查系统依赖是否满足（API key、SDK import 等）。

        运行前调用，返回 False 时该系统被跳过并记录警告。
        默认返回 True（乐观假设）。
        """
        return True

    def get_max_concurrent(self) -> int:
        """最大并发请求数。默认 1（串行），避免 API 限流。"""
        return 1

    def get_request_delay(self) -> float:
        """每次请求间延迟（秒）。默认 1.0。"""
        return 1.0

    def extra_metadata(self) -> Dict[str, Any]:
        """附加到 BaselineResult.metadata 的系统级元数据。
        
        如: {"model": "gpt-4o-2024-05-13", "temperature": 0}
        """
        return {}
