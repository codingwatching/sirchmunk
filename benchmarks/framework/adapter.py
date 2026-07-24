"""framework/adapter.py — BenchmarkAdapter 抽象基类

所有 benchmark 必须实现此接口以接入研究流水线。
依赖倒置：UnifiedExperimentRunner 只依赖本抽象，不依赖具体 benchmark。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

from .schema import BenchmarkSample


class BenchmarkAdapter(ABC):
    """Benchmark 接入适配器抽象基类。

    实现者需覆盖全部 abstractmethod。
    可选 hook（带默认实现）：``extra_result_fields``。
    """

    # ------------------------------------------------------------------
    # 必须实现
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def name(self) -> str:
        """Benchmark 名称，用于日志与文件命名。"""

    @property
    @abstractmethod
    def env_file(self) -> str:
        """主配置 .env 文件的绝对路径。"""

    @abstractmethod
    def load_samples(self, limit: int = 0, seed: int = 42) -> List[BenchmarkSample]:
        """加载问题集。

        Args:
            limit: 0 表示全量；>0 表示随机采样 limit 条（使用 seed）。
            seed:  随机种子，保证复现性。

        Returns:
            BenchmarkSample 列表。
        """

    @abstractmethod
    def validate_corpus(self) -> Tuple[int, List[str]]:
        """验证语料库完整性。

        Returns:
            (found_count, missing_doc_names)
        """

    @abstractmethod
    def get_search_paths(self, sample: BenchmarkSample) -> List[str]:
        """返回针对该样本的搜索路径列表。

        singleDoc 模式返回单个 PDF 路径；
        sharedCorpus 模式返回整个语料目录。
        """

    @abstractmethod
    def get_run_config(self) -> Dict[str, Any]:
        """返回可序列化的运行配置字典，用于 config_hash 计算与实验记录。"""

    @abstractmethod
    def build_searcher(self) -> Any:
        """构建 AgenticSearch 实例（或兼容接口的搜索器）。"""

    @abstractmethod
    def build_judge(self) -> Optional[Any]:
        """构建 LLM Judge 实例；无 judge 时返回 None。"""

    @abstractmethod
    def get_output_dir(self) -> str:
        """输出目录的绝对路径。"""

    @abstractmethod
    def get_work_path(self) -> str:
        """AgenticSearch work_path 的绝对路径。"""

    # ------------------------------------------------------------------
    # 可选 hook（提供默认实现）
    # ------------------------------------------------------------------

    def extra_result_fields(self, sample: BenchmarkSample) -> Dict[str, Any]:
        """返回需要写入结果 JSONL 的 benchmark 特有字段。

        默认将 sample.metadata 全量透传。
        子类可覆盖以精细控制。
        """
        return dict(sample.metadata)

    def get_max_concurrent(self) -> int:
        """最大并发数，默认 3。"""
        return 3

    def get_request_delay(self) -> float:
        """每次请求间延迟（秒），默认 0.5。"""
        return 0.5

    def get_search_kwargs(self) -> Dict[str, Any]:
        """传递给 searcher.search() 的额外关键字参数。"""
        return {}

    def get_protocol_spec(self, run_id: str, seed: int, limit: int) -> Dict[str, Any]:
        """Return an optional machine-readable experiment protocol override."""
        return {}

    def get_dataset_manifest(self) -> Dict[str, Any]:
        """Return optional dataset/corpus provenance metadata."""
        return {}

    def enrich_telemetry(self, sample: BenchmarkSample, prediction: str, telemetry: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Return optional per-sample telemetry additions after judging."""
        return {}

    def get_analysis_schema(self) -> Dict[str, Any]:
        """Return benchmark-specific metadata keys for badcase analysis."""
        return {"primary_group_key": "group"}

    def get_config_schema(self) -> Dict[str, Any]:
        """Return benchmark-specific config keys and global config boundaries."""
        return {
            "global_keys": [
                "LLM_BASE_URL",
                "LLM_API_KEY",
                "LLM_MODEL_NAME",
                "LLM_TIMEOUT",
                "EMBEDDING_MODEL_ID",
                "EMBEDDING_CACHE_DIR",
                "SIRCHMUNK_WORK_PATH",
                "GREP_CONCURRENT_LIMIT",
            ],
            "top_k_env_key": "TOP_K_FILES",
            "mode_env_key": "MODE",
        }

    def get_cache_policy(self) -> Dict[str, Any]:
        """Return adapter-specific cache policy hints for P3 queue runs."""
        return {
            "cache_names": [".cache", "knowledge", "history", "compile", "rga"],
            "cache_paths": [".cache/rga", ".cache/knowledge", ".cache/compile"],
            "compiled_markers": ["compile", "compiled"],
        }

    def get_metric_aggregator(self):
        """Return optional benchmark-specific metrics aggregator callable."""
        return None
