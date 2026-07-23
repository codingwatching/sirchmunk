"""baselines/sdk_baseline.py — SDK 通用包装 + 已发表结果导入

SdkBaseline:
    包装任意 Python SDK / 框架（不要求有 REST API）。
    通过 predict_fn 回调注入竞品系统的调用逻辑，框架只负责调度和计时。

ManualImportAdapter:
    从预计算的 JSONL 文件导入竞品的预测结果。
    两种使用场景：
    a) 竞品提供了原始 predictions（需要用我们的 Judge 重新评分，保证 Judge 一致性）
    b) 竞品只有发表的汇总数字（直接传入 metrics_dict 跳过 Judge）

接入新竞品的最快路径：
    1. 如果竞品有 Python 包，用 SdkBaseline 包装
    2. 如果只有已发表数字，用 PaperTableGenerator.add_published_metrics() 直接写入表格
    3. 如果有原始 predictions JSONL，用 ManualImportAdapter 加载后重新过 Judge
"""
from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .base_adapter import BaselineAdapter, BaselinePrediction


class SdkBaseline(BaselineAdapter):
    """通用 Python SDK/框架竞品包装器。

    使用方法（以假想的 NaiveRAG 为例）::

        # 1. 初始化竞品系统（只需实例化一次）
        from some_rag_package import NaiveRAGSystem
        system = NaiveRAGSystem(
            model="gpt-4o-mini",
            top_k=5,
        )

        # 2. 定义 predict_fn：接受 (system, question, context_paths) → str
        def naive_rag_predict(sys, question, paths):
            return sys.retrieve_and_answer(question, document_paths=paths)

        # 3. 包装为 SdkBaseline
        baseline = SdkBaseline(
            name="naive_rag_v1",
            citation_name="Naive RAG (Gao et al., 2023)",
            system=system,
            predict_fn=naive_rag_predict,
            is_async=False,      # 若 predict_fn 为 async 则设为 True
            max_concurrent=2,
            metadata={"model": "gpt-4o-mini", "top_k": 5},
        )

    若竞品的 predict 是异步的::

        async def async_predict(sys, question, paths):
            return await sys.apredict(question, paths)

        baseline = SdkBaseline(..., predict_fn=async_predict, is_async=True)
    """

    def __init__(
        self,
        name: str,
        citation_name: str,
        system: Any,
        predict_fn: Callable,
        is_async: bool = False,
        max_concurrent: int = 1,
        request_delay: float = 0.5,
        tokens_fn: Optional[Callable] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Args:
            name:           系统内部 ID，不含空格，用于文件命名。
            citation_name:  论文表格展示名称。
            system:         竞品系统实例（任意类型）。
            predict_fn:     调用签名 (system, question: str, paths: List[str]) -> str
                            若 is_async=True，则为 async callable。
            is_async:       predict_fn 是否为协程函数。
            max_concurrent: 最大并发请求数。
            request_delay:  每次请求间延迟（秒）。
            tokens_fn:      可选，提取 token 数量的函数
                            签名: (system, question, result) -> int
            metadata:       写入 BaselineResult.metadata 的系统元数据。
        """
        self._name = name
        self._citation = citation_name
        self._system = system
        self._predict_fn = predict_fn
        self._is_async = is_async
        self._max_concurrent = max_concurrent
        self._delay = request_delay
        self._tokens_fn = tokens_fn
        self._metadata = metadata or {}

    @property
    def name(self) -> str:
        return self._name

    @property
    def citation_name(self) -> str:
        return self._citation

    async def predict(self, question: str, context_paths: List[str]) -> BaselinePrediction:
        t0 = time.monotonic()
        try:
            if self._is_async:
                answer = await self._predict_fn(self._system, question, context_paths)
            else:
                loop = asyncio.get_event_loop()
                answer = await loop.run_in_executor(
                    None,
                    lambda: self._predict_fn(self._system, question, context_paths)
                )
        except Exception as exc:
            answer = f"[SdkBaseline error: {exc}]"

        elapsed = time.monotonic() - t0

        tokens = 0
        if self._tokens_fn:
            try:
                tokens = int(self._tokens_fn(self._system, question, answer))
            except Exception:
                pass

        return BaselinePrediction(
            answer=str(answer),
            elapsed=elapsed,
            tokens_used=tokens,
            metadata=dict(self._metadata),
        )

    def get_max_concurrent(self) -> int:
        return self._max_concurrent

    def get_request_delay(self) -> float:
        return self._delay

    def extra_metadata(self) -> Dict[str, Any]:
        return dict(self._metadata)


class ManualImportAdapter(BaselineAdapter):
    """从预计算结果 JSONL 导入竞品预测，用我们的 Judge 重新评分。

    使用场景：竞品系统无 API / SDK，但已产出一份 JSONL 文件，每行含
    {"sample_id": "...", "prediction": "...", "elapsed": 3.2}

    这样可保证所有系统使用同一个 Judge 评分（论文公平性要求）。

    JSONL 格式（每行一条）::

        {"sample_id": "hotpotqa_id_001", "prediction": "Paris", "elapsed": 5.2}

    注意：sample_id 必须与 GoldenSet 中的 sample_id 匹配，否则该样本被跳过。
    """

    def __init__(
        self,
        name: str,
        citation_name: str,
        predictions_path: str,
        default_elapsed: float = 0.0,
    ) -> None:
        """
        Args:
            name:              系统内部 ID。
            citation_name:     论文表格展示名称。
            predictions_path:  JSONL 文件路径，每行含 sample_id + prediction。
            default_elapsed:   若 JSONL 中无 elapsed 字段，使用此默认值。
        """
        self._name = name
        self._citation = citation_name
        self._default_elapsed = default_elapsed
        self._predictions: Dict[str, Dict[str, Any]] = {}
        self._load(predictions_path)

    def _load(self, path: str) -> None:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(
                f"ManualImportAdapter: predictions file not found: {path}"
            )
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                    sid = (
                        row.get("sample_id")
                        or row.get("hotpot_id")
                        or row.get("id")
                        or ""
                    )
                    if sid:
                        self._predictions[sid] = row
                except (json.JSONDecodeError, KeyError):
                    pass

    @property
    def name(self) -> str:
        return self._name

    @property
    def citation_name(self) -> str:
        return self._citation

    async def predict(self, question: str, context_paths: List[str]) -> BaselinePrediction:
        # 通过 question 无法反查，此方法不使用
        # 实际评估由 BaselineEvaluationSuite 调用 predict_by_id()
        return BaselinePrediction(
            answer="",
            elapsed=self._default_elapsed,
            metadata={"import_adapter": True},
        )

    def predict_by_id(self, sample_id: str) -> Optional[BaselinePrediction]:
        """通过 sample_id 查取预导入的预测结果。

        Returns:
            BaselinePrediction，若 sample_id 不存在则返回 None。
        """
        row = self._predictions.get(sample_id)
        if row is None:
            return None
        return BaselinePrediction(
            answer=str(row.get("prediction") or row.get("raw_prediction") or ""),
            elapsed=float(row.get("elapsed", self._default_elapsed)),
            tokens_used=int(row.get("tokens_used", 0)),
            metadata={"imported_from": "jsonl"},
        )

    @property
    def loaded_count(self) -> int:
        """已成功加载的预测条数。"""
        return len(self._predictions)

    def get_request_delay(self) -> float:
        return 0.0   # 纯内存查询，无需延迟
