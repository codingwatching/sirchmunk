"""Indexing SDK baseline wrapper for index-heavy RAG competitors.

Use this adapter for systems such as LightRAG, GraphRAG, or RAPTOR where
preprocessing/index construction is a first-class lifecycle phase rather than
part of per-query latency.
"""
from __future__ import annotations

import asyncio
import time
from typing import Any, Callable, Dict, List, Optional

from .base_adapter import BaselineAdapter, BaselinePrediction, BaselineSetupResult


class IndexingSdkBaseline(BaselineAdapter):
    """Generic wrapper for baselines with explicit build/index lifecycle.

    Callback signatures are intentionally simple and duck-typed:

    - ``prepare_fn(system, golden_set, bm_adapter) -> dict | BaselineSetupResult``
    - ``validate_fn(system, corpus_manifest) -> dict``
    - ``predict_fn(system, question, context_paths) -> str | BaselinePrediction``
    - ``cleanup_fn(system) -> None``
    """

    def __init__(
        self,
        name: str,
        citation_name: str,
        system: Any,
        predict_fn: Callable,
        *,
        prepare_fn: Optional[Callable] = None,
        validate_fn: Optional[Callable] = None,
        cleanup_fn: Optional[Callable] = None,
        is_async_predict: bool = False,
        is_async_prepare: bool = False,
        max_concurrent: int = 1,
        request_delay: float = 1.0,
        tokens_fn: Optional[Callable] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._name = name
        self._citation = citation_name
        self._system = system
        self._predict_fn = predict_fn
        self._prepare_fn = prepare_fn
        self._validate_fn = validate_fn
        self._cleanup_fn = cleanup_fn
        self._is_async_predict = is_async_predict
        self._is_async_prepare = is_async_prepare
        self._max_concurrent = max_concurrent
        self._request_delay = request_delay
        self._tokens_fn = tokens_fn
        self._metadata = metadata or {}
        self._setup_result = BaselineSetupResult(build_completed=False, index_ready=False)
        self._index_ready = False
        self._lifecycle_metadata: Dict[str, Any] = {}

    @property
    def name(self) -> str:
        return self._name

    @property
    def citation_name(self) -> str:
        return self._citation

    async def prepare(self, golden_set: Any = None, bm_adapter: Any = None) -> BaselineSetupResult:
        t0 = time.monotonic()
        if self._prepare_fn is None:
            self._setup_result = BaselineSetupResult(
                setup_seconds=0.0,
                build_completed=True,
                index_ready=True,
                metadata={"prepare_fn_missing": True},
            )
            self._index_ready = True
            return self._setup_result

        try:
            if self._is_async_prepare:
                raw = await self._prepare_fn(self._system, golden_set, bm_adapter)
            else:
                loop = asyncio.get_event_loop()
                raw = await loop.run_in_executor(
                    None,
                    lambda: self._prepare_fn(self._system, golden_set, bm_adapter),
                )
            setup = _coerce_setup_result(raw)
            if setup.setup_seconds <= 0:
                setup.setup_seconds = time.monotonic() - t0
            setup.build_completed = bool(setup.build_completed)
            setup.index_ready = bool(setup.index_ready)
            self._setup_result = setup
            self._index_ready = setup.index_ready
            self._lifecycle_metadata.update(setup.metadata)
            return setup
        except Exception as exc:
            self._setup_result = BaselineSetupResult(
                setup_seconds=time.monotonic() - t0,
                build_completed=False,
                index_ready=False,
                failure_reason=self.classify_failure(exc),
                failure_message=str(exc),
                metadata=dict(self._metadata),
            )
            self._index_ready = False
            raise

    async def predict(self, question: str, context_paths: List[str]) -> BaselinePrediction:
        t0 = time.monotonic()
        try:
            if self._is_async_predict:
                raw = await self._predict_fn(self._system, question, context_paths)
            else:
                loop = asyncio.get_event_loop()
                raw = await loop.run_in_executor(
                    None,
                    lambda: self._predict_fn(self._system, question, context_paths),
                )
            pred = _coerce_prediction(raw, elapsed=time.monotonic() - t0)
        except Exception as exc:
            pred = BaselinePrediction(
                answer=f"[IndexingSdkBaseline error: {exc}]",
                elapsed=time.monotonic() - t0,
                metadata={"error": str(exc), "failure_reason": self.classify_failure(exc)},
            )

        if self._tokens_fn and pred.tokens_used <= 0:
            try:
                pred.tokens_used = int(self._tokens_fn(self._system, question, pred.answer))
            except Exception:
                pass
        pred.metadata.update(self.extra_metadata())
        pred.metadata["setup_metrics"] = self.collect_setup_metrics()
        return pred

    async def cleanup(self) -> None:
        if self._cleanup_fn is None:
            return None
        result = self._cleanup_fn(self._system)
        if asyncio.iscoroutine(result):
            await result
        return None

    def collect_setup_metrics(self) -> Dict[str, Any]:
        return {
            "setup_seconds": self._setup_result.setup_seconds,
            "preprocessing_seconds": self._setup_result.preprocessing_seconds,
            "index_build_seconds": self._setup_result.index_build_seconds,
            "storage_bytes": self._setup_result.storage_bytes,
            "indexed_documents": self._setup_result.indexed_documents,
            "expected_documents": self._setup_result.expected_documents,
            "build_completed": self._setup_result.build_completed,
            "index_ready": self._setup_result.index_ready,
            "failure_reason": self._setup_result.failure_reason,
            "failure_message": self._setup_result.failure_message,
            "peak_ram_bytes": self._setup_result.peak_ram_bytes,
            "preprocess_llm_tokens": self._setup_result.preprocess_llm_tokens,
            "api_cost_usd": self._setup_result.api_cost_usd,
            "artifact_dir": self._setup_result.artifact_dir,
            "metadata": self._setup_result.metadata,
        }

    def is_index_ready(self) -> bool:
        return bool(self._index_ready)

    def validate_index(self, corpus_manifest: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if self._validate_fn is None:
            return {
                "index_ready": self._index_ready,
                "indexed_documents": self._setup_result.indexed_documents,
                "expected_documents": self._setup_result.expected_documents,
            }
        result = self._validate_fn(self._system, corpus_manifest or {})
        if not isinstance(result, dict):
            return {"index_ready": bool(result)}
        if "index_ready" in result:
            self._index_ready = bool(result.get("index_ready"))
        return result

    def get_lifecycle_metadata(self) -> Dict[str, Any]:
        return {**self._metadata, **self._lifecycle_metadata}

    def get_max_concurrent(self) -> int:
        return self._max_concurrent

    def get_request_delay(self) -> float:
        return self._request_delay

    def extra_metadata(self) -> Dict[str, Any]:
        return dict(self._metadata)


def _coerce_setup_result(raw: Any) -> BaselineSetupResult:
    if isinstance(raw, BaselineSetupResult):
        return raw
    if isinstance(raw, dict):
        allowed = set(BaselineSetupResult.__dataclass_fields__)
        payload = {k: v for k, v in raw.items() if k in allowed}
        setup = BaselineSetupResult(**payload)
        metadata = {k: v for k, v in raw.items() if k not in allowed}
        setup.metadata.update(metadata)
        return setup
    return BaselineSetupResult(build_completed=bool(raw), index_ready=bool(raw))


def _coerce_prediction(raw: Any, *, elapsed: float) -> BaselinePrediction:
    if isinstance(raw, BaselinePrediction):
        if raw.elapsed <= 0:
            raw.elapsed = elapsed
        return raw
    if isinstance(raw, dict):
        answer = str(raw.get("answer") or raw.get("prediction") or raw.get("response") or "")
        return BaselinePrediction(
            answer=answer,
            elapsed=float(raw.get("elapsed", elapsed) or elapsed),
            tokens_used=int(raw.get("tokens_used", 0) or 0),
            metadata={k: v for k, v in raw.items() if k not in {"answer", "prediction", "response", "elapsed", "tokens_used"}},
        )
    return BaselinePrediction(answer=str(raw), elapsed=elapsed)


__all__ = ["IndexingSdkBaseline"]
