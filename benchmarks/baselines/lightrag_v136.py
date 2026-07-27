"""LightRAG v1.3.6 dedicated lifecycle baseline adapter.

This adapter treats LightRAG as an index-heavy related-work system.  It builds a
stage-local LightRAG working directory from the provided D_n search corpus and
reports preprocessing/index/storage lifecycle metrics separately from query
latency.
"""
from __future__ import annotations

import asyncio
import inspect
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .base_adapter import BaselineAdapter, BaselinePrediction, BaselineSetupResult

_TEXT_EXTS = {
    "", ".txt", ".md", ".rst", ".csv", ".tsv", ".json", ".jsonl",
    ".yaml", ".yml", ".xml", ".html", ".htm", ".log",
}
_VALID_QUERY_MODES = {"naive", "local", "global", "hybrid", "mix"}


class LightRAGV136Baseline(BaselineAdapter):
    """LightRAG v1.3.6 SDK baseline with explicit build/index lifecycle."""

    def __init__(
        self,
        *,
        query_mode: str = "hybrid",
        working_dir: str = "",
        max_files: int = 0,
        max_file_chars: int = 300_000,
        llm_model_name: str = "",
        embedding_model_name: str = "",
        embedding_dim: int = 1536,
        embedding_max_token_size: int = 8192,
        top_k: int = 60,
        name: str = "lightrag_v136",
        citation_name: str = "LightRAG v1.3.6 (hybrid)",
    ) -> None:
        normalized_mode = (query_mode or "hybrid").strip().lower()
        if normalized_mode not in _VALID_QUERY_MODES:
            raise ValueError(f"Unsupported LightRAG query mode: {query_mode}")
        self._query_mode = normalized_mode
        self._working_dir = working_dir
        self._max_files = max(int(max_files or 0), 0)
        self._max_file_chars = max(int(max_file_chars or 0), 0)
        self._llm_model_name = llm_model_name
        self._embedding_model_name = embedding_model_name
        self._embedding_dim = int(embedding_dim or 1536)
        self._embedding_max_token_size = int(embedding_max_token_size or 8192)
        self._top_k = int(top_k or 60)
        self._name = name if normalized_mode == "hybrid" else f"{name}_{normalized_mode}"
        self._citation = citation_name if normalized_mode == "hybrid" else f"LightRAG v1.3.6 ({normalized_mode})"
        self._setup = BaselineSetupResult(build_completed=False, index_ready=False)
        self._rag = None
        self._last_paths: List[str] = []
        self._dependency_error = ""

    @property
    def name(self) -> str:
        return self._name

    @property
    def citation_name(self) -> str:
        return self._citation

    def is_available(self) -> bool:
        ok, error = _check_lightrag_available()
        self._dependency_error = error
        return ok

    async def prepare(self, golden_set: Any = None, bm_adapter: Any = None) -> BaselineSetupResult:
        start = time.monotonic()
        ok, error = _check_lightrag_available()
        if not ok:
            self._setup = BaselineSetupResult(
                setup_seconds=0.0,
                build_completed=False,
                index_ready=False,
                failure_reason="dependency_missing",
                failure_message=error,
                index_required=True,
                rebuild_required=True,
                query_ready_immediately=False,
                metadata=self.extra_metadata(),
            )
            return self._setup

        working_dir = self._resolve_working_dir(bm_adapter)
        working_dir.mkdir(parents=True, exist_ok=True)
        paths = _collect_context_paths(golden_set, bm_adapter)
        self._last_paths = list(paths)
        expected_docs = _count_files(paths, max_files=self._max_files)
        index_start = time.monotonic()
        indexed_docs = 0
        bytes_read = 0
        try:
            self._rag = await self._build_rag(working_dir)
            for file_path in _iter_files(paths, max_files=self._max_files):
                text = _read_text(file_path, self._max_file_chars)
                if not text.strip():
                    continue
                await _maybe_await(self._rag.insert(text))
                indexed_docs += 1
                try:
                    bytes_read += file_path.stat().st_size
                except OSError:
                    pass
            index_seconds = time.monotonic() - index_start
            storage_bytes = _dir_size(working_dir)
            partial = expected_docs > indexed_docs > 0
            ready = indexed_docs > 0 and not partial
            self._setup = BaselineSetupResult(
                setup_seconds=time.monotonic() - start,
                preprocessing_seconds=index_seconds,
                index_build_seconds=index_seconds,
                storage_bytes=storage_bytes,
                indexed_documents=indexed_docs,
                expected_documents=expected_docs,
                build_completed=True,
                index_ready=ready,
                index_required=True,
                rebuild_required=True,
                query_ready_immediately=False,
                partial_index=partial,
                artifact_dir=str(working_dir),
                metadata={
                    **self.extra_metadata(),
                    "working_dir": str(working_dir),
                    "input_paths": paths,
                    "bytes_read": bytes_read,
                },
            )
            return self._setup
        except Exception as exc:
            self._setup = BaselineSetupResult(
                setup_seconds=time.monotonic() - start,
                preprocessing_seconds=time.monotonic() - index_start,
                index_build_seconds=time.monotonic() - index_start,
                storage_bytes=_dir_size(working_dir),
                indexed_documents=indexed_docs,
                expected_documents=expected_docs,
                build_completed=False,
                index_ready=False,
                failure_reason=self.classify_failure(exc),
                failure_message=str(exc),
                index_required=True,
                rebuild_required=True,
                query_ready_immediately=False,
                artifact_dir=str(working_dir),
                metadata={**self.extra_metadata(), "working_dir": str(working_dir)},
            )
            raise

    async def predict(self, question: str, context_paths: List[str]) -> BaselinePrediction:
        start = time.monotonic()
        if self._rag is None:
            return BaselinePrediction(
                answer="LightRAG v1.3.6 baseline is not prepared.",
                elapsed=time.monotonic() - start,
                metadata={"failure_reason": "index_not_ready", **self.extra_metadata()},
            )
        try:
            query_param = self._query_param()
            answer = await _maybe_await(self._rag.query(question, param=query_param))
            return BaselinePrediction(
                answer=str(answer or ""),
                elapsed=time.monotonic() - start,
                metadata={
                    **self.extra_metadata(),
                    "setup_metrics": self.collect_setup_metrics(),
                    "query_mode": self._query_mode,
                },
            )
        except Exception as exc:
            return BaselinePrediction(
                answer=f"[LightRAG v1.3.6 error: {exc}]",
                elapsed=time.monotonic() - start,
                metadata={"failure_reason": self.classify_failure(exc), "error": str(exc), **self.extra_metadata()},
            )

    async def cleanup(self) -> None:
        if self._rag is not None and hasattr(self._rag, "finalize_storages"):
            await _maybe_await(self._rag.finalize_storages())

    async def update_index(self, mutation: Any, bm_adapter: Any = None) -> Dict[str, Any]:
        return {
            "update_supported": False,
            "rebuild_required": True,
            "query_ready_immediately": False,
            "failure_reason": "full_rebuild_required",
            "baseline_type": "lightrag_v136",
            "query_mode": self._query_mode,
        }

    def estimate_update_cost(self, mutation: Any) -> Dict[str, Any]:
        return {"rebuild_required": True, "query_ready_immediately": False}

    def collect_setup_metrics(self) -> Dict[str, Any]:
        return {
            "setup_seconds": self._setup.setup_seconds,
            "preprocessing_seconds": self._setup.preprocessing_seconds,
            "index_build_seconds": self._setup.index_build_seconds,
            "storage_bytes": self._setup.storage_bytes,
            "indexed_documents": self._setup.indexed_documents,
            "expected_documents": self._setup.expected_documents,
            "build_completed": self._setup.build_completed,
            "index_ready": self._setup.index_ready,
            "failure_reason": self._setup.failure_reason,
            "failure_message": self._setup.failure_message,
            "artifact_dir": self._setup.artifact_dir,
            "index_required": self._setup.index_required,
            "rebuild_required": self._setup.rebuild_required,
            "query_ready_immediately": self._setup.query_ready_immediately,
            "partial_index": self._setup.partial_index,
            "metadata": self._setup.metadata,
        }

    def validate_index(self, corpus_manifest: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return {
            "index_ready": self._setup.index_ready,
            "indexed_documents": self._setup.indexed_documents,
            "expected_documents": self._setup.expected_documents,
            "validation_errors": ["partial_index"] if self._setup.partial_index else [],
            "working_dir": self._setup.artifact_dir,
            "query_mode": self._query_mode,
        }

    def is_index_ready(self) -> bool:
        return bool(self._setup.index_ready)

    def is_index_required(self) -> bool:
        return True

    def is_query_ready_immediately(self) -> bool:
        return False

    def extra_metadata(self) -> Dict[str, Any]:
        return {
            "baseline_type": "lightrag_v136",
            "lightrag_version_ref": "v1.3.6",
            "query_mode": self._query_mode,
            "index_required": True,
            "rebuild_required": True,
            "query_ready_immediately": False,
            "llm_model_name": self._llm_model_name or os.environ.get("LLM_MODEL_NAME", ""),
            "embedding_model_name": self._embedding_model_name or os.environ.get("EMBEDDING_MODEL_ID", ""),
            "working_dir": self._working_dir,
        }

    def _resolve_working_dir(self, bm_adapter: Any = None) -> Path:
        if self._working_dir:
            return Path(self._working_dir).expanduser().resolve()
        base = ""
        if bm_adapter is not None:
            try:
                base = bm_adapter.get_work_path()
            except Exception:
                base = ""
        if not base:
            base = os.environ.get("SIRCHMUNK_WORK_PATH", ".work")
        return Path(base).expanduser().resolve() / "baselines" / self._name

    async def _build_rag(self, working_dir: Path):
        LightRAG, _, initialize_pipeline_status = _import_lightrag_core()
        llm_func = _build_lightrag_llm_func(self._llm_model_name)
        embedding_func = _build_lightrag_embedding_func(
            self._embedding_model_name,
            embedding_dim=self._embedding_dim,
            max_token_size=self._embedding_max_token_size,
        )
        rag = LightRAG(
            working_dir=str(working_dir),
            llm_model_func=llm_func,
            embedding_func=embedding_func,
        )
        if hasattr(rag, "initialize_storages"):
            await _maybe_await(rag.initialize_storages())
        if initialize_pipeline_status is not None:
            await _maybe_await(initialize_pipeline_status())
        return rag

    def _query_param(self):
        _, QueryParam, _ = _import_lightrag_core()
        return QueryParam(mode=self._query_mode, top_k=self._top_k)


def _check_lightrag_available() -> tuple[bool, str]:
    try:
        _import_lightrag_core()
        _import_lightrag_openai()
        return True, ""
    except Exception as exc:
        return False, str(exc)


def _import_lightrag_core():
    from lightrag import LightRAG, QueryParam
    try:
        from lightrag.kg.shared_storage import initialize_pipeline_status
    except Exception:
        initialize_pipeline_status = None
    return LightRAG, QueryParam, initialize_pipeline_status


def _import_lightrag_openai():
    from lightrag.llm.openai import openai_complete_if_cache, openai_embed
    return openai_complete_if_cache, openai_embed


def _build_lightrag_llm_func(model_name: str = ""):
    openai_complete_if_cache, _ = _import_lightrag_openai()
    model = model_name or os.environ.get("LLM_MODEL_NAME") or "gpt-4o-mini"
    api_key = os.environ.get("LLM_API_KEY", "")
    base_url = os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1")

    async def llm_model_func(prompt, system_prompt=None, history_messages=None, keyword_extraction=False, **kwargs):
        return await openai_complete_if_cache(
            model,
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages or [],
            api_key=api_key,
            base_url=base_url,
            **kwargs,
        )

    return llm_model_func


def _build_lightrag_embedding_func(model_name: str = "", *, embedding_dim: int, max_token_size: int):
    _, openai_embed = _import_lightrag_openai()
    model = model_name or os.environ.get("EMBEDDING_MODEL_ID") or "text-embedding-3-small"
    api_key = os.environ.get("LLM_API_KEY", "")
    base_url = os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1")
    try:
        from lightrag.utils import EmbeddingFunc
    except Exception:
        EmbeddingFunc = None

    async def embed_func(texts: list[str]):
        return await openai_embed(texts, model=model, api_key=api_key, base_url=base_url)

    if EmbeddingFunc is None:
        return embed_func
    return EmbeddingFunc(
        embedding_dim=embedding_dim,
        max_token_size=max_token_size,
        func=embed_func,
    )


def _collect_context_paths(golden_set: Any, bm_adapter: Any) -> List[str]:
    paths: List[str] = []
    if bm_adapter is None:
        return paths
    for sample_dict in getattr(golden_set, "samples", []) or []:
        try:
            from framework.schema import BenchmarkSample
            sample = BenchmarkSample(
                sample_id=sample_dict["sample_id"],
                question=sample_dict["question"],
                gold_answer=sample_dict["gold_answer"],
                metadata=sample_dict.get("metadata", {}),
            )
            paths.extend(bm_adapter.get_search_paths(sample))
        except Exception:
            continue
    return sorted(set(paths))


def _iter_files(paths: Iterable[str], *, max_files: int = 0) -> Iterable[Path]:
    seen = 0
    for raw in paths:
        path = Path(raw)
        if path.is_file() and _is_text_file(path):
            yield path
            seen += 1
        elif path.is_dir():
            for child in path.rglob("*"):
                if child.is_file() and not child.name.startswith(".") and _is_text_file(child):
                    yield child
                    seen += 1
                    if max_files and seen >= max_files:
                        return
        if max_files and seen >= max_files:
            return


def _count_files(paths: Iterable[str], *, max_files: int = 0) -> int:
    return sum(1 for _ in _iter_files(paths, max_files=max_files))


def _is_text_file(path: Path) -> bool:
    return path.suffix.lower() in _TEXT_EXTS


def _read_text(path: Path, max_chars: int) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
        return text[:max_chars] if max_chars else text
    except OSError:
        return ""


def _dir_size(path: Path) -> int:
    total = 0
    if not path.exists():
        return 0
    for child in path.rglob("*"):
        if child.is_file():
            try:
                total += child.stat().st_size
            except OSError:
                pass
    return total


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


__all__ = ["LightRAGV136Baseline"]
