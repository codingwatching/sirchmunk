"""Hybrid-RAG baseline: BM25 + dense retrieval fusion + shared LLM synthesis.

This is the P1 paper-facing RAG baseline.  It stays intentionally small: fixed
chunks, BM25 scoring, lightweight dense search, reciprocal-rank fusion, and the
same benchmark LLM used by other generated-answer baselines.
"""
from __future__ import annotations

import os
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .base_adapter import BaselineAdapter, BaselinePrediction, BaselineSetupResult
from .bm25_rag import (
    _bm25_score,
    _chunk_text,
    _collect_context_paths,
    _count_files,
    _extract_text,
    _iter_files,
    _tokenize,
)
from .dense_index import LightweightDenseIndex

_HYBRID_RAG_PROMPT = """You are answering a question using hybrid retrieved document evidence.

Question:
{question}

Retrieved evidence chunks:
{evidence}

Instructions:
- Answer only using the retrieved evidence.
- Return a concise final answer span when possible.
- If evidence is insufficient, provide the most plausible concise answer supported by the retrieved evidence instead of refusing; answer with your best supported candidate.
"""


class HybridRAGBaseline(BaselineAdapter):
    """Fixed-chunk BM25 + dense fusion RAG baseline."""

    def __init__(
        self,
        *,
        chunk_words: int = 220,
        chunk_overlap: int = 40,
        bm25_top_k: int = 20,
        dense_top_k: int = 20,
        final_top_k: int = 5,
        rrf_k: int = 60,
        max_files: int = 5000,
        max_file_chars: int = 300_000,
        max_chunks: int = 80_000,
        dense_backend: str = "hash",
        dense_model_id: str = "",
        dense_dim: int = 256,
        llm: Optional[Any] = None,
        name: str = "hybrid_rag",
        citation_name: str = "Hybrid-RAG",
    ) -> None:
        self._chunk_words = int(chunk_words or 220)
        self._chunk_overlap = int(chunk_overlap or 40)
        self._bm25_top_k = int(bm25_top_k or 20)
        self._dense_top_k = int(dense_top_k or 20)
        self._final_top_k = int(final_top_k or 5)
        self._rrf_k = int(rrf_k or 60)
        self._max_files = int(max_files or 5000)
        self._max_file_chars = int(max_file_chars or 300_000)
        self._max_chunks = int(max_chunks or 80_000)
        self._dense_backend = (dense_backend or os.environ.get("HYBRID_RAG_DENSE_BACKEND") or "hash").strip().lower()
        self._dense_model_id = dense_model_id or os.environ.get("EMBEDDING_MODEL_ID", "")
        self._dense_dim = int(dense_dim or 256)
        self._llm = llm
        self._name = name
        self._citation = citation_name
        self._chunks: List[Dict[str, Any]] = []
        self._df: Counter[str] = Counter()
        self._avgdl = 0.0
        self._dense_index = LightweightDenseIndex(
            backend=self._dense_backend,
            model_id=self._dense_model_id,
            dim=self._dense_dim,
        )
        self._setup = BaselineSetupResult()

    @property
    def name(self) -> str:
        return self._name

    @property
    def citation_name(self) -> str:
        return self._citation

    async def prepare(self, golden_set: Any = None, bm_adapter: Any = None) -> BaselineSetupResult:
        start = time.monotonic()
        if self._llm is None and bm_adapter is not None:
            try:
                self._llm = bm_adapter.build_searcher().llm
            except Exception:
                self._llm = None

        self._chunks = []
        self._df = Counter()
        self._avgdl = 0.0
        self._dense_index = LightweightDenseIndex(
            backend=self._dense_backend,
            model_id=self._dense_model_id,
            dim=self._dense_dim,
        )

        paths = _collect_context_paths(golden_set, bm_adapter)
        expected_docs = _count_files(paths, limit=self._max_files + 1) if paths else 0
        chunks: List[Dict[str, Any]] = []
        bytes_read = 0
        indexed_docs = 0

        for file_path in _iter_files(paths, self._max_files):
            if len(chunks) >= self._max_chunks:
                break
            text = await _extract_text(file_path, self._max_file_chars)
            if not text.strip():
                continue
            indexed_docs += 1
            try:
                bytes_read += min(file_path.stat().st_size, self._max_file_chars)
            except OSError:
                pass
            for chunk_text in _chunk_text(text, self._chunk_words, self._chunk_overlap):
                tokens = _tokenize(chunk_text)
                if not tokens:
                    continue
                chunk = {
                    "chunk_id": len(chunks),
                    "path": str(file_path),
                    "text": chunk_text,
                    "tokens": tokens,
                    "tf": Counter(tokens),
                    "length": len(tokens),
                }
                chunks.append(chunk)
                for term in set(tokens):
                    self._df[term] += 1
                if len(chunks) >= self._max_chunks:
                    break

        self._chunks = chunks
        self._avgdl = sum(c["length"] for c in chunks) / len(chunks) if chunks else 0.0
        dense_start = time.monotonic()
        await self._dense_index.add(chunks)
        self._sync_dense_runtime_config()
        dense_seconds = time.monotonic() - dense_start
        elapsed = time.monotonic() - start
        partial_index = expected_docs > indexed_docs > 0
        storage_bytes = bytes_read + _estimate_dense_storage_bytes(self._dense_index.indexed_documents, self._dense_dim)
        self._setup = BaselineSetupResult(
            setup_seconds=elapsed,
            preprocessing_seconds=elapsed,
            index_build_seconds=elapsed,
            storage_bytes=storage_bytes,
            indexed_documents=indexed_docs,
            expected_documents=expected_docs,
            build_completed=True,
            index_ready=bool(chunks) and not partial_index,
            index_required=True,
            rebuild_required=True,
            query_ready_immediately=False,
            partial_index=partial_index,
            metadata={
                **self.extra_metadata(),
                "chunk_count": len(chunks),
                "dense_index_seconds": dense_seconds,
                "dense_indexed_items": self._dense_index.indexed_documents,
                "partial_index": partial_index,
            },
        )
        return self._setup

    async def predict(self, question: str, context_paths: List[str]) -> BaselinePrediction:
        start = time.monotonic()
        if not self._chunks:
            await self.prepare(golden_set=None, bm_adapter=_SinglePathAdapter(context_paths))

        bm25_ranked = self._rank_bm25(question)[: self._bm25_top_k]
        dense_ranked = await self._rank_dense(question)
        fused = self._fuse(bm25_ranked, dense_ranked)[: self._final_top_k]
        evidence = _format_hybrid_evidence(fused)

        if not evidence:
            answer = "No hybrid RAG evidence found."
            tokens = 0
        elif self._llm is None:
            answer = (
                "Hybrid-RAG evidence excerpt fallback: "
                + fused[0][1]["text"][:1200]
            )
            tokens = 0
        else:
            prompt = _HYBRID_RAG_PROMPT.format(question=question, evidence=evidence)
            resp = await self._llm.achat(messages=[{"role": "user", "content": prompt}], stream=False)
            answer = resp.content or ""
            usage = getattr(resp, "usage", {}) or {}
            tokens = int(usage.get("total_tokens", 0) or 0) if isinstance(usage, dict) else 0

        elapsed = time.monotonic() - start
        top_paths = [chunk["path"] for _, chunk, _ in fused]
        return BaselinePrediction(
            answer=answer,
            elapsed=elapsed,
            tokens_used=tokens,
            metadata={
                **self.extra_metadata(),
                "top_chunks": [
                    {"path": chunk["path"], "score": round(score, 6), "source": source}
                    for score, chunk, source in fused
                ],
                "read_file_ids": top_paths,
                "evidence_sources": top_paths,
                "evidence_snippets": [chunk["text"][:1800] for _, chunk, _ in fused],
                "setup_metrics": self.collect_setup_metrics(),
            },
        )

    def _rank_bm25(self, question: str) -> List[Tuple[float, Dict[str, Any]]]:
        query_terms = _tokenize(question)
        ranked = [
            (_bm25_score(query_terms, chunk, self._df, len(self._chunks), self._avgdl), chunk)
            for chunk in self._chunks
        ]
        return sorted(ranked, key=lambda x: x[0], reverse=True)

    async def _rank_dense(self, question: str) -> List[Tuple[float, Dict[str, Any]]]:
        results = await self._dense_index.search(question, top_k=self._dense_top_k)
        return [(result.score, result.item) for result in results]

    def _fuse(
        self,
        bm25_ranked: List[Tuple[float, Dict[str, Any]]],
        dense_ranked: List[Tuple[float, Dict[str, Any]]],
    ) -> List[Tuple[float, Dict[str, Any], str]]:
        scores: Dict[int, float] = {}
        chunks: Dict[int, Dict[str, Any]] = {}
        sources: Dict[int, List[str]] = {}
        for source, ranked in (("bm25", bm25_ranked), ("dense", dense_ranked)):
            for rank, (_, chunk) in enumerate(ranked, 1):
                chunk_id = int(chunk.get("chunk_id", id(chunk)))
                scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (self._rrf_k + rank)
                chunks[chunk_id] = chunk
                sources.setdefault(chunk_id, []).append(source)
        fused = [
            (score, chunks[chunk_id], "+".join(sorted(set(sources.get(chunk_id, [])))))
            for chunk_id, score in scores.items()
        ]
        return sorted(fused, key=lambda x: x[0], reverse=True)

    def _sync_dense_runtime_config(self) -> None:
        """Persist actual dense backend after possible embedding fallback for cache identity."""
        self._dense_backend = self._dense_index.backend
        self._dense_model_id = self._dense_index.model_id
        self._dense_dim = self._dense_index.dim

    def collect_setup_metrics(self) -> Dict[str, Any]:
        return {
            "setup_seconds": self._setup.setup_seconds,
            "preprocessing_seconds": self._setup.preprocessing_seconds,
            "index_build_seconds": self._setup.index_build_seconds,
            "storage_bytes": self._setup.storage_bytes,
            "indexed_documents": self._setup.indexed_documents,
            "expected_documents": self._setup.expected_documents,
            "index_ready": self._setup.index_ready,
            "index_required": self._setup.index_required,
            "rebuild_required": self._setup.rebuild_required,
            "query_ready_immediately": self._setup.query_ready_immediately,
            "partial_index": self._setup.partial_index,
            "metadata": self._setup.metadata,
        }

    def is_index_ready(self) -> bool:
        return bool(self._chunks) and not self._setup.partial_index

    def is_index_required(self) -> bool:
        return True

    def is_query_ready_immediately(self) -> bool:
        return False

    async def update_index(self, mutation: Any, bm_adapter: Any = None) -> Dict[str, Any]:
        return {
            "update_supported": False,
            "rebuild_required": True,
            "query_ready_immediately": False,
            "failure_reason": "full_rebuild_required",
            "baseline_type": "hybrid_rag",
        }

    def estimate_update_cost(self, mutation: Any) -> Dict[str, Any]:
        return {"rebuild_required": True, "query_ready_immediately": False}

    def validate_index(self, corpus_manifest: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return {
            "index_ready": self.is_index_ready(),
            "indexed_documents": self._setup.indexed_documents,
            "expected_documents": self._setup.expected_documents,
            "validation_errors": ["partial_index"] if self._setup.partial_index else [],
            "partial_index": self._setup.partial_index,
            **self._dense_index.metadata(),
        }

    def extra_metadata(self) -> Dict[str, Any]:
        return {
            "baseline_type": "hybrid_rag",
            "fusion_method": "rrf",
            "rrf_k": self._rrf_k,
            "bm25_top_k": self._bm25_top_k,
            "dense_top_k": self._dense_top_k,
            "final_top_k": self._final_top_k,
            "chunk_words": self._chunk_words,
            "chunk_overlap": self._chunk_overlap,
            "max_files": self._max_files,
            "max_chunks": self._max_chunks,
            "index_required": True,
            "rebuild_required": True,
            "query_ready_immediately": False,
            **self._dense_index.metadata(),
        }


def _format_hybrid_evidence(ranked: List[Tuple[float, Dict[str, Any], str]]) -> str:
    blocks = []
    for i, (score, chunk, source) in enumerate(ranked, 1):
        if score <= 0:
            continue
        blocks.append(
            f"[Chunk {i} | fused_score={score:.6f} | source={source} | file={Path(chunk['path']).name}]\n"
            f"{chunk['text'][:1800]}"
        )
    return "\n\n---\n\n".join(blocks)


def _estimate_dense_storage_bytes(items: int, dim: int) -> int:
    return max(int(items or 0), 0) * max(int(dim or 0), 0) * 8


class _SinglePathAdapter:
    def __init__(self, paths: List[str]) -> None:
        self._paths = paths

    def _corpus_paths(self) -> List[str]:
        return self._paths
