"""Lightweight dense index utilities for benchmark baselines.

The default backend is deterministic hashed vectors so smoke/lifecycle checks do
not require downloading embedding models.  When ``backend='sirchmunk_embedding'``
is selected, the index uses Sirchmunk's EmbeddingUtil and records that backend in
baseline metadata.  This keeps P1 reproducible while leaving room for stronger
embedding-backed runs.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence


@dataclass
class DenseSearchResult:
    score: float
    item: Dict[str, Any]


class LightweightDenseIndex:
    """Small in-memory cosine index used by Dense/Hybrid RAG baselines."""

    def __init__(
        self,
        *,
        backend: str = "hash",
        model_id: str = "",
        dim: int = 256,
        batch_size: int = 32,
        embedding_timeout: float = 300.0,
    ) -> None:
        self.backend = (backend or "hash").strip().lower()
        self.model_id = model_id or "hashing-vectorizer"
        self.dim = int(dim or 256)
        self.batch_size = max(int(batch_size or 32), 1)
        self.embedding_timeout = float(embedding_timeout or 300.0)
        self._items: List[Dict[str, Any]] = []
        self._vectors: List[List[float]] = []
        self._embedding_util = None
        self._backend_error = ""

    @property
    def indexed_documents(self) -> int:
        return len(self._items)

    @property
    def backend_error(self) -> str:
        return self._backend_error

    async def add(self, items: Sequence[Dict[str, Any]], *, text_key: str = "text") -> None:
        if not items:
            return
        texts = [str(item.get(text_key, "") or "") for item in items]
        vectors = await self._embed(texts)
        for item, vector in zip(items, vectors):
            if not vector:
                continue
            self._items.append(item)
            self._vectors.append(_normalize(vector))

    async def search(self, query: str, *, top_k: int = 5) -> List[DenseSearchResult]:
        if not self._items or not query:
            return []
        query_vectors = await self._embed([query])
        if not query_vectors:
            return []
        q = _normalize(query_vectors[0])
        scored = [DenseSearchResult(score=_dot(q, vec), item=item) for item, vec in zip(self._items, self._vectors)]
        return sorted(scored, key=lambda r: r.score, reverse=True)[: max(int(top_k or 5), 1)]

    def metadata(self) -> Dict[str, Any]:
        return {
            "dense_backend": self.backend,
            "dense_model_id": self.model_id,
            "dense_dim": self.dim,
            "dense_indexed_items": self.indexed_documents,
            "dense_backend_error": self._backend_error,
        }

    async def _embed(self, texts: Sequence[str]) -> List[List[float]]:
        if self.backend in {"sirchmunk", "sirchmunk_embedding", "embedding_util"}:
            try:
                if self._embedding_util is None:
                    from sirchmunk.utils.embedding_util import EmbeddingUtil
                    self._embedding_util = EmbeddingUtil(model_id=self.model_id if self.model_id != "hashing-vectorizer" else None)
                vectors = await self._embedding_util.embed(list(texts))
                return [[float(x) for x in vector] for vector in vectors]
            except Exception as exc:
                self._backend_error = str(exc)[:300]
                # Deterministic fallback keeps smoke tests runnable and visible.
                self.backend = "hash"
                self.model_id = "hashing-vectorizer-fallback"
        return [_hash_vector(text, self.dim) for text in texts]


def _hash_vector(text: str, dim: int) -> List[float]:
    vec = [0.0] * max(int(dim or 256), 1)
    for token in _tokens(text):
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        bucket = int.from_bytes(digest[:4], "big") % len(vec)
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vec[bucket] += sign
    return _normalize(vec)


def _tokens(text: str) -> Iterable[str]:
    token = []
    for ch in (text or "").lower():
        if ch.isalnum():
            token.append(ch)
        elif token:
            yield "".join(token)
            token = []
    if token:
        yield "".join(token)


def _normalize(vector: Sequence[float]) -> List[float]:
    norm = math.sqrt(sum(float(x) * float(x) for x in vector))
    if norm <= 0:
        return [0.0 for _ in vector]
    return [float(x) / norm for x in vector]


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(float(x) * float(y) for x, y in zip(a, b))
