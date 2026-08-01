"""baselines/bm25_rag.py — BM25-RAG baseline

论文主表 baseline：固定 chunk → BM25 检索 top-k chunks → 同一 LLM 生成答案。

与 LocalBM25Baseline 的区别：
- LocalBM25Baseline 是 quickstart/local smoke baseline，可选轻量 LLM 合成，但不作为论文主表 baseline。
- BM25RAGBaseline 使用固定 chunk 级 BM25 选择 chunks，再调用生成 LLM；适合作为论文主表 sparse RAG baseline。

公平性：
- prepare() 记录 index build / preprocessing / storage 成本。
- predict() 只统计 query-time latency。
- 使用 BenchmarkAdapter 的 search paths，和 LENS 使用同一 GoldenSet 文档。
"""
from __future__ import annotations

import math
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .base_adapter import BaselineAdapter, BaselinePrediction, BaselineSetupResult

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
_TEXT_EXTS = {
    ".txt", ".md", ".rst", ".csv", ".tsv", ".json", ".yaml", ".yml",
    ".xml", ".html", ".htm", ".log",
}

_BM25_RAG_PROMPT = """You are answering a question using retrieved document evidence.

Question:
{question}

Retrieved evidence chunks:
{evidence}

Instructions:
- Answer only using the retrieved evidence.
- If the evidence is insufficient, provide the most plausible concise answer supported by the retrieved evidence instead of refusing; answer with your best supported candidate.
- For numerical answers, preserve units and fiscal years exactly.
- Keep the answer concise.
"""


class BM25RAGBaseline(BaselineAdapter):
    """Fixed-chunk BM25-RAG baseline.

    This is the main sparse retrieval baseline for the LENS paper.
    """

    def __init__(
        self,
        *,
        chunk_words: int = 220,
        chunk_overlap: int = 40,
        top_k_chunks: int = 5,
        max_files: int = 5000,
        max_file_chars: int = 300_000,
        max_chunks: int = 80_000,
        llm: Optional[Any] = None,
        name: str = "bm25_rag",
        citation_name: str = "BM25-RAG",
    ) -> None:
        self._chunk_words = chunk_words
        self._chunk_overlap = chunk_overlap
        self._top_k_chunks = top_k_chunks
        self._max_files = max_files
        self._max_file_chars = max_file_chars
        self._max_chunks = max_chunks
        self._llm = llm
        self._name = name
        self._citation = citation_name
        self._chunks: List[Dict[str, Any]] = []
        self._df: Counter[str] = Counter()
        self._avgdl = 0.0
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

        # A BM25 index is corpus-specific. Always reset before building a new D_n index.
        self._chunks = []
        self._df = Counter()
        self._avgdl = 0.0

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
            for chunk in _chunk_text(text, self._chunk_words, self._chunk_overlap):
                tokens = _tokenize(chunk)
                if not tokens:
                    continue
                tf = Counter(tokens)
                chunks.append({
                    "path": str(file_path),
                    "text": chunk,
                    "tokens": tokens,
                    "tf": tf,
                    "length": len(tokens),
                })
                for term in set(tokens):
                    self._df[term] += 1
                if len(chunks) >= self._max_chunks:
                    break

        self._chunks = chunks
        self._avgdl = sum(c["length"] for c in chunks) / len(chunks) if chunks else 0.0
        elapsed = time.monotonic() - start
        partial_index = expected_docs > indexed_docs > 0
        self._setup = BaselineSetupResult(
            setup_seconds=elapsed,
            preprocessing_seconds=elapsed,
            index_build_seconds=elapsed,
            storage_bytes=bytes_read,
            indexed_documents=indexed_docs,
            expected_documents=expected_docs,
            build_completed=True,
            index_ready=bool(chunks),
            index_required=True,
            rebuild_required=True,
            query_ready_immediately=False,
            partial_index=partial_index,
            metadata={
                "baseline_type": "bm25_rag",
                "chunk_words": self._chunk_words,
                "chunk_overlap": self._chunk_overlap,
                "top_k_chunks": self._top_k_chunks,
                "max_files": self._max_files,
                "max_chunks": self._max_chunks,
                "partial_index": partial_index,
                "rebuild_required": True,
                "index_required": True,
            },
        )
        return self._setup

    async def predict(self, question: str, context_paths: List[str]) -> BaselinePrediction:
        start = time.monotonic()
        if not self._chunks:
            await self.prepare(golden_set=None, bm_adapter=_SinglePathAdapter(context_paths))

        ranked = self._rank_chunks(question)[: self._top_k_chunks]
        evidence = _format_evidence(ranked)

        if not evidence:
            answer = "No BM25 evidence found."
            tokens = 0
        elif self._llm is None:
            # LLM unavailable fallback: return best chunk excerpt.
            answer = ranked[0][1]["text"][:1200]
            tokens = 0
        else:
            prompt = _BM25_RAG_PROMPT.format(question=question, evidence=evidence)
            resp = await self._llm.achat(
                messages=[{"role": "user", "content": prompt}],
                stream=False,
            )
            answer = resp.content or ""
            usage = getattr(resp, "usage", {}) or {}
            tokens = int(usage.get("total_tokens", 0) or 0)

        elapsed = time.monotonic() - start
        top_paths = [chunk["path"] for _, chunk in ranked]
        return BaselinePrediction(
            answer=answer,
            elapsed=elapsed,
            tokens_used=tokens,
            metadata={
                "baseline_type": "bm25_rag",
                "top_chunks": [
                    {"path": chunk["path"], "score": round(score, 4)}
                    for score, chunk in ranked
                ],
                "read_file_ids": top_paths,
                "evidence_sources": top_paths,
                "setup_metrics": self.collect_setup_metrics(),
            },
        )

    def _rank_chunks(self, question: str) -> List[Tuple[float, Dict[str, Any]]]:
        query_terms = _tokenize(question)
        ranked = [
            (_bm25_score(query_terms, chunk, self._df, len(self._chunks), self._avgdl), chunk)
            for chunk in self._chunks
        ]
        return sorted(ranked, key=lambda x: x[0], reverse=True)

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
        return bool(self._chunks)

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
            "baseline_type": "bm25_rag",
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
        }

    def extra_metadata(self) -> Dict[str, Any]:
        return {"baseline_type": "bm25_rag", "index_required": True, "rebuild_required": True}


def _count_files(paths: Iterable[str], *, limit: int = 0) -> int:
    count = 0
    for raw in paths:
        path = Path(raw)
        if path.is_file():
            count += 1
        elif path.is_dir():
            for child in path.rglob("*"):
                if child.is_file() and not child.name.startswith("."):
                    count += 1
                    if limit and count >= limit:
                        return count
        if limit and count >= limit:
            return count
    return count


async def _extract_text(path: Path, max_chars: int) -> str:
    try:
        if path.suffix.lower() in _TEXT_EXTS or not path.suffix:
            return path.read_text(encoding="utf-8", errors="ignore")[:max_chars]
        from sirchmunk.utils.file_utils import fast_extract
        result = await fast_extract(path)
        return (result.content if result and result.content else "")[:max_chars]
    except Exception:
        return ""


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
    if not paths and hasattr(bm_adapter, "_corpus_paths"):
        try:
            paths.extend(bm_adapter._corpus_paths())
        except Exception:
            pass
    return sorted(set(paths))


def _iter_files(paths: Iterable[str], max_files: int) -> Iterable[Path]:
    seen = 0
    for raw in paths:
        path = Path(raw)
        if path.is_file():
            yield path
            seen += 1
        elif path.is_dir():
            for child in path.rglob("*"):
                if child.is_file() and not child.name.startswith("."):
                    yield child
                    seen += 1
                    if seen >= max_files:
                        return
        if seen >= max_files:
            return


def _chunk_text(text: str, chunk_words: int, overlap: int) -> Iterable[str]:
    words = text.split()
    if not words:
        return
    step = max(1, chunk_words - overlap)
    for start in range(0, len(words), step):
        part = words[start:start + chunk_words]
        if part:
            yield " ".join(part)


def _tokenize(text: str) -> List[str]:
    return [m.group(0).lower() for m in _TOKEN_RE.finditer(text or "")]


def _bm25_score(query_terms: List[str], doc: Dict[str, Any], df: Counter[str], n_docs: int, avgdl: float) -> float:
    if not query_terms or not doc or n_docs <= 0 or avgdl <= 0:
        return 0.0
    k1 = 1.5
    b = 0.75
    score = 0.0
    dl = doc["length"]
    for term in query_terms:
        freq = doc["tf"].get(term, 0)
        if freq <= 0:
            continue
        idf = math.log(1 + (n_docs - df.get(term, 0) + 0.5) / (df.get(term, 0) + 0.5))
        denom = freq + k1 * (1 - b + b * dl / avgdl)
        score += idf * freq * (k1 + 1) / denom
    return score


def _format_evidence(ranked: List[Tuple[float, Dict[str, Any]]]) -> str:
    blocks = []
    for i, (score, chunk) in enumerate(ranked, 1):
        if score <= 0:
            continue
        blocks.append(
            f"[Chunk {i} | score={score:.4f} | source={Path(chunk['path']).name}]\n"
            f"{chunk['text'][:1800]}"
        )
    return "\n\n---\n\n".join(blocks)


class _SinglePathAdapter:
    def __init__(self, paths: List[str]) -> None:
        self._paths = paths

    def _corpus_paths(self) -> List[str]:
        return self._paths
