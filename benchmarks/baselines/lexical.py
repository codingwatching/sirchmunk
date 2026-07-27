"""Local retrieval baselines for P1 fair comparison.

These baselines are dependency-light and deterministic. They account for
preparation/indexing time separately from per-query latency so setup cost can be
reported fairly against Sirchmunk's direct-to-data path.
"""
from __future__ import annotations

import math
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .base_adapter import BaselineAdapter, BaselinePrediction, BaselineSetupResult

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


class LocalBM25Baseline(BaselineAdapter):
    def __init__(
        self,
        *,
        max_files: int = 20000,
        max_file_bytes: int = 256_000,
        top_k: int = 5,
        llm: Optional[Any] = None,
        use_llm_synthesis: bool = True,
        name: str = "bm25_local",
        citation_name: str = "BM25 + LLM (local)",
    ) -> None:
        self._max_files = max_files
        self._max_file_bytes = max_file_bytes
        self._top_k = top_k
        self._llm = llm
        self._use_llm_synthesis = use_llm_synthesis
        self._name = name
        self._citation = citation_name
        self._docs: List[Dict[str, Any]] = []
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
        paths = _collect_context_paths(golden_set, bm_adapter)
        docs = []
        corpus_bytes = 0
        for file_path in _iter_files(paths, self._max_files):
            try:
                raw = file_path.read_text(encoding="utf-8", errors="ignore")[: self._max_file_bytes]
            except OSError:
                continue
            tokens = _tokenize(raw)
            if not tokens:
                continue
            tf = Counter(tokens)
            docs.append({"path": str(file_path), "text": raw, "tokens": tokens, "tf": tf, "length": len(tokens)})
            corpus_bytes += min(file_path.stat().st_size, self._max_file_bytes)
            for term in set(tokens):
                self._df[term] += 1
        self._docs = docs
        self._avgdl = sum(d["length"] for d in docs) / len(docs) if docs else 0.0
        elapsed = time.monotonic() - start
        self._setup = BaselineSetupResult(
            setup_seconds=elapsed,
            preprocessing_seconds=elapsed,
            index_build_seconds=elapsed,
            storage_bytes=corpus_bytes,
            indexed_documents=len(docs),
            metadata={
                "max_files": self._max_files,
                "max_file_bytes": self._max_file_bytes,
                "top_k": self._top_k,
                "llm_synthesis_enabled": bool(self._use_llm_synthesis and self._llm is not None),
            },
        )
        return self._setup

    async def predict(self, question: str, context_paths: List[str]) -> BaselinePrediction:
        start = time.monotonic()
        if not self._docs:
            await self.prepare(golden_set=None, bm_adapter=_SinglePathAdapter(context_paths))
        query_terms = _tokenize(question)
        ranked = sorted(
            ((_bm25_score(query_terms, doc, self._df, len(self._docs), self._avgdl), doc) for doc in self._docs),
            key=lambda x: x[0],
            reverse=True,
        )[: self._top_k]
        evidence_texts = [doc["text"][:1800] for score, doc in ranked if score > 0]
        top_paths = [doc["path"] for score, doc in ranked if score > 0]
        fallback = _best_sentence(question, [doc["text"] for score, doc in ranked if score > 0]) or "No lexical BM25 evidence found."
        answer, tokens = await _synthesize_answer(
            self._llm if self._use_llm_synthesis else None,
            question,
            evidence_texts,
            fallback=fallback,
            source="BM25",
        )
        return BaselinePrediction(
            answer=answer,
            elapsed=time.monotonic() - start,
            tokens_used=tokens,
            metadata={
                "baseline_type": "bm25",
                "llm_synthesis_used": bool(tokens),
                "top_docs": [{"path": doc["path"], "score": score} for score, doc in ranked],
                "read_file_ids": top_paths,
                "evidence_sources": top_paths,
                "evidence_snippets": evidence_texts,
                "setup_metrics": self.collect_setup_metrics(),
            },
        )

    def collect_setup_metrics(self) -> Dict[str, Any]:
        return _setup_to_dict(self._setup)


class NaiveRAGBaseline(BaselineAdapter):
    def __init__(
        self,
        *,
        chunk_words: int = 220,
        chunk_overlap: int = 40,
        max_files: int = 5000,
        max_chunks: int = 50000,
        llm: Optional[Any] = None,
        use_llm_synthesis: bool = True,
        name: str = "naive_rag_local",
        citation_name: str = "Naive RAG + LLM (chunk lexical)",
    ) -> None:
        self._chunk_words = chunk_words
        self._chunk_overlap = chunk_overlap
        self._max_files = max_files
        self._max_chunks = max_chunks
        self._llm = llm
        self._use_llm_synthesis = use_llm_synthesis
        self._chunks: List[Dict[str, Any]] = []
        self._name = name
        self._citation = citation_name
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
        paths = _collect_context_paths(golden_set, bm_adapter)
        chunks: List[Dict[str, Any]] = []
        bytes_read = 0
        for file_path in _iter_files(paths, self._max_files):
            if len(chunks) >= self._max_chunks:
                break
            try:
                text = file_path.read_text(encoding="utf-8", errors="ignore")
                bytes_read += file_path.stat().st_size
            except OSError:
                continue
            words = text.split()
            step = max(1, self._chunk_words - self._chunk_overlap)
            for start_idx in range(0, len(words), step):
                chunk_words = words[start_idx:start_idx + self._chunk_words]
                if not chunk_words:
                    continue
                chunk_text = " ".join(chunk_words)
                chunks.append({"path": str(file_path), "text": chunk_text, "tokens": set(_tokenize(chunk_text))})
                if len(chunks) >= self._max_chunks:
                    break
        self._chunks = chunks
        elapsed = time.monotonic() - start
        self._setup = BaselineSetupResult(
            setup_seconds=elapsed,
            preprocessing_seconds=elapsed,
            index_build_seconds=elapsed,
            storage_bytes=bytes_read,
            indexed_documents=len(chunks),
            metadata={
                "chunk_words": self._chunk_words,
                "chunk_overlap": self._chunk_overlap,
                "max_chunks": self._max_chunks,
                "llm_synthesis_enabled": bool(self._use_llm_synthesis and self._llm is not None),
            },
        )
        return self._setup

    async def predict(self, question: str, context_paths: List[str]) -> BaselinePrediction:
        start = time.monotonic()
        if not self._chunks:
            await self.prepare(golden_set=None, bm_adapter=_SinglePathAdapter(context_paths))
        q = set(_tokenize(question))
        ranked = sorted(
            ((len(q & chunk["tokens"]) / max(len(q), 1), chunk) for chunk in self._chunks),
            key=lambda x: x[0],
            reverse=True,
        )[:5]
        evidence_texts = [chunk["text"][:1800] for score, chunk in ranked if score > 0]
        top_paths = [chunk["path"] for score, chunk in ranked if score > 0]
        fallback = ranked[0][1]["text"][:1000] if ranked and ranked[0][0] > 0 else "No chunk evidence found."
        answer, tokens = await _synthesize_answer(
            self._llm if self._use_llm_synthesis else None,
            question,
            evidence_texts,
            fallback=fallback,
            source="Naive RAG",
        )
        return BaselinePrediction(
            answer=answer,
            elapsed=time.monotonic() - start,
            tokens_used=tokens,
            metadata={
                "baseline_type": "naive_rag_chunk_lexical",
                "llm_synthesis_used": bool(tokens),
                "top_chunks": [{"path": chunk["path"], "score": score} for score, chunk in ranked],
                "read_file_ids": top_paths,
                "evidence_sources": top_paths,
                "evidence_snippets": evidence_texts,
                "setup_metrics": self.collect_setup_metrics(),
            },
        )

    def collect_setup_metrics(self) -> Dict[str, Any]:
        return _setup_to_dict(self._setup)


def _collect_context_paths(golden_set: Any, bm_adapter: Any) -> List[str]:
    paths: List[str] = []
    samples = getattr(golden_set, "samples", []) if golden_set is not None else []
    if bm_adapter is None:
        return paths
    for sample_dict in samples[:20]:
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
                if child.is_file():
                    yield child
                    seen += 1
                    if seen >= max_files:
                        return
        if seen >= max_files:
            return


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


def _best_sentence(question: str, texts: List[str]) -> str:
    q = set(_tokenize(question))
    best = (0, "")
    for text in texts:
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", text):
            score = len(q & set(_tokenize(sentence)))
            if score > best[0] and len(sentence.strip()) > 20:
                best = (score, sentence.strip())
    return best[1]


def _setup_to_dict(setup: BaselineSetupResult) -> Dict[str, Any]:
    return {
        "setup_seconds": setup.setup_seconds,
        "preprocessing_seconds": setup.preprocessing_seconds,
        "index_build_seconds": setup.index_build_seconds,
        "storage_bytes": setup.storage_bytes,
        "indexed_documents": setup.indexed_documents,
        "metadata": setup.metadata,
    }


async def _synthesize_answer(
    llm: Optional[Any],
    question: str,
    evidence_texts: List[str],
    *,
    fallback: str,
    source: str,
) -> Tuple[str, int]:
    """Use the shared benchmark LLM for lightweight answer synthesis when available."""
    if llm is None or not evidence_texts:
        return fallback, 0
    evidence = "\n\n---\n\n".join(
        f"[Evidence {idx}]\n{text[:1600]}"
        for idx, text in enumerate(evidence_texts[:5], 1)
        if text.strip()
    )
    if not evidence.strip():
        return fallback, 0
    prompt = f"""You are a HotpotQA answerer using retrieved evidence from {source}.

Question:
{question}

Retrieved evidence:
{evidence}

Instructions:
- Answer using only the retrieved evidence.
- Return a concise final answer span, not a paragraph.
- If the evidence is insufficient, return the best supported short answer rather than a refusal.
"""
    try:
        resp = await llm.achat(messages=[{"role": "user", "content": prompt}], stream=False)
        answer = (getattr(resp, "content", "") or "").strip()
        usage = getattr(resp, "usage", {}) or {}
        tokens = int(usage.get("total_tokens", 0) or 0) if isinstance(usage, dict) else 0
        return (answer or fallback), tokens
    except Exception:
        return fallback, 0


class _SinglePathAdapter:
    def __init__(self, paths: List[str]) -> None:
        self._paths = paths

    def _corpus_paths(self) -> List[str]:
        return self._paths
