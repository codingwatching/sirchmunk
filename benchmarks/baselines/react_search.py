"""baselines/react_search.py — ReAct Search baseline

Main-table paper baseline: a plain ReAct tool-use search loop.

Scope:
- Uses LLM + ToolRegistry + keyword_search / file_read / dir_scan.
- knowledge_query and tree_navigate stay disabled so the baseline cannot consume
  LENS cache or tree-index assets.
- Never calls AgenticSearch.search(mode="DEEP"), so LENS multi-signal priors,
  requirement decomposition, self-correction and persistence stay out of scope.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base_adapter import BaselineAdapter, BaselinePrediction, BaselineSetupResult


def _compact_search_history(history: Any, *, limit: int = 16, text_limit: int = 240) -> List[Dict[str, Any]]:
    """Keep ReAct telemetry bounded while preserving query-budget counts."""
    if not isinstance(history, list):
        return []
    compact: List[Dict[str, Any]] = []
    for index, item in enumerate(history[:limit]):
        if isinstance(item, dict):
            compact.append({
                "index": index,
                "tool": str(item.get("tool") or item.get("tool_name") or "")[:80],
                "summary": str(item.get("summary") or item.get("query") or item.get("arguments") or item)[:text_limit],
            })
        else:
            compact.append({"index": index, "summary": str(item)[:text_limit]})
    return compact


def _compact_retrieval_logs(logs: Any, *, limit: int = 16) -> List[Dict[str, Any]]:
    """Drop bulky retrieval payloads from ReAct JSONL artifacts."""
    if not isinstance(logs, list):
        return []
    compact: List[Dict[str, Any]] = []
    for index, log in enumerate(logs[:limit]):
        data = log.to_dict() if hasattr(log, "to_dict") else log
        if isinstance(data, dict):
            compact.append({
                "index": index,
                "tool_name": str(data.get("tool_name") or data.get("tool") or "")[:80],
                "tokens": int(data.get("tokens") or 0),
                "timestamp": str(data.get("timestamp") or "")[:80],
            })
        else:
            compact.append({"index": index, "summary": str(data)[:240]})
    return compact


class ReActSearchBaseline(BaselineAdapter):
    """Pure ReAct + retrieval tools baseline."""

    def __init__(
        self,
        *,
        llm: Optional[Any] = None,
        work_path: str = "",
        max_loops: int = 8,
        max_token_budget: int = 64_000,
        max_results: int = 8,
        max_chars_per_file: int = 24_000,
        enable_dir_scan: bool = True,
        dir_scan_max_files: int = 200,
        name: str = "react",
        citation_name: str = "ReAct Search",
    ) -> None:
        self._llm = llm
        self._work_path = work_path
        self._max_loops = max_loops
        self._max_token_budget = max_token_budget
        self._max_results = max_results
        self._max_chars_per_file = max_chars_per_file
        self._enable_dir_scan = enable_dir_scan
        self._dir_scan_max_files = dir_scan_max_files
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
        if not self._work_path and bm_adapter is not None:
            try:
                self._work_path = bm_adapter.get_work_path()
            except Exception:
                self._work_path = ""
        elapsed = time.monotonic() - start
        self._setup = BaselineSetupResult(
            setup_seconds=elapsed,
            preprocessing_seconds=0.0,
            index_build_seconds=0.0,
            storage_bytes=0,
            indexed_documents=0,
            expected_documents=0,
            build_completed=True,
            index_ready=True,
            index_required=False,
            rebuild_required=False,
            query_ready_immediately=True,
            metadata={
                "baseline_type": "react",
                "max_loops": self._max_loops,
                "max_token_budget": self._max_token_budget,
                "tools": ["keyword_search", "file_read"] + (["dir_scan"] if self._enable_dir_scan else []),
                "knowledge_query_enabled": False,
                "tree_navigate_enabled": False,
                "index_required": False,
                "rebuild_required": False,
                "query_ready_immediately": True,
            },
        )
        return self._setup

    async def predict(self, question: str, context_paths: List[str]) -> BaselinePrediction:
        start = time.monotonic()
        if self._llm is None:
            return BaselinePrediction(
                answer="ReAct baseline unavailable: LLM is not configured.",
                elapsed=time.monotonic() - start,
                metadata={"baseline_type": "react", "error": "missing_llm"},
            )

        from sirchmunk.agentic.dir_scan_tool import DirScanTool
        from sirchmunk.agentic.react_agent import ReActSearchAgent
        from sirchmunk.agentic.tools import FileReadTool, KeywordSearchTool, ToolRegistry
        from sirchmunk.retrieve.text_retriever import GrepRetriever
        from sirchmunk.scan.dir_scanner import DirectoryScanner

        registry = ToolRegistry()
        retriever = GrepRetriever(work_path=self._work_path or None)
        registry.register(KeywordSearchTool(
            retriever=retriever,
            paths=context_paths,
            max_results=self._max_results,
        ))
        registry.register(FileReadTool(max_chars_per_file=self._max_chars_per_file))
        if self._enable_dir_scan:
            scanner = DirectoryScanner(
                llm=self._llm,
                max_files=self._dir_scan_max_files,
                max_preview_chars=600,
                max_workers=4,
            )
            registry.register(DirScanTool(scanner=scanner, paths=context_paths))

        agent = ReActSearchAgent(
            llm=self._llm,
            tool_registry=registry,
            max_loops=self._max_loops,
            max_token_budget=self._max_token_budget,
            # Align the answer register with the LENS answer contract so EM and
            # judge metrics compare systems at the same output granularity:
            # verbose answers are penalized by EM yet favored by semantic
            # judges, which otherwise contaminates both readings.
            answer_style_instruction=(
                "Your final answer inside <ANSWER></ANSWER> must be the minimal "
                "concise answer span only (a name, date, phrase, or yes/no) — "
                "no explanations, no full sentences, no multiple candidates. "
                "If evidence is partial, give the best supported concise answer "
                "instead of refusing."
            ),
        )
        answer, ctx = await agent.run(query=question)
        elapsed = time.monotonic() - start

        search_history = getattr(ctx, "search_history", [])
        retrieval_logs = getattr(ctx, "retrieval_logs", [])
        return BaselinePrediction(
            answer=answer,
            elapsed=elapsed,
            tokens_used=int(getattr(ctx, "total_llm_tokens", 0) or 0),
            metadata={
                "baseline_type": "react",
                "tools": registry.tool_names,
                "loop_count": getattr(ctx, "loop_count", 0),
                "read_file_ids": sorted(getattr(ctx, "read_file_ids", set()) or []),
                "search_history": _compact_search_history(search_history),
                "search_history_count": len(search_history) if isinstance(search_history, list) else 0,
                "retrieval_logs": _compact_retrieval_logs(retrieval_logs),
                "retrieval_log_count": len(retrieval_logs) if isinstance(retrieval_logs, list) else 0,
                "setup_metrics": self.collect_setup_metrics(),
            },
        )

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
            "metadata": self._setup.metadata,
        }

    def is_index_required(self) -> bool:
        return False

    def is_query_ready_immediately(self) -> bool:
        return True

    async def update_index(self, mutation: Any, bm_adapter: Any = None) -> Dict[str, Any]:
        return {
            "update_supported": True,
            "rebuild_required": False,
            "query_ready_immediately": True,
            "baseline_type": "react",
        }

    def estimate_update_cost(self, mutation: Any) -> Dict[str, Any]:
        return {"rebuild_required": False, "query_ready_immediately": True}

    def validate_index(self, corpus_manifest: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return {
            "index_ready": True,
            "indexed_documents": 0,
            "expected_documents": 0,
            "validation_errors": [],
            "index_required": False,
        }

    def extra_metadata(self) -> Dict[str, Any]:
        return {
            "baseline_type": "react",
            "index_required": False,
            "rebuild_required": False,
            "query_ready_immediately": True,
        }

    def get_max_concurrent(self) -> int:
        # ReAct baseline is LLM-call heavy; keep conservative default.
        return 1
