"""ablations/lens_ablation_adapter.py — LENS 消融 BaselineAdapter

将 LENS full / ablation profile 包装为 BaselineAdapter，使其进入与 BM25-RAG、
ReAct Search 相同的 BaselineEvaluationSuite / PaperTableGenerator 流程。

关键原则：
- 不修改 run_research_loop.py 的自改进循环。
- 默认不修改 search.py；通过独立 AgenticSearch 实例 + 实例级 method patch 实现消融。
- 每个 profile 使用独立 work_path: {bm_work_path}/ablations/{profile.name}/，避免缓存污染。
"""
from __future__ import annotations

import types
from pathlib import Path
import time
from typing import Any, Dict, List, Optional

from baselines.base_adapter import BaselineAdapter, BaselinePrediction, BaselineSetupResult
from .lens_profile import LensSearchProfile


class LensAblationAdapter(BaselineAdapter):
    """Wrap AgenticSearch(DEEP) with a LENS ablation profile."""

    def __init__(
        self,
        bm_adapter: Any,
        profile: LensSearchProfile,
        *,
        max_loops: int = 10,
        max_token_budget: int = 128_000,
        top_k_files: int = 5,
        name_prefix: str = "ablation",
    ) -> None:
        self._bm_adapter = bm_adapter
        self._profile = profile
        self._max_loops = max_loops
        self._max_token_budget = max_token_budget
        self._top_k_files = top_k_files
        self._name = f"{name_prefix}_{profile.name}"
        self._searcher = None
        self._setup = BaselineSetupResult()

    @property
    def name(self) -> str:
        return self._name

    @property
    def citation_name(self) -> str:
        return self._profile.citation_name

    async def prepare(self, golden_set: Any = None, bm_adapter: Any = None) -> BaselineSetupResult:
        self._searcher = self._build_profile_searcher()
        self._setup = BaselineSetupResult(
            setup_seconds=0.0,
            preprocessing_seconds=0.0,
            index_build_seconds=0.0,
            storage_bytes=0,
            indexed_documents=0,
            expected_documents=len(getattr(golden_set, "samples", []) or []),
            build_completed=True,
            index_ready=True,
            index_required=False,
            rebuild_required=False,
            query_ready_immediately=True,
            metadata={
                "baseline_type": "lens_ablation",
                "profile": self._profile.name,
                "description": self._profile.description,
                "profile_flags": self._profile.__dict__,
                "index_required": False,
                "rebuild_required": False,
                "query_ready_immediately": True,
            },
        )
        return self._setup

    async def predict(self, question: str, context_paths: List[str]) -> BaselinePrediction:
        if self._searcher is None:
            self._searcher = self._build_profile_searcher()

        kwargs = {
            "mode": "DEEP",
            "max_loops": self._max_loops,
            "max_token_budget": self._max_token_budget,
            "top_k_files": self._top_k_files,
            "enable_dir_scan": self._profile.enable_dir_scan,
            **self._profile.search_kwargs,
        }
        t0 = time.monotonic()
        result = await self._searcher.search(
            query=question,
            paths=context_paths,
            return_context=True,
            **kwargs,
        )
        elapsed = time.monotonic() - t0
        answer = getattr(result, "answer", "") or str(result)
        tokens = int(getattr(result, "total_llm_tokens", 0) or 0)
        telemetry = result.to_dict() if hasattr(result, "to_dict") else {}
        return BaselinePrediction(
            answer=answer,
            elapsed=elapsed,
            tokens_used=tokens,
            metadata={
                "baseline_type": "lens_ablation",
                "profile": self._profile.name,
                "profile_description": self._profile.description,
                "telemetry": telemetry,
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
            "baseline_type": "lens_ablation",
            "profile": self._profile.name,
        }

    def estimate_update_cost(self, mutation: Any) -> Dict[str, Any]:
        return {"rebuild_required": False, "query_ready_immediately": True}

    def validate_index(self, corpus_manifest: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return {"index_ready": True, "indexed_documents": 0, "expected_documents": 0, "index_required": False}

    def extra_metadata(self) -> Dict[str, Any]:
        return {
            "baseline_type": "lens_ablation",
            "profile": self._profile.name,
            "index_required": False,
            "rebuild_required": False,
            "query_ready_immediately": True,
        }

    def get_max_concurrent(self) -> int:
        return 1

    def _build_profile_searcher(self):
        from sirchmunk.search import AgenticSearch

        base_searcher = self._bm_adapter.build_searcher()
        llm = getattr(base_searcher, "llm", None)
        base_work = Path(self._bm_adapter.get_work_path())
        work_path = base_work / "ablations" / self._profile.name
        searcher = AgenticSearch(
            llm=llm,
            work_path=str(work_path),
            reuse_knowledge=self._profile.enable_knowledge_reuse,
            verbose=False,
        )
        _apply_profile_patches(searcher, self._profile)
        return searcher


# ---------------------------------------------------------------------------
# Instance-level patches
# ---------------------------------------------------------------------------

def _apply_profile_patches(searcher: Any, profile: LensSearchProfile) -> None:
    """Apply safe instance-level patches according to profile flags."""
    if not profile.enable_cluster_reuse:
        searcher._try_reuse_cluster = types.MethodType(_no_reuse_cluster, searcher)
        searcher._try_soft_reuse = types.MethodType(_no_soft_reuse, searcher)

    if not profile.enable_knowledge_probe:
        searcher._probe_knowledge_cache = types.MethodType(_empty_knowledge_probe, searcher)

    if not profile.enable_spec_cache:
        searcher._load_spec_context = types.MethodType(_empty_spec_context, searcher)
        searcher._save_spec_context = types.MethodType(_noop_async, searcher)

    if not profile.enable_tree_probe:
        searcher._probe_tree_index = types.MethodType(_empty_list_async, searcher)

    if not profile.enable_compile_hints:
        searcher._probe_compile_hints = types.MethodType(_empty_compile_hints, searcher)

    if not profile.enable_summary_index:
        searcher._probe_summary_index = types.MethodType(_empty_list_async, searcher)

    if not profile.enable_catalog_probe:
        searcher._probe_catalog_for_deep = types.MethodType(_empty_list_async, searcher)

    if not profile.enable_dir_scan:
        searcher._probe_dir_scan = types.MethodType(_empty_dir_scan, searcher)
        searcher._rank_dir_scan_candidates = types.MethodType(_empty_list_async, searcher)

    if not profile.enable_sequential_exploration:
        searcher._agentic_retrieve = types.MethodType(_one_shot_retrieve_factory(profile), searcher)

    if not profile.enable_persistence:
        searcher._save_cluster_with_embedding = types.MethodType(_noop_async, searcher)
        searcher._save_spec_context = types.MethodType(_noop_async, searcher)


async def _no_reuse_cluster(self, query: str, paths: Optional[List[str]] = None):
    return None


async def _no_soft_reuse(self, query: str, paths: Optional[List[str]] = None):
    return None


async def _empty_knowledge_probe(self, query: str):
    from sirchmunk.search import KnowledgeProbeResult
    return KnowledgeProbeResult([], [], "")


async def _empty_spec_context(self, paths: List[str], stale_hours: float = 72.0):
    return ""


async def _empty_compile_hints(self, queries: List[str], scope: Any = None):
    from sirchmunk.search import CompileHints
    return CompileHints([], [])


async def _empty_dir_scan(self, paths: List[str], enable_dir_scan: bool = False, **kwargs):
    return None


async def _empty_list_async(self, *args, **kwargs):
    return []


async def _noop_async(self, *args, **kwargs):
    return None


def _one_shot_retrieve_factory(profile: LensSearchProfile):
    async def _one_shot_retrieve(self, query, data_reqs, target_files, context):
        from sirchmunk.search import RetrievalResult
        evidence_blocks: List[str] = []
        max_files = min(profile.one_shot_max_files, len(target_files or []))
        for fp in list(target_files or [])[:max_files]:
            text = await _extract_text_for_ablation(Path(fp), profile.one_shot_max_chars_per_file)
            if text.strip():
                evidence_blocks.append(
                    f"[One-shot evidence: {Path(fp).name}]\n{text[:profile.one_shot_max_chars_per_file]}"
                )
                try:
                    context.mark_file_read(str(fp))
                except Exception:
                    pass
        evidence = "\n\n---\n\n".join(evidence_blocks)
        return RetrievalResult(
            evidence=evidence,
            pages_extracted={},
            is_complete=bool(evidence),
            rounds_used=1,
        )
    return _one_shot_retrieve


async def _extract_text_for_ablation(path: Path, max_chars: int) -> str:
    try:
        if path.suffix.lower() in {".txt", ".md", ".rst", ".csv", ".json", ".xml", ".html", ".htm", ".log"} or not path.suffix:
            return path.read_text(encoding="utf-8", errors="ignore")[:max_chars]
        from sirchmunk.utils.file_utils import fast_extract
        result = await fast_extract(path)
        return (result.content if result and result.content else "")[:max_chars]
    except Exception:
        return ""
