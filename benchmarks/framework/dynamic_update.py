"""Dynamic corpus update utilities for freshness experiments.

This module models add/delete/update mutations over raw document corpora and
records update costs separately from query-time quality.  It is deliberately
lightweight: baselines can optionally implement ``update_index``; otherwise the
study records that a full rebuild is required.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shutil
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from framework.time_utils import local_timestamp, now_local_iso


class UpdateOperation(str, Enum):
    ADD = "add"
    DELETE = "delete"
    UPDATE = "update"
    MIXED = "mixed"


@dataclass
class CorpusMutation:
    """Description of a corpus mutation for dynamic update studies."""

    mutation_id: str
    operation: UpdateOperation
    doc_ids: List[str] = field(default_factory=list)
    delta_docs_dir: str = ""
    mutation_ratio: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["operation"] = self.operation.value
        return data


@dataclass
class CorpusVersionManifest:
    version_id: str
    source_dir: str
    version_dir: str
    mutation: Dict[str, Any]
    doc_count: int
    total_bytes: int
    checksum: str
    created_at: str
    materialize_mode: str = "symlink"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DynamicUpdateResult:
    baseline_name: str
    mutation_id: str
    operation: str
    update_completed: bool
    rebuild_required: bool
    update_time_seconds: float
    query_ready_immediately: bool = False
    failure_reason: str = "none"
    failure_message: str = ""
    freshness_before: Optional[float] = None
    freshness_after: Optional[float] = None
    query_quality_after_update: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DynamicUpdateManager:
    """Create corpus versions and measure baseline update cost."""

    def __init__(
        self,
        base_corpus_dir: str | Path,
        work_dir: str | Path,
        *,
        materialize_mode: str = "symlink",
    ) -> None:
        self.base_corpus_dir = Path(base_corpus_dir).expanduser().resolve()
        self.work_dir = Path(work_dir).expanduser().resolve()
        self.materialize_mode = materialize_mode
        self.work_dir.mkdir(parents=True, exist_ok=True)
        if not self.base_corpus_dir.exists():
            raise FileNotFoundError(f"Base corpus directory not found: {self.base_corpus_dir}")

    def create_version(self, mutation: CorpusMutation, *, version_id: str = "") -> CorpusVersionManifest:
        """Create a mutated corpus version via symlink/copy materialization."""
        version_id = version_id or f"v_{mutation.mutation_id}_{_timestamp()}"
        version_dir = self.work_dir / version_id
        if version_dir.exists():
            shutil.rmtree(version_dir)
        version_dir.mkdir(parents=True, exist_ok=True)

        self._materialize_base(version_dir)
        self._apply_mutation(version_dir, mutation)
        docs = _list_files(version_dir)
        checksum = _checksum_paths(version_dir, docs)
        manifest = CorpusVersionManifest(
            version_id=version_id,
            source_dir=str(self.base_corpus_dir),
            version_dir=str(version_dir),
            mutation=mutation.to_dict(),
            doc_count=len(docs),
            total_bytes=sum(_safe_size(p) for p in docs),
            checksum=checksum,
            created_at=now_local_iso(),
            materialize_mode=self.materialize_mode,
        )
        (version_dir / "corpus_version_manifest.json").write_text(
            json.dumps(manifest.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return manifest

    async def evaluate_baseline_update(
        self,
        baseline: Any,
        mutation: CorpusMutation,
        *,
        bm_adapter: Any = None,
    ) -> DynamicUpdateResult:
        """Run optional baseline.update_index() and measure update cost."""
        t0 = time.monotonic()
        update_fn = getattr(baseline, "update_index", None)
        if not callable(update_fn):
            return DynamicUpdateResult(
                baseline_name=getattr(baseline, "name", "unknown"),
                mutation_id=mutation.mutation_id,
                operation=mutation.operation.value,
                update_completed=False,
                rebuild_required=True,
                update_time_seconds=0.0,
                query_ready_immediately=False,
                failure_reason="update_not_supported",
                failure_message="Baseline does not implement update_index(); full rebuild required.",
                metadata=getattr(baseline, "extra_metadata", lambda: {})(),
            )
        try:
            result = update_fn(mutation, bm_adapter=bm_adapter)
            if asyncio.iscoroutine(result):
                result = await result
            elapsed = time.monotonic() - t0
            metadata = result if isinstance(result, dict) else {}
            if metadata.get("update_supported") is False:
                return DynamicUpdateResult(
                    baseline_name=getattr(baseline, "name", "unknown"),
                    mutation_id=mutation.mutation_id,
                    operation=mutation.operation.value,
                    update_completed=False,
                    rebuild_required=True,
                    update_time_seconds=elapsed,
                    query_ready_immediately=bool(metadata.get("query_ready_immediately", False)),
                    failure_reason=str(metadata.get("failure_reason") or "update_not_supported"),
                    failure_message="Baseline does not support incremental update; full rebuild required.",
                    metadata=metadata,
                )
            return DynamicUpdateResult(
                baseline_name=getattr(baseline, "name", "unknown"),
                mutation_id=mutation.mutation_id,
                operation=mutation.operation.value,
                update_completed=True,
                rebuild_required=bool(metadata.get("rebuild_required", False)),
                update_time_seconds=elapsed,
                query_ready_immediately=bool(metadata.get("query_ready_immediately", False)),
                metadata=metadata,
            )
        except Exception as exc:
            return DynamicUpdateResult(
                baseline_name=getattr(baseline, "name", "unknown"),
                mutation_id=mutation.mutation_id,
                operation=mutation.operation.value,
                update_completed=False,
                rebuild_required=True,
                update_time_seconds=time.monotonic() - t0,
                query_ready_immediately=False,
                failure_reason=_classify_update_failure(exc),
                failure_message=str(exc),
            )

    def save_update_results(self, results: Iterable[DynamicUpdateResult], path: str | Path) -> str:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as fp:
            for result in results:
                fp.write(json.dumps(result.to_dict(), ensure_ascii=False) + "\n")
        return str(p)

    def _materialize_base(self, version_dir: Path) -> None:
        for source in _list_files(self.base_corpus_dir):
            rel = source.relative_to(self.base_corpus_dir)
            target = version_dir / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            if self.materialize_mode == "copy":
                shutil.copy2(source, target)
            elif self.materialize_mode == "symlink":
                os.symlink(source, target)
            else:
                raise ValueError("materialize_mode must be symlink or copy")

    def _apply_mutation(self, version_dir: Path, mutation: CorpusMutation) -> None:
        if mutation.operation in (UpdateOperation.DELETE, UpdateOperation.MIXED):
            for doc_id in mutation.doc_ids:
                target = version_dir / doc_id
                if target.exists() or target.is_symlink():
                    target.unlink()
        if mutation.operation in (UpdateOperation.ADD, UpdateOperation.UPDATE, UpdateOperation.MIXED):
            delta_dir = Path(mutation.delta_docs_dir).expanduser().resolve() if mutation.delta_docs_dir else None
            if not delta_dir or not delta_dir.exists():
                if mutation.operation in (UpdateOperation.ADD, UpdateOperation.UPDATE):
                    raise FileNotFoundError("delta_docs_dir is required for add/update mutations")
                return
            for source in _list_files(delta_dir):
                rel = source.relative_to(delta_dir)
                if mutation.doc_ids and str(rel) not in set(mutation.doc_ids):
                    continue
                target = version_dir / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists() or target.is_symlink():
                    target.unlink()
                shutil.copy2(source, target)


def _list_files(root: Path) -> List[Path]:
    return sorted([p for p in root.rglob("*") if p.is_file() or p.is_symlink()], key=lambda p: str(p))


def _checksum_paths(root: Path, files: List[Path]) -> str:
    rels = [str(p.relative_to(root)) for p in files]
    raw = json.dumps(rels, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


def _safe_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


def _timestamp() -> str:
    return local_timestamp()


def _classify_update_failure(exc: Exception) -> str:
    text = str(exc).lower()
    if "timeout" in text:
        return "timeout"
    if "oom" in text or "out of memory" in text:
        return "oom"
    if "disk" in text or "no space" in text:
        return "disk_exceeded"
    return "update_failed"


__all__ = [
    "UpdateOperation",
    "CorpusMutation",
    "CorpusVersionManifest",
    "DynamicUpdateResult",
    "DynamicUpdateManager",
]
