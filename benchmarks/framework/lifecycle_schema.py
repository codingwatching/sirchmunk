"""Lifecycle schemas for baseline feasibility and full-corpus experiments.

These data contracts are intentionally separate from ``framework.schema`` so
that baseline build/index lifecycle governance can evolve without disrupting
existing ResearchOps experiment records.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


class BaselinePhase(str, Enum):
    """Lifecycle phase for index-heavy baseline systems."""

    PENDING = "pending"
    PREPARING = "preparing"
    INDEXING = "indexing"
    VALIDATING = "validating"
    READY = "ready"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    SKIPPED = "skipped"


class FailureReason(str, Enum):
    """Structured failure reasons for build/index feasibility reporting."""

    NONE = "none"
    TIMEOUT = "timeout"
    OOM = "oom"
    DISK_EXCEEDED = "disk_exceeded"
    API_BUDGET_EXCEEDED = "api_budget_exceeded"
    DEPENDENCY_MISSING = "dependency_missing"
    BUILD_CRASH = "build_crash"
    INDEX_VALIDATION_FAILED = "index_validation_failed"
    PARTIAL_INDEX = "partial_index"
    IMPORT_MISSING = "import_missing"
    UNAVAILABLE = "unavailable"
    UNKNOWN = "unknown"


@dataclass
class ResourceBudget:
    """Declared resource budget for baseline build/index phases."""

    wall_clock_seconds: float = 0.0
    max_ram_bytes: int = 0
    max_disk_bytes: int = 0
    max_llm_tokens: int = 0
    max_api_cost_usd: float = 0.0
    retry_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BaselineIndexValidation:
    """Result of validating whether a baseline index is query-ready."""

    index_ready: bool = True
    indexed_documents: int = 0
    expected_documents: int = 0
    validation_errors: list[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_partial(self) -> bool:
        return (
            self.expected_documents > 0
            and self.indexed_documents > 0
            and self.indexed_documents < self.expected_documents
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BaselineLifecycleRecord:
    """Single baseline full-corpus lifecycle record."""

    run_id: str
    benchmark: str
    baseline_name: str
    citation_name: str = ""
    corpus_id: str = ""
    corpus_scale: str = "fullwiki"
    corpus_size_docs: int = 0
    indexed_documents: int = 0
    index_required: bool = True
    phase: BaselinePhase = BaselinePhase.PENDING
    build_completed: bool = False
    index_ready: bool = False
    query_eligible: bool = False
    rebuild_required: bool = False
    query_ready_immediately: bool = False
    partial_index: bool = False
    build_time_seconds: float = 0.0
    preprocessing_seconds: float = 0.0
    index_build_seconds: float = 0.0
    peak_ram_bytes: int = 0
    disk_bytes: int = 0
    preprocess_llm_tokens: int = 0
    api_cost_usd: float = 0.0
    failure_reason: FailureReason = FailureReason.NONE
    failure_message: str = ""
    artifact_dir: str = ""
    started_at: str = ""
    ended_at: str = ""
    resource_budget: Optional[ResourceBudget] = None
    validation: Optional[BaselineIndexValidation] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["phase"] = self.phase.value
        data["failure_reason"] = self.failure_reason.value
        if self.resource_budget is not None:
            data["resource_budget"] = self.resource_budget.to_dict()
        if self.validation is not None:
            data["validation"] = self.validation.to_dict()
        return data


__all__ = [
    "BaselinePhase",
    "FailureReason",
    "ResourceBudget",
    "BaselineIndexValidation",
    "BaselineLifecycleRecord",
]
