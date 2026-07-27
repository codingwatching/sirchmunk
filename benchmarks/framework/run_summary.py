"""Standard run-summary contracts for ResearchOps control-layer P0."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List

from .asset_registry import AssetRecord
from .control_phase import ControlBlock, ExperimentStage
from .time_utils import now_local_iso


class ControlRunStatus(str, Enum):
    """Top-level status values for a controlled benchmark run."""

    PLANNED = "planned"
    RUNNING = "running"
    PARTIAL = "partial"
    SUCCESS = "success"
    BLOCKED = "blocked"
    FAILED = "failed"
    CANCELLED = "cancelled"
    UNKNOWN = "unknown"


@dataclass
class GateSummary:
    """Compact gate outcome embedded into ``run_summary.json``."""

    name: str
    passed: bool
    severity: str = "error"
    blocking: bool = True
    issue_count: int = 0
    messages: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GateSummary":
        payload = dict(data or {})
        payload["passed"] = bool(payload.get("passed", False))
        payload["blocking"] = bool(payload.get("blocking", True))
        payload["issue_count"] = int(payload.get("issue_count", 0) or 0)
        payload["messages"] = _as_str_list(payload.get("messages"))
        payload["details"] = dict(payload.get("details") or {})
        allowed = set(cls.__dataclass_fields__)  # type: ignore[attr-defined]
        return cls(**{key: value for key, value in payload.items() if key in allowed})

    @classmethod
    def from_gate_result(cls, gate_result: Any) -> "GateSummary":
        """Build from a control-gate result-like object without importing it."""
        payload = gate_result.to_dict() if hasattr(gate_result, "to_dict") else dict(gate_result)
        issues = payload.get("issues") or []
        messages: List[str] = []
        for issue in issues:
            if isinstance(issue, dict):
                message = str(issue.get("message", ""))
                if message:
                    messages.append(message)
        return cls(
            name=str(payload.get("name", "")),
            passed=bool(payload.get("passed", False)),
            severity=str(payload.get("severity", "error")),
            blocking=bool(payload.get("blocking", True)),
            issue_count=int(payload.get("issue_count", len(issues)) or 0),
            messages=messages,
            details=dict(payload.get("details") or {}),
        )


@dataclass
class StageSummary:
    """One block/stage execution summary."""

    block: str
    stage: str
    status: ControlRunStatus = ControlRunStatus.UNKNOWN
    started_at: str = ""
    ended_at: str = ""
    run_ids: List[str] = field(default_factory=list)
    artifact_paths: Dict[str, str] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["status"] = self.status.value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StageSummary":
        payload = dict(data or {})
        payload["status"] = _coerce_status(payload.get("status"))
        payload["run_ids"] = _as_str_list(payload.get("run_ids"))
        payload["artifact_paths"] = {
            str(key): str(value)
            for key, value in dict(payload.get("artifact_paths") or {}).items()
        }
        payload["metrics"] = dict(payload.get("metrics") or {})
        allowed = set(cls.__dataclass_fields__)  # type: ignore[attr-defined]
        return cls(**{key: value for key, value in payload.items() if key in allowed})


@dataclass
class ControlRunSummary:
    """Machine-readable summary emitted by future total-control scripts."""

    control_run_id: str
    benchmark: str
    block: str
    stage: str
    status: ControlRunStatus = ControlRunStatus.PLANNED
    summary_version: int = 1
    created_at: str = field(default_factory=now_local_iso)
    updated_at: str = field(default_factory=now_local_iso)
    env_file: str = ""
    output_dir: str = ""
    config_hash: str = ""
    protocol_hash: str = ""
    sample_id_checksum: str = ""
    gates: List[GateSummary] = field(default_factory=list)
    stages: List[StageSummary] = field(default_factory=list)
    assets: List[Dict[str, Any]] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    paths: Dict[str, str] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def blocked(self) -> bool:
        return any(gate.blocking and not gate.passed for gate in self.gates)

    @property
    def paper_ready(self) -> bool:
        return self.status == ControlRunStatus.SUCCESS and not self.blocked

    def add_gate(self, gate: GateSummary | Any) -> None:
        summary = gate if isinstance(gate, GateSummary) else GateSummary.from_gate_result(gate)
        self.gates.append(summary)
        self.updated_at = now_local_iso()
        if summary.blocking and not summary.passed:
            self.status = ControlRunStatus.BLOCKED

    def add_stage(self, stage: StageSummary) -> None:
        self.stages.append(stage)
        self.updated_at = now_local_iso()

    def add_asset(self, asset: AssetRecord | Dict[str, Any]) -> None:
        payload = asset.to_dict() if hasattr(asset, "to_dict") else dict(asset)
        self.assets.append(payload)
        self.updated_at = now_local_iso()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "control_run_id": self.control_run_id,
            "benchmark": self.benchmark,
            "block": self.block,
            "stage": self.stage,
            "status": self.status.value,
            "summary_version": self.summary_version,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "env_file": self.env_file,
            "output_dir": self.output_dir,
            "config_hash": self.config_hash,
            "protocol_hash": self.protocol_hash,
            "sample_id_checksum": self.sample_id_checksum,
            "blocked": self.blocked,
            "paper_ready": self.paper_ready,
            "gates": [gate.to_dict() for gate in self.gates],
            "stages": [stage.to_dict() for stage in self.stages],
            "assets": list(self.assets),
            "metrics": self.metrics,
            "paths": self.paths,
            "recommendations": list(self.recommendations),
            "metadata": self.metadata,
        }

    def save(self, path: str | Path) -> str:
        return save_summary(self, path)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ControlRunSummary":
        payload = dict(data or {})
        summary = cls(
            control_run_id=str(payload.get("control_run_id", "")),
            benchmark=str(payload.get("benchmark", "")),
            block=str(payload.get("block", "")),
            stage=str(payload.get("stage", "")),
            status=_coerce_status(payload.get("status")),
            summary_version=int(payload.get("summary_version", 1) or 1),
            created_at=str(payload.get("created_at", "")) or now_local_iso(),
            updated_at=str(payload.get("updated_at", "")) or now_local_iso(),
            env_file=str(payload.get("env_file", "")),
            output_dir=str(payload.get("output_dir", "")),
            config_hash=str(payload.get("config_hash", "")),
            protocol_hash=str(payload.get("protocol_hash", "")),
            sample_id_checksum=str(payload.get("sample_id_checksum", "")),
            metrics=dict(payload.get("metrics") or {}),
            paths={str(k): str(v) for k, v in dict(payload.get("paths") or {}).items()},
            recommendations=_as_str_list(payload.get("recommendations")),
            metadata=dict(payload.get("metadata") or {}),
        )
        summary.gates = [GateSummary.from_dict(row) for row in payload.get("gates", [])]
        summary.stages = [StageSummary.from_dict(row) for row in payload.get("stages", [])]
        summary.assets = [dict(row) for row in payload.get("assets", []) if isinstance(row, dict)]
        return summary



def create_control_run_summary(
    *,
    control_run_id: str,
    benchmark: str,
    block: ControlBlock | str,
    stage: ExperimentStage | str,
    env_file: str = "",
    output_dir: str = "",
    paths: Dict[str, str] | None = None,
    metadata: Dict[str, Any] | None = None,
) -> ControlRunSummary:
    """Create a summary skeleton from control-layer identifiers."""
    resolved_block = block.value if isinstance(block, ControlBlock) else str(block)
    resolved_stage = stage.value if isinstance(stage, ExperimentStage) else str(stage)
    return ControlRunSummary(
        control_run_id=control_run_id,
        benchmark=benchmark,
        block=resolved_block,
        stage=resolved_stage,
        env_file=env_file,
        output_dir=output_dir,
        paths=paths or {},
        metadata=metadata or {},
    )


def save_summary(summary: ControlRunSummary, path: str | Path) -> str:
    """Write ``run_summary.json`` atomically."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    summary.updated_at = now_local_iso()
    tmp.write_text(json.dumps(summary.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(target)
    return str(target)


def load_summary(path: str | Path) -> ControlRunSummary:
    """Load a previously saved ``run_summary.json``."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return ControlRunSummary.from_dict(data)


def summarize_assets(records: Iterable[AssetRecord]) -> Dict[str, Any]:
    """Return compact asset counts for dashboard/status output."""
    total = 0
    by_status: Dict[str, int] = {}
    by_type: Dict[str, int] = {}
    ready_reusable = 0
    for record in records:
        total += 1
        status = record.status.value
        asset_type = record.asset_type.value
        by_status[status] = by_status.get(status, 0) + 1
        by_type[asset_type] = by_type.get(asset_type, 0) + 1
        if record.reuse_allowed and status == "ready":
            ready_reusable += 1
    return {
        "total": total,
        "ready_reusable": ready_reusable,
        "by_status": by_status,
        "by_type": by_type,
    }


def _coerce_status(value: Any) -> ControlRunStatus:
    if isinstance(value, ControlRunStatus):
        return value
    try:
        return ControlRunStatus(str(value))
    except ValueError:
        return ControlRunStatus.UNKNOWN


def _as_str_list(value: Any) -> List[str]:
    if value is None or value == "":
        return []
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()]


__all__ = [
    "ControlRunStatus",
    "GateSummary",
    "StageSummary",
    "ControlRunSummary",
    "create_control_run_summary",
    "save_summary",
    "load_summary",
    "summarize_assets",
]
