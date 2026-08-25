"""Append-only asset registry for ResearchOps control-layer P0.

The registry is intentionally lightweight: it records where reusable assets live
and which hashes make them safe to reuse.  It does not move, delete, rebuild, or
validate assets by itself; those operations remain owned by lifecycle managers
and future control scripts.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .time_utils import now_local_iso


class AssetType(str, Enum):
    """Canonical asset categories for benchmark control runs."""

    CORPUS = "corpus"
    GOLDEN_SET = "golden_set"
    SAMPLE_IDS = "sample_ids"
    BASELINE_ASSET = "baseline_asset"
    INDEX = "index"
    EMBEDDING = "embedding"
    GRAPH = "graph"
    RUN_ARTIFACT = "run_artifact"
    PAPER_TABLE = "paper_table"
    REPORT = "report"
    OTHER = "other"


class AssetStatus(str, Enum):
    """Lifecycle state of one registered asset."""

    PLANNED = "planned"
    BUILDING = "building"
    READY = "ready"
    FAILED = "failed"
    SKIPPED = "skipped"
    STALE = "stale"
    UNKNOWN = "unknown"


@dataclass
class AssetRecord:
    """One asset registry row.

    ``asset_id`` is stable for a given benchmark/method/type/hash tuple, while
    the JSONL registry remains append-only so status changes are reconstructable.
    """

    asset_id: str
    asset_type: AssetType
    status: AssetStatus
    benchmark: str
    method: str = ""
    stage: str = ""
    block: str = ""
    path: str = ""
    run_id: str = ""
    task_id: str = ""
    corpus_id: str = ""
    corpus_hash: str = ""
    config_hash: str = ""
    protocol_hash: str = ""
    sample_id_checksum: str = ""
    build_completed: bool = False
    index_ready: bool = False
    query_eligible: bool = False
    reuse_allowed: bool = True
    failure_reason: str = "none"
    failure_message: str = ""
    setup_seconds: float = 0.0
    storage_bytes: int = 0
    created_at: str = field(default_factory=now_local_iso)
    updated_at: str = field(default_factory=now_local_iso)
    dependencies: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["asset_type"] = self.asset_type.value
        data["status"] = self.status.value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AssetRecord":
        payload = dict(data or {})
        payload["asset_type"] = _coerce_asset_type(payload.get("asset_type"))
        payload["status"] = _coerce_asset_status(payload.get("status"))
        payload["dependencies"] = _as_str_list(payload.get("dependencies"))
        payload["metadata"] = dict(payload.get("metadata") or {})
        allowed = set(cls.__dataclass_fields__)  # type: ignore[attr-defined]
        return cls(**{key: value for key, value in payload.items() if key in allowed})

    @classmethod
    def from_lifecycle_record(
        cls,
        record: Any,
        *,
        asset_id: str = "",
        stage: str = "",
        block: str = "assets",
        corpus_hash: str = "",
        config_hash: str = "",
    ) -> "AssetRecord":
        """Create an asset record from a baseline lifecycle record-like object."""
        payload = record.to_dict() if hasattr(record, "to_dict") else dict(record)
        benchmark = str(payload.get("benchmark", ""))
        method = str(payload.get("baseline_name") or payload.get("method") or "")
        status = AssetStatus.READY if payload.get("query_eligible") else AssetStatus.FAILED
        failure_reason = str(payload.get("failure_reason") or "none")
        if failure_reason not in {"", "none"}:
            status = AssetStatus.FAILED
        path = str(payload.get("artifact_dir") or "")
        resolved_corpus_hash = str(
            corpus_hash
            or payload.get("corpus_hash")
            or payload.get("corpus_id")
            or ""
        )
        resolved_config_hash = str(config_hash or payload.get("config_hash") or "")
        resolved_asset_id = asset_id or compute_asset_id(
            benchmark=benchmark,
            method=method,
            asset_type=AssetType.BASELINE_ASSET,
            corpus_hash=resolved_corpus_hash,
            config_hash=resolved_config_hash,
            path=path,
        )
        return cls(
            asset_id=resolved_asset_id,
            asset_type=AssetType.BASELINE_ASSET,
            status=status,
            benchmark=benchmark,
            method=method,
            stage=stage,
            block=block,
            path=path,
            run_id=str(payload.get("run_id", "")),
            corpus_id=str(payload.get("corpus_id", "")),
            corpus_hash=resolved_corpus_hash,
            config_hash=resolved_config_hash,
            build_completed=bool(payload.get("build_completed", False)),
            index_ready=bool(payload.get("index_ready", False)),
            query_eligible=bool(payload.get("query_eligible", False)),
            failure_reason=failure_reason or "none",
            failure_message=str(payload.get("failure_message", "")),
            setup_seconds=float(payload.get("build_time_seconds") or 0.0),
            storage_bytes=int(payload.get("disk_bytes") or 0),
            metadata={
                "corpus_scale": payload.get("corpus_scale", ""),
                "indexed_documents": payload.get("indexed_documents", 0),
                "partial_index": payload.get("partial_index", False),
                "rebuild_required": payload.get("rebuild_required", False),
            },
        )


class AssetRegistry:
    """Append-only JSONL registry with latest-state reconstruction."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: AssetRecord) -> AssetRecord:
        record.updated_at = now_local_iso()
        with self.path.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
        return record

    def record_asset(
        self,
        *,
        asset_type: AssetType | str,
        benchmark: str,
        status: AssetStatus | str = AssetStatus.READY,
        method: str = "",
        path: str = "",
        stage: str = "",
        block: str = "",
        run_id: str = "",
        corpus_hash: str = "",
        config_hash: str = "",
        sample_id_checksum: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AssetRecord:
        resolved_type = _coerce_asset_type(asset_type)
        resolved_status = _coerce_asset_status(status)
        asset_id = compute_asset_id(
            benchmark=benchmark,
            method=method,
            asset_type=resolved_type,
            corpus_hash=corpus_hash,
            config_hash=config_hash,
            sample_id_checksum=sample_id_checksum,
            path=path,
        )
        record = AssetRecord(
            asset_id=asset_id,
            asset_type=resolved_type,
            status=resolved_status,
            benchmark=benchmark,
            method=method,
            stage=stage,
            block=block,
            path=path,
            run_id=run_id,
            corpus_hash=corpus_hash,
            config_hash=config_hash,
            sample_id_checksum=sample_id_checksum,
            metadata=metadata or {},
        )
        return self.append(record)

    def latest(self) -> Dict[str, AssetRecord]:
        latest: Dict[str, AssetRecord] = {}
        for record in self.iter_records():
            if record.asset_id:
                latest[record.asset_id] = record
        return latest

    def get(self, asset_id: str) -> Optional[AssetRecord]:
        return self.latest().get(asset_id)

    def list(
        self,
        *,
        benchmark: str = "",
        method: str = "",
        asset_type: AssetType | str | None = None,
        status: AssetStatus | str | None = None,
        stage: str = "",
        corpus_hash: str = "",
        config_hash: str = "",
        sample_id_checksum: str = "",
        reusable_only: bool = False,
    ) -> List[AssetRecord]:
        records = list(self.latest().values())
        if benchmark:
            records = [row for row in records if row.benchmark == benchmark]
        if method:
            records = [row for row in records if row.method == method]
        if asset_type is not None:
            resolved_type = _coerce_asset_type(asset_type)
            records = [row for row in records if row.asset_type == resolved_type]
        if status is not None:
            resolved_status = _coerce_asset_status(status)
            records = [row for row in records if row.status == resolved_status]
        if stage:
            records = [row for row in records if row.stage == stage]
        if corpus_hash:
            records = [row for row in records if row.corpus_hash == corpus_hash]
        if config_hash:
            records = [row for row in records if row.config_hash == config_hash]
        if sample_id_checksum:
            records = [row for row in records if row.sample_id_checksum == sample_id_checksum]
        if reusable_only:
            records = [row for row in records if row.reuse_allowed and row.status == AssetStatus.READY]
        return sorted(records, key=lambda row: row.updated_at)

    def resolve_reusable(
        self,
        *,
        benchmark: str,
        method: str,
        asset_type: AssetType | str,
        corpus_hash: str = "",
        config_hash: str = "",
        sample_id_checksum: str = "",
    ) -> Optional[AssetRecord]:
        """Return the newest ready asset matching all declared reuse hashes."""
        rows = self.list(
            benchmark=benchmark,
            method=method,
            asset_type=asset_type,
            status=AssetStatus.READY,
            corpus_hash=corpus_hash,
            config_hash=config_hash,
            sample_id_checksum=sample_id_checksum,
            reusable_only=True,
        )
        return rows[-1] if rows else None

    def iter_records(self) -> Iterable[AssetRecord]:
        if not self.path.exists():
            return []
        rows: List[AssetRecord] = []
        with self.path.open(encoding="utf-8") as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(data, dict):
                    rows.append(AssetRecord.from_dict(data))
        return rows


def compute_asset_id(
    *,
    benchmark: str,
    method: str,
    asset_type: AssetType | str,
    corpus_hash: str = "",
    config_hash: str = "",
    sample_id_checksum: str = "",
    path: str = "",
) -> str:
    """Compute a stable short ID for an asset identity tuple."""
    resolved_type = _coerce_asset_type(asset_type)
    payload = {
        "benchmark": benchmark,
        "method": method,
        "asset_type": resolved_type.value,
        "corpus_hash": corpus_hash,
        "config_hash": config_hash,
        "sample_id_checksum": sample_id_checksum,
        "path": path,
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    prefix = f"{benchmark or 'asset'}:{method or resolved_type.value}"
    return f"{prefix}:{digest}"


def _coerce_asset_type(value: Any) -> AssetType:
    if isinstance(value, AssetType):
        return value
    try:
        return AssetType(str(value))
    except ValueError:
        return AssetType.OTHER


def _coerce_asset_status(value: Any) -> AssetStatus:
    if isinstance(value, AssetStatus):
        return value
    try:
        return AssetStatus(str(value))
    except ValueError:
        return AssetStatus.UNKNOWN


def _as_str_list(value: Any) -> List[str]:
    if value is None or value == "":
        return []
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()]


__all__ = [
    "AssetType",
    "AssetStatus",
    "AssetRecord",
    "AssetRegistry",
    "compute_asset_id",
]
