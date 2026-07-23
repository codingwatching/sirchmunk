"""Sample-level checkpointing for resumable ResearchOps runs."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set


_CHECKPOINT_COMPLETED = "completed"
_CHECKPOINT_FAILED = "failed"
_CHECKPOINT_SKIPPED = "skipped"
_KNOWN_STATUSES = {_CHECKPOINT_COMPLETED, _CHECKPOINT_FAILED, _CHECKPOINT_SKIPPED}


@dataclass
class CheckpointRecord:
    sample_id: str
    status: str  # completed | failed | skipped
    row: Dict[str, Any] = field(default_factory=dict)
    error: str = ""
    attempts: int = 0
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "status": self.status,
            "row": self.row,
            "error": self.error,
            "attempts": self.attempts,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CheckpointRecord":
        return cls(
            sample_id=str(data.get("sample_id", "")),
            status=str(data.get("status", "")),
            row=data.get("row", {}) if isinstance(data.get("row", {}), dict) else {},
            error=str(data.get("error", "")),
            attempts=int(data.get("attempts", 0) or 0),
            updated_at=str(data.get("updated_at") or datetime.now(timezone.utc).isoformat()),
        )


class CheckpointManager:
    """Append-only JSONL checkpoint with latest-record reconstruction."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._records: Optional[Dict[str, CheckpointRecord]] = None

    def load(self) -> Dict[str, CheckpointRecord]:
        if self._records is not None:
            return self._records
        records: Dict[str, CheckpointRecord] = {}
        if self.path.exists():
            with self.path.open(encoding="utf-8") as fp:
                for line in fp:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = CheckpointRecord.from_dict(json.loads(line))
                    except json.JSONDecodeError:
                        continue
                    if rec.sample_id:
                        records[rec.sample_id] = rec
        self._records = records
        return records

    def reload(self) -> Dict[str, CheckpointRecord]:
        """Clear the in-memory cache and reconstruct latest records from disk."""
        self._records = None
        return self.load()

    def latest(self, sample_id: str) -> Optional[CheckpointRecord]:
        """Return the latest checkpoint record for a sample, if present."""
        return self.load().get(str(sample_id))

    def ids_by_status(
        self,
        status: str,
        sample_ids: Optional[Iterable[str]] = None,
    ) -> Set[str]:
        allowed = _sample_id_filter(sample_ids)
        return {
            sid
            for sid, rec in self.load().items()
            if rec.status == status and (allowed is None or sid in allowed)
        }

    def completed_ids(self, sample_ids: Optional[Iterable[str]] = None) -> Set[str]:
        return self.ids_by_status(_CHECKPOINT_COMPLETED, sample_ids=sample_ids)

    def failed_ids(self, sample_ids: Optional[Iterable[str]] = None) -> Set[str]:
        return self.ids_by_status(_CHECKPOINT_FAILED, sample_ids=sample_ids)

    def skipped_ids(self, sample_ids: Optional[Iterable[str]] = None) -> Set[str]:
        return self.ids_by_status(_CHECKPOINT_SKIPPED, sample_ids=sample_ids)

    def pending_ids(
        self,
        sample_ids: Iterable[str],
        *,
        skip_statuses: Iterable[str] = (_CHECKPOINT_COMPLETED,),
    ) -> Set[str]:
        all_ids = {str(sid) for sid in sample_ids}
        skipped = set(skip_statuses)
        known_done = {
            sid
            for sid, rec in self.load().items()
            if sid in all_ids and rec.status in skipped
        }
        return all_ids - known_done

    def load_completed_rows(self, sample_ids: Optional[Iterable[str]] = None) -> List[Dict[str, Any]]:
        records = self.load()
        if sample_ids is None:
            return [rec.row for rec in records.values() if rec.status == _CHECKPOINT_COMPLETED and rec.row]

        rows: List[Dict[str, Any]] = []
        seen: Set[str] = set()
        for sample_id in sample_ids:
            sid = str(sample_id)
            if sid in seen:
                continue
            seen.add(sid)
            rec = records.get(sid)
            if rec and rec.status == _CHECKPOINT_COMPLETED and rec.row:
                rows.append(rec.row)
        return rows

    def summary(self, sample_ids: Optional[Iterable[str]] = None) -> Dict[str, int]:
        allowed = _sample_id_filter(sample_ids)
        counts = {
            _CHECKPOINT_COMPLETED: 0,
            _CHECKPOINT_FAILED: 0,
            _CHECKPOINT_SKIPPED: 0,
            "unknown": 0,
        }
        for sid, rec in self.load().items():
            if allowed is not None and sid not in allowed:
                continue
            counts[rec.status if rec.status in _KNOWN_STATUSES else "unknown"] += 1
        counts["known"] = sum(counts.values())
        if allowed is not None:
            counts["pending"] = max(len(allowed) - counts["known"], 0)
        return counts

    def mark_completed(self, sample_id: str, row: Dict[str, Any], attempts: int = 1) -> None:
        self._append(CheckpointRecord(sample_id=sample_id, status=_CHECKPOINT_COMPLETED, row=row, attempts=attempts))

    def mark_failed(self, sample_id: str, error: str, attempts: int = 1, row: Optional[Dict[str, Any]] = None) -> None:
        self._append(CheckpointRecord(sample_id=sample_id, status=_CHECKPOINT_FAILED, row=row or {}, error=error, attempts=attempts))

    def mark_skipped(self, sample_id: str, reason: str) -> None:
        self._append(CheckpointRecord(sample_id=sample_id, status=_CHECKPOINT_SKIPPED, error=reason))

    def compact(self) -> None:
        """Rewrite checkpoint with only latest record per sample."""
        records = self.load()
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as fp:
            for rec in records.values():
                fp.write(json.dumps(rec.to_dict(), ensure_ascii=False) + "\n")
        tmp.replace(self.path)

    def _append(self, record: CheckpointRecord) -> None:
        if not record.sample_id:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
        records = self.load()
        records[record.sample_id] = record


def _sample_id_filter(sample_ids: Optional[Iterable[str]]) -> Optional[Set[str]]:
    if sample_ids is None:
        return None
    return {str(sample_id) for sample_id in sample_ids}


def rows_to_prediction_results(rows: Iterable[Dict[str, Any]]):
    """Convert checkpoint rows to PredictionResult without importing at module import time."""
    from .schema import PredictionResult

    results = []
    for row in rows:
        results.append(PredictionResult(
            sample_id=row.get("sample_id") or row.get("hotpot_id", ""),
            prediction=row.get("prediction") or row.get("raw_prediction", ""),
            judge_correct=bool(row.get("judge_correct", False)),
            coverage=bool(row.get("coverage", False)),
            elapsed=float(row.get("elapsed", 0.0) or 0.0),
            telemetry=row.get("telemetry", {}) if isinstance(row.get("telemetry", {}), dict) else {},
            error=row.get("error"),
            raw=row,
        ))
    return results
