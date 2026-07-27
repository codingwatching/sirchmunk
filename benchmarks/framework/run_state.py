"""Run state machine for long-running ResearchOps experiments."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from .time_utils import now_local_iso


class RunStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    PARTIAL = "PARTIAL"
    FAILED = "FAILED"
    SUCCESS = "SUCCESS"
    REPORTING = "REPORTING"
    REPORTED = "REPORTED"
    CANCELLED = "CANCELLED"


_TERMINAL = {RunStatus.FAILED, RunStatus.SUCCESS, RunStatus.REPORTED, RunStatus.CANCELLED}
_ALLOWED = {
    RunStatus.PENDING: {RunStatus.RUNNING, RunStatus.CANCELLED},
    RunStatus.RUNNING: {RunStatus.PARTIAL, RunStatus.FAILED, RunStatus.SUCCESS, RunStatus.CANCELLED},
    RunStatus.PARTIAL: {RunStatus.RUNNING, RunStatus.FAILED, RunStatus.SUCCESS, RunStatus.CANCELLED},
    RunStatus.SUCCESS: {RunStatus.REPORTING, RunStatus.REPORTED},
    RunStatus.REPORTING: {RunStatus.REPORTED, RunStatus.FAILED},
    RunStatus.FAILED: {RunStatus.RUNNING},
    RunStatus.REPORTED: set(),
    RunStatus.CANCELLED: set(),
}


@dataclass
class RunState:
    run_id: str
    benchmark: str
    status: RunStatus = RunStatus.PENDING
    system: str = "sirchmunk"
    seed: int = 42
    cache_mode: str = "cold"
    total_samples: int = 0
    completed_samples: int = 0
    failed_samples: int = 0
    retry_count: int = 0
    started_at: Optional[str] = None
    updated_at: str = field(default_factory=now_local_iso)
    ended_at: Optional[str] = None
    artifact_dir: str = ""
    results_path: str = ""
    checkpoint_path: str = ""
    last_error: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_terminal(self) -> bool:
        return self.status in _TERMINAL

    def progress(self) -> float:
        if self.total_samples <= 0:
            return 0.0
        return self.completed_samples / self.total_samples

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["status"] = self.status.value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RunState":
        payload = dict(data)
        payload["status"] = RunStatus(payload.get("status", RunStatus.PENDING))
        return cls(**payload)


class RunStateStore:
    """Persistent run state store backed by a JSON object file."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def get(self, run_id: str) -> Optional[RunState]:
        data = self._load()
        row = data.get(run_id)
        return RunState.from_dict(row) if row else None

    def upsert(self, state: RunState) -> RunState:
        state.updated_at = now_local_iso()
        data = self._load()
        data[state.run_id] = state.to_dict()
        self._save(data)
        return state

    def transition(self, run_id: str, status: RunStatus, **updates) -> RunState:
        state = self.get(run_id)
        if state is None:
            state = RunState(run_id=run_id, benchmark=str(updates.pop("benchmark", "unknown")))
        current = state.status
        if status != current and status not in _ALLOWED.get(current, set()):
            raise ValueError(f"Invalid run transition: {current.value} -> {status.value}")
        state.status = status
        if status == RunStatus.RUNNING and not state.started_at:
            state.started_at = now_local_iso()
        if status in _TERMINAL:
            state.ended_at = now_local_iso()
        for key, value in updates.items():
            if hasattr(state, key):
                setattr(state, key, value)
            else:
                state.metadata[key] = value
        return self.upsert(state)

    def list(self, *, status: RunStatus | None = None) -> List[RunState]:
        states = [RunState.from_dict(row) for row in self._load().values()]
        if status is not None:
            states = [s for s in states if s.status == status]
        return sorted(states, key=lambda s: s.updated_at)

    def _load(self) -> Dict[str, Dict[str, Any]]:
        if not self.path.exists():
            return {}
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}

    def _save(self, data: Dict[str, Dict[str, Any]]) -> None:
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.path)


class RunStateMachine:
    """Explicit run state machine facade.

    RunStateStore keeps persistence; this facade owns lifecycle semantics so
    queue/orchestrator code does not need to know the raw transition table.
    """

    def __init__(self, store: RunStateStore) -> None:
        self.store = store

    def start(self, run_id: str, **updates) -> RunState:
        return self.store.transition(run_id, RunStatus.RUNNING, **updates)

    def mark_partial(self, run_id: str, **updates) -> RunState:
        return self.store.transition(run_id, RunStatus.PARTIAL, **updates)

    def mark_failed(self, run_id: str, error: str, **updates) -> RunState:
        updates["last_error"] = error
        return self.store.transition(run_id, RunStatus.FAILED, **updates)

    def mark_success(self, run_id: str, **updates) -> RunState:
        return self.store.transition(run_id, RunStatus.SUCCESS, **updates)

    def begin_reporting(self, run_id: str, **updates) -> RunState:
        return self.store.transition(run_id, RunStatus.REPORTING, **updates)

    def mark_reported(self, run_id: str, **updates) -> RunState:
        return self.store.transition(run_id, RunStatus.REPORTED, **updates)

    def cancel(self, run_id: str, reason: str = "", **updates) -> RunState:
        if reason:
            updates["last_error"] = reason
        return self.store.transition(run_id, RunStatus.CANCELLED, **updates)

    @staticmethod
    def allowed_next(status: RunStatus | str) -> List[str]:
        current = RunStatus(status)
        return sorted(next_status.value for next_status in _ALLOWED.get(current, set()))


__all__ = ["RunStatus", "RunState", "RunStateStore", "RunStateMachine"]
