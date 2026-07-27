"""Long-term experiment registry for ResearchOps P3.

The registry is append-only JSONL with latest-record reconstruction. This keeps
writes robust for long-running unattended experiments while still allowing the
caller to query the latest state for each run/task.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .time_utils import now_local_iso


@dataclass
class ExperimentRegistryRecord:
    run_id: str
    task_id: str = ""
    benchmark: str = ""
    system: str = "sirchmunk"
    seed: int = 42
    cache_mode: str = "cold"
    stage: str = "frozen"
    status: str = "unknown"
    protocol_hash: str = ""
    git_commit: str = ""
    config_hash: str = ""
    artifact_dir: str = ""
    results_path: str = ""
    metrics_path: str = ""
    checkpoint_path: str = ""
    started_at: str = ""
    ended_at: str = ""
    updated_at: str = field(default_factory=now_local_iso)
    metrics: Dict[str, Any] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)
    error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentRegistryRecord":
        payload = dict(data)
        allowed = set(cls.__dataclass_fields__)  # type: ignore[attr-defined]
        payload = {key: value for key, value in payload.items() if key in allowed}
        return cls(**payload)


class ExperimentRegistry:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: ExperimentRegistryRecord) -> None:
        record.updated_at = now_local_iso()
        with self.path.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")

    def record_run(
        self,
        *,
        task_id: str,
        status: str,
        meta: Dict[str, Any],
        metrics: Optional[Dict[str, Any]] = None,
        error: str = "",
    ) -> ExperimentRegistryRecord:
        record = ExperimentRegistryRecord(
            run_id=str(meta.get("run_id", "")),
            task_id=task_id,
            benchmark=str(meta.get("benchmark", "")),
            system=str(meta.get("system", "sirchmunk")),
            seed=int(meta.get("seed", 42) or 42),
            cache_mode=str((meta.get("cache_report") or {}).get("mode") or meta.get("cache_mode") or ""),
            stage=str(meta.get("stage", "")),
            status=status,
            protocol_hash=str(meta.get("protocol_hash", "")),
            git_commit=str(meta.get("git_commit", "")),
            config_hash=str(meta.get("config_hash", "")),
            artifact_dir=str(meta.get("artifact_dir", "")),
            results_path=str(meta.get("results_path", "")),
            metrics_path=str(meta.get("metrics_path", "")),
            checkpoint_path=str(meta.get("checkpoint_path", "")),
            started_at=str(meta.get("task_started_at") or meta.get("started_at") or ""),
            ended_at=str(meta.get("task_ended_at") or meta.get("ended_at") or ""),
            metrics=metrics or {},
            meta=meta,
            error=error,
        )
        self.append(record)
        return record

    def latest(self) -> Dict[str, ExperimentRegistryRecord]:
        latest: Dict[str, ExperimentRegistryRecord] = {}
        for record in self.iter_records():
            key = record.run_id or record.task_id
            if key:
                latest[key] = record
        return latest

    def get(self, run_id: str) -> Optional[ExperimentRegistryRecord]:
        return self.latest().get(run_id)

    def list(
        self,
        *,
        benchmark: str = "",
        system: str = "",
        stage: str = "",
        status: str = "",
    ) -> List[ExperimentRegistryRecord]:
        rows = list(self.latest().values())
        if benchmark:
            rows = [row for row in rows if row.benchmark == benchmark]
        if system:
            rows = [row for row in rows if row.system == system]
        if stage:
            rows = [row for row in rows if row.stage == stage]
        if status:
            rows = [row for row in rows if row.status == status]
        return sorted(rows, key=lambda row: row.updated_at)

    def iter_records(self) -> Iterable[ExperimentRegistryRecord]:
        if not self.path.exists():
            return []
        rows: List[ExperimentRegistryRecord] = []
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
                    rows.append(ExperimentRegistryRecord.from_dict(data))
        return rows
