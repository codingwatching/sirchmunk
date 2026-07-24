"""Experiment queue and executor for ResearchOps P3."""
from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .experiment_registry import ExperimentRegistry
from .guards import TimeoutGuard
from .registry import load_benchmark_adapter
from .runner import UnifiedExperimentRunner


class QueueTaskStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    FAILED = "FAILED"
    PARTIAL = "PARTIAL"
    SUCCESS = "SUCCESS"
    SKIPPED = "SKIPPED"
    CANCELLED = "CANCELLED"


_TERMINAL = {
    QueueTaskStatus.FAILED,
    QueueTaskStatus.PARTIAL,
    QueueTaskStatus.SUCCESS,
    QueueTaskStatus.SKIPPED,
    QueueTaskStatus.CANCELLED,
}


@dataclass
class QueueTask:
    task_id: str
    benchmark: str
    env_file: str
    system: str = "sirchmunk"
    seed: int = 42
    cache_mode: str = "cold"
    stage: str = "frozen"
    limit: int = 0
    priority: int = 100
    run_id: str = ""
    kind: str = "sirchmunk"
    status: QueueTaskStatus = QueueTaskStatus.PENDING
    dependencies: List[str] = field(default_factory=list)
    config_overrides: Dict[str, Any] = field(default_factory=dict)
    command: List[str] = field(default_factory=list)
    attempts: int = 0
    max_attempts: int = 1
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    started_at: str = ""
    ended_at: str = ""
    artifact_dir: str = ""
    results_path: str = ""
    metrics_path: str = ""
    checkpoint_path: str = ""
    error: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_terminal(self) -> bool:
        return self.status in _TERMINAL

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["status"] = self.status.value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QueueTask":
        payload = dict(data)
        payload["status"] = QueueTaskStatus(payload.get("status", QueueTaskStatus.PENDING))
        allowed = set(cls.__dataclass_fields__)  # type: ignore[attr-defined]
        payload = {key: value for key, value in payload.items() if key in allowed}
        return cls(**payload)


class ExperimentQueue:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def add_task(self, task: QueueTask, *, replace: bool = False) -> QueueTask:
        tasks = self._load_map()
        if task.task_id in tasks and not replace:
            return tasks[task.task_id]
        task.updated_at = _now()
        tasks[task.task_id] = task
        self._save_map(tasks)
        return task

    def add_matrix(
        self,
        *,
        benchmarks: Dict[str, str],
        systems: Sequence[str],
        seeds: Sequence[int],
        cache_modes: Sequence[str],
        stage: str = "frozen",
        limit: int = 0,
        priority: int = 100,
        max_attempts: int = 1,
        config_overrides: Optional[Dict[str, Any]] = None,
        commands_by_system: Optional[Dict[str, List[str]]] = None,
        replace: bool = False,
    ) -> List[QueueTask]:
        created: List[QueueTask] = []
        if stage == "frozen":
            invalid_cache_modes = [str(mode) for mode in cache_modes if str(mode).lower() not in {"cold", "compiled"}]
            if invalid_cache_modes:
                raise ValueError(
                    "Frozen queue tasks must use cache_modes cold or compiled; "
                    f"invalid={invalid_cache_modes}"
                )
        command_map = commands_by_system or {}
        for benchmark, env_file in benchmarks.items():
            for system in systems:
                for seed in seeds:
                    for cache_mode in cache_modes:
                        command = list(command_map.get(system, []))
                        task = QueueTask(
                            task_id=_task_id(benchmark, system, seed, cache_mode, stage, limit),
                            benchmark=benchmark,
                            env_file=env_file,
                            system=system,
                            seed=int(seed),
                            cache_mode=cache_mode,
                            stage=stage,
                            limit=int(limit),
                            priority=int(priority),
                            run_id=_run_id(benchmark, system, seed, cache_mode, stage),
                            kind="sirchmunk" if system.lower() == "sirchmunk" and not command else "external",
                            config_overrides=dict(config_overrides or {}),
                            command=command,
                            max_attempts=max(int(max_attempts or 1), 1),
                        )
                        created.append(self.add_task(task, replace=replace))
        return created

    def list(self, *, status: str = "") -> List[QueueTask]:
        rows = list(self._load_map().values())
        if status:
            rows = [row for row in rows if row.status.value == status.upper()]
        return sorted(rows, key=lambda row: (row.priority, row.created_at, row.task_id))

    def get(self, task_id: str) -> Optional[QueueTask]:
        return self._load_map().get(task_id)

    def update(self, task: QueueTask) -> QueueTask:
        tasks = self._load_map()
        task.updated_at = _now()
        tasks[task.task_id] = task
        self._save_map(tasks)
        return task

    def cancel(self, task_id: str) -> bool:
        task = self.get(task_id)
        if task is None or task.is_terminal:
            return False
        task.status = QueueTaskStatus.CANCELLED
        task.ended_at = _now()
        self.update(task)
        return True

    def ready(self) -> List[QueueTask]:
        tasks = self._load_map()
        successful = {tid for tid, task in tasks.items() if task.status == QueueTaskStatus.SUCCESS}
        ready = []
        for task in tasks.values():
            if task.status != QueueTaskStatus.PENDING:
                continue
            if all(dep in successful for dep in task.dependencies):
                ready.append(task)
        return sorted(ready, key=lambda row: (row.priority, row.created_at, row.task_id))

    def summary(self) -> Dict[str, int]:
        counts = {status.value: 0 for status in QueueTaskStatus}
        for task in self._load_map().values():
            counts[task.status.value] = counts.get(task.status.value, 0) + 1
        counts["TOTAL"] = sum(counts.values())
        return counts

    def _load_map(self) -> Dict[str, QueueTask]:
        if not self.path.exists():
            return {}
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
        rows = raw.get("tasks", []) if isinstance(raw, dict) else []
        out: Dict[str, QueueTask] = {}
        for item in rows:
            if isinstance(item, dict):
                task = QueueTask.from_dict(item)
                out[task.task_id] = task
        return out

    def _save_map(self, tasks: Dict[str, QueueTask]) -> None:
        payload = {
            "version": 1,
            "updated_at": _now(),
            "summary": self._summary_from_map(tasks),
            "tasks": [task.to_dict() for task in sorted(tasks.values(), key=lambda row: (row.priority, row.created_at, row.task_id))],
        }
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.path)

    @staticmethod
    def _summary_from_map(tasks: Dict[str, QueueTask]) -> Dict[str, int]:
        counts = {status.value: 0 for status in QueueTaskStatus}
        for task in tasks.values():
            counts[task.status.value] = counts.get(task.status.value, 0) + 1
        counts["TOTAL"] = sum(counts.values())
        return counts


class QueueExecutor:
    def __init__(
        self,
        queue: ExperimentQueue,
        *,
        registry: Optional[ExperimentRegistry] = None,
        max_concurrent: int = 1,
        resume: bool = True,
    ) -> None:
        self.queue = queue
        self.registry = registry
        self.max_concurrent = max(int(max_concurrent or 1), 1)
        self.resume = resume

    async def run_pending(self, *, max_tasks: int = 0) -> List[QueueTask]:
        ready = self.queue.ready()
        if max_tasks:
            ready = ready[:max_tasks]
        semaphore = asyncio.Semaphore(self.max_concurrent)
        results: List[QueueTask] = []

        async def _run(task: QueueTask) -> None:
            async with semaphore:
                results.append(await self.run_task(task))

        await asyncio.gather(*[asyncio.create_task(_run(task)) for task in ready])
        return results

    async def run_task(self, task: QueueTask) -> QueueTask:
        task.status = QueueTaskStatus.RUNNING
        task.attempts += 1
        task.started_at = task.started_at or _now()
        task.updated_at = _now()
        task.error = ""
        self.queue.update(task)
        try:
            if task.kind == "external" and task.command:
                meta = await self._run_external(task)
                task.meta = meta
                task.status = QueueTaskStatus.SUCCESS if meta.get("returncode") == 0 else QueueTaskStatus.FAILED
                task.ended_at = _now()
                meta["task_started_at"] = task.started_at
                meta["task_ended_at"] = task.ended_at
                if task.status == QueueTaskStatus.FAILED:
                    task.error = _external_error(meta)
                    if task.attempts < task.max_attempts:
                        task.status = QueueTaskStatus.PENDING
                if self.registry:
                    self.registry.record_run(
                        task_id=task.task_id,
                        status=task.status.value,
                        meta=meta,
                        metrics={},
                        error=task.error,
                    )
            elif task.kind == "external":
                raise RuntimeError(
                    f"external system '{task.system}' requires an explicit command; "
                    "use sirchmunk system for UnifiedExperimentRunner tasks"
                )
            else:
                meta, metrics = await self._run_sirchmunk(task)
                task.meta = meta
                task.artifact_dir = str(meta.get("artifact_dir", ""))
                task.results_path = str(meta.get("results_path", ""))
                task.metrics_path = str(meta.get("metrics_path", ""))
                task.checkpoint_path = str(meta.get("checkpoint_path", ""))
                failed = int(((metrics or {}).get("checkpoint") or {}).get("failed", 0) or 0)
                task.status = QueueTaskStatus.PARTIAL if failed else QueueTaskStatus.SUCCESS
                task.ended_at = _now()
                meta["task_started_at"] = task.started_at
                meta["task_ended_at"] = task.ended_at
                if self.registry:
                    self.registry.record_run(
                        task_id=task.task_id,
                        status=task.status.value,
                        meta=meta,
                        metrics=metrics,
                    )
        except Exception as exc:
            task.error = str(exc)
            task.status = QueueTaskStatus.PENDING if task.attempts < task.max_attempts else QueueTaskStatus.FAILED
            if self.registry:
                failure_meta = {
                    "run_id": task.run_id or task.task_id,
                    "benchmark": task.benchmark,
                    "system": task.system,
                    "seed": task.seed,
                    "cache_mode": task.cache_mode,
                    "stage": task.stage,
                    "task_started_at": task.started_at,
                    "task_ended_at": _now(),
                }
                self.registry.record_run(
                    task_id=task.task_id,
                    status=task.status.value,
                    meta=failure_meta,
                    error=task.error,
                )
        finally:
            if task.status in _TERMINAL:
                task.ended_at = _now()
            task.updated_at = _now()
            self.queue.update(task)
        return task

    async def _run_sirchmunk(self, task: QueueTask) -> tuple[Dict[str, Any], Dict[str, Any]]:
        adapter = load_benchmark_adapter(task.benchmark, task.env_file)
        overrides = dict(task.config_overrides)
        overrides.update({
            "cache_mode": task.cache_mode,
            "stage": task.stage,
            "frozen_evaluation": task.stage == "frozen",
        })
        runner = UnifiedExperimentRunner(adapter)
        run_coro = runner.run(
            limit=task.limit,
            seed=task.seed,
            run_id=task.run_id or task.task_id,
            resume=self.resume,
            config_overrides=overrides,
            stage=task.stage,
            system_name=task.system,
        )
        timeout_seconds = _safe_float(overrides.get("system_timeout_seconds") or overrides.get("SYSTEM_TIMEOUT_SECONDS"))
        if timeout_seconds:
            results, meta = await TimeoutGuard().run_system(run_coro, timeout_seconds)
        else:
            results, meta = await run_coro
        meta["seed"] = task.seed
        meta["cache_mode"] = task.cache_mode
        meta["task_id"] = task.task_id
        metrics = _read_json(meta.get("metrics_path", ""))
        metrics.setdefault("n", len(results))
        return meta, metrics

    @staticmethod
    async def _run_external(task: QueueTask) -> Dict[str, Any]:
        proc = await asyncio.create_subprocess_exec(
            *task.command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        return {
            "run_id": task.run_id or task.task_id,
            "benchmark": task.benchmark,
            "system": task.system,
            "seed": task.seed,
            "cache_mode": task.cache_mode,
            "stage": task.stage,
            "returncode": proc.returncode,
            "stdout_tail": stdout.decode("utf-8", errors="replace")[-4000:],
            "stderr_tail": stderr.decode("utf-8", errors="replace")[-4000:],
        }


def _external_error(meta: Dict[str, Any]) -> str:
    returncode = meta.get("returncode")
    stderr_tail = str(meta.get("stderr_tail", "")).strip()
    stdout_tail = str(meta.get("stdout_tail", "")).strip()
    detail = stderr_tail or stdout_tail
    if detail:
        return f"external command failed with returncode={returncode}: {detail[:500]}"
    return f"external command failed with returncode={returncode}"


def _read_json(path: str | Path) -> Dict[str, Any]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        return {}


def _task_id(benchmark: str, system: str, seed: int, cache_mode: str, stage: str, limit: int) -> str:
    raw = f"{benchmark}|{system}|{seed}|{cache_mode}|{stage}|{limit}"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]
    return f"{benchmark}_{system}_{seed}_{cache_mode}_{stage}_{digest}".replace("/", "_")


def _run_id(benchmark: str, system: str, seed: int, cache_mode: str, stage: str) -> str:
    safe_system = system.replace("/", "_").replace(":", "_")
    return f"{benchmark}_{safe_system}_seed{seed}_{cache_mode}_{stage}"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0
