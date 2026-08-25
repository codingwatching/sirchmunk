"""Budget and timeout guards for ResearchOps runs."""
from __future__ import annotations

import asyncio
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Dict, Iterable


class BudgetExceeded(RuntimeError):
    pass


class SampleTimeout(RuntimeError):
    pass


class SystemTimeout(RuntimeError):
    pass


class BenchmarkTimeout(RuntimeError):
    pass


class GlobalTimeout(RuntimeError):
    pass


@dataclass
class GuardConfig:
    max_runtime_seconds: float = 0.0
    max_total_tokens: int = 0
    max_api_cost_usd: float = 0.0
    max_disk_usage_bytes: int = 0
    min_free_disk_bytes: int = 0
    sample_timeout_seconds: float = 0.0
    system_timeout_seconds: float = 0.0
    benchmark_timeout_seconds: float = 0.0
    global_timeout_seconds: float = 0.0

    @classmethod
    def from_run_config(cls, config: Dict[str, Any]) -> "GuardConfig":
        return cls(
            max_runtime_seconds=_float_config(config, "max_runtime_seconds", "MAX_RUNTIME_SECONDS"),
            max_total_tokens=_int_config(config, "max_total_tokens", "MAX_TOTAL_TOKENS"),
            max_api_cost_usd=_float_config(config, "max_api_cost_usd", "MAX_API_COST_USD"),
            max_disk_usage_bytes=_int_config(config, "max_disk_usage_bytes", "MAX_DISK_USAGE_BYTES"),
            min_free_disk_bytes=_int_config(config, "min_free_disk_bytes", "MIN_FREE_DISK_BYTES"),
            sample_timeout_seconds=_float_config(config, "sample_timeout_seconds", "SAMPLE_TIMEOUT_SECONDS"),
            system_timeout_seconds=_float_config(config, "system_timeout_seconds", "SYSTEM_TIMEOUT_SECONDS"),
            benchmark_timeout_seconds=_float_config(config, "benchmark_timeout_seconds", "BENCHMARK_TIMEOUT_SECONDS"),
            global_timeout_seconds=_float_config(config, "global_timeout_seconds", "GLOBAL_TIMEOUT_SECONDS"),
        )


class BudgetGuard:
    def __init__(self, config: GuardConfig, *, output_dir: str | Path) -> None:
        self.config = config
        self.output_dir = Path(output_dir)
        self.started = time.monotonic()

    def check_before_sample(self, results: Iterable[Any]) -> Dict[str, Any]:
        usage = self.current_usage(results)
        self._check_runtime(usage["elapsed_seconds"])
        self._check_tokens(usage["total_tokens"])
        self._check_api_cost(usage["api_cost_usd"])
        self._check_disk_usage(usage["disk_usage_bytes"])
        self._check_free_disk(usage["free_disk_bytes"])
        return usage

    def current_usage(self, results: Iterable[Any]) -> Dict[str, Any]:
        return {
            "elapsed_seconds": time.monotonic() - self.started,
            "total_tokens": self.consumed_tokens(results),
            "api_cost_usd": self.consumed_api_cost(results),
            "disk_usage_bytes": _path_size(self.output_dir) if self.config.max_disk_usage_bytes else 0,
            "free_disk_bytes": _free_disk_bytes(self.output_dir) if self.config.min_free_disk_bytes else 0,
        }

    @staticmethod
    def consumed_tokens(results: Iterable[Any]) -> int:
        total = 0
        for result in results:
            telemetry = getattr(result, "telemetry", {}) or {}
            if not isinstance(telemetry, dict):
                continue
            total += _safe_int(telemetry.get("total_tokens", 0))
            total += _safe_int(telemetry.get("judge_tokens", 0))
        return total

    @staticmethod
    def consumed_api_cost(results: Iterable[Any]) -> float:
        total = 0.0
        for result in results:
            telemetry = getattr(result, "telemetry", {}) or {}
            if not isinstance(telemetry, dict):
                continue
            total += _safe_float(
                telemetry.get("api_cost_usd")
                or telemetry.get("cost_usd")
                or telemetry.get("estimated_cost_usd")
                or 0.0
            )
        return total

    def _check_runtime(self, elapsed: float) -> None:
        limit = self.config.max_runtime_seconds
        if limit and elapsed > limit:
            raise BudgetExceeded(f"max_runtime_seconds exceeded: elapsed={elapsed:.2f}>{limit}")

    def _check_tokens(self, total: int) -> None:
        limit = self.config.max_total_tokens
        if limit and total >= limit:
            raise BudgetExceeded(f"max_total_tokens exceeded: {total}>={limit}")

    def _check_api_cost(self, total: float) -> None:
        limit = self.config.max_api_cost_usd
        if limit and total >= limit:
            raise BudgetExceeded(f"max_api_cost_usd exceeded: {total:.6f}>={limit}")

    def _check_disk_usage(self, used: int) -> None:
        limit = self.config.max_disk_usage_bytes
        if limit and used >= limit:
            raise BudgetExceeded(f"max_disk_usage_bytes exceeded: {used}>={limit}")

    def _check_free_disk(self, free_bytes: int) -> None:
        minimum = self.config.min_free_disk_bytes
        if minimum and free_bytes < minimum:
            raise BudgetExceeded(f"min_free_disk_bytes not satisfied: free={free_bytes}<required={minimum}")


class TimeoutGuard:
    def __init__(self, sample_timeout_seconds: float = 0.0) -> None:
        self.sample_timeout_seconds = max(_safe_float(sample_timeout_seconds), 0.0)

    async def run_sample(self, awaitable: Awaitable[Any]) -> Any:
        return await self._run_with_timeout(
            awaitable,
            timeout_seconds=self.sample_timeout_seconds,
            exception_type=SampleTimeout,
            label="sample",
        )

    async def run_system(self, awaitable: Awaitable[Any], timeout_seconds: float) -> Any:
        return await self._run_with_timeout(
            awaitable,
            timeout_seconds=timeout_seconds,
            exception_type=SystemTimeout,
            label="system",
        )

    async def run_benchmark(self, awaitable: Awaitable[Any], timeout_seconds: float) -> Any:
        return await self._run_with_timeout(
            awaitable,
            timeout_seconds=timeout_seconds,
            exception_type=BenchmarkTimeout,
            label="benchmark",
        )

    async def run_global(self, awaitable: Awaitable[Any], timeout_seconds: float) -> Any:
        return await self._run_with_timeout(
            awaitable,
            timeout_seconds=timeout_seconds,
            exception_type=GlobalTimeout,
            label="global",
        )

    @staticmethod
    async def _run_with_timeout(
        awaitable: Awaitable[Any],
        *,
        timeout_seconds: float,
        exception_type: type[RuntimeError],
        label: str,
    ) -> Any:
        timeout_seconds = max(_safe_float(timeout_seconds), 0.0)
        if timeout_seconds <= 0:
            return await awaitable
        try:
            return await asyncio.wait_for(awaitable, timeout=timeout_seconds)
        except asyncio.TimeoutError as exc:
            raise exception_type(f"{label} timeout after {timeout_seconds}s") from exc


def _path_size(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        try:
            return path.stat().st_size
        except OSError:
            return 0
    total = 0
    for child in path.rglob("*"):
        if child.is_file():
            try:
                total += child.stat().st_size
            except OSError:
                pass
    return total


def _free_disk_bytes(path: Path) -> int:
    return shutil.disk_usage(str(_existing_path(path))).free


def _existing_path(path: Path) -> Path:
    current = path if path.exists() else path.parent
    while not current.exists() and current != current.parent:
        current = current.parent
    return current


def _config_value(config: Dict[str, Any], *keys: str, default: Any = 0) -> Any:
    for key in keys:
        value = config.get(key)
        if value is not None and value != "":
            return value
    return default


def _float_config(config: Dict[str, Any], *keys: str) -> float:
    return max(_safe_float(_config_value(config, *keys)), 0.0)


def _int_config(config: Dict[str, Any], *keys: str) -> int:
    return max(_safe_int(_config_value(config, *keys)), 0)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default
