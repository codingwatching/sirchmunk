"""Retry policy for transient ResearchOps failures."""
from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Iterable, List, Optional


_RETRYABLE_MARKERS = (
    "timeout",
    "timed out",
    "temporarily",
    "rate limit",
    "429",
    "connection reset",
    "connection aborted",
    "server disconnected",
    "bad gateway",
    "service unavailable",
)


@dataclass
class RetryConfig:
    max_attempts: int = 3
    base_delay_seconds: float = 0.5
    max_delay_seconds: float = 8.0
    jitter_seconds: float = 0.2
    retryable_markers: Iterable[str] = field(default_factory=lambda: _RETRYABLE_MARKERS)

    def __post_init__(self) -> None:
        self.max_attempts = max(int(self.max_attempts or 1), 1)
        self.base_delay_seconds = max(float(self.base_delay_seconds or 0.0), 0.0)
        self.max_delay_seconds = max(float(self.max_delay_seconds or 0.0), self.base_delay_seconds)
        self.jitter_seconds = max(float(self.jitter_seconds or 0.0), 0.0)
        markers = self.retryable_markers
        if isinstance(markers, str):
            markers = [marker.strip() for marker in markers.split(",")]
        self.retryable_markers = tuple(
            str(marker).lower()
            for marker in (markers or ())
            if str(marker).strip()
        )


@dataclass
class RetryResult:
    value: Any
    attempts: int
    retried: bool = False
    last_error: str = ""
    errors: List[str] = field(default_factory=list)


class RetryExhausted(RuntimeError):
    """Raised when retryable exceptions keep failing until attempts are exhausted."""

    def __init__(
        self,
        *,
        attempts: int,
        last_error: str,
        errors: Iterable[str],
        cause: Optional[BaseException] = None,
    ) -> None:
        self.attempts = attempts
        self.last_error = last_error
        self.errors = list(errors)
        self.cause = cause
        super().__init__(f"retry exhausted after {attempts} attempt(s): {last_error}")


class RetryPolicy:
    def __init__(self, config: Optional[RetryConfig] = None) -> None:
        self.config = config or RetryConfig()

    async def run(
        self,
        operation: Callable[[], Awaitable[Any]],
        *,
        is_retryable_result: Optional[Callable[[Any], bool]] = None,
    ) -> RetryResult:
        attempts = 0
        errors: List[str] = []
        max_attempts = self.config.max_attempts
        while attempts < max_attempts:
            attempts += 1
            try:
                value = await operation()
            except Exception as exc:
                last_error = str(exc) or exc.__class__.__name__
                errors.append(last_error)
                retryable = self.is_retryable_error(last_error)
                if attempts >= max_attempts and retryable:
                    raise RetryExhausted(
                        attempts=attempts,
                        last_error=last_error,
                        errors=errors,
                        cause=exc,
                    ) from exc
                if attempts >= max_attempts or not retryable:
                    raise
                await self._sleep(attempts)
                continue

            if is_retryable_result and is_retryable_result(value):
                last_error = _result_error(value)
                errors.append(last_error)
                if attempts >= max_attempts:
                    return RetryResult(
                        value=value,
                        attempts=attempts,
                        retried=attempts > 1,
                        last_error=last_error,
                        errors=errors,
                    )
                await self._sleep(attempts)
                continue

            return RetryResult(
                value=value,
                attempts=attempts,
                retried=attempts > 1,
                last_error=errors[-1] if errors else "",
                errors=errors,
            )
        raise RetryExhausted(
            attempts=attempts,
            last_error=errors[-1] if errors else "unknown retry exhaustion",
            errors=errors,
        )

    def is_retryable_result(self, result: Any) -> bool:
        error = _result_error(result)
        return bool(error) and self.is_retryable_error(error)

    def is_retryable_error(self, error: str) -> bool:
        lower = (error or "").lower()
        return any(marker.lower() in lower for marker in self.config.retryable_markers)

    async def _sleep(self, attempts: int) -> None:
        delay = self._delay(attempts)
        if delay > 0:
            await asyncio.sleep(delay)

    def _delay(self, attempts: int) -> float:
        delay = min(self.config.max_delay_seconds, self.config.base_delay_seconds * (2 ** max(attempts - 1, 0)))
        if self.config.jitter_seconds > 0:
            delay += random.random() * self.config.jitter_seconds
        return delay


def is_retryable_prediction_result(result: Any) -> bool:
    return RetryPolicy().is_retryable_result(result)


def _result_error(result: Any) -> str:
    if result is None:
        return "none result"
    error = getattr(result, "error", "") or ""
    telemetry = getattr(result, "telemetry", {}) or {}
    if not error and isinstance(telemetry, dict):
        error = str(telemetry.get("error") or telemetry.get("error_type") or "")
    return str(error)
