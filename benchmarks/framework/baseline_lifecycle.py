"""Baseline lifecycle orchestration for full-corpus feasibility studies."""
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional

from baselines.base_adapter import BaselineAdapter, BaselineSetupResult
from framework.lifecycle_schema import (
    BaselineIndexValidation,
    BaselineLifecycleRecord,
    BaselinePhase,
    FailureReason,
    ResourceBudget,
)
from framework.time_utils import now_local_iso


class BaselineLifecycleManager:
    """Run and record baseline build/index lifecycle phases.

    This manager deliberately stays outside ``UnifiedExperimentRunner`` so
    query evaluation remains separate from preprocessing feasibility studies.
    """

    def __init__(
        self,
        output_dir: str | Path,
        *,
        resource_budget: Optional[ResourceBudget] = None,
    ) -> None:
        self.output_dir = Path(output_dir).resolve()
        self.lifecycle_dir = self.output_dir / "lifecycle"
        self.lifecycle_dir.mkdir(parents=True, exist_ok=True)
        self.resource_budget = resource_budget or ResourceBudget()
        self.records_path = self.lifecycle_dir / "baseline_lifecycle.jsonl"

    async def run_build(
        self,
        baseline: BaselineAdapter,
        *,
        run_id: str,
        benchmark: str,
        corpus_manifest: Optional[Dict[str, Any]] = None,
        golden_set: Any = None,
        bm_adapter: Any = None,
        corpus_scale: str = "fullwiki",
    ) -> BaselineLifecycleRecord:
        """Run baseline preparation and index validation under a budget."""
        started = _now()
        t0 = time.monotonic()
        corpus_manifest = corpus_manifest or {}
        corpus_id = str(corpus_manifest.get("corpus_id") or corpus_manifest.get("id") or "")
        expected_docs = _safe_int(
            corpus_manifest.get("doc_count")
            or corpus_manifest.get("total_documents")
            or corpus_manifest.get("num_docs")
            or 0
        )
        artifact_dir = str(self.lifecycle_dir / baseline.name)
        Path(artifact_dir).mkdir(parents=True, exist_ok=True)

        baseline_metadata = dict(baseline.extra_metadata())
        record = BaselineLifecycleRecord(
            run_id=run_id,
            benchmark=benchmark,
            baseline_name=baseline.name,
            citation_name=baseline.citation_name,
            corpus_id=corpus_id,
            corpus_scale=corpus_scale,
            corpus_size_docs=expected_docs,
            index_required=bool(baseline_metadata.get("index_required", True)),
            phase=BaselinePhase.PREPARING,
            artifact_dir=artifact_dir,
            started_at=started,
            resource_budget=self.resource_budget,
            metadata=baseline_metadata,
        )

        if not baseline.is_available():
            record.phase = BaselinePhase.SKIPPED
            record.failure_reason = FailureReason.UNAVAILABLE
            record.failure_message = "Baseline dependencies are unavailable."
            record.ended_at = _now()
            self.save_record(record)
            return record

        try:
            record.phase = BaselinePhase.INDEXING
            setup = await self._run_prepare_with_timeout(
                baseline,
                golden_set=golden_set,
                bm_adapter=bm_adapter,
            )
            self._apply_setup(record, setup)
            if (
                record.failure_reason != FailureReason.NONE
                or not record.build_completed
                or not record.index_ready
            ):
                record.phase = BaselinePhase.FAILED
                if record.failure_reason == FailureReason.NONE:
                    record.failure_reason = FailureReason.INDEX_VALIDATION_FAILED
                if not record.failure_message:
                    record.failure_message = "Baseline setup did not produce a query-ready index."
                record.query_eligible = False
                return record

            record.phase = BaselinePhase.VALIDATING
            validation = _coerce_validation(
                baseline.validate_index(corpus_manifest),
                expected_documents=expected_docs,
            )
            if validation.indexed_documents == 0 and record.indexed_documents:
                validation.indexed_documents = record.indexed_documents
            if validation.expected_documents == 0 and record.corpus_size_docs:
                validation.expected_documents = record.corpus_size_docs
            record.validation = validation
            record.index_ready = bool(validation.index_ready and baseline.is_index_ready())
            record.query_eligible = record.index_ready

            if validation.is_partial:
                record.phase = BaselinePhase.FAILED
                record.failure_reason = FailureReason.PARTIAL_INDEX
                record.failure_message = (
                    f"Indexed {validation.indexed_documents}/"
                    f"{validation.expected_documents} documents."
                )
            elif not record.index_ready:
                record.phase = BaselinePhase.FAILED
                record.failure_reason = FailureReason.INDEX_VALIDATION_FAILED
                record.failure_message = "; ".join(validation.validation_errors) or "Index is not query-ready."
            else:
                record.phase = BaselinePhase.READY
                record.failure_reason = FailureReason.NONE
                record.failure_message = ""
        except asyncio.TimeoutError as exc:
            self._mark_failed(record, FailureReason.TIMEOUT, exc)
        except Exception as exc:  # baseline-specific crashes are formal lifecycle outcomes
            reason = _coerce_failure_reason(baseline.classify_failure(exc))
            self._mark_failed(record, reason, exc)
        finally:
            record.build_time_seconds = record.build_time_seconds or (time.monotonic() - t0)
            record.ended_at = _now()
            record.metadata.update(baseline.get_lifecycle_metadata())
            self.save_record(record)

        return record

    async def _run_prepare_with_timeout(
        self,
        baseline: BaselineAdapter,
        *,
        golden_set: Any,
        bm_adapter: Any,
    ) -> BaselineSetupResult:
        timeout = max(float(self.resource_budget.wall_clock_seconds or 0.0), 0.0)
        coro = baseline.prepare(golden_set=golden_set, bm_adapter=bm_adapter)
        if timeout > 0:
            return await asyncio.wait_for(coro, timeout=timeout)
        return await coro

    @staticmethod
    def _apply_setup(record: BaselineLifecycleRecord, setup: BaselineSetupResult) -> None:
        record.build_completed = bool(getattr(setup, "build_completed", True))
        record.index_ready = bool(getattr(setup, "index_ready", True))
        record.query_eligible = record.index_ready
        record.rebuild_required = bool(getattr(setup, "rebuild_required", False))
        record.query_ready_immediately = bool(getattr(setup, "query_ready_immediately", False))
        record.partial_index = bool(getattr(setup, "partial_index", False))
        record.build_time_seconds = float(getattr(setup, "setup_seconds", 0.0) or 0.0)
        record.preprocessing_seconds = float(getattr(setup, "preprocessing_seconds", 0.0) or 0.0)
        record.index_build_seconds = float(getattr(setup, "index_build_seconds", 0.0) or 0.0)
        record.disk_bytes = int(getattr(setup, "storage_bytes", 0) or 0)
        record.indexed_documents = int(getattr(setup, "indexed_documents", 0) or 0)
        record.corpus_size_docs = int(getattr(setup, "expected_documents", 0) or record.corpus_size_docs)
        record.peak_ram_bytes = int(getattr(setup, "peak_ram_bytes", 0) or 0)
        record.preprocess_llm_tokens = int(getattr(setup, "preprocess_llm_tokens", 0) or 0)
        record.api_cost_usd = float(getattr(setup, "api_cost_usd", 0.0) or 0.0)
        record.artifact_dir = str(getattr(setup, "artifact_dir", "") or record.artifact_dir)
        setup_failure = str(getattr(setup, "failure_reason", "none") or "none")
        if setup_failure != "none":
            record.failure_reason = _coerce_failure_reason(setup_failure)
            record.failure_message = str(getattr(setup, "failure_message", "") or "")
        metadata = getattr(setup, "metadata", None)
        if isinstance(metadata, dict):
            record.metadata.update(metadata)

    @staticmethod
    def _mark_failed(record: BaselineLifecycleRecord, reason: FailureReason, exc: Exception) -> None:
        record.phase = BaselinePhase.FAILED
        record.build_completed = False
        record.index_ready = False
        record.query_eligible = False
        record.failure_reason = reason
        record.failure_message = str(exc)

    def save_record(self, record: BaselineLifecycleRecord) -> str:
        """Append one lifecycle record and persist a latest JSON snapshot."""
        self.lifecycle_dir.mkdir(parents=True, exist_ok=True)
        payload = record.to_dict()
        with self.records_path.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(payload, ensure_ascii=False) + "\n")
        latest_path = self.lifecycle_dir / f"{record.baseline_name}_latest.json"
        latest_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return str(self.records_path)


def _coerce_validation(value: Any, *, expected_documents: int = 0) -> BaselineIndexValidation:
    if isinstance(value, BaselineIndexValidation):
        if value.expected_documents == 0 and expected_documents:
            value.expected_documents = expected_documents
        return value
    if isinstance(value, dict):
        return BaselineIndexValidation(
            index_ready=bool(value.get("index_ready", True)),
            indexed_documents=_safe_int(value.get("indexed_documents", 0)),
            expected_documents=_safe_int(value.get("expected_documents", expected_documents)),
            validation_errors=list(value.get("validation_errors", []) or []),
            metadata={k: v for k, v in value.items() if k not in {"index_ready", "indexed_documents", "expected_documents", "validation_errors"}},
        )
    return BaselineIndexValidation(index_ready=bool(value), expected_documents=expected_documents)


def _coerce_failure_reason(value: Any) -> FailureReason:
    if isinstance(value, FailureReason):
        return value
    try:
        return FailureReason(str(value))
    except ValueError:
        return FailureReason.UNKNOWN


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return default


def _now() -> str:
    return now_local_iso()


__all__ = ["BaselineLifecycleManager"]
