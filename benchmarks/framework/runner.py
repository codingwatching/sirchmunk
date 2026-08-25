"""framework/runner.py — UnifiedExperimentRunner

Unified asynchronous experiment executor, decoupled from any concrete benchmark through
the BenchmarkAdapter interface.
Each run() returns List[PredictionResult] and writes the raw results to output/ JSONL.
It also collects git_commit + config_hash to keep runs reproducible.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .adapter import BenchmarkAdapter
from .artifact import RunArtifactManager
from .cache_manager import CacheManager
from .checkpoint import CheckpointManager, rows_to_prediction_results
from .guards import (
    BenchmarkTimeout,
    BudgetExceeded,
    BudgetGuard,
    GuardConfig,
    SampleTimeout,
    TimeoutGuard,
)
from .protocol import ProtocolValidator, default_protocol
from .retry import RetryConfig, RetryExhausted, RetryPolicy
from .run_state import RunState, RunStateStore, RunStatus
from .schema import PredictionResult
from .time_utils import local_timestamp, now_local_iso, now_utc_iso

logger = logging.getLogger(__name__)


def _get_git_commit() -> str:
    """Return the first 12 characters of the current HEAD commit hash, or 'unknown'."""
    try:
        import subprocess
        result = subprocess.run(
            ["git", "rev-parse", "--short=12", "HEAD"],
            capture_output=True, text=True, check=False,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "unknown"


def _config_hash(config: dict) -> str:
    """Compute the first 16 characters of the config dict SHA-256 to identify a config version."""
    try:
        serialized = json.dumps(config, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(serialized.encode()).hexdigest()[:16]
    except Exception:
        return "unknown"


def _sample_id_checksum(sample_ids: List[str]) -> str:
    try:
        serialized = json.dumps(sorted(str(sample_id) for sample_id in sample_ids), ensure_ascii=False)
        return hashlib.sha256(serialized.encode()).hexdigest()[:32]
    except Exception:
        return "unknown"


def _validate_stage_config(config: Dict[str, Any]) -> None:
    stage = str(config.get("stage") or "")
    if stage not in ("exploration", "frozen"):
        raise ValueError(f"Invalid experiment stage: {stage}. Expected exploration or frozen.")
    if stage != "frozen":
        return

    cache_mode = str(config.get("cache_mode") or config.get("CACHE_MODE") or "none").lower()
    if cache_mode not in {"cold", "compiled"}:
        raise ValueError("Frozen evaluation must use cache_mode='cold' or cache_mode='compiled'.")
    if _config_bool(config, "cache_dry_run", "CACHE_DRY_RUN", default=False):
        raise ValueError("Frozen evaluation cannot use cache dry-run mode.")
    if _config_bool(config, "enable_eval_feedback", "HOTPOT_ENABLE_EVAL_FEEDBACK", default=False):
        raise ValueError("Frozen evaluation must disable eval feedback to avoid test-set tuning.")

    memory_enabled = _config_bool(config, "enable_memory", "SIRCHMUNK_ENABLE_MEMORY", default=False)
    if memory_enabled:
        memory_state = _config_value(
            config,
            "memory_state_version",
            "fixed_memory_state_version",
            "SIRCHMUNK_MEMORY_STATE_VERSION",
            default="",
        )
        if not str(memory_state).strip():
            raise ValueError("Frozen evaluation with memory enabled must record a fixed memory_state_version.")
        if not _config_bool(config, "memory_read_only", "frozen_memory_read_only", "SIRCHMUNK_MEMORY_READ_ONLY", default=False):
            raise ValueError("Frozen evaluation with memory enabled must mark memory as read-only.")
        if _config_bool(config, "enable_memory_updates", "memory_write_enabled", "SIRCHMUNK_ENABLE_MEMORY_UPDATES", default=False):
            raise ValueError("Frozen evaluation cannot enable adaptive memory updates.")

    if _config_bool(config, "enable_llm_judge", "HOTPOT_ENABLE_LLM_JUDGE", default=False) and not _config_bool(
        config,
        "allow_frozen_llm_judge_auxiliary",
        "ALLOW_FROZEN_LLM_JUDGE_AUXILIARY",
        default=False,
    ):
        raise ValueError("Frozen evaluation cannot enable LLM judge unless allow_frozen_llm_judge_auxiliary=true.")


class UnifiedExperimentRunner:
    """Unified experiment executor.

    Design principles:
    - Depends only on the BenchmarkAdapter interface; never imports concrete benchmark code
    - Per-question execution is fully obtained through the adapter build_searcher /
      build_judge
    - Results stream to output/ JSONL so a crash cannot lose them
    - Collects and returns experiment metadata (git_commit, config_hash, run_id)
    """

    def __init__(self, adapter: BenchmarkAdapter) -> None:
        self._adapter = adapter

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(
        self,
        limit: int = 0,
        seed: int = 42,
        run_id: Optional[str] = None,
        resume: bool = True,
        config_overrides: Optional[Dict[str, Any]] = None,
        stage: str = "exploration",
        system_name: str = "sirchmunk",
    ) -> tuple[List[PredictionResult], dict]:
        """Run the full experiment.

        Args:
            limit: 0 means the full set; >0 samples randomly.
            seed:  random seed.
            run_id: externally supplied run_id; a timestamp ID is generated when None.
            resume: when True, rerunning the same run_id reads the checkpoint and skips
                finished samples.
            config_overrides: runtime overrides passed in by the P3 queue; the adapter and
                env files are left untouched.
            stage: exploration or frozen; separates tuning from frozen evaluation in
                artifacts and the registry.
            system_name: system name, defaults to sirchmunk.

        Returns:
            (results, meta) where meta holds run_id / git_commit / config_hash /
            results_path / timestamp.
        """
        adapter = self._adapter
        ts = local_timestamp()
        run_id = run_id or f"{adapter.name}_{ts}"

        git_commit = _get_git_commit()
        config = dict(adapter.get_run_config())
        if config_overrides:
            config.update(config_overrides)
        config["stage"] = stage
        config["system_name"] = system_name
        config.setdefault("frozen_evaluation", stage == "frozen")
        _validate_stage_config(config)

        # Load samples
        samples = adapter.load_samples(limit=limit, seed=seed)
        sample_ids = [sample.sample_id for sample in samples]
        config["sample_count"] = len(samples)
        config["sample_id_checksum"] = _sample_id_checksum(sample_ids)
        cfg_hash = _config_hash(config)

        logger.info("=" * 60)
        logger.info("[Runner] %s  run_id=%s", adapter.name.upper(), run_id)
        logger.info("[Runner] git=%s  config_hash=%s", git_commit, cfg_hash)
        logger.info("[Runner] %d samples loaded  sample_id_checksum=%s", len(samples), config["sample_id_checksum"])
        logger.info("=" * 60)

        # Validate the corpus
        found, missing = adapter.validate_corpus()
        logger.info("[Runner] Corpus: %d found, %d missing", found, len(missing))
        if missing:
            logger.warning("[Runner] Missing: %s", missing[:5])

        # Build searcher and judge as shared singletons to avoid re-initialization
        searcher = adapter.build_searcher()
        judge = adapter.build_judge()

        guard_config = GuardConfig.from_run_config(config)
        retry_config_kwargs: Dict[str, Any] = {
            "max_attempts": _config_int(config, "retry_max_attempts", "RETRY_MAX_ATTEMPTS", default=3),
            "base_delay_seconds": _config_float(config, "retry_base_delay_seconds", "RETRY_BASE_DELAY_SECONDS", default=0.5),
            "max_delay_seconds": _config_float(config, "retry_max_delay_seconds", "RETRY_MAX_DELAY_SECONDS", default=8.0),
            "jitter_seconds": _config_float(config, "retry_jitter_seconds", "RETRY_JITTER_SECONDS", default=0.2),
        }
        retry_markers = _retryable_markers_from_config(
            config.get("retryable_markers") or config.get("RETRYABLE_MARKERS")
        )
        if retry_markers:
            retry_config_kwargs["retryable_markers"] = retry_markers
        retry_policy = RetryPolicy(RetryConfig(**retry_config_kwargs))

        # Prepare the output paths and the ResearchOps artifact directory
        out_dir = Path(adapter.get_output_dir())
        out_dir.mkdir(parents=True, exist_ok=True)
        results_path = str(out_dir / f"results_{ts}.jsonl")
        artifact = RunArtifactManager(out_dir, run_id)
        artifact.create()
        checkpoint = CheckpointManager(artifact.checkpoints_dir / "samples.jsonl")
        state_store = RunStateStore(out_dir / "run_states.json")
        existing_state = state_store.get(run_id)
        if existing_state is not None and existing_state.status in (RunStatus.SUCCESS, RunStatus.REPORTED) and resume:
            completed_rows = checkpoint.load_completed_rows(sample_ids)
            if len(completed_rows) >= len(samples):
                logger.info("[Runner] Run %s already completed; loading %d rows from checkpoint", run_id, len(completed_rows))
                return rows_to_prediction_results(completed_rows), {
                    "run_id": run_id,
                    "benchmark": adapter.name,
                    "system": system_name,
                    "stage": stage,
                    "timestamp": now_local_iso(),
                    "timestamp_utc": now_utc_iso(),
                    "git_commit": git_commit,
                    "config_hash": cfg_hash,
                    "results_path": existing_state.results_path,
                    "artifact_dir": existing_state.artifact_dir or str(artifact.run_dir),
                    "protocol_path": str(artifact.run_dir / "protocol.yaml"),
                    "manifest_path": str(artifact.run_dir / "manifest.json"),
                    "metrics_path": str(artifact.metrics_path),
                    "checkpoint_path": str(checkpoint.path),
                    "state_path": str(state_store.path),
                    "total_samples": len(completed_rows),
                    "checkpoint_summary": checkpoint.summary(sample_ids),
                    "seed": seed,
                    "sample_id_checksum": config.get("sample_id_checksum", ""),
                    "cache_mode": str(config.get("cache_mode", "")),
                    "cache_report": _read_json_file(artifact.run_dir / "cache_report.json"),
                }
            logger.warning(
                "[Runner] Run %s was marked %s but checkpoint is incomplete; rerunning pending samples",
                run_id,
                existing_state.status.value,
            )
            existing_state.status = RunStatus.PENDING
            existing_state.ended_at = None
            state_store.upsert(existing_state)
        if existing_state is None:
            state_store.upsert(RunState(
                run_id=run_id,
                benchmark=adapter.name,
                status=RunStatus.PENDING,
                system=system_name,
                seed=seed,
                cache_mode=str(config.get("cache_mode", "declared_by_adapter")),
                total_samples=len(samples),
                checkpoint_path=str(checkpoint.path),
            ))
        state_store.transition(
            run_id,
            RunStatus.RUNNING,
            benchmark=adapter.name,
            system=system_name,
            total_samples=len(samples),
            checkpoint_path=str(checkpoint.path),
        )
        budget_guard = BudgetGuard(guard_config, output_dir=out_dir)
        timeout_guard = TimeoutGuard(guard_config.sample_timeout_seconds)
        cache_report = _prepare_cache(adapter, config)
        artifact.save_cache_report(cache_report)
        config["cache_report"] = cache_report

        protocol = _build_protocol(adapter, run_id=run_id, seed=seed, limit=limit, config=config)
        protocol_ok, protocol_errors = ProtocolValidator.validate(protocol)
        if not protocol_ok:
            raise ValueError("Invalid experiment protocol: " + "; ".join(protocol_errors))
        protocol_path = artifact.save_protocol(protocol)
        dataset_manifest = _safe_adapter_call(adapter, "get_dataset_manifest", default={})
        manifest_path = artifact.save_manifest(
            benchmark=adapter.name,
            git_commit=git_commit,
            config_hash=cfg_hash,
            config=config,
            dataset_manifest=dataset_manifest if isinstance(dataset_manifest, dict) else {},
            env_file=getattr(adapter, "env_file", ""),
        )

        # Concurrent execution
        semaphore = asyncio.Semaphore(adapter.get_max_concurrent())
        completed_rows = checkpoint.load_completed_rows(sample_ids) if resume else []
        results: List[PredictionResult] = rows_to_prediction_results(completed_rows)
        if completed_rows:
            with open(results_path, "a", encoding="utf-8") as fp:
                for row in completed_rows:
                    fp.write(json.dumps(row, ensure_ascii=False) + "\n")
            logger.info("[Runner] Resumed %d completed samples from checkpoint", len(completed_rows))
        completed_ids = checkpoint.completed_ids(sample_ids) if resume else set()
        pending_samples = [s for s in samples if s.sample_id not in completed_ids]
        completed = len(results)
        total = len(samples)

        async def _run_one(sample, idx):
            nonlocal completed
            async with semaphore:
                try:
                    budget_usage = budget_guard.check_before_sample(results)
                    retry_result = await retry_policy.run(
                        lambda: timeout_guard.run_sample(self._run_single(
                            sample=sample,
                            searcher=searcher,
                            judge=judge,
                            adapter=adapter,
                        )),
                        is_retryable_result=retry_policy.is_retryable_result,
                    )
                    result = retry_result.value
                    result.telemetry = result.telemetry or {}
                    result.telemetry["retry_attempts"] = retry_result.attempts
                    result.telemetry["retried"] = retry_result.retried
                    result.telemetry["budget_before_sample"] = budget_usage
                    if retry_result.last_error:
                        result.telemetry["last_retry_error"] = retry_result.last_error
                    if retry_result.errors:
                        result.telemetry["retry_errors"] = retry_result.errors
                    if result.error:
                        result.telemetry["retry_exhausted"] = (
                            retry_result.attempts >= retry_policy.config.max_attempts
                            and retry_policy.is_retryable_result(result)
                        )
                except RetryExhausted as exc:
                    result = self._error_result(
                        sample,
                        exc,
                        retry_attempts=exc.attempts,
                        retried=exc.attempts > 1,
                        retry_errors=exc.errors,
                        last_retry_error=exc.last_error,
                    )
                except BudgetExceeded as exc:
                    result = self._error_result(sample, exc, retry_attempts=0, retried=False)
                except SampleTimeout as exc:
                    result = self._error_result(sample, exc, retry_attempts=1, retried=False)
                except Exception as exc:
                    result = self._error_result(sample, exc, retry_attempts=1, retried=False)
                # Stream results in append mode so a crash cannot lose them
                raw_row = self._result_to_row(result, sample, adapter)
                per_sample_eval = self._result_to_per_sample_eval(result, sample)
                raw_row["per_sample_eval"] = per_sample_eval
                result.raw = raw_row
                with open(results_path, "a", encoding="utf-8") as fp:
                    fp.write(json.dumps(raw_row, ensure_ascii=False) + "\n")
                artifact.append_prediction(raw_row)
                artifact.append_per_sample_eval(per_sample_eval)
                attempts = max(_safe_int((result.telemetry or {}).get("retry_attempts"), default=1), 0)
                if result.error:
                    checkpoint.mark_failed(sample.sample_id, result.error, attempts=attempts, row=raw_row)
                else:
                    checkpoint.mark_completed(sample.sample_id, raw_row, attempts=attempts)
                results.append(result)

                completed += 1
                acc_tag = "✓" if result.judge_correct else "✗"
                cov_tag = "cov" if result.coverage else "no-cov"
                logger.info(
                    "[%d/%d] %s  [acc:%s] [%s]  %.1fs",
                    completed, total, result.sample_id,
                    acc_tag, cov_tag, result.elapsed,
                )

                # Inter-request delay
                delay = adapter.get_request_delay()
                if delay > 0:
                    await asyncio.sleep(delay)

                return result

        tasks = [asyncio.create_task(_run_one(s, i)) for i, s in enumerate(pending_samples)]
        if tasks:
            gather = asyncio.gather(*tasks)
            try:
                if guard_config.benchmark_timeout_seconds:
                    await timeout_guard.run_benchmark(gather, guard_config.benchmark_timeout_seconds)
                else:
                    await gather
            except BenchmarkTimeout as exc:
                logger.warning("[Runner] Benchmark timeout: %s", exc)
                await _cancel_pending_tasks(tasks)
                seen_ids = {r.sample_id for r in results}
                for sample in pending_samples:
                    if sample.sample_id in seen_ids:
                        continue
                    result = self._error_result(sample, exc, retry_attempts=0, retried=False)
                    raw_row = self._result_to_row(result, sample, adapter)
                    per_sample_eval = self._result_to_per_sample_eval(result, sample)
                    raw_row["per_sample_eval"] = per_sample_eval
                    result.raw = raw_row
                    with open(results_path, "a", encoding="utf-8") as fp:
                        fp.write(json.dumps(raw_row, ensure_ascii=False) + "\n")
                    artifact.append_prediction(raw_row)
                    artifact.append_per_sample_eval(per_sample_eval)
                    checkpoint.mark_failed(sample.sample_id, result.error or str(exc), attempts=0, row=raw_row)
                    results.append(result)
                    seen_ids.add(sample.sample_id)
        checkpoint.compact()

        checkpoint_summary = checkpoint.summary(sample_ids)
        metrics = _aggregate_runner_metrics(results)
        benchmark_metrics = _adapter_metrics(adapter, results)
        if benchmark_metrics:
            metrics["benchmark_specific"] = benchmark_metrics
            for key in (
                "official_exact_match",
                "official_f1_correct",
                "llm_assisted_accuracy",
                "llm_judge_usage_rate",
                "source_grounding_accuracy",
                "by_type",
                "by_level",
            ):
                if key in benchmark_metrics:
                    metrics[key] = benchmark_metrics[key]
        metrics["checkpoint"] = checkpoint_summary
        metrics["cache"] = cache_report
        self._save_analysis_artifacts(artifact, results, metrics)
        artifact.save_metrics(metrics)
        failed_count = sum(1 for r in results if r.error)
        retry_count = sum(
            max(_safe_int((r.telemetry or {}).get("retry_attempts", 1), default=1) - 1, 0)
            for r in results
        )
        final_status = RunStatus.SUCCESS if failed_count == 0 else RunStatus.PARTIAL
        state_store.transition(
            run_id,
            final_status,
            completed_samples=len(results) - failed_count,
            failed_samples=failed_count,
            retry_count=retry_count,
            artifact_dir=str(artifact.run_dir),
            results_path=results_path,
            checkpoint_path=str(checkpoint.path),
        )

        meta = {
            "run_id": run_id,
            "benchmark": adapter.name,
            "system": system_name,
            "stage": stage,
            "timestamp": now_local_iso(),
            "timestamp_utc": now_utc_iso(),
            "git_commit": git_commit,
            "config_hash": cfg_hash,
            "results_path": results_path,
            "artifact_dir": str(artifact.run_dir),
            "protocol_path": protocol_path,
            "manifest_path": manifest_path,
            "metrics_path": str(artifact.metrics_path),
            "checkpoint_path": str(checkpoint.path),
            "state_path": str(state_store.path),
            "total_samples": len(results),
            "checkpoint_summary": checkpoint_summary,
            "sample_id_checksum": config.get("sample_id_checksum", ""),
            "cache_report": cache_report,
        }
        logger.info("[Runner] Done. results_path=%s", results_path)
        return results, meta

    @classmethod
    def load_results_from_jsonl(cls, results_path: str) -> List[PredictionResult]:
        """Load a list of PredictionResult from an existing JSONL (--skip-run mode)."""
        results = []
        with open(results_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                results.append(PredictionResult(
                    sample_id=row.get("sample_id") or row.get("hotpot_id", ""),
                    prediction=row.get("prediction") or row.get("raw_prediction", ""),
                    judge_correct=bool(row.get("judge_correct", False)),
                    coverage=bool(row.get("coverage", False)),
                    elapsed=float(row.get("elapsed", 0.0)),
                    telemetry=row.get("telemetry", {}),
                    error=row.get("error"),
                    raw=row,
                ))
        return results

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _run_single(self, sample, searcher, judge, adapter) -> PredictionResult:
        """Run search plus evaluation for a single sample."""
        t0 = time.time()
        prediction = ""
        telemetry: dict = {}
        error: Optional[str] = None
        judge_correct = False
        coverage = False

        try:
            search_paths = adapter.get_search_paths(sample)
            search_kwargs = adapter.get_search_kwargs()

            result = await searcher.search(
                query=sample.question,
                paths=search_paths,
                return_context=True,
                **search_kwargs,
            )
            prediction = getattr(result, "answer", "") or str(result)

            read_files = list(getattr(result, "read_file_ids", None) or set())
            retrieval_logs = [
                log.to_dict() if hasattr(log, "to_dict") else dict(log)
                for log in (getattr(result, "retrieval_logs", None) or [])
            ]
            evidence_sources = _extract_evidence_sources(getattr(result, "cluster", None))
            evidence_snippets = _extract_evidence_snippets(getattr(result, "cluster", None))
            telemetry = {
                "read_file_ids": read_files,
                "retrieval_logs": retrieval_logs,
                "evidence_sources": evidence_sources,
                "evidence_snippets": evidence_snippets,
                "total_tokens": getattr(result, "total_llm_tokens", 0),
                "loop_count": getattr(result, "loop_count", 0),
                "llm_calls": len(getattr(result, "llm_usages", None) or []),
                "num_files_read": len(read_files),
                "search_mode": search_kwargs.get("mode", ""),
            }
            result_telemetry = getattr(result, "telemetry", None)
            if isinstance(result_telemetry, dict):
                telemetry.update(result_telemetry)
        except Exception as exc:
            error = str(exc)
            logger.error("[Runner] Error on %s: %s", sample.sample_id, error)

        elapsed = time.time() - t0

        # LLM judge evaluation
        if judge is not None:
            try:
                judge_result = await judge.judge(
                    prediction=prediction,
                    gold_answer=sample.gold_answer,
                    question=sample.question,
                )
                judge_correct = judge_result.get("equivalent", False)
                telemetry["judge_tokens"] = judge_result.get("tokens_used", 0)
                for key in (
                    "em",
                    "f1",
                    "official_em",
                    "official_f1",
                    "official_exact_match",
                    "official_f1_correct",
                    "confidence",
                    "reasoning",
                    "short_prediction",
                    "normalized_prediction",
                    "normalized_gold",
                    "llm_judge_used",
                    "llm_equivalent",
                ):
                    if key in judge_result:
                        telemetry[key] = judge_result[key]
            except Exception as e:
                logger.warning("[Runner] Judge failed for %s: %s", sample.sample_id, e)

            try:
                cov_result = await judge.judge_coverage(
                    prediction=prediction,
                    question=sample.question,
                )
                coverage = cov_result.get("has_coverage", False)
                telemetry["judge_tokens"] = (
                    telemetry.get("judge_tokens", 0)
                    + cov_result.get("tokens_used", 0)
                )
                if "confidence" in cov_result:
                    telemetry["coverage_confidence"] = cov_result["confidence"]
                if "reasoning" in cov_result:
                    telemetry["coverage_reasoning"] = cov_result["reasoning"]
            except Exception as e:
                logger.warning("[Runner] Coverage failed for %s: %s", sample.sample_id, e)

        try:
            enrich = getattr(adapter, "enrich_telemetry", None)
            if callable(enrich):
                extra = enrich(
                    sample=sample,
                    prediction=prediction,
                    telemetry=telemetry,
                    elapsed=elapsed,
                    judge_correct=judge_correct,
                    coverage=coverage,
                )
                if isinstance(extra, dict):
                    telemetry.update(extra)
        except Exception as exc:
            logger.warning("[Runner] Telemetry enrichment failed for %s: %s", sample.sample_id, exc)

        return PredictionResult(
            sample_id=sample.sample_id,
            prediction=prediction,
            judge_correct=judge_correct,
            coverage=coverage,
            elapsed=elapsed,
            telemetry=telemetry,
            error=error,
        )

    @staticmethod
    def _save_analysis_artifacts(artifact: RunArtifactManager, results: List[PredictionResult], metrics: Dict[str, Any]) -> None:
        try:
            from evaluation.badcase_analysis import (
                build_answer_type_consistency,
                build_badcase_outputs,
                quality_gate,
            )
        except Exception as exc:
            logger.warning("[Runner] Failed to import analysis builders: %s", exc)
            return
        rows = [getattr(result, "raw", None) for result in results]
        rows = [row for row in rows if isinstance(row, dict)]
        if not rows:
            return
        try:
            badcase_rows, taxonomy = build_badcase_outputs(rows)
            answer_type = build_answer_type_consistency(rows)
            target_slot_checks: List[bool] = []
            for row in rows:
                telemetry = row.get("telemetry", {}) or {}
                if isinstance(telemetry, dict) and telemetry.get("target_slot"):
                    target_slot_checks.append(bool(telemetry.get("target_slot_verified")))
            if target_slot_checks:
                metrics["target_slot_verification_rate"] = round(
                    sum(1 for ok in target_slot_checks if ok) / len(target_slot_checks) * 100,
                    2,
                )
                metrics["target_slot_checked_samples"] = len(target_slot_checks)
            quality = quality_gate(metrics)
            metrics["quality_gate"] = quality
            artifact.save_analysis_jsonl("badcases.jsonl", badcase_rows)
            artifact.save_analysis_json("failure_taxonomy.json", taxonomy)
            artifact.save_analysis_json("answer_type_consistency.json", answer_type)
            artifact.save_analysis_json("quality_gate.json", quality)
        except Exception as exc:
            logger.warning("[Runner] Failed to save analysis artifacts: %s", exc)

    @staticmethod
    def _result_to_row(result: PredictionResult, sample, adapter) -> dict:
        """Merge a PredictionResult with the sample metadata into a JSONL row."""
        row = {
            "sample_id": result.sample_id,
            "question": sample.question,
            "gold_answer": sample.gold_answer,
            "prediction": result.prediction,
            "judge_correct": result.judge_correct,
            "coverage": result.coverage,
            "elapsed": result.elapsed,
            "telemetry": result.telemetry,
            "error": result.error,
        }
        row.update(adapter.extra_result_fields(sample))
        return row

    @staticmethod
    def _result_to_per_sample_eval(result: PredictionResult, sample) -> dict:
        telemetry = result.telemetry or {}
        return {
            "sample_id": result.sample_id,
            "question": sample.question,
            "gold_answer": sample.gold_answer,
            "prediction": result.prediction,
            "error": result.error,
            "official_em": telemetry.get("official_em", telemetry.get("em", 0.0)),
            "official_f1": telemetry.get("official_f1", telemetry.get("f1", 0.0)),
            "official_exact_match": bool(telemetry.get("official_exact_match", False)),
            "official_f1_correct": bool(telemetry.get("official_f1_correct", False)),
            "llm_judge_used": bool(telemetry.get("llm_judge_used", False)),
            "llm_equivalent": telemetry.get("llm_equivalent"),
            "llm_confidence": telemetry.get("confidence"),
            "llm_reasoning": telemetry.get("reasoning", ""),
            "coverage": result.coverage,
            "evidence_recall": telemetry.get("evidence_recall", 0.0),
            "supporting_fact_hit": telemetry.get("supporting_fact_hit", False),
            "supporting_fact_title_recall": telemetry.get("supporting_fact_title_recall"),
            "supporting_sentence_recall": telemetry.get("supporting_sentence_recall"),
            "supporting_sentence_completion_rate": telemetry.get("supporting_sentence_completion_rate"),
            "supporting_sentence_completion_complete": telemetry.get("supporting_sentence_completion_complete"),
            "evidence_completion_needed": telemetry.get("evidence_completion_needed"),
            "missing_supporting_fact_titles": telemetry.get("missing_supporting_fact_titles", []),
            "missing_supporting_sentences": telemetry.get("missing_supporting_sentences", []),
            "answer_source_grounded": telemetry.get("answer_source_grounded", False),
            "target_slot_verified": telemetry.get("target_slot_verified"),
            "target_slot_failure_reason": telemetry.get("target_slot_failure_reason", ""),
            "target_slot": telemetry.get("target_slot", ""),
            "target_slot_expected_type": telemetry.get("target_slot_expected_type", ""),
            "evidence_reranking_applied": telemetry.get("evidence_reranking_applied", False),
            "evidence_reranking_scores": telemetry.get("evidence_reranking_scores", {}),
            "search_mode": telemetry.get("search_mode", ""),
            "num_files_read": telemetry.get("num_files_read", 0),
            "loop_count": telemetry.get("loop_count", 0),
            "total_tokens": telemetry.get("total_tokens", 0),
            "judge_tokens": telemetry.get("judge_tokens", 0),
            "error_type": telemetry.get("error_type", ""),
        }

    @staticmethod
    def _error_result(sample, exc: Exception, **telemetry: Any) -> PredictionResult:
        payload = {"error_type": exc.__class__.__name__, "error": str(exc)}
        payload.update({key: value for key, value in telemetry.items() if value is not None})
        return PredictionResult(
            sample_id=sample.sample_id,
            prediction="",
            judge_correct=False,
            coverage=False,
            elapsed=0.0,
            telemetry=payload,
            error=str(exc),
        )


async def _cancel_pending_tasks(tasks: List[asyncio.Task]) -> None:
    for task in tasks:
        if not task.done():
            task.cancel()
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)


def _extract_evidence_sources(cluster: Any) -> List[str]:
    sources: List[str] = []
    if cluster is None:
        return sources
    for ev in getattr(cluster, "evidences", []) or []:
        source = getattr(ev, "file_or_url", None)
        if source:
            sources.append(str(source))
    return sources


def _extract_evidence_snippets(cluster: Any) -> List[str]:
    snippets: List[str] = []
    if cluster is None:
        return snippets
    content = getattr(cluster, "content", None)
    if isinstance(content, str) and content.strip():
        snippets.append(content)
    for ev in getattr(cluster, "evidences", []) or []:
        for snippet in getattr(ev, "snippets", []) or []:
            if isinstance(snippet, str):
                snippets.append(snippet)
            elif isinstance(snippet, dict):
                text = snippet.get("snippet") or snippet.get("text") or snippet.get("content")
                if text:
                    snippets.append(str(text))
    return snippets[:50]


def _build_protocol(
    adapter: BenchmarkAdapter,
    *,
    run_id: str,
    seed: int,
    limit: int,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    hook = getattr(adapter, "get_protocol_spec", None)
    protocol: Dict[str, Any] = {}
    if callable(hook):
        try:
            candidate = hook(run_id=run_id, seed=seed, limit=limit)
            if isinstance(candidate, dict):
                protocol = dict(candidate)
        except Exception as exc:
            logger.warning("[Runner] adapter protocol hook failed: %s", exc)
    if not protocol:
        protocol = default_protocol(
            run_id=run_id,
            benchmark=adapter.name,
            config=config,
            seed=seed,
        ).to_dict()
    return _normalize_protocol(protocol, adapter=adapter, run_id=run_id, seed=seed, limit=limit, config=config)


def _normalize_protocol(
    protocol: Dict[str, Any],
    *,
    adapter: BenchmarkAdapter,
    run_id: str,
    seed: int,
    limit: int,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    normalized = dict(protocol)
    normalized["run_id"] = run_id
    normalized["benchmark"] = adapter.name
    normalized["protocol_schema_version"] = 2
    normalized.setdefault("protocol_version", f"{adapter.name}-v2")
    normalized["systems"] = [str(config.get("system_name") or "sirchmunk")]
    normalized["seeds"] = [seed]
    normalized["limit"] = limit
    normalized["stage"] = config.get("stage", "")
    normalized["sample_count"] = config.get("sample_count", 0)
    normalized["sample_id_checksum"] = config.get("sample_id_checksum", "")
    normalized["cache_policy"] = {
        **(normalized.get("cache_policy", {}) if isinstance(normalized.get("cache_policy"), dict) else {}),
        "mode": config.get("cache_mode", config.get("CACHE_MODE", "declared_by_adapter")),
        "allow_clear": _config_bool(config, "cache_allow_clear", "CACHE_ALLOW_CLEAR", default=False),
        "dry_run": _config_bool(config, "cache_dry_run", "CACHE_DRY_RUN", default=False),
    }
    normalized["frozen_constraints"] = {
        "frozen_evaluation": config.get("stage") == "frozen",
        "eval_feedback_disabled": not _config_bool(config, "enable_eval_feedback", "HOTPOT_ENABLE_EVAL_FEEDBACK", default=False),
        "memory_read_only": _config_bool(config, "memory_read_only", "frozen_memory_read_only", "SIRCHMUNK_MEMORY_READ_ONLY", default=False),
        "llm_judge_auxiliary_only": _config_bool(config, "allow_frozen_llm_judge_auxiliary", "ALLOW_FROZEN_LLM_JUDGE_AUXILIARY", default=False),
    }
    normalized["resource_limits"] = {
        key: config.get(key)
        for key in (
            "max_runtime_seconds",
            "max_total_tokens",
            "max_api_cost_usd",
            "max_disk_usage_bytes",
            "min_free_disk_bytes",
            "sample_timeout_seconds",
            "system_timeout_seconds",
            "benchmark_timeout_seconds",
            "global_timeout_seconds",
        )
        if config.get(key) not in (None, "", 0, 0.0)
    }
    normalized["config"] = config
    return normalized


def _safe_adapter_call(adapter: BenchmarkAdapter, method_name: str, default=None):
    hook = getattr(adapter, method_name, None)
    if not callable(hook):
        return default
    try:
        return hook()
    except Exception as exc:
        logger.warning("[Runner] adapter.%s failed: %s", method_name, exc)
        return default


def _adapter_metrics(adapter: BenchmarkAdapter, results: List[PredictionResult]) -> Dict[str, Any]:
    aggregator = _safe_adapter_call(adapter, "get_metric_aggregator", default=None)
    if not callable(aggregator):
        return {}
    try:
        metrics = aggregator(results)
        return metrics if isinstance(metrics, dict) else {}
    except Exception as exc:
        logger.warning("[Runner] adapter metric aggregator failed: %s", exc)
        return {}


def _read_json_file(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        return {}


def _prepare_cache(adapter: BenchmarkAdapter, config: Dict[str, Any]) -> Dict[str, Any]:
    policy = _safe_adapter_call(adapter, "get_cache_policy", default={})
    if not isinstance(policy, dict):
        policy = {}
    manager = CacheManager(
        adapter.get_work_path(),
        cache_names=policy.get("cache_names"),
        cache_paths=policy.get("cache_paths"),
        compiled_markers=policy.get("compiled_markers"),
    )
    mode = config.get("cache_mode") or config.get("CACHE_MODE") or policy.get("mode") or "none"
    report = manager.prepare(
        mode,
        allow_clear=_config_bool(config, "cache_allow_clear", "CACHE_ALLOW_CLEAR", default=False),
        dry_run=_config_bool(config, "cache_dry_run", "CACHE_DRY_RUN", default=False),
    ).to_dict()
    report["policy"] = policy
    return report


def _aggregate_runner_metrics(results: List[PredictionResult]) -> Dict[str, Any]:
    n = len(results)
    if n == 0:
        return {"n": 0}
    correct = sum(1 for r in results if r.judge_correct)
    coverage = sum(1 for r in results if r.coverage)
    latencies = sorted(float(r.elapsed or 0.0) for r in results)
    telemetry = [r.telemetry or {} for r in results]
    em = sum(float(t.get("em", 0.0) or 0.0) for t in telemetry) / n
    f1 = sum(float(t.get("f1", 0.0) or 0.0) for t in telemetry) / n
    evidence = sum(float(t.get("evidence_recall", 0.0) or 0.0) for t in telemetry) / n
    # Evidence recall reported at both levels, because the headline
    # ``evidence_recall`` switches basis depending on whether a system exposes
    # evidence snippets, which makes it unsafe to compare across systems alone.
    ev_fact = sum(float(t.get("evidence_recall_fact_level", 0.0) or 0.0) for t in telemetry) / n
    sent_values = [
        float(t["evidence_recall_sentence_level"]) for t in telemetry
        if isinstance(t.get("evidence_recall_sentence_level"), (int, float))
    ]
    ev_sentence = (sum(sent_values) / len(sent_values)) if sent_values else None
    # Answers actually traceable to retrieved evidence, and exact match counted
    # only over those. An open-book system can score on parametric recall alone,
    # which credits memorisation rather than retrieval; restricting EM to
    # grounded answers measures what the retrieval chain contributed.
    grounded = [t for t in telemetry if t.get("answer_source_grounded")]
    grounded_em = (
        sum(float(t.get("em", 0.0) or 0.0) for t in grounded) / len(grounded)
        if grounded else None
    )
    total_tokens = sum(_safe_int(t.get("total_tokens", 0)) for t in telemetry)
    judge_tokens = sum(_safe_int(t.get("judge_tokens", 0)) for t in telemetry)
    retry_attempts = [max(_safe_int(t.get("retry_attempts", 1), default=1), 0) for t in telemetry]
    total_retries = sum(max(attempts - 1, 0) for attempts in retry_attempts)
    retried_samples = sum(1 for attempts in retry_attempts if attempts > 1)
    retry_exhausted_samples = sum(
        1 for t in telemetry
        if t.get("error_type") == "RetryExhausted" or t.get("retry_exhausted")
    )
    system_failures = sum(1 for r in results if r.error)
    answer_failures = sum(1 for r in results if not r.error and not r.judge_correct)
    failure_types: Dict[str, int] = {}
    for r in results:
        if not r.error:
            continue
        telemetry_row = r.telemetry or {}
        error_type = str(telemetry_row.get("error_type") or "system_error")
        failure_types[error_type] = failure_types.get(error_type, 0) + 1
    return {
        "n": n,
        "accuracy": round(correct / n * 100, 2),
        "coverage": round(coverage / n * 100, 2),
        "em": round(em * 100, 2),
        "f1": round(f1 * 100, 2),
        "evidence_recall": round(evidence * 100, 2),
        "evidence_recall_fact_level": round(ev_fact * 100, 2),
        "evidence_recall_sentence_level": (
            round(ev_sentence * 100, 2) if ev_sentence is not None else None
        ),
        "answer_source_grounded_rate": round(len(grounded) / n * 100, 2),
        "grounded_em": round(grounded_em * 100, 2) if grounded_em is not None else None,
        "grounded_n": len(grounded),
        "latency": {
            "avg": round(sum(latencies) / n, 2),
            "p50": round(_percentile(latencies, 50), 2),
            "p95": round(_percentile(latencies, 95), 2),
        },
        "token_usage": {
            "total_tokens": total_tokens + judge_tokens,
            "search_tokens": total_tokens,
            "judge_tokens": judge_tokens,
            "avg_tokens_per_question": round((total_tokens + judge_tokens) / n, 1),
        },
        "retries": {
            "total_retries": total_retries,
            "retried_samples": retried_samples,
            "retry_exhausted_samples": retry_exhausted_samples,
        },
        "failure_classification": {
            "system_failures": system_failures,
            "answer_failures": answer_failures,
            "system_failure_types": failure_types,
        },
    }


def _config_value(config: Dict[str, Any], *keys: str, default: Any = 0) -> Any:
    for key in keys:
        value = config.get(key)
        if value is not None and value != "":
            return value
    return default


def _config_int(config: Dict[str, Any], *keys: str, default: int = 0) -> int:
    return _safe_int(_config_value(config, *keys, default=default), default=default)


def _config_float(config: Dict[str, Any], *keys: str, default: float = 0.0) -> float:
    return _safe_float(_config_value(config, *keys, default=default), default=default)


def _config_bool(config: Dict[str, Any], *keys: str, default: bool = False) -> bool:
    value = _config_value(config, *keys, default=default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in ("1", "true", "yes", "y", "on"):
        return True
    if text in ("0", "false", "no", "n", "off", ""):
        return False
    return default


def _retryable_markers_from_config(value: Any) -> Optional[List[str]]:
    if value is None or value == "":
        return None
    if isinstance(value, str):
        return [marker.strip() for marker in value.split(",") if marker.strip()]
    if isinstance(value, (list, tuple, set)):
        return [str(marker).strip() for marker in value if str(marker).strip()]
    return [str(value).strip()]


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


def _percentile(values: List[float], percentile: int) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    idx = (len(values) - 1) * percentile / 100
    lo = int(idx)
    hi = min(lo + 1, len(values) - 1)
    weight = idx - lo
    return values[lo] * (1 - weight) + values[hi] * weight
