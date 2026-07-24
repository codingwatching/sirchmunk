"""Run artifact management for ResearchOps P0."""
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from .protocol import protocol_to_text


_SENSITIVE_ENV_MARKERS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL", "AUTH")


class RunArtifactManager:
    """Create and populate a reproducible run artifact directory."""

    def __init__(self, output_dir: str | Path, run_id: str) -> None:
        self.output_dir = Path(output_dir).resolve()
        self.run_id = run_id
        self.run_dir = self.output_dir / "runs" / run_id
        self.results_dir = self.run_dir / "results"
        self.analysis_dir = self.run_dir / "analysis"
        self.reports_dir = self.run_dir / "reports"
        self.logs_dir = self.run_dir / "logs"
        self.checkpoints_dir = self.run_dir / "checkpoints"
        self.artifacts_dir = self.run_dir / "artifacts"

    def create(self) -> None:
        for path in (
            self.results_dir,
            self.analysis_dir,
            self.reports_dir,
            self.logs_dir,
            self.checkpoints_dir,
            self.artifacts_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)

    @property
    def predictions_path(self) -> Path:
        return self.results_dir / "predictions.jsonl"

    @property
    def per_sample_eval_path(self) -> Path:
        return self.results_dir / "per_sample_eval.jsonl"

    @property
    def metrics_path(self) -> Path:
        return self.results_dir / "metrics.json"

    def save_protocol(self, protocol: Dict[str, Any]) -> str:
        self.create()
        path = self.run_dir / "protocol.yaml"
        path.write_text(protocol_to_text(protocol), encoding="utf-8")
        return str(path)

    def save_manifest(
        self,
        *,
        benchmark: str,
        git_commit: str,
        config_hash: str,
        config: Dict[str, Any],
        dataset_manifest: Optional[Dict[str, Any]] = None,
        env_file: str = "",
    ) -> str:
        self.create()
        git_snapshot = self._git_snapshot()
        system_specs = self._system_specs()
        manifest = {
            "run_id": self.run_id,
            "benchmark": benchmark,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "git_commit": git_commit,
            "git_branch": git_snapshot.get("branch", "unknown"),
            "git_dirty": git_snapshot.get("dirty", True),
            "git_diff_hash": git_snapshot.get("diff_hash", "unknown"),
            "config_hash": config_hash,
            "python": sys.version,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "system_specs": system_specs,
            "config": config,
            "dataset_manifest": dataset_manifest or {},
            "env_file": env_file,
            "env_snapshot": self._safe_env_snapshot(),
            "env_snapshot_version": 2,
        }
        path = self.run_dir / "manifest.json"
        path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        (self.run_dir / "config_snapshot.json").write_text(
            json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        (self.run_dir / "git_snapshot.json").write_text(
            json.dumps(git_snapshot, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        (self.run_dir / "system_specs.json").write_text(
            json.dumps(system_specs, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        if dataset_manifest is not None:
            (self.run_dir / "dataset_manifest.json").write_text(
                json.dumps(dataset_manifest, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        if env_file and Path(env_file).exists():
            self._write_sanitized_env_snapshot(Path(env_file), self.run_dir / "env_snapshot.txt")
        return str(path)

    def save_metrics(self, metrics: Dict[str, Any]) -> str:
        self.create()
        self.metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
        return str(self.metrics_path)

    def save_cache_report(self, cache_report: Dict[str, Any]) -> str:
        self.create()
        path = self.run_dir / "cache_report.json"
        path.write_text(json.dumps(cache_report, indent=2, ensure_ascii=False), encoding="utf-8")
        return str(path)

    def append_prediction(self, row: Dict[str, Any]) -> None:
        self.create()
        with self.predictions_path.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")

    def append_per_sample_eval(self, row: Dict[str, Any]) -> None:
        self.create()
        with self.per_sample_eval_path.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")

    @staticmethod
    def _safe_env_snapshot() -> Dict[str, str]:
        allow_prefixes = (
            "HOTPOT_",
            "MECHANISM_",
            "SIRCHMUNK_",
            "LLM_MODEL",
            "LLM_BASE_URL",
            "EMBEDDING_MODEL",
        )
        snapshot: Dict[str, str] = {}
        for key, value in os.environ.items():
            if not key.startswith(allow_prefixes):
                continue
            if _is_sensitive_env_key(key):
                snapshot[key] = "<redacted>"
            else:
                snapshot[key] = value
        return snapshot

    @staticmethod
    def _write_sanitized_env_snapshot(source: Path, target: Path) -> None:
        sanitized = []
        for raw_line in source.read_text(encoding="utf-8").splitlines():
            stripped = raw_line.strip()
            if not stripped or stripped.startswith("#") or "=" not in raw_line:
                sanitized.append(raw_line)
                continue
            key, _, value = raw_line.partition("=")
            if _is_sensitive_env_key(key.strip()):
                sanitized.append(f"{key}=<redacted>")
            else:
                sanitized.append(f"{key}={value}")
        target.write_text("\n".join(sanitized) + "\n", encoding="utf-8")

    @staticmethod
    def _git_snapshot() -> Dict[str, Any]:
        def _run(args: list[str]) -> str:
            try:
                result = subprocess.run(args, capture_output=True, text=True, check=False)
                if result.returncode == 0:
                    return result.stdout.strip()
            except Exception:
                pass
            return ""

        commit = _run(["git", "rev-parse", "HEAD"])
        branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
        status = _run(["git", "status", "--porcelain"])
        diff = _run(["git", "diff", "--no-ext-diff"])
        import hashlib

        hash_source = diff if diff else status
        return {
            "commit": commit or "unknown",
            "branch": branch or "unknown",
            "dirty": bool(status),
            "status_porcelain": status.splitlines(),
            "diff_hash": hashlib.sha256(hash_source.encode("utf-8")).hexdigest()[:16] if hash_source else "clean",
        }

    @staticmethod
    def _system_specs() -> Dict[str, Any]:
        return {
            "python_executable": sys.executable,
            "python_version": sys.version,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        }


def _is_sensitive_env_key(key: str) -> bool:
    upper = (key or "").upper()
    return any(marker in upper for marker in _SENSITIVE_ENV_MARKERS)
