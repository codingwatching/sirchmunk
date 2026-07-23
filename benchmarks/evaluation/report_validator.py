"""Academic report validation gate.

The validator is intentionally conservative: it blocks publication-ready claims
when critical provenance, pairing, or setup-cost evidence is missing, while
keeping warnings for non-fatal issues such as a dirty working tree.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class ValidationIssue:
    severity: str  # error | warning | info
    check: str
    message: str
    path: str = ""


@dataclass
class ValidationReport:
    passed: bool
    issues: List[ValidationIssue] = field(default_factory=list)

    @property
    def errors(self) -> List[ValidationIssue]:
        return [issue for issue in self.issues if issue.severity == "error"]

    @property
    def warnings(self) -> List[ValidationIssue]:
        return [issue for issue in self.issues if issue.severity == "warning"]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "issues": [asdict(issue) for issue in self.issues],
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
        }


class AcademicReportValidator:
    """Validate whether an artifact/table package supports paper claims."""

    REQUIRED_RUN_FILES = [
        "protocol.yaml",
        "manifest.json",
        "config_snapshot.json",
        "git_snapshot.json",
        "system_specs.json",
        "dataset_manifest.json",
        "results/metrics.json",
        "results/predictions.jsonl",
    ]

    def validate(
        self,
        *,
        run_dir: str | Path | None = None,
        table_json: str | Path | None = None,
    ) -> ValidationReport:
        issues: List[ValidationIssue] = []
        run_path = Path(run_dir).resolve() if run_dir else None
        table_path = Path(table_json).resolve() if table_json else None

        manifest: Dict[str, Any] = {}
        protocol: Dict[str, Any] = {}
        metrics: Dict[str, Any] = {}

        if run_path:
            issues.extend(self._validate_run_dir(run_path))
            manifest = _read_json(run_path / "manifest.json")
            protocol = _read_json(run_path / "protocol.yaml")
            metrics = _read_json(run_path / "results" / "metrics.json")
            issues.extend(self._validate_manifest(run_path, manifest))
            issues.extend(self._validate_protocol(run_path, protocol))
            issues.extend(self._validate_metrics(run_path, metrics))
            issues.extend(self._validate_predictions(run_path / "results" / "predictions.jsonl"))
        else:
            issues.append(ValidationIssue("warning", "run_dir", "No run artifact directory supplied; provenance checks are limited."))

        if table_path:
            table = _read_json(table_path)
            issues.extend(self._validate_table(table_path, table))
        else:
            issues.append(ValidationIssue("warning", "table_json", "No paper table JSON supplied; baseline comparability checks are limited."))

        passed = not any(issue.severity == "error" for issue in issues)
        return ValidationReport(passed=passed, issues=issues)

    def _validate_run_dir(self, run_path: Path) -> List[ValidationIssue]:
        issues: List[ValidationIssue] = []
        if not run_path.exists():
            return [ValidationIssue("error", "run_dir", "Run artifact directory does not exist.", str(run_path))]
        for rel in self.REQUIRED_RUN_FILES:
            path = run_path / rel
            if not path.exists():
                issues.append(ValidationIssue("error", "artifact_file", f"Missing required artifact file: {rel}", str(path)))
        return issues

    def _validate_manifest(self, run_path: Path, manifest: Dict[str, Any]) -> List[ValidationIssue]:
        issues: List[ValidationIssue] = []
        if not manifest:
            return [ValidationIssue("error", "manifest", "Manifest is missing or invalid JSON.", str(run_path / "manifest.json"))]
        for key in ("git_commit", "git_branch", "git_dirty", "git_diff_hash", "system_specs", "dataset_manifest", "config"):
            if key not in manifest:
                issues.append(ValidationIssue("error", "manifest", f"Manifest missing key: {key}", str(run_path / "manifest.json")))
        if manifest.get("git_dirty"):
            issues.append(ValidationIssue("warning", "git_dirty", "Working tree was dirty; report remains reproducible only with git_snapshot status.", str(run_path / "git_snapshot.json")))
        config = manifest.get("config", {}) or {}
        if config.get("llm_fallback") is True:
            issues.append(ValidationIssue("error", "llm_fallback", "llm_fallback=True is not allowed for pure retrieval evaluation."))
        if "cache_mode" not in config:
            issues.append(ValidationIssue("warning", "cache_policy", "cache_mode is not explicitly recorded in run config."))
        if config.get("stage") not in ("frozen", "exploration"):
            issues.append(ValidationIssue("warning", "stage", "run stage is not explicitly recorded as frozen/exploration."))
        if config.get("stage") == "exploration":
            issues.append(ValidationIssue("warning", "stage", "Exploration runs should not be used as final frozen-evaluation claims."))
        cache_report = _read_json(run_path / "cache_report.json")
        if not cache_report:
            issues.append(ValidationIssue("warning", "cache_report", "cache_report.json is missing or invalid."))
        elif cache_report.get("mode") == "none":
            issues.append(ValidationIssue("warning", "cache_policy", "cache mode is 'none'; cold/warm policy was not applied."))
        return issues

    def _validate_protocol(self, run_path: Path, protocol: Dict[str, Any]) -> List[ValidationIssue]:
        issues: List[ValidationIssue] = []
        if not protocol:
            return [ValidationIssue("error", "protocol", "Protocol is missing or invalid.", str(run_path / "protocol.yaml"))]
        for key in ("run_id", "benchmark", "systems", "metrics", "seeds"):
            if key not in protocol:
                issues.append(ValidationIssue("error", "protocol", f"Protocol missing key: {key}", str(run_path / "protocol.yaml")))
        if "cache_policy" not in protocol:
            issues.append(ValidationIssue("warning", "protocol", "Protocol lacks cache_policy."))
        metric_groups = protocol.get("metrics", {}) or {}
        if not any(name in metric_groups for name in ("researchops", "setup", "dynamic", "fidelity", "mechanism")):
            issues.append(ValidationIssue("warning", "mechanism_metrics", "Protocol does not explicitly include mechanism metrics."))
        return issues

    def _validate_metrics(self, run_path: Path, metrics: Dict[str, Any]) -> List[ValidationIssue]:
        issues: List[ValidationIssue] = []
        if not metrics:
            return [ValidationIssue("error", "metrics", "Metrics file is missing or invalid.", str(run_path / "results" / "metrics.json"))]
        if int(metrics.get("n", 0) or 0) <= 0:
            issues.append(ValidationIssue("error", "metrics", "Metrics report zero evaluated samples."))
        if "latency" not in metrics:
            issues.append(ValidationIssue("warning", "metrics", "Latency distribution is missing from metrics."))
        if "token_usage" not in metrics:
            issues.append(ValidationIssue("warning", "metrics", "Token usage is missing from metrics."))
        failure_info = metrics.get("failure_classification", {}) or {}
        system_failures = int(failure_info.get("system_failures", 0) or 0)
        total = int(metrics.get("n", 0) or 0)
        if total and system_failures / total > 0.05:
            issues.append(ValidationIssue("error", "system_failure_rate", f"System failure rate is too high: {system_failures}/{total}."))
        elif system_failures:
            issues.append(ValidationIssue("warning", "system_failure_rate", f"System failures present: {system_failures}/{total}."))
        checkpoint = metrics.get("checkpoint", {}) or {}
        if int(checkpoint.get("pending", 0) or 0) > 0:
            issues.append(ValidationIssue("error", "checkpoint", "Run has pending samples in checkpoint summary."))
        return issues

    def _validate_predictions(self, path: Path) -> List[ValidationIssue]:
        issues: List[ValidationIssue] = []
        if not path.exists():
            return issues
        count = 0
        malformed = 0
        with path.open(encoding="utf-8") as fp:
            for line in fp:
                if not line.strip():
                    continue
                count += 1
                try:
                    row = json.loads(line)
                    if "telemetry" not in row:
                        issues.append(ValidationIssue("warning", "prediction_telemetry", "A prediction row lacks telemetry.", str(path)))
                        break
                except json.JSONDecodeError:
                    malformed += 1
        if count == 0:
            issues.append(ValidationIssue("error", "predictions", "Predictions file is empty.", str(path)))
        if malformed:
            issues.append(ValidationIssue("error", "predictions", f"Predictions file contains {malformed} malformed JSON lines.", str(path)))
        return issues

    def _validate_table(self, table_path: Path, table: Dict[str, Any]) -> List[ValidationIssue]:
        issues: List[ValidationIssue] = []
        systems = table.get("systems", []) if isinstance(table, dict) else []
        if not systems:
            return [ValidationIssue("error", "table", "Paper table JSON has no systems.", str(table_path))]
        ours = [s for s in systems if s.get("is_ours")]
        if not ours:
            issues.append(ValidationIssue("error", "table", "No system is marked as ours.", str(table_path)))
        sample_sizes = {int(s.get("n", 0) or 0) for s in systems if not s.get("is_published_only")}
        if len(sample_sizes) > 1:
            issues.append(ValidationIssue("error", "paired_samples", f"Non-published systems have different sample sizes: {sorted(sample_sizes)}", str(table_path)))
        for system in systems:
            if system.get("is_ours") or system.get("is_published_only"):
                continue
            if not system.get("setup_metrics"):
                issues.append(ValidationIssue("error", "baseline_setup", f"Baseline '{system.get('system_name')}' lacks setup_metrics.", str(table_path)))
            if system.get("p_value") is None and ours:
                issues.append(ValidationIssue("warning", "significance", f"Baseline '{system.get('system_name')}' lacks paired p_value.", str(table_path)))
        return issues


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
