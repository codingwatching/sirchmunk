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
        "results/per_sample_eval.jsonl",
    ]
    FROZEN_CACHE_MODES = {"cold", "compiled"}
    MAX_BASELINE_FAILURE_RATE = 0.05
    MIN_IMPORT_COVERAGE = 95.0

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
        stage = str(config.get("stage") or "").lower()
        if config.get("llm_fallback") is True:
            issues.append(ValidationIssue("error", "llm_fallback", "llm_fallback=True is not allowed for pure retrieval evaluation."))
        if "cache_mode" not in config:
            issues.append(ValidationIssue("warning", "cache_policy", "cache_mode is not explicitly recorded in run config."))
        if stage not in ("frozen", "exploration"):
            issues.append(ValidationIssue("error", "stage", "Publication artifacts must explicitly record stage=frozen."))
        elif stage == "exploration":
            issues.append(ValidationIssue("error", "stage", "Exploration runs cannot be used as final frozen-evaluation claims."))
        if stage == "frozen":
            cache_mode = str(config.get("cache_mode") or config.get("CACHE_MODE") or "").lower()
            if cache_mode not in self.FROZEN_CACHE_MODES:
                issues.append(ValidationIssue("error", "cache_policy", "Frozen evaluation must use cache_mode cold or compiled."))
            if _config_bool(config, "cache_dry_run", "CACHE_DRY_RUN"):
                issues.append(ValidationIssue("error", "cache_policy", "Frozen evaluation cannot use cache dry-run mode."))
            if _config_bool(config, "enable_eval_feedback", "HOTPOT_ENABLE_EVAL_FEEDBACK"):
                issues.append(ValidationIssue("error", "eval_feedback", "Frozen evaluation must disable eval feedback to avoid test-set tuning."))
            if _config_bool(config, "enable_memory", "SIRCHMUNK_ENABLE_MEMORY"):
                memory_state = _config_value(
                    config,
                    "memory_state_version",
                    "fixed_memory_state_version",
                    "SIRCHMUNK_MEMORY_STATE_VERSION",
                    default="",
                )
                if not str(memory_state).strip():
                    issues.append(ValidationIssue("error", "memory_state", "Frozen evaluation with memory enabled must record a fixed memory_state_version."))
                if not _config_bool(config, "memory_read_only", "frozen_memory_read_only", "SIRCHMUNK_MEMORY_READ_ONLY"):
                    issues.append(ValidationIssue("error", "memory_state", "Frozen evaluation with memory enabled must mark memory as read-only."))
                if _config_bool(config, "enable_memory_updates", "memory_write_enabled", "SIRCHMUNK_ENABLE_MEMORY_UPDATES"):
                    issues.append(ValidationIssue("error", "memory_state", "Frozen evaluation cannot enable adaptive memory updates."))
            llm_judge_enabled = _config_bool(config, "enable_llm_judge", "HOTPOT_ENABLE_LLM_JUDGE")
            llm_judge_allowed = _config_bool(config, "allow_frozen_llm_judge_auxiliary", "ALLOW_FROZEN_LLM_JUDGE_AUXILIARY")
            if llm_judge_enabled and not llm_judge_allowed:
                issues.append(ValidationIssue("error", "llm_judge", "Frozen evaluation cannot enable LLM judge unless explicitly marked auxiliary."))
            elif llm_judge_enabled:
                issues.append(ValidationIssue("warning", "llm_judge", "LLM judge is enabled as auxiliary; official EM/F1 must remain the primary paper metric."))
        if int(manifest.get("env_snapshot_version", 0) or 0) < 2:
            issues.append(ValidationIssue("warning", "env_snapshot", "Env snapshot predates sanitized snapshot format v2."))
        env_snapshot_path = run_path / "env_snapshot.txt"
        if env_snapshot_path.exists() and _env_snapshot_has_unredacted_secret(env_snapshot_path):
            issues.append(ValidationIssue("error", "env_snapshot_secret", "env_snapshot.txt contains an unredacted secret-like key.", str(env_snapshot_path)))
        cache_report = _read_json(run_path / "cache_report.json")
        if not cache_report:
            severity = "error" if stage == "frozen" else "warning"
            issues.append(ValidationIssue(severity, "cache_report", "cache_report.json is missing or invalid."))
        elif stage == "frozen":
            report_mode = str(cache_report.get("mode") or "").lower()
            if report_mode not in self.FROZEN_CACHE_MODES:
                issues.append(ValidationIssue("error", "cache_policy", "Frozen cache_report mode must be cold or compiled."))
            if cache_report.get("dry_run"):
                issues.append(ValidationIssue("error", "cache_policy", "Frozen cache_report cannot be dry-run."))
            warnings = cache_report.get("warnings") or []
            if warnings:
                issues.append(ValidationIssue("error", "cache_policy", f"Frozen cache report contains warnings: {warnings}", str(run_path / "cache_report.json")))
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
        systems = protocol.get("systems", [])
        seeds = protocol.get("seeds", [])
        if not isinstance(systems, list) or not systems:
            issues.append(ValidationIssue("error", "protocol", "Protocol must declare at least one fixed system.", str(run_path / "protocol.yaml")))
        if not isinstance(seeds, list) or not seeds:
            issues.append(ValidationIssue("error", "protocol", "Protocol must declare fixed seeds.", str(run_path / "protocol.yaml")))
        if "cache_policy" not in protocol:
            issues.append(ValidationIssue("warning", "protocol", "Protocol lacks cache_policy."))
        metric_groups = protocol.get("metrics", {}) or {}
        if not any(name in metric_groups for name in ("researchops", "setup", "dynamic", "fidelity", "mechanism")):
            issues.append(ValidationIssue("warning", "mechanism_metrics", "Protocol does not explicitly include mechanism metrics."))

        config = protocol.get("config", {}) if isinstance(protocol.get("config", {}), dict) else {}
        stage = str(protocol.get("stage") or config.get("stage") or "").lower()
        if stage != "frozen":
            issues.append(ValidationIssue("error", "protocol_stage", "Publication protocol must record stage=frozen.", str(run_path / "protocol.yaml")))
            return issues
        if int(protocol.get("protocol_schema_version", 0) or 0) < 2:
            issues.append(ValidationIssue("error", "protocol_schema", "Frozen protocol requires protocol_schema_version >= 2.", str(run_path / "protocol.yaml")))
        cache_policy = protocol.get("cache_policy", {}) if isinstance(protocol.get("cache_policy", {}), dict) else {}
        cache_mode = str(cache_policy.get("mode") or config.get("cache_mode") or config.get("CACHE_MODE") or "").lower()
        if cache_mode not in self.FROZEN_CACHE_MODES:
            issues.append(ValidationIssue("error", "protocol_cache_policy", "Frozen protocol cache_policy.mode must be cold or compiled.", str(run_path / "protocol.yaml")))
        if cache_policy.get("dry_run") or _config_bool(config, "cache_dry_run", "CACHE_DRY_RUN"):
            issues.append(ValidationIssue("error", "protocol_cache_policy", "Frozen protocol cannot use cache dry-run mode.", str(run_path / "protocol.yaml")))
        if not (protocol.get("sample_id_checksum") or config.get("sample_id_checksum")):
            issues.append(ValidationIssue("error", "protocol_samples", "Frozen protocol must record sample_id_checksum.", str(run_path / "protocol.yaml")))
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
        if "official_exact_match" not in metrics or "official_f1_correct" not in metrics:
            issues.append(ValidationIssue("warning", "official_metrics", "Official EM/F1-derived metrics are missing from metrics.json."))
        failure_info = metrics.get("failure_classification", {}) or {}
        system_failures = int(failure_info.get("system_failures", 0) or 0)
        failure_types = failure_info.get("system_failure_types", {}) or {}
        total = int(metrics.get("n", 0) or 0)
        if total and system_failures / total > 0.05:
            issues.append(ValidationIssue("error", "system_failure_rate", f"System failure rate is too high: {system_failures}/{total}."))
        elif system_failures:
            issues.append(ValidationIssue("warning", "system_failure_rate", f"System failures present: {system_failures}/{total}."))
        timeout_failures = _failure_count_matching(failure_types, "timeout")
        budget_failures = _failure_count_matching(failure_types, "budget")
        if total and timeout_failures / total > self.MAX_BASELINE_FAILURE_RATE:
            issues.append(ValidationIssue("error", "timeout_failure_rate", f"Timeout failure rate is too high: {timeout_failures}/{total}."))
        elif timeout_failures:
            issues.append(ValidationIssue("warning", "timeout_failure_rate", f"Timeout failures present: {timeout_failures}/{total}."))
        if total and budget_failures / total > self.MAX_BASELINE_FAILURE_RATE:
            issues.append(ValidationIssue("error", "budget_failure_rate", f"Budget failure rate is too high: {budget_failures}/{total}."))
        elif budget_failures:
            issues.append(ValidationIssue("warning", "budget_failure_rate", f"Budget failures present: {budget_failures}/{total}."))
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
        non_published = [s for s in systems if not s.get("is_published_only")]
        missing_checksums = [s.get("system_name") for s in non_published if not s.get("sample_id_checksum")]
        if missing_checksums:
            issues.append(ValidationIssue("error", "sample_id_checksum", f"Systems lack sample_id_checksum: {missing_checksums}", str(table_path)))
        checksums = {s.get("sample_id_checksum") for s in non_published if s.get("sample_id_checksum")}
        if len(checksums) > 1:
            issues.append(ValidationIssue("error", "paired_sample_ids", "Non-published systems do not share the same sample_id set.", str(table_path)))
        for system in systems:
            if system.get("is_ours") or system.get("is_published_only"):
                continue
            if not system.get("setup_metrics"):
                issues.append(ValidationIssue("error", "baseline_setup", f"Baseline '{system.get('system_name')}' lacks setup_metrics.", str(table_path)))
            if system.get("p_value") is None and ours:
                issues.append(ValidationIssue("warning", "significance", f"Baseline '{system.get('system_name')}' lacks paired p_value.", str(table_path)))

            failure_counts = system.get("failure_counts") or {}
            failure_total = sum(_safe_int(value) for value in failure_counts.values()) if isinstance(failure_counts, dict) else 0
            failure_rate = _safe_float(system.get("failure_rate"), (failure_total / max(int(system.get("n", 0) or 0), 1)) * 100)
            if failure_rate > self.MAX_BASELINE_FAILURE_RATE * 100:
                issues.append(ValidationIssue("error", "baseline_failure_rate", f"Baseline '{system.get('system_name')}' failure rate is too high: {failure_rate:.1f}%.", str(table_path)))
            elif failure_total:
                issues.append(ValidationIssue("warning", "baseline_failures", f"Baseline '{system.get('system_name')}' has classified failures: {failure_counts}.", str(table_path)))
            timeout_count = _failure_count_matching(failure_counts, "timeout") if isinstance(failure_counts, dict) else 0
            budget_count = _failure_count_matching(failure_counts, "budget") if isinstance(failure_counts, dict) else 0
            sample_n = int(system.get("n", 0) or 0)
            if sample_n and timeout_count / sample_n > self.MAX_BASELINE_FAILURE_RATE:
                issues.append(ValidationIssue("error", "baseline_timeout_rate", f"Baseline '{system.get('system_name')}' timeout rate is too high: {timeout_count}/{sample_n}.", str(table_path)))
            if sample_n and budget_count / sample_n > self.MAX_BASELINE_FAILURE_RATE:
                issues.append(ValidationIssue("error", "baseline_budget_rate", f"Baseline '{system.get('system_name')}' budget-exceeded rate is too high: {budget_count}/{sample_n}.", str(table_path)))

            imported = _as_bool(system.get("imported_baseline")) or system.get("import_coverage") is not None
            if imported:
                import_coverage = _safe_float(system.get("import_coverage"), default=-1.0)
                if import_coverage < 0:
                    issues.append(ValidationIssue("error", "import_coverage", f"Imported baseline '{system.get('system_name')}' lacks import_coverage.", str(table_path)))
                elif import_coverage < self.MIN_IMPORT_COVERAGE:
                    issues.append(ValidationIssue("error", "import_coverage", f"Imported baseline '{system.get('system_name')}' coverage is too low: {import_coverage:.1f}%.", str(table_path)))
                missing_samples = int(system.get("missing_samples", 0) or 0)
                if missing_samples and not system.get("missing_sample_ids"):
                    issues.append(ValidationIssue("warning", "import_missing_samples", f"Imported baseline '{system.get('system_name')}' has missing samples without sample id details.", str(table_path)))
        return issues


def _failure_count_matching(failure_counts: Dict[str, Any], marker: str) -> int:
    if not isinstance(failure_counts, dict):
        return 0
    needle = marker.lower()
    return sum(_safe_int(count) for name, count in failure_counts.items() if needle in str(name).lower())


def _config_value(config: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        value = config.get(key)
        if value is not None and value != "":
            return value
    return default


def _config_bool(config: Dict[str, Any], *keys: str, default: bool = False) -> bool:
    return _as_bool(_config_value(config, *keys, default=default), default=default)


def _as_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if value is None:
        return default
    text = str(value).strip().lower()
    if text in ("1", "true", "yes", "y", "on"):
        return True
    if text in ("0", "false", "no", "n", "off", ""):
        return False
    return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _env_snapshot_has_unredacted_secret(path: Path) -> bool:
    markers = ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL", "AUTH")
    try:
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            stripped = raw_line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, _, value = stripped.partition("=")
            if any(marker in key.upper() for marker in markers) and value and value != "<redacted>":
                return True
    except OSError:
        return False
    return False
