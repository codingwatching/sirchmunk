"""Business-level quality gates for ResearchOps control-layer P0.

Runtime guards in ``guards.py`` stop runs when budgets/timeouts are exceeded.
This module is intentionally different: it checks whether a control-layer flow
has enough declared protocol, sampling, asset, evaluation, and report evidence
to proceed or to support paper-facing claims.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .asset_registry import AssetRegistry, AssetType
from .control_phase import ControlBlock, ExperimentStage
from .param_schema import ControlRunConfig, ParamSeverity, validate_control_config
from .protocol import ProtocolLoader, ProtocolValidator


class GateName(str, Enum):
    """Canonical P0 gate names."""

    PARAMS = "gate_0_params"
    ASSETS = "gate_1_assets"
    SAMPLING = "gate_2_sampling"
    FROZEN_RUN = "gate_3_frozen_run"
    EVALUATION = "gate_4_evaluation"
    REPORT = "gate_5_report"


class GateSeverity(str, Enum):
    """Severity for gate issues."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass
class GateIssue:
    """One issue emitted by a P0 control gate."""

    severity: GateSeverity
    gate: GateName
    message: str
    path: str = ""
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["severity"] = self.severity.value
        data["gate"] = self.gate.value
        return data


@dataclass
class GateResult:
    """Result of one control gate."""

    name: GateName
    passed: bool = True
    severity: GateSeverity = GateSeverity.ERROR
    blocking: bool = True
    issues: List[GateIssue] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)

    @property
    def errors(self) -> List[GateIssue]:
        return [issue for issue in self.issues if issue.severity == GateSeverity.ERROR]

    @property
    def warnings(self) -> List[GateIssue]:
        return [issue for issue in self.issues if issue.severity == GateSeverity.WARNING]

    def add_issue(
        self,
        severity: GateSeverity | str,
        message: str,
        *,
        path: str = "",
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        resolved = severity if isinstance(severity, GateSeverity) else GateSeverity(str(severity))
        self.issues.append(
            GateIssue(
                severity=resolved,
                gate=self.name,
                message=message,
                path=path,
                details=details or {},
            )
        )
        if resolved == GateSeverity.ERROR:
            self.passed = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name.value,
            "passed": self.passed,
            "severity": self.severity.value,
            "blocking": self.blocking,
            "issue_count": len(self.issues),
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "issues": [issue.to_dict() for issue in self.issues],
            "details": self.details,
        }


@dataclass
class ControlGateReport:
    """Combined P0 gate report."""

    results: List[GateResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not any(result.blocking and not result.passed for result in self.results)

    @property
    def issues(self) -> List[GateIssue]:
        return [issue for result in self.results for issue in result.issues]

    @property
    def errors(self) -> List[GateIssue]:
        return [issue for issue in self.issues if issue.severity == GateSeverity.ERROR]

    @property
    def warnings(self) -> List[GateIssue]:
        return [issue for issue in self.issues if issue.severity == GateSeverity.WARNING]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "gate_count": len(self.results),
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "results": [result.to_dict() for result in self.results],
        }


class ControlGateError(RuntimeError):
    """Raised when blocking P0 gates fail."""


def evaluate_control_gates(
    config: ControlRunConfig,
    *,
    asset_registry: AssetRegistry | None = None,
    protocol: Dict[str, Any] | None = None,
    run_dir: str | Path | None = None,
    table_json: str | Path | None = None,
    metrics: Dict[str, Any] | None = None,
) -> ControlGateReport:
    """Evaluate all P0 gates in canonical order."""
    results = [
        gate_0_params(config),
        gate_1_assets(config, asset_registry=asset_registry),
        gate_2_sampling(config),
        gate_3_frozen_run(config, protocol=protocol, run_dir=run_dir),
        gate_4_evaluation(config, metrics=metrics),
        gate_5_report(config, run_dir=run_dir, table_json=table_json),
    ]
    return ControlGateReport(results=results)


def ensure_control_gates_pass(
    config: ControlRunConfig,
    **kwargs: Any,
) -> ControlGateReport:
    """Return a report or raise with all blocking gate errors joined."""
    report = evaluate_control_gates(config, **kwargs)
    if report.passed:
        return report
    messages = "; ".join(
        f"{issue.gate.value}:{issue.path}: {issue.message}"
        for issue in report.errors
    )
    raise ControlGateError(messages)


def gate_0_params(config: ControlRunConfig) -> GateResult:
    """Validate block/stage/parameter schema before any expensive work."""
    result = GateResult(GateName.PARAMS, details={"block": config.block.value, "stage": config.stage.value})
    validation = validate_control_config(config)
    result.details["validation"] = validation.to_dict()
    for issue in validation.issues:
        severity = GateSeverity.ERROR if issue.severity == ParamSeverity.ERROR else GateSeverity.WARNING
        result.add_issue(severity, issue.message, path=issue.path)
    return result


def gate_1_assets(
    config: ControlRunConfig,
    *,
    asset_registry: AssetRegistry | None = None,
) -> GateResult:
    """Validate asset registry and reusable baseline asset boundaries."""
    result = GateResult(GateName.ASSETS)
    methods = _declared_asset_methods(config)
    registry = asset_registry or _load_asset_registry(config.assets.asset_registry)
    result.details.update(
        {
            "methods": methods,
            "reuse_assets": config.assets.reuse_assets,
            "force_rebuild": config.assets.force_rebuild,
            "asset_registry": config.assets.asset_registry,
        }
    )

    if config.assets.force_rebuild and config.assets.reuse_assets:
        result.add_issue(
            GateSeverity.ERROR,
            "force_rebuild and reuse_assets cannot both be true",
            path="assets.force_rebuild",
        )
    if config.block == ControlBlock.ASSETS and not methods:
        result.add_issue(
            GateSeverity.ERROR,
            "assets block requires at least one method",
            path="assets.methods",
        )
    if not methods:
        return result
    if not config.assets.reuse_assets:
        result.add_issue(
            GateSeverity.INFO,
            "asset reuse disabled; future control script must build fresh assets",
            path="assets.reuse_assets",
        )
        return result
    if registry is None:
        severity = GateSeverity.ERROR if config.stage == ExperimentStage.FROZEN else GateSeverity.WARNING
        result.add_issue(
            severity,
            "asset reuse requested but no readable asset registry is declared",
            path="assets.asset_registry",
        )
        return result

    ready_methods: Dict[str, str] = {}
    missing_methods: List[str] = []
    for method in methods:
        reusable = registry.resolve_reusable(
            benchmark=config.benchmark,
            method=method,
            asset_type=AssetType.BASELINE_ASSET,
            corpus_hash=config.assets.corpus_hash,
            config_hash=config.assets.config_hash,
        )
        if reusable is None:
            missing_methods.append(method)
        else:
            ready_methods[method] = reusable.asset_id
    result.details["ready_methods"] = ready_methods
    if missing_methods:
        severity = GateSeverity.ERROR if config.stage == ExperimentStage.FROZEN else GateSeverity.WARNING
        result.add_issue(
            severity,
            "no ready reusable asset found for methods: " + ", ".join(missing_methods),
            path="assets.methods",
            details={"missing_methods": missing_methods},
        )
    return result


def gate_2_sampling(config: ControlRunConfig) -> GateResult:
    """Validate sampling protocol and fixed-ID evidence for paper-claim blocks."""
    sampling = config.sampling
    result = GateResult(
        GateName.SAMPLING,
        details={
            "method": sampling.method,
            "target_n": sampling.target_n,
            "sample_ids_file": sampling.sample_ids_file,
            "sampling_protocol": sampling.sampling_protocol,
            "golden_set_file": sampling.golden_set_file,
        },
    )
    method = str(sampling.method or "").lower()
    if config.block not in {ControlBlock.MAIN, ControlBlock.ABLATION}:
        return result

    sample_sources = [
        sampling.sample_ids_file,
        sampling.golden_set_file,
        sampling.sampling_protocol,
    ]
    has_sample_source = any(str(path).strip() for path in sample_sources)
    if config.stage == ExperimentStage.FROZEN and method != "full" and not has_sample_source:
        result.add_issue(
            GateSeverity.ERROR,
            "frozen paper-claim runs require fixed sample IDs, GoldenSet, or sampling protocol",
            path="sampling.sample_ids_file",
        )
    if method == "fixed_ids" and not _path_exists(sampling.sample_ids_file):
        result.add_issue(
            GateSeverity.ERROR,
            "fixed_ids sampling requires an existing sample_ids_file",
            path="sampling.sample_ids_file",
        )
    if sampling.sampling_protocol and not _path_exists(sampling.sampling_protocol):
        result.add_issue(
            GateSeverity.ERROR,
            "declared sampling_protocol file does not exist",
            path="sampling.sampling_protocol",
        )
    if sampling.golden_set_file and not _path_exists(sampling.golden_set_file):
        result.add_issue(
            GateSeverity.ERROR,
            "declared golden_set_file does not exist",
            path="sampling.golden_set_file",
        )
    if method == "stratified" and not sampling.strata:
        result.add_issue(
            GateSeverity.WARNING,
            "stratified sampling should declare strata for auditability",
            path="sampling.strata",
        )
    return result


def gate_3_frozen_run(
    config: ControlRunConfig,
    *,
    protocol: Dict[str, Any] | None = None,
    run_dir: str | Path | None = None,
) -> GateResult:
    """Validate frozen-run protocol constraints and artifact boundary."""
    result = GateResult(GateName.FROZEN_RUN)
    resolved_protocol = protocol or _load_protocol_from_run_dir(run_dir)
    result.details["has_protocol"] = bool(resolved_protocol)
    result.details["run_dir"] = str(run_dir or "")

    if config.stage != ExperimentStage.FROZEN:
        if config.block in {ControlBlock.MAIN, ControlBlock.ABLATION}:
            result.add_issue(
                GateSeverity.ERROR,
                "paper-claim blocks must use stage=frozen",
                path="stage",
            )
        return result

    if not resolved_protocol:
        result.add_issue(
            GateSeverity.WARNING,
            "no protocol supplied yet; frozen protocol validation is pending",
            path="protocol",
        )
        return result

    valid, errors = ProtocolValidator.validate(resolved_protocol)
    result.details["protocol_valid"] = valid
    if not valid:
        for message in errors:
            result.add_issue(GateSeverity.ERROR, message, path="protocol")
    protocol_stage = str(
        resolved_protocol.get("stage")
        or (resolved_protocol.get("config") or {}).get("stage")
        or ""
    ).lower()
    if protocol_stage and protocol_stage != ExperimentStage.FROZEN.value:
        result.add_issue(
            GateSeverity.ERROR,
            "frozen control run cannot use a non-frozen protocol",
            path="protocol.stage",
        )
    return result


def gate_4_evaluation(
    config: ControlRunConfig,
    *,
    metrics: Dict[str, Any] | None = None,
) -> GateResult:
    """Validate evaluation completeness at the control boundary."""
    result = GateResult(
        GateName.EVALUATION,
        details={
            "systems": config.evaluation.systems,
            "baselines": config.evaluation.baselines,
            "has_metrics": bool(metrics),
        },
    )
    if config.block not in {ControlBlock.MAIN, ControlBlock.ABLATION}:
        return result
    if not config.evaluation.systems:
        result.add_issue(
            GateSeverity.ERROR,
            "formal evaluation requires at least one system",
            path="evaluation.systems",
        )
    if not config.evaluation.run_evaluation:
        result.add_issue(
            GateSeverity.ERROR,
            "formal paper-claim blocks must run evaluation",
            path="evaluation.run_evaluation",
        )
    if metrics:
        n_samples = _metric_number(metrics, "n", "total_samples", "num_samples")
        if n_samples <= 0:
            result.add_issue(
                GateSeverity.ERROR,
                "metrics do not record a positive sample count",
                path="metrics.n",
            )
        failure_rate = _metric_number(metrics, "failure_rate", default=-1.0)
        if failure_rate > 0.05 and config.stage == ExperimentStage.FROZEN:
            result.add_issue(
                GateSeverity.WARNING,
                "frozen evaluation failure_rate exceeds 5%",
                path="metrics.failure_rate",
                details={"failure_rate": failure_rate},
            )
    else:
        result.add_issue(
            GateSeverity.INFO,
            "no metrics supplied yet; evaluation gate will be final after run completion",
            path="metrics",
        )
    return result


def gate_5_report(
    config: ControlRunConfig,
    *,
    run_dir: str | Path | None = None,
    table_json: str | Path | None = None,
) -> GateResult:
    """Validate report/table artifacts and, when available, academic validator output."""
    result = GateResult(
        GateName.REPORT,
        details={
            "generate": config.report.generate,
            "formats": config.report.formats,
            "run_dir": str(run_dir or ""),
            "table_json": str(table_json or ""),
        },
    )
    if config.block == ControlBlock.REPORT and not config.report.generate:
        result.add_issue(
            GateSeverity.ERROR,
            "report block requires report.generate=true",
            path="report.generate",
        )
    if not config.report.generate:
        return result
    if config.block in {ControlBlock.MAIN, ControlBlock.ABLATION, ControlBlock.REPORT}:
        if table_json and not _path_exists(table_json):
            result.add_issue(
                GateSeverity.ERROR,
                "declared table_json does not exist",
                path="table_json",
            )
        if run_dir and not Path(run_dir).exists():
            result.add_issue(
                GateSeverity.ERROR,
                "declared run_dir does not exist",
                path="run_dir",
            )
        if config.report.require_validator_pass and (run_dir or table_json):
            validator_report = _run_academic_validator(run_dir=run_dir, table_json=table_json)
            result.details["validator"] = validator_report
            if not validator_report.get("passed", False):
                result.add_issue(
                    GateSeverity.ERROR,
                    "academic report validator did not pass",
                    path="report.validator",
                    details=validator_report,
                )
        elif config.stage == ExperimentStage.FROZEN:
            result.add_issue(
                GateSeverity.WARNING,
                "no run_dir/table_json supplied yet; report validator is pending",
                path="report.validator",
            )
    return result


def _declared_asset_methods(config: ControlRunConfig) -> List[str]:
    methods = list(config.assets.methods)
    if config.block in {ControlBlock.MAIN, ControlBlock.ABLATION}:
        for method in config.evaluation.baselines:
            if method not in methods:
                methods.append(method)
    return methods


def _load_asset_registry(path: str) -> AssetRegistry | None:
    if not path:
        return None
    registry_path = Path(path)
    if not registry_path.exists():
        return None
    return AssetRegistry(registry_path)


def _load_protocol_from_run_dir(run_dir: str | Path | None) -> Dict[str, Any]:
    if not run_dir:
        return {}
    path = Path(run_dir) / "protocol.yaml"
    if not path.exists():
        return {}
    try:
        return ProtocolLoader.load(path)
    except Exception:
        return {}


def _run_academic_validator(
    *,
    run_dir: str | Path | None,
    table_json: str | Path | None,
) -> Dict[str, Any]:
    try:
        try:
            from evaluation.report_validator import AcademicReportValidator
        except ImportError:
            from benchmarks.evaluation.report_validator import AcademicReportValidator
    except Exception as exc:
        return _validator_error("validator_import", exc)
    try:
        report = AcademicReportValidator().validate(run_dir=run_dir, table_json=table_json)
        return report.to_dict()
    except Exception as exc:
        return _validator_error("validator_runtime", exc)


def _validator_error(check: str, exc: Exception) -> Dict[str, Any]:
    return {
        "passed": False,
        "error_count": 1,
        "warning_count": 0,
        "issues": [
            {
                "severity": "error",
                "check": check,
                "message": str(exc),
                "path": "",
            }
        ],
    }


def _path_exists(path: str | Path | None) -> bool:
    return bool(path) and Path(path).exists()


def _metric_number(metrics: Dict[str, Any], *keys: str, default: float = 0.0) -> float:
    for key in keys:
        if key in metrics:
            try:
                return float(metrics.get(key) or 0.0)
            except (TypeError, ValueError):
                return default
    return default


def gate_report_to_json(report: ControlGateReport, path: str | Path) -> str:
    """Persist a gate report for inspection by future status/report commands."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    return str(target)


def failed_gate_names(results: Iterable[GateResult]) -> List[str]:
    """Return blocking failed gate names in deterministic order."""
    return [result.name.value for result in results if result.blocking and not result.passed]


__all__ = [
    "GateName",
    "GateSeverity",
    "GateIssue",
    "GateResult",
    "ControlGateReport",
    "ControlGateError",
    "evaluate_control_gates",
    "ensure_control_gates_pass",
    "gate_0_params",
    "gate_1_assets",
    "gate_2_sampling",
    "gate_3_frozen_run",
    "gate_4_evaluation",
    "gate_5_report",
    "gate_report_to_json",
    "failed_gate_names",
]
