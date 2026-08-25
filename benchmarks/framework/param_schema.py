"""Unified parameter schema for ResearchOps control-layer P0.

The control layer deliberately keeps configuration contracts separate from the
future CLI implementation.  This module defines the machine-readable config
objects that ``run_benchmark.py`` and sibling scripts can consume later without
having to duplicate argument parsing rules across multiple entry points.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from .control_phase import ControlBlock, ExperimentStage, validate_block_stage
from .lifecycle_schema import ResourceBudget


_ALLOWED_SAMPLING_METHODS = {
    "simple_random",
    "stratified",
    "full",
    "diagnostic_rare",
    "fixed_ids",
}
_FROZEN_CACHE_MODES = {"cold", "compiled"}


class ParamSeverity(str, Enum):
    """Severity for control-parameter validation issues."""

    ERROR = "error"
    WARNING = "warning"


@dataclass
class ParamValidationIssue:
    """One control-parameter validation issue."""

    severity: ParamSeverity
    path: str
    message: str

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["severity"] = self.severity.value
        return data


@dataclass
class ParamValidationResult:
    """Validation result for a :class:`ControlRunConfig`."""

    issues: List[ParamValidationIssue] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not any(issue.severity == ParamSeverity.ERROR for issue in self.issues)

    @property
    def errors(self) -> List[ParamValidationIssue]:
        return [issue for issue in self.issues if issue.severity == ParamSeverity.ERROR]

    @property
    def warnings(self) -> List[ParamValidationIssue]:
        return [issue for issue in self.issues if issue.severity == ParamSeverity.WARNING]

    def add_error(self, path: str, message: str) -> None:
        self.issues.append(ParamValidationIssue(ParamSeverity.ERROR, path, message))

    def add_warning(self, path: str, message: str) -> None:
        self.issues.append(ParamValidationIssue(ParamSeverity.WARNING, path, message))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "issues": [issue.to_dict() for issue in self.issues],
        }


@dataclass
class SamplingConfig:
    """Sampling and GoldenSet controls shared by main and ablation runs."""

    method: str = "stratified"
    seed: int = 42
    target_n: int = 0
    strata: List[str] = field(default_factory=list)
    allocation: str = "proportional"
    min_per_stratum: int = 1
    split: str = "validation"
    population_size: int = 0
    expected_population_size: int = 0
    sample_ids_file: str = ""
    sampling_protocol: str = ""
    golden_set_file: str = ""
    monitored_fields: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any] | None) -> "SamplingConfig":
        payload = _filter_dataclass_fields(cls, data or {})
        payload["strata"] = _as_str_list(payload.get("strata"))
        payload["monitored_fields"] = _as_str_list(payload.get("monitored_fields"))
        payload["seed"] = _as_int(payload.get("seed"), 42)
        payload["target_n"] = _as_int(payload.get("target_n"), 0)
        payload["min_per_stratum"] = _as_int(payload.get("min_per_stratum"), 1)
        payload["population_size"] = _as_int(payload.get("population_size"), 0)
        payload["expected_population_size"] = _as_int(
            payload.get("expected_population_size"), 0
        )
        return cls(**payload)


@dataclass
class AssetsConfig:
    """Baseline/index-heavy asset preparation controls."""

    methods: List[str] = field(default_factory=list)
    corpus_scale: str = "fullwiki"
    corpus_dir: str = ""
    corpus_id: str = ""
    corpus_hash: str = ""
    config_hash: str = ""
    asset_registry: str = ""
    force_rebuild: bool = False
    reuse_assets: bool = True
    validate_only: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any] | None) -> "AssetsConfig":
        payload = _filter_dataclass_fields(cls, data or {})
        payload["methods"] = _as_str_list(payload.get("methods"))
        payload["force_rebuild"] = _as_bool(payload.get("force_rebuild"), False)
        payload["reuse_assets"] = _as_bool(payload.get("reuse_assets"), True)
        payload["validate_only"] = _as_bool(payload.get("validate_only"), False)
        return cls(**payload)


@dataclass
class EvaluationConfig:
    """Formal evaluation controls that will be consumed by paper runs."""

    systems: List[str] = field(default_factory=lambda: ["sirchmunk"])
    baselines: List[str] = field(default_factory=list)
    cache_mode: str = "cold"
    limit: int = 0
    run_evaluation: bool = True
    resume: bool = True
    imported_predictions_dir: str = ""
    table_json: str = ""
    run_dir: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any] | None) -> "EvaluationConfig":
        payload = _filter_dataclass_fields(cls, data or {})
        payload["systems"] = _as_str_list(payload.get("systems")) or ["sirchmunk"]
        payload["baselines"] = _as_str_list(payload.get("baselines"))
        payload["limit"] = _as_int(payload.get("limit"), 0)
        payload["run_evaluation"] = _as_bool(payload.get("run_evaluation"), True)
        payload["resume"] = _as_bool(payload.get("resume"), True)
        return cls(**payload)


@dataclass
class ReportConfig:
    """Report/table generation controls for publication artifacts."""

    generate: bool = True
    formats: List[str] = field(default_factory=lambda: ["markdown", "latex"])
    report_dir: str = ""
    report_title: str = ""
    require_validator_pass: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any] | None) -> "ReportConfig":
        payload = _filter_dataclass_fields(cls, data or {})
        payload["generate"] = _as_bool(payload.get("generate"), True)
        payload["formats"] = _as_str_list(payload.get("formats")) or ["markdown", "latex"]
        payload["require_validator_pass"] = _as_bool(
            payload.get("require_validator_pass"), True
        )
        return cls(**payload)


@dataclass
class ControlRunConfig:
    """Top-level P0 config consumed by future benchmark-control scripts."""

    benchmark: str
    block: ControlBlock = ControlBlock.SMOKE_TUNE
    stage: ExperimentStage = ExperimentStage.EXPLORATION
    env_file: str = ""
    output_dir: str = ""
    run_id: str = ""
    seed: int = 42
    dry_run: bool = False
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    assets: AssetsConfig = field(default_factory=AssetsConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    report: ReportConfig = field(default_factory=ReportConfig)
    resource_budget: Optional[ResourceBudget] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["block"] = self.block.value
        data["stage"] = self.stage.value
        if self.resource_budget is not None:
            data["resource_budget"] = self.resource_budget.to_dict()
        return data

    def to_protocol_config(self) -> Dict[str, Any]:
        """Return the subset suitable for embedding in protocol/config snapshots."""
        data = self.to_dict()
        data.pop("metadata", None)
        return data

    def validate(self) -> ParamValidationResult:
        return validate_control_config(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ControlRunConfig":
        payload = dict(data or {})
        block = _coerce_enum(ControlBlock, payload.get("block"), ControlBlock.SMOKE_TUNE)
        default_stage = _default_stage_for_block(block)
        stage = _coerce_enum(ExperimentStage, payload.get("stage"), default_stage)
        return cls(
            benchmark=str(payload.get("benchmark", "")),
            block=block,
            stage=stage,
            env_file=str(payload.get("env_file", "")),
            output_dir=str(payload.get("output_dir", "")),
            run_id=str(payload.get("run_id", "")),
            seed=_as_int(payload.get("seed"), 42),
            dry_run=_as_bool(payload.get("dry_run"), False),
            sampling=SamplingConfig.from_dict(payload.get("sampling")),
            assets=AssetsConfig.from_dict(payload.get("assets")),
            evaluation=EvaluationConfig.from_dict(payload.get("evaluation")),
            report=ReportConfig.from_dict(payload.get("report")),
            resource_budget=resource_budget_from_dict(payload.get("resource_budget")),
            metadata=dict(payload.get("metadata") or {}),
        )


@dataclass
class ControlReusePolicy:
    """Explicit reuse boundary for assets and run artifacts."""

    reuse_assets: bool = True
    force_rebuild_assets: bool = False
    reuse_run_results: bool = False
    allow_cross_stage_reuse: bool = False
    require_config_hash_match: bool = True
    require_corpus_hash_match: bool = True
    require_sample_checksum_match: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_config(cls, config: ControlRunConfig) -> "ControlReusePolicy":
        return cls(
            reuse_assets=config.assets.reuse_assets,
            force_rebuild_assets=config.assets.force_rebuild,
            reuse_run_results=False,
            allow_cross_stage_reuse=False,
            require_config_hash_match=True,
            require_corpus_hash_match=True,
            require_sample_checksum_match=config.stage == ExperimentStage.FROZEN,
        )


class ControlConfigError(ValueError):
    """Raised when a control config cannot pass P0 parameter validation."""



def validate_control_config(config: ControlRunConfig) -> ParamValidationResult:
    """Validate a future total-control run before any expensive work starts."""
    result = ParamValidationResult()
    if not str(config.benchmark).strip():
        result.add_error("benchmark", "benchmark is required")
    try:
        validate_block_stage(config.block, config.stage)
    except ValueError as exc:
        result.add_error("stage", str(exc))

    if not str(config.output_dir).strip():
        result.add_warning(
            "output_dir",
            "output_dir is empty; future scripts should derive it from the adapter",
        )
    if config.stage == ExperimentStage.FROZEN and config.dry_run:
        result.add_error("dry_run", "frozen control runs cannot be dry runs")
    if config.stage == ExperimentStage.FROZEN:
        cache_mode = str(config.evaluation.cache_mode or "").lower()
        if cache_mode not in _FROZEN_CACHE_MODES:
            result.add_error(
                "evaluation.cache_mode",
                "frozen evaluation cache_mode must be one of: cold, compiled",
            )

    _validate_sampling_config(config, result)
    _validate_assets_config(config, result)
    _validate_evaluation_config(config, result)
    _validate_report_config(config, result)
    _validate_resource_budget(config.resource_budget, result)
    return result


def ensure_valid_control_config(config: ControlRunConfig) -> ControlRunConfig:
    """Return ``config`` or raise with all P0 parameter errors joined."""
    validation = validate_control_config(config)
    if validation.ok:
        return config
    messages = "; ".join(f"{issue.path}: {issue.message}" for issue in validation.errors)
    raise ControlConfigError(messages)


def resource_budget_from_dict(data: Any) -> Optional[ResourceBudget]:
    """Coerce a mapping into the existing lifecycle ``ResourceBudget`` schema."""
    if data is None or data == "":
        return None
    if isinstance(data, ResourceBudget):
        return data
    if not isinstance(data, dict):
        return None
    return ResourceBudget(
        wall_clock_seconds=_as_float(data.get("wall_clock_seconds"), 0.0),
        max_ram_bytes=_as_int(data.get("max_ram_bytes"), 0),
        max_disk_bytes=_as_int(data.get("max_disk_bytes"), 0),
        max_llm_tokens=_as_int(data.get("max_llm_tokens"), 0),
        max_api_cost_usd=_as_float(data.get("max_api_cost_usd"), 0.0),
        retry_count=_as_int(data.get("retry_count"), 0),
    )


def _validate_sampling_config(
    config: ControlRunConfig,
    result: ParamValidationResult,
) -> None:
    sampling = config.sampling
    method = str(sampling.method or "").lower()
    if method not in _ALLOWED_SAMPLING_METHODS:
        allowed = ", ".join(sorted(_ALLOWED_SAMPLING_METHODS))
        result.add_error("sampling.method", f"unknown sampling method: {method}; allowed: {allowed}")
    if sampling.seed < 0:
        result.add_error("sampling.seed", "seed must be non-negative")
    if sampling.target_n < 0:
        result.add_error("sampling.target_n", "target_n must be non-negative")
    if sampling.min_per_stratum < 0:
        result.add_error("sampling.min_per_stratum", "min_per_stratum must be non-negative")
    if method == "fixed_ids" and not sampling.sample_ids_file:
        result.add_error("sampling.sample_ids_file", "fixed_ids sampling requires sample_ids_file")
    if method == "stratified" and not sampling.strata:
        result.add_warning("sampling.strata", "stratified sampling should declare strata")
    if config.block in {ControlBlock.MAIN, ControlBlock.ABLATION}:
        has_fixed_sample = bool(sampling.sample_ids_file or sampling.golden_set_file)
        if method not in {"full", "fixed_ids"} and sampling.target_n <= 0:
            result.add_error(
                "sampling.target_n",
                "paper-claim blocks require target_n, full, or fixed_ids sampling",
            )
        if config.stage == ExperimentStage.FROZEN and not has_fixed_sample:
            result.add_warning(
                "sampling.sample_ids_file",
                "frozen paper runs should persist or load fixed sample IDs",
            )


def _validate_assets_config(
    config: ControlRunConfig,
    result: ParamValidationResult,
) -> None:
    assets = config.assets
    if assets.force_rebuild and assets.reuse_assets:
        result.add_error(
            "assets.force_rebuild",
            "force_rebuild and reuse_assets cannot both be true",
        )
    if config.block == ControlBlock.ASSETS and not assets.methods:
        result.add_error("assets.methods", "assets block requires at least one method")
    if assets.asset_registry:
        suffix = Path(assets.asset_registry).suffix
        if suffix and suffix != ".jsonl":
            result.add_warning("assets.asset_registry", "asset registry should be a JSONL file")


def _validate_evaluation_config(
    config: ControlRunConfig,
    result: ParamValidationResult,
) -> None:
    evaluation = config.evaluation
    if evaluation.limit < 0:
        result.add_error("evaluation.limit", "limit must be non-negative")
    if config.block in {ControlBlock.MAIN, ControlBlock.ABLATION} and not evaluation.systems:
        result.add_error("evaluation.systems", "paper-claim blocks require systems")
    if config.block == ControlBlock.MAIN and not evaluation.run_evaluation:
        result.add_error("evaluation.run_evaluation", "main block must run evaluation")
    if config.stage == ExperimentStage.FROZEN and evaluation.limit > 0:
        result.add_warning(
            "evaluation.limit",
            "frozen runs should prefer sampling protocol over ad hoc limit",
        )


def _validate_report_config(
    config: ControlRunConfig,
    result: ParamValidationResult,
) -> None:
    report = config.report
    if config.block == ControlBlock.REPORT and not report.generate:
        result.add_error("report.generate", "report block requires generate=true")
    allowed = {"markdown", "latex", "json"}
    unknown = sorted(set(report.formats) - allowed)
    if unknown:
        result.add_warning("report.formats", f"unknown report formats: {unknown}")


def _validate_resource_budget(
    budget: Optional[ResourceBudget],
    result: ParamValidationResult,
) -> None:
    if budget is None:
        return
    values = budget.to_dict()
    for key, value in values.items():
        try:
            number = float(value or 0)
        except (TypeError, ValueError):
            result.add_error(f"resource_budget.{key}", "resource budget values must be numeric")
            continue
        if number < 0:
            result.add_error(f"resource_budget.{key}", "resource budget values must be non-negative")


def _default_stage_for_block(block: ControlBlock) -> ExperimentStage:
    if block in {ControlBlock.MAIN, ControlBlock.ABLATION}:
        return ExperimentStage.FROZEN
    return ExperimentStage.EXPLORATION


def _coerce_enum(enum_cls: type[Enum], value: Any, default: Any) -> Any:
    if isinstance(value, enum_cls):
        return value
    if value is None or value == "":
        return default
    return enum_cls(str(value))


def _filter_dataclass_fields(cls: Any, data: Dict[str, Any]) -> Dict[str, Any]:
    allowed = set(cls.__dataclass_fields__)  # type: ignore[attr-defined]
    return {key: value for key, value in dict(data).items() if key in allowed}


def _as_str_list(value: Any) -> List[str]:
    if value is None or value == "":
        return []
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()]


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


__all__ = [
    "ParamSeverity",
    "ParamValidationIssue",
    "ParamValidationResult",
    "SamplingConfig",
    "AssetsConfig",
    "EvaluationConfig",
    "ReportConfig",
    "ControlRunConfig",
    "ControlReusePolicy",
    "ControlConfigError",
    "validate_control_config",
    "ensure_valid_control_config",
    "resource_budget_from_dict",
]
