"""benchmarks/framework — Research Loop Framework

Unified flow: benchmark onboarding -> experiment execution -> badcase analysis ->
improvement hypotheses -> manual confirmation -> iterate.
Supports both a single benchmark (ResearchOrchestrator) and multi-benchmark joint
optimization (MultiAdapterOrchestrator).
Quick import example::

    from benchmarks.framework import (
        BenchmarkAdapter,
        BadCaseAnalyzer,
        ImprovementAdvisor,
        HumanConfirmLoop,
        ExperimentTracker,
        ResearchOrchestrator,
        UnifiedExperimentRunner,
        RunArtifactManager,
        ExperimentProtocol,
        MetricEngine,
        StatisticalAnalyzer,
        load_benchmark_adapter,
        supported_benchmarks,
        # Joint optimization
        MultiAdapterOrchestrator,
        ParetoTracker,
        ShadowEvaluator,
    )
"""
from .adapter import BenchmarkAdapter
from .analyzer import BadCaseAnalyzer
from .artifact import RunArtifactManager
from .cache_manager import CacheActionReport, CacheManager, CacheMode
from .checkpoint import CheckpointManager, CheckpointRecord
from .asset_registry import AssetRecord, AssetRegistry, AssetStatus, AssetType, compute_asset_id
from .control_gates import (
    ControlGateError,
    ControlGateReport,
    GateIssue,
    GateName,
    GateResult,
    GateSeverity,
    ensure_control_gates_pass,
    evaluate_control_gates,
    failed_gate_names,
    gate_0_params,
    gate_1_assets,
    gate_2_sampling,
    gate_3_frozen_run,
    gate_4_evaluation,
    gate_5_report,
    gate_report_to_json,
)
from .control_phase import (
    ControlBlock,
    ControlOutputLayout,
    ExperimentStage,
    allowed_stages,
    for_benchmark_output_dir,
    validate_block_stage,
)
from .experiment_queue import ExperimentQueue, QueueExecutor, QueueTask, QueueTaskStatus
from .experiment_registry import ExperimentRegistry, ExperimentRegistryRecord
from .guards import (
    BenchmarkTimeout,
    BudgetExceeded,
    BudgetGuard,
    GlobalTimeout,
    GuardConfig,
    SampleTimeout,
    SystemTimeout,
    TimeoutGuard,
)
from .retry import RetryConfig, RetryExhausted, RetryPolicy, RetryResult
from .advisor import ImprovementAdvisor
from .confirm import HumanConfirmLoop
from .metric_engine import MetricEngine
from .multi_orchestrator import MultiAdapterOrchestrator
from .orchestrator import ResearchOrchestrator
from .pareto import MultiDelta, MultiMetricsPoint, ParetoTracker
from .param_schema import (
    AssetsConfig,
    ControlConfigError,
    ControlReusePolicy,
    ControlRunConfig,
    EvaluationConfig,
    ParamSeverity,
    ParamValidationIssue,
    ParamValidationResult,
    ReportConfig,
    SamplingConfig,
    ensure_valid_control_config,
    resource_budget_from_dict,
    validate_control_config,
)
from .protocol import ExperimentProtocol, ProtocolLoader, ProtocolValidator
from .registry import load_benchmark_adapter, supported_benchmarks
from .runner import UnifiedExperimentRunner
from .run_state import RunState, RunStateMachine, RunStateStore, RunStatus
from .run_summary import (
    ControlRunStatus,
    ControlRunSummary,
    GateSummary,
    StageSummary,
    create_control_run_summary,
    load_summary,
    save_summary,
    summarize_assets,
)
from .statistical_analyzer import StatisticalAnalyzer
from .schema import (
    BadCase,
    BadCaseReport,
    BenchmarkSample,
    ChangeType,
    ConfigLayer,
    ExperimentRecord,
    ImpactLevel,
    ImprovementHypothesis,
    PredictionResult,
    RootCause,
)
from .shadow import BmShadowResult, ShadowEvaluator, ShadowImpactMatrix
from .tracker import ExperimentTracker
from .dynamic_stage_runner import DynamicStageBinding, StageExecutionRecord, build_stage_bindings, validate_result_reuse

__all__ = [
    "BenchmarkAdapter",
    "BadCaseAnalyzer",
    "ImprovementAdvisor",
    "HumanConfirmLoop",
    "MultiAdapterOrchestrator",
    "ResearchOrchestrator",
    "UnifiedExperimentRunner",
    "ExperimentTracker",
    "RunArtifactManager",
    "CacheActionReport",
    "CacheManager",
    "CacheMode",
    "CheckpointManager",
    "CheckpointRecord",
    "AssetRecord",
    "AssetRegistry",
    "AssetStatus",
    "AssetType",
    "compute_asset_id",
    "ControlGateError",
    "ControlGateReport",
    "GateIssue",
    "GateName",
    "GateResult",
    "GateSeverity",
    "ensure_control_gates_pass",
    "evaluate_control_gates",
    "failed_gate_names",
    "gate_0_params",
    "gate_1_assets",
    "gate_2_sampling",
    "gate_3_frozen_run",
    "gate_4_evaluation",
    "gate_5_report",
    "gate_report_to_json",
    "ControlBlock",
    "ControlOutputLayout",
    "ExperimentStage",
    "allowed_stages",
    "for_benchmark_output_dir",
    "validate_block_stage",
    "ExperimentQueue",
    "QueueExecutor",
    "QueueTask",
    "QueueTaskStatus",
    "ExperimentRegistry",
    "ExperimentRegistryRecord",
    "BudgetGuard",
    "BudgetExceeded",
    "GuardConfig",
    "TimeoutGuard",
    "SampleTimeout",
    "SystemTimeout",
    "BenchmarkTimeout",
    "GlobalTimeout",
    "RetryConfig",
    "RetryPolicy",
    "RetryResult",
    "RetryExhausted",
    "RunState",
    "RunStateMachine",
    "RunStateStore",
    "RunStatus",
    "ControlRunStatus",
    "ControlRunSummary",
    "GateSummary",
    "StageSummary",
    "create_control_run_summary",
    "load_summary",
    "save_summary",
    "summarize_assets",
    "ExperimentProtocol",
    "MetricEngine",
    "StatisticalAnalyzer",
    "ProtocolLoader",
    "ProtocolValidator",
    "load_benchmark_adapter",
    "supported_benchmarks",
    # Pareto joint optimization
    "MultiDelta",
    "MultiMetricsPoint",
    "ParetoTracker",
    "AssetsConfig",
    "ControlConfigError",
    "ControlReusePolicy",
    "ControlRunConfig",
    "EvaluationConfig",
    "ParamSeverity",
    "ParamValidationIssue",
    "ParamValidationResult",
    "ReportConfig",
    "SamplingConfig",
    "ensure_valid_control_config",
    "resource_budget_from_dict",
    "validate_control_config",
    "BmShadowResult",
    "ShadowEvaluator",
    "ShadowImpactMatrix",
    # dynamic evaluation
    "StageExecutionRecord",
    "DynamicStageBinding",
    "build_stage_bindings",
    "validate_result_reuse",
    # schema
    "BadCase",
    "BadCaseReport",
    "BenchmarkSample",
    "ChangeType",
    "ConfigLayer",
    "ExperimentRecord",
    "ImpactLevel",
    "ImprovementHypothesis",
    "PredictionResult",
    "RootCause",
]
