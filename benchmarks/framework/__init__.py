"""benchmarks/framework — Research Loop Framework

统一 benchmark 接入 → 实验执行 → badcase 分析 → 改进建议 → 人工确认 → 循环迭代。
支持单 benchmark （ResearchOrchestrator）和多 benchmark 联合优化（MultiAdapterOrchestrator）。
快速导入示例::

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
        # 联合优化
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
from .protocol import ExperimentProtocol, ProtocolLoader, ProtocolValidator
from .registry import load_benchmark_adapter, supported_benchmarks
from .runner import UnifiedExperimentRunner
from .run_state import RunState, RunStateMachine, RunStateStore, RunStatus
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
    "ExperimentProtocol",
    "MetricEngine",
    "StatisticalAnalyzer",
    "ProtocolLoader",
    "ProtocolValidator",
    "load_benchmark_adapter",
    "supported_benchmarks",
    # Pareto 联合优化
    "MultiDelta",
    "MultiMetricsPoint",
    "ParetoTracker",
    "BmShadowResult",
    "ShadowEvaluator",
    "ShadowImpactMatrix",
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
