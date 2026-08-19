"""framework/schema.py — shared data contracts

Cross-benchmark data structures that do not depend on any concrete benchmark
implementation. Every field carries an explicit type annotation so static checking and
documentation generation both work.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Change-type enumeration
# ---------------------------------------------------------------------------


class ChangeType(str, Enum):
    """Change type of an improvement hypothesis.

    CONFIG_CHANGE  — only .env variables change; the framework can apply it automatically
    PROMPT_FIX     — change a prompt template such as prompts.py; needs a human/Qoder
    PIPELINE_PATCH — change core retrieval logic under src/; needs a human/Qoder
    SKIP           — skip this round and continue with the next iteration
    """
    CONFIG_CHANGE = "config_change"
    PROMPT_FIX = "prompt_fix"
    PIPELINE_PATCH = "pipeline_patch"
    SKIP = "skip"


class ConfigLayer(int, Enum):
    """Three-layer config isolation.

    GLOBAL   (0) — Layer 0: global change affecting every benchmark.
                    Covers src/sirchmunk/ code changes, shared prompts and global env
                    values such as LLM_MODEL_NAME.
                    Acceptance: no benchmark may show a Pareto regression.

    FAMILY   (1) — Layer 1: family change affecting benchmarks of the same kind.
                    Reserved for future extension, not enabled yet.

    SPECIFIC (2) — Layer 2: benchmark-specific change affecting this benchmark only.
                    Covers dedicated config such as HOTPOT_TOP_K_FILES or HOTPOT_MODE.
                    Acceptance: an improvement on this benchmark is enough, no joint
                    evaluation required.
    """
    GLOBAL   = 0
    FAMILY   = 1
    SPECIFIC = 2


class RootCause(str, Enum):
    """Root-cause classification."""
    RETRIEVAL_FAILURE = "retrieval_failure"   # No relevant file or passage was found
    EVIDENCE_PARTIAL  = "evidence_partial"    # Partial evidence was found but it is incomplete
    SYNTHESIS_ERROR   = "synthesis_error"     # Reasoning or computation error
    JUDGE_SUSPECT     = "judge_suspect"       # The judge may have mis-scored it (false negative)
    DATA_QUALITY      = "data_quality"        # The question or the gold answer itself is questionable
    UNKNOWN           = "unknown"


class ImpactLevel(str, Enum):
    LOW    = "low"
    MEDIUM = "medium"
    HIGH   = "high"


# ---------------------------------------------------------------------------
# Core data structures
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkSample:
    """Generic QA sample shared across benchmarks."""
    sample_id: str
    question: str
    gold_answer: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    """metadata 存放 benchmark 特有字段，如 type, level, supporting_facts 等。"""


@dataclass
class PredictionResult:
    """Execution result of a single sample."""
    sample_id: str
    prediction: str
    judge_correct: bool
    coverage: bool
    elapsed: float                          # Seconds
    telemetry: Dict[str, Any] = field(default_factory=dict)
    """telemetry 含 total_tokens / loop_count / num_files_read 等。"""
    error: Optional[str] = None
    # Keep benchmark-specific raw fields for BadCaseAnalyzer
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BadCase:
    """Classification result of a single failing sample."""
    sample_id: str
    question: str
    gold_answer: str
    prediction: str
    failure_type: str               # refusal / wrong_value / no_coverage / partial_answer
    root_cause: RootCause
    evidence: str = ""              # Short description of the signals supporting this root cause
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BadCaseReport:
    """Badcase analysis report."""
    total_samples: int
    total_badcases: int
    accuracy: float
    coverage: float
    badcases: List[BadCase] = field(default_factory=list)

    # Failure-type distribution {failure_type: count}
    failure_type_breakdown: Dict[str, int] = field(default_factory=dict)
    # Root-cause distribution {root_cause: count}
    root_cause_breakdown: Dict[str, int] = field(default_factory=dict)
    # Statistics stratified by question_type
    by_question_type: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # LLM-induced failure patterns from a single call, may be empty
    pattern_summary: str = ""
    # Suspect judge samples that need manual review
    judge_suspect_ids: List[str] = field(default_factory=list)


@dataclass
class ImprovementHypothesis:
    """A single improvement hypothesis."""
    hypothesis_id: str
    title: str
    root_cause: str
    change_type: ChangeType
    description: str
    estimated_impact: ImpactLevel
    risk_level: str = "low"         # low / medium / high

    # Three-layer config isolation (Layer 0=global / 1=family / 2=benchmark-specific)
    config_layer: ConfigLayer = ConfigLayer.SPECIFIC

    # Filled for CONFIG_CHANGE: {env_key: new_value}
    config_changes: Dict[str, str] = field(default_factory=dict)
    # Affected .env file paths, relative to benchmarks/
    env_file: str = ""

    # Filled for PROMPT_FIX / PIPELINE_PATCH: textual description of the change
    code_guidance: str = ""
    # Estimated share of badcases affected
    estimated_coverage_fraction: float = 0.0


@dataclass
class ExperimentRecord:
    """One complete experiment snapshot, persisted to experiments.jsonl."""
    run_id: str
    benchmark: str
    timestamp: str                  # ISO 8601
    git_commit: str
    config_hash: str
    metrics: Dict[str, Any] = field(default_factory=dict)
    results_path: str = ""
    notes: str = ""
    is_regression: bool = False     # Flagged when accuracy drops more than 2% below the previous run
