"""framework/schema.py — 通用数据契约

跨 benchmark 通用数据结构，不依赖任何具体 benchmark 实现。
所有字段均为显式类型标注，以便静态检查和文档自动生成。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# 变更类型枚举
# ---------------------------------------------------------------------------


class ChangeType(str, Enum):
    """改进建议的变更类型。

    CONFIG_CHANGE  — 仅修改 .env 变量，框架可自动执行
    PROMPT_FIX     — 修改 prompts.py 等 prompt 模板，需人工/Qoder 执行
    PIPELINE_PATCH — 修改核心检索逻辑（src/），需人工/Qoder 执行
    SKIP           — 跳过本轮改动，继续下一次迭代
    """
    CONFIG_CHANGE = "config_change"
    PROMPT_FIX = "prompt_fix"
    PIPELINE_PATCH = "pipeline_patch"
    SKIP = "skip"


class ConfigLayer(int, Enum):
    """Config 三层隔离层级。

    GLOBAL   (0) — Layer 0: 全局变更，影响所有 benchmark。
                    包括: src/sirchmunk/ 代码修改、共享 prompt、
                    全局 env（LLM_MODEL_NAME 等）。
                    接受条件: 所有 benchmark 均不出现 Pareto 退化。

    FAMILY   (1) — Layer 1: 家族变更，影响同类 benchmark。
                    预留用于未来扩展，当前尚未启用。

    SPECIFIC (2) — Layer 2: benchmark 专属变更，仅影响本 benchmark。
                    包括: HOTPOT_TOP_K_FILES、HOTPOT_MODE 等专属配置。
                    接受条件: 本 benchmark 指标提升即可，无需联合评估。
    """
    GLOBAL   = 0
    FAMILY   = 1
    SPECIFIC = 2


class RootCause(str, Enum):
    """根因分类。"""
    RETRIEVAL_FAILURE = "retrieval_failure"   # 没有找到相关文件/段落
    EVIDENCE_PARTIAL  = "evidence_partial"    # 找到部分证据但不完整
    SYNTHESIS_ERROR   = "synthesis_error"     # 推理/计算错误
    JUDGE_SUSPECT     = "judge_suspect"       # Judge 可能误判（false negative）
    DATA_QUALITY      = "data_quality"        # 问题或金标准本身有疑问
    UNKNOWN           = "unknown"


class ImpactLevel(str, Enum):
    LOW    = "low"
    MEDIUM = "medium"
    HIGH   = "high"


# ---------------------------------------------------------------------------
# 核心数据结构
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkSample:
    """通用 QA 样本，跨 benchmark 共享。"""
    sample_id: str
    question: str
    gold_answer: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    """metadata 存放 benchmark 特有字段，如 type, level, supporting_facts 等。"""


@dataclass
class PredictionResult:
    """单样本执行结果。"""
    sample_id: str
    prediction: str
    judge_correct: bool
    coverage: bool
    elapsed: float                          # 秒
    telemetry: Dict[str, Any] = field(default_factory=dict)
    """telemetry 含 total_tokens / loop_count / num_files_read 等。"""
    error: Optional[str] = None
    # 保留原始 benchmark 特有字段供 BadCaseAnalyzer 使用
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BadCase:
    """单个失败样本的分类结果。"""
    sample_id: str
    question: str
    gold_answer: str
    prediction: str
    failure_type: str               # refusal / wrong_value / no_coverage / partial_answer
    root_cause: RootCause
    evidence: str = ""              # 支持该根因判断的关键信号，简短描述
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BadCaseReport:
    """坏案例分析报告。"""
    total_samples: int
    total_badcases: int
    accuracy: float
    coverage: float
    badcases: List[BadCase] = field(default_factory=list)

    # 失败类型分布 {failure_type: count}
    failure_type_breakdown: Dict[str, int] = field(default_factory=dict)
    # 根因分布 {root_cause: count}
    root_cause_breakdown: Dict[str, int] = field(default_factory=dict)
    # 按 question_type 分层统计
    by_question_type: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # LLM 归纳的失败模式（单次 LLM call，可为空字符串）
    pattern_summary: str = ""
    # 需人工审查的可疑 judge 样本
    judge_suspect_ids: List[str] = field(default_factory=list)


@dataclass
class ImprovementHypothesis:
    """单条改进假设。"""
    hypothesis_id: str
    title: str
    root_cause: str
    change_type: ChangeType
    description: str
    estimated_impact: ImpactLevel
    risk_level: str = "low"         # low / medium / high

    # Config 三层隔离层级（Layer 0=全局 / 1=家族 / 2=benchmark专属）
    config_layer: ConfigLayer = ConfigLayer.SPECIFIC

    # CONFIG_CHANGE 时填充：{env_key: new_value}
    config_changes: Dict[str, str] = field(default_factory=dict)
    # 受影响的 .env 文件路径（相对于 benchmarks/）
    env_file: str = ""

    # PROMPT_FIX / PIPELINE_PATCH 时填充：文字描述修改点
    code_guidance: str = ""
    # 预计影响的 badcase 比例
    estimated_coverage_fraction: float = 0.0


@dataclass
class ExperimentRecord:
    """一次完整实验快照，持久化到 experiments.jsonl。"""
    run_id: str
    benchmark: str
    timestamp: str                  # ISO 8601
    git_commit: str
    config_hash: str
    metrics: Dict[str, Any] = field(default_factory=dict)
    results_path: str = ""
    notes: str = ""
    is_regression: bool = False     # 若 accuracy < 上一次 - 2% 则标记
