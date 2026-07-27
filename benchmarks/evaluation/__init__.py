"""evaluation — 竞品横向评估与论文表格生成包

与 framework/ 自改进循环完全隔离，专用于：
  1. 管理固定测试集（GoldenSet）
  2. 对所有竞品运行评估（BaselineEvaluationSuite）
  3. 统计分析（Bootstrap CI, McNemar, Bonferroni）
  4. 生成论文级比较表格（LaTeX + Markdown + JSON）
"""
from .error_appendix import ErrorAppendixGenerator
from .figure_generator import FigureGenerator
from .golden_set import GoldenSet, GoldenSetManager
from .report_generator import ReportGenerator
from .report_validator import AcademicReportValidator, ValidationReport
from .reproducibility import ReproducibilityChecklist
from .statistics import (
    bonferroni_correction,
    bootstrap_ci,
    cohens_h,
    holm_correction,
    mcnemar_test,
    paired_bootstrap_delta,
    significance_marker,
)
from .suite import BaselineEvaluationSuite
from .table_generator import PaperTableGenerator, SystemEntry
from .v4_table_generator import V4PaperTableGenerator

__all__ = [
    "GoldenSet",
    "GoldenSetManager",
    "BaselineEvaluationSuite",
    "PaperTableGenerator",
    "V4PaperTableGenerator",
    "SystemEntry",
    "ReportGenerator",
    "AcademicReportValidator",
    "ValidationReport",
    "ReproducibilityChecklist",
    "ErrorAppendixGenerator",
    "FigureGenerator",
    "bootstrap_ci",
    "mcnemar_test",
    "bonferroni_correction",
    "holm_correction",
    "paired_bootstrap_delta",
    "cohens_h",
    "significance_marker",
]
