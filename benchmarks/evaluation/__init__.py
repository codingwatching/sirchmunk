"""evaluation — competitor comparison and paper table generation package

Fully isolated from the framework/ self-improvement loop and dedicated to:
  1. managing the frozen test set (GoldenSet)
  2. running the evaluation for every competitor (BaselineEvaluationSuite)
  3. statistical analysis (bootstrap CI, McNemar, Bonferroni)
  4. generating paper-grade comparison tables (LaTeX + Markdown + JSON)
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
from .dynamic_table_generator import DynamicPaperTableGenerator

__all__ = [
    "GoldenSet",
    "GoldenSetManager",
    "BaselineEvaluationSuite",
    "PaperTableGenerator",
    "DynamicPaperTableGenerator",
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
