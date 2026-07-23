"""Class-based statistical analysis facade for ResearchOps reports."""
from __future__ import annotations

from typing import Any, Dict, List

from evaluation.statistics import (
    bonferroni_correction,
    bootstrap_ci,
    cohens_h,
    holm_correction,
    mcnemar_test,
    paired_bootstrap_delta,
    significance_marker,
)


class StatisticalAnalyzer:
    """Run paired statistical tests for paper tables."""

    def compare_correctness(
        self,
        ours: List[bool],
        baseline: List[bool],
        *,
        alpha: float = 0.05,
    ) -> Dict[str, Any]:
        chi2, p_value, significant = mcnemar_test(ours, baseline)
        ours_acc, ours_lo, ours_hi = bootstrap_ci(ours)
        base_acc, base_lo, base_hi = bootstrap_ci(baseline)
        return {
            "ours_accuracy": ours_acc,
            "ours_ci": [ours_lo, ours_hi],
            "baseline_accuracy": base_acc,
            "baseline_ci": [base_lo, base_hi],
            "mcnemar_chi2": chi2,
            "p_value": p_value,
            "significant": significant and p_value < alpha,
            "significance_marker": significance_marker(p_value),
            "effect_size_cohens_h": cohens_h(ours_acc, base_acc),
        }

    def compare_many(
        self,
        ours: List[bool],
        baselines: Dict[str, List[bool]],
        *,
        correction: str = "holm",
    ) -> Dict[str, Dict[str, Any]]:
        raw: Dict[str, Dict[str, Any]] = {}
        p_values: List[float] = []
        names: List[str] = []
        for name, values in baselines.items():
            if len(values) != len(ours):
                raw[name] = {"error": "sample_length_mismatch", "n": len(values)}
                continue
            result = self.compare_correctness(ours, values)
            raw[name] = result
            p_values.append(result["p_value"])
            names.append(name)
        corrected = holm_correction(p_values) if correction == "holm" else bonferroni_correction(p_values)
        for name, is_sig in zip(names, corrected):
            raw[name]["multiple_comparison_correction"] = correction
            raw[name]["corrected_significant"] = is_sig
            raw[name]["significance_marker_corrected"] = significance_marker(raw[name]["p_value"]) if is_sig else ""
        return raw

    def paired_delta_ci(self, metric_a: List[float], metric_b: List[float]) -> Dict[str, Any]:
        observed, low, high = paired_bootstrap_delta(metric_a, metric_b)
        return {"delta": observed, "ci": [low, high]}
