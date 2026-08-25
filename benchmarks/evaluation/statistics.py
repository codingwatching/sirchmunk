"""evaluation/statistics.py — paper-grade statistical analysis

Provides the statistical rigor behind competitor comparisons:
  bootstrap_ci()          : bootstrap confidence interval (95% CI of accuracy)
  paired_bootstrap_delta(): paired bootstrap CI of the difference
  mcnemar_test()          : McNemar paired significance test on two correct vectors
  bonferroni_correction() : Bonferroni multiple-comparison correction
  holm_correction()       : Holm-Bonferroni multiple-comparison correction
  cohens_h()              : effect size of a difference between two proportions

Every function is pure (no side effects) and depends on no framework/ code.

McNemar test rationale:
  Applies to a paired comparison of two classifiers on the same samples.
  H0: the two systems have the same error pattern.
  The statistic is based on the count of discordant pairs where one is right and the
  other is wrong.

Bootstrap CI rationale:
  Resample the correct list with replacement n_bootstrap times, compute the accuracy of
  each resample, and take the alpha/2 and 1-alpha/2 percentiles as the CI.
"""
from __future__ import annotations

import math
import random
from typing import List, Tuple


def bootstrap_ci(
    correct: List[bool],
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Compute the bootstrap confidence interval of accuracy.

    Args:
        correct:     per-sample judge_correct booleans.
        n_bootstrap: number of bootstrap resamples (default 1000).
        alpha:       significance level (default 0.05, i.e. 95% CI).
        seed:        random seed that keeps results reproducible.

    Returns:
        (accuracy, lower_bound, upper_bound), where accuracy = mean(correct) and the CI
        comes from percentiles of the bootstrap distribution.
    """
    n = len(correct)
    if n == 0:
        return 0.0, 0.0, 0.0

    accuracy = sum(correct) / n
    rng = random.Random(seed)

    bootstrap_accuracies: List[float] = []
    for _ in range(n_bootstrap):
        sample = [correct[rng.randint(0, n - 1)] for _ in range(n)]
        bootstrap_accuracies.append(sum(sample) / n)

    bootstrap_accuracies.sort()
    lower_idx = int(alpha / 2 * n_bootstrap)
    upper_idx = int((1 - alpha / 2) * n_bootstrap)
    lower = bootstrap_accuracies[max(0, lower_idx)]
    upper = bootstrap_accuracies[min(n_bootstrap - 1, upper_idx)]

    return accuracy, lower, upper


def paired_bootstrap_delta(
    metric_a: List[float],
    metric_b: List[float],
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Paired bootstrap CI for mean(metric_a - metric_b)."""
    if len(metric_a) != len(metric_b):
        raise ValueError(f"Sample length mismatch: len(a)={len(metric_a)}, len(b)={len(metric_b)}")
    n = len(metric_a)
    if n == 0:
        return 0.0, 0.0, 0.0
    observed = sum(a - b for a, b in zip(metric_a, metric_b)) / n
    rng = random.Random(seed)
    samples: List[float] = []
    for _ in range(n_bootstrap):
        total = 0.0
        for _ in range(n):
            idx = rng.randint(0, n - 1)
            total += metric_a[idx] - metric_b[idx]
        samples.append(total / n)
    samples.sort()
    lower_idx = int(alpha / 2 * n_bootstrap)
    upper_idx = int((1 - alpha / 2) * n_bootstrap)
    return observed, samples[max(0, lower_idx)], samples[min(n_bootstrap - 1, upper_idx)]


def mcnemar_test(
    correct_a: List[bool],
    correct_b: List[bool],
) -> Tuple[float, float, bool]:
    """McNemar paired significance test.

    Applies to a comparison of two systems on the same test set (paired design).
    H0: the two systems have the same error rate (n01 = n10).

    Uses the Yates continuity-corrected form to avoid over-significance on small samples:
      chi2 = (|n01 - n10| - 1)^2 / (n01 + n10)

    Args:
        correct_a: judge_correct list of system A (same order).
        correct_b: judge_correct list of system B (same order).

    Returns:
        (chi_squared, p_value, is_significant_at_0.05)

    Raises:
        ValueError: when the two lists differ in length or the sample size is too small.
    """
    if len(correct_a) != len(correct_b):
        raise ValueError(
            f"Sample length mismatch: len(a)={len(correct_a)}, len(b)={len(correct_b)}"
        )

    n = len(correct_a)
    if n < 2:
        return 0.0, 1.0, False

    # Compute n01 and n10 of the 2x2 contingency table
    n01 = sum(1 for a, b in zip(correct_a, correct_b) if not a and b)  # A wrong, B correct
    n10 = sum(1 for a, b in zip(correct_a, correct_b) if a and not b)  # A correct, B wrong

    discordant = n01 + n10
    if discordant == 0:
        # Fully identical, no discriminative information
        return 0.0, 1.0, False

    # Yates continuity correction
    chi_sq = (abs(n01 - n10) - 1.0) ** 2 / discordant

    # p-value from the chi-squared CDF (df=1 approximation)
    p_value = _chi2_pvalue(chi_sq, df=1)
    is_significant = p_value < 0.05

    return chi_sq, p_value, is_significant


def bonferroni_correction(
    p_values: List[float],
    alpha: float = 0.05,
) -> List[bool]:
    """Bonferroni multiple-comparison correction.

    When running k hypothesis tests, the adjusted significance threshold is alpha / k in
    order to control the family-wise type I error rate.

    Args:
        p_values: p-values of each comparison, e.g. LENS vs every baseline.
        alpha:    family-wise significance level (default 0.05).

    Returns:
        A boolean list where True means still significant after correction.
    """
    k = len(p_values)
    if k == 0:
        return []
    adjusted_alpha = alpha / k
    return [p < adjusted_alpha for p in p_values]


def holm_correction(
    p_values: List[float],
    alpha: float = 0.05,
) -> List[bool]:
    """Holm-Bonferroni correction, more powerful than plain Bonferroni."""
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    rejected = [False] * len(p_values)
    m = len(p_values)
    for rank, (idx, p_value) in enumerate(indexed):
        threshold = alpha / max(m - rank, 1)
        if p_value <= threshold:
            rejected[idx] = True
        else:
            break
    return rejected


def cohens_h(p1: float, p2: float) -> float:
    """Compute the Cohen's h effect size between two proportions.

    Cohen's h = 2 * arcsin(sqrt(p1)) - 2 * arcsin(sqrt(p2))

    Effect-size reading (absolute value):
      h < 0.2:  small effect
      h < 0.5:  medium effect
      h >= 0.5: large effect

    Args:
        p1: accuracy of system A in [0, 1].
        p2: accuracy of system B in [0, 1].

    Returns:
        The signed Cohen's h value; positive means p1 > p2.
    """
    p1 = max(0.0, min(1.0, p1))
    p2 = max(0.0, min(1.0, p2))
    return 2 * math.asin(math.sqrt(p1)) - 2 * math.asin(math.sqrt(p2))


def significance_marker(p_value: float) -> str:
    """Convert a p-value into the significance marker conventional in papers.

    Returns:
        "***" (p<0.001), "**" (p<0.01), "*" (p<0.05), "" (n.s.)
    """
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return ""


# ---------------------------------------------------------------------------
# Chi-squared CDF approximation (avoids a scipy dependency)
# ---------------------------------------------------------------------------

def _chi2_pvalue(chi_sq: float, df: int = 1) -> float:
    """Approximate p-value of the chi-squared distribution with df=1.

    Uses the normal approximation: for df=1, chi2(1) = Z^2, hence
    p = P(chi2 > chi_sq) ~= 2 * P(Z > sqrt(chi_sq)) = 2 * (1 - Phi(sqrt(chi_sq)))

    The error is about +/-0.001, which meets paper-grade statistical precision.
    For higher precision, switch to scipy.stats.chi2.sf(chi_sq, df).
    """
    if chi_sq <= 0:
        return 1.0
    if df != 1:
        # For df > 1 use the Wilson-Hilferty approximation (more accurate as df grows)
        x = (chi_sq / df) ** (1 / 3)
        mu = 1 - 2 / (9 * df)
        sigma = math.sqrt(2 / (9 * df))
        z = (x - mu) / sigma
        return 1 - _normal_cdf(z)

    z = math.sqrt(chi_sq)
    return 2 * (1 - _normal_cdf(z))


def _normal_cdf(z: float) -> float:
    """CDF of the standard normal distribution, implemented with math.erf."""
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))
