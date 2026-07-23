"""evaluation/statistics.py — 论文级统计分析

为竞品对比提供统计严谨性支撑：
  bootstrap_ci()        : Bootstrap 置信区间（accuracy 的 95% CI）
  paired_bootstrap_delta(): 配对Bootstrap差值置信区间
  mcnemar_test()        : McNemar 配对显著性检验（两系统的 correct 向量）
  bonferroni_correction(): Bonferroni 多重比较校正
  holm_correction()     : Holm-Bonferroni 多重比较校正
  cohens_h()            : 两比例差异的效应量

所有函数均为纯函数（无副作用），不依赖任何 framework/ 代码。

McNemar 检验原理：
  适用于两个分类器在相同样本上的配对对比。
  H0: 两系统的错误模式无差异。
  检验统计量基于"一方对一方错"的不一致对数量。

Bootstrap CI 原理：
  对 correct 列表做有放回重采样 n_bootstrap 次，
  计算每次采样的 accuracy，取 α/2 和 1-α/2 百分位作为 CI。
"""
from __future__ import annotations

import math
import random
from typing import List, Optional, Tuple


def bootstrap_ci(
    correct: List[bool],
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """计算 accuracy 的 Bootstrap 置信区间。

    Args:
        correct:     每个样本的 judge_correct 布尔列表。
        n_bootstrap: Bootstrap 重采样次数（默认 1000）。
        alpha:       显著性水平（默认 0.05，即 95% CI）。
        seed:        随机种子，保证复现性。

    Returns:
        (accuracy, lower_bound, upper_bound)
        其中 accuracy = mean(correct)，CI 由 Bootstrap 分布的百分位给出。
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
    """McNemar 配对显著性检验。

    适用于两个系统在相同测试集上的比较（配对设计）。
    H0: 两系统的错误率无差异（n01 = n10）。

    使用 Yates 连续性校正版本（避免小样本过度显著）：
      χ² = (|n01 - n10| - 1)² / (n01 + n10)

    Args:
        correct_a: 系统 A 的 judge_correct 列表（同序）。
        correct_b: 系统 B 的 judge_correct 列表（同序）。

    Returns:
        (chi_squared, p_value, is_significant_at_0.05)

    Raises:
        ValueError: 若两列表长度不一致或样本量不足。
    """
    if len(correct_a) != len(correct_b):
        raise ValueError(
            f"Sample length mismatch: len(a)={len(correct_a)}, len(b)={len(correct_b)}"
        )

    n = len(correct_a)
    if n < 2:
        return 0.0, 1.0, False

    # 计算四格联表的 n01 和 n10
    n01 = sum(1 for a, b in zip(correct_a, correct_b) if not a and b)  # A错B对
    n10 = sum(1 for a, b in zip(correct_a, correct_b) if a and not b)  # A对B错

    discordant = n01 + n10
    if discordant == 0:
        # 完全一致，无法区分
        return 0.0, 1.0, False

    # Yates 连续性校正
    chi_sq = (abs(n01 - n10) - 1.0) ** 2 / discordant

    # 使用卡方分布的 CDF 计算 p 值（df=1 的近似）
    p_value = _chi2_pvalue(chi_sq, df=1)
    is_significant = p_value < 0.05

    return chi_sq, p_value, is_significant


def bonferroni_correction(
    p_values: List[float],
    alpha: float = 0.05,
) -> List[bool]:
    """Bonferroni 多重比较校正。

    当进行 k 次假设检验时，为控制总体 I 类错误率，
    调整后的显著性阈值为 alpha / k。

    Args:
        p_values: 各次比较的 p 值列表（如 LENS vs 每个 baseline）。
        alpha:    总体显著性水平（默认 0.05）。

    Returns:
        布尔列表，True 表示在校正后仍显著。
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
    """计算两个比例之间的 Cohen's h 效应量。

    Cohen's h = 2 * arcsin(sqrt(p1)) - 2 * arcsin(sqrt(p2))

    效应量解释（绝对值）：
      h < 0.2:  small effect
      h < 0.5:  medium effect
      h >= 0.5: large effect

    Args:
        p1: 系统 A 的准确率 [0, 1]。
        p2: 系统 B 的准确率 [0, 1]。

    Returns:
        Cohen's h 值（有符号，正值表示 p1 > p2）。
    """
    p1 = max(0.0, min(1.0, p1))
    p2 = max(0.0, min(1.0, p2))
    return 2 * math.asin(math.sqrt(p1)) - 2 * math.asin(math.sqrt(p2))


def significance_marker(p_value: float) -> str:
    """将 p 值转换为论文惯用的显著性标记。

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
# Chi-squared CDF approximation（避免依赖 scipy）
# ---------------------------------------------------------------------------

def _chi2_pvalue(chi_sq: float, df: int = 1) -> float:
    """df=1 的卡方分布 p 值近似计算。

    使用正态近似：对于 df=1，χ²(1) = Z²，所以
    p = P(χ² > chi_sq) ≈ 2 * P(Z > sqrt(chi_sq)) = 2 * (1 - Φ(sqrt(chi_sq)))

    误差约 ±0.001，满足论文级统计精度要求。
    如需更高精度，可换用 scipy.stats.chi2.sf(chi_sq, df)。
    """
    if chi_sq <= 0:
        return 1.0
    if df != 1:
        # 对 df > 1，使用 Wilson-Hilferty 近似（df 越大越准）
        x = (chi_sq / df) ** (1 / 3)
        mu = 1 - 2 / (9 * df)
        sigma = math.sqrt(2 / (9 * df))
        z = (x - mu) / sigma
        return 1 - _normal_cdf(z)

    z = math.sqrt(chi_sq)
    return 2 * (1 - _normal_cdf(z))


def _normal_cdf(z: float) -> float:
    """标准正态分布的 CDF，用 math.erf 实现。"""
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))
