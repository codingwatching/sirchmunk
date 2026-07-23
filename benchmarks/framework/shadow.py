"""framework/shadow.py — ShadowEvaluator

轻量级跨 benchmark 预评估，专用于 CONFIG_CHANGE 类型的 Layer 0/1 假设。

核心思路：
  在正式全量实验之前，用每个 benchmark 约 10% 的样本估算变更效果。
  成本：~10-15 题 × N benchmarks，vs 全量 150 题 × N benchmarks。
  仅支持 CONFIG_CHANGE（能通过覆盖 search_kwargs 实现）。
  PIPELINE_PATCH / PROMPT_FIX 无法通过此方式预估，跳过并提示人工评审。

工作机制：
  1. 从 hypothesis.config_changes 提取可映射为 search_kwargs 的键
  2. 对每个 adapter，以 base_kwargs + overrides 运行小批量
  3. 与 baseline 对比，返回 ImpactMatrix
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .schema import ChangeType, ConfigLayer, ImprovementHypothesis, PredictionResult

logger = logging.getLogger(__name__)

# env key → search kwarg 映射规则：剥去 benchmark 前缀后 lowercase
# 支持的 search kwargs（与 AgenticSearch.search() 参数对应）
_SEARCH_KWARG_NAMES = frozenset({
    "mode", "top_k_files", "max_token_budget", "enable_dir_scan",
})

# benchmark 专属前缀列表（新增 benchmark 时在此添加前缀即可）
_BM_PREFIXES = ("HOTPOT_", "MECHANISM_")


@dataclass
class BmShadowResult:
    """单个 benchmark 的 shadow 评估结果。"""
    benchmark: str
    n_samples: int
    accuracy: float
    coverage: float
    avg_latency: float
    applicable: bool = True    # False 表示该假设的 config_changes 对本 bm 无适用 kwarg
    error: Optional[str] = None

    def accuracy_delta(self, baseline: float) -> float:
        return self.accuracy - baseline

    def coverage_delta(self, baseline: float) -> float:
        return self.coverage - baseline


@dataclass
class ShadowImpactMatrix:
    """假设在所有 benchmark 上的预估影响矩阵。"""
    hypothesis_id: str
    results: Dict[str, BmShadowResult] = field(default_factory=dict)
    # {bm: {accuracy_delta, coverage_delta}}（vs baseline_vector）
    deltas: Dict[str, Dict[str, float]] = field(default_factory=dict)

    @property
    def pareto_status(self) -> str:
        """根据 accuracy delta 判断 Pareto 状态。"""
        applicable = {bm: r for bm, r in self.results.items() if r.applicable}
        if not applicable:
            return "unknown"
        deltas = [self.deltas.get(bm, {}).get("accuracy_delta", 0)
                  for bm in applicable]
        if all(d >= 0 for d in deltas) and any(d > 0 for d in deltas):
            return "dominant"
        if all(d < -2.0 for d in deltas):
            return "harmful"
        if any(d > 0 for d in deltas) and any(d < -2.0 for d in deltas):
            return "trade_off"
        return "neutral"

    def print_summary(self, hypothesis_title: str = "") -> None:
        title = hypothesis_title or self.hypothesis_id
        icon = {"dominant": "✅", "harmful": "❌", "trade_off": "⚖️ ", "neutral": "➖",
                "unknown": "❓"}.get(self.pareto_status, "❓")
        print(f"\n  Shadow Eval [{self.hypothesis_id}] {title[:50]}")
        print(f"  Pareto预估: {icon} {self.pareto_status.upper()}")
        print(f"  {'Benchmark':<25} {'n':>4} {'Δacc%':>7} {'Δcov%':>7}")
        print("  " + "─" * 46)
        for bm, r in sorted(self.results.items()):
            if not r.applicable:
                print(f"  {bm:<25} {'N/A':>4} {'n/a':>7} {'n/a':>7}  (not applicable)")
                continue
            if r.error:
                print(f"  {bm:<25} {'err':>4}  (error: {r.error[:40]})")
                continue
            d = self.deltas.get(bm, {})
            delta_acc = d.get("accuracy_delta", 0)
            delta_cov = d.get("coverage_delta", 0)
            s = lambda v: f"+{v:.1f}" if v >= 0 else f"{v:.1f}"
            print(f"  {bm:<25} {r.n_samples:>4} {s(delta_acc):>7} {s(delta_cov):>7}")
        print()


def _env_key_to_search_kwarg(key: str) -> Optional[str]:
    """将 env key 映射为 search kwarg 名称。

    规则：剥去 benchmark 前缀（HOTPOT_、MECHANISM_ 等）后 lowercase。
    不在 _SEARCH_KWARG_NAMES 白名单中的 key 返回 None。
    """
    upper_key = key.upper()
    for prefix in _BM_PREFIXES:
        if upper_key.startswith(prefix):
            remainder = upper_key[len(prefix):].lower()
            if remainder in _SEARCH_KWARG_NAMES:
                return remainder
    return None


def _coerce_kwarg(kwarg_name: str, value: str) -> Any:
    """将字符串 value 强制转为 search kwarg 期望的类型。"""
    if kwarg_name == "top_k_files":
        return int(value)
    if kwarg_name == "max_token_budget":
        return int(value)
    if kwarg_name == "enable_dir_scan":
        return str(value).lower() in ("true", "1", "yes")
    # mode 等字符串类型直接返回
    return value


class ShadowEvaluator:
    """跨 benchmark 轻量预评估器。

    仅支持 CONFIG_CHANGE 类型且 config_changes 能映射到 search_kwargs 的假设。
    PIPELINE_PATCH / PROMPT_FIX 无法预评估，返回空结果并记录日志。

    Usage::

        evaluator = ShadowEvaluator()
        matrix = await evaluator.evaluate(
            hypothesis, adapters,
            baseline_vector={"financebench": {"accuracy": 24.0, "coverage": 40.0}},
            sample_fraction=0.10,
        )
        matrix.print_summary(hypothesis.title)
    """

    async def evaluate(
        self,
        hypothesis: ImprovementHypothesis,
        adapters: List[Any],          # List[BenchmarkAdapter]
        baseline_vector: Dict[str, Dict[str, float]],
        sample_fraction: float = 0.10,
        seed: int = 99,
    ) -> ShadowImpactMatrix:
        """对所有 adapter 并发运行小批量评估，返回影响矩阵。

        Args:
            hypothesis:       要评估的改进假设。
            adapters:         所有已注册的 BenchmarkAdapter 列表。
            baseline_vector:  当前基线指标向量 {bm: {accuracy, coverage, ...}}。
            sample_fraction:  采样比例（默认 10%）。
            seed:             用于复现的随机种子。

        Returns:
            ShadowImpactMatrix，含每个 benchmark 的 delta 估算。
        """
        matrix = ShadowImpactMatrix(hypothesis_id=hypothesis.hypothesis_id)

        # 只支持 CONFIG_CHANGE
        if hypothesis.change_type != ChangeType.CONFIG_CHANGE:
            logger.info(
                "[Shadow] %s: skip non-CONFIG_CHANGE (%s)",
                hypothesis.hypothesis_id, hypothesis.change_type.value,
            )
            for adapter in adapters:
                matrix.results[adapter.name] = BmShadowResult(
                    benchmark=adapter.name, n_samples=0,
                    accuracy=0, coverage=0, avg_latency=0, applicable=False,
                )
            return matrix

        # 并发运行每个 adapter 的 shadow eval
        tasks = {
            adapter.name: asyncio.create_task(
                self._eval_single_adapter(
                    hypothesis, adapter, baseline_vector.get(adapter.name, {}),
                    sample_fraction, seed,
                )
            )
            for adapter in adapters
        }

        for bm_name, task in tasks.items():
            try:
                result = await task
            except Exception as exc:
                logger.warning("[Shadow] %s on %s failed: %s",
                               hypothesis.hypothesis_id, bm_name, exc)
                result = BmShadowResult(
                    benchmark=bm_name, n_samples=0,
                    accuracy=0, coverage=0, avg_latency=0,
                    applicable=True, error=str(exc)[:80],
                )
            matrix.results[bm_name] = result

        # 计算 delta vs baseline
        for bm, r in matrix.results.items():
            if not r.applicable or r.error:
                continue
            baseline_acc = baseline_vector.get(bm, {}).get("accuracy", 0)
            baseline_cov = baseline_vector.get(bm, {}).get("coverage", 0)
            matrix.deltas[bm] = {
                "accuracy_delta": round(r.accuracy - baseline_acc, 2),
                "coverage_delta": round(r.coverage - baseline_cov, 2),
            }

        return matrix

    async def _eval_single_adapter(
        self,
        hypothesis: ImprovementHypothesis,
        adapter: Any,
        baseline: Dict[str, float],
        sample_fraction: float,
        seed: int,
    ) -> BmShadowResult:
        """运行单个 adapter 的 shadow 评估。"""
        # 构建 search_kwargs 覆盖
        overrides: Dict[str, Any] = {}
        for env_key, new_val in hypothesis.config_changes.items():
            kwarg = _env_key_to_search_kwarg(env_key)
            if kwarg is not None:
                overrides[kwarg] = _coerce_kwarg(kwarg, new_val)

        if not overrides:
            return BmShadowResult(
                benchmark=adapter.name, n_samples=0,
                accuracy=0, coverage=0, avg_latency=0, applicable=False,
            )

        # 计算样本数（最少 5 题，最多 30 题）
        full_samples = adapter.load_samples(limit=0, seed=seed)
        n = max(5, min(30, int(len(full_samples) * sample_fraction)))
        samples = adapter.load_samples(limit=n, seed=seed)

        # 使用原搜索器，覆盖 search_kwargs
        searcher = adapter.build_searcher()
        judge = adapter.build_judge()
        base_kwargs = dict(adapter.get_search_kwargs())
        base_kwargs.update(overrides)

        # 顺序执行（shadow eval 不需要并发，降低 API 压力）
        correct = 0
        coverage_count = 0
        elapsed_list = []
        errors = 0

        for sample in samples:
            try:
                search_paths = adapter.get_search_paths(sample)
                result = await searcher.search(
                    query=sample.question,
                    paths=search_paths,
                    return_context=True,
                    **base_kwargs,
                )
                prediction = getattr(result, "answer", "") or str(result)

                jc, cov = False, False
                if judge:
                    try:
                        jr = await judge.judge(
                            prediction=prediction,
                            gold_answer=sample.gold_answer,
                            question=sample.question,
                        )
                        jc = jr.get("equivalent", False)
                        cr = await judge.judge_coverage(
                            prediction=prediction, question=sample.question
                        )
                        cov = cr.get("has_coverage", False)
                    except Exception:
                        pass

                if jc:
                    correct += 1
                if cov:
                    coverage_count += 1

                elapsed = getattr(result, "elapsed", 0)
                if elapsed:
                    elapsed_list.append(elapsed)

            except Exception as exc:
                errors += 1
                logger.debug("[Shadow] sample error on %s: %s", adapter.name, exc)

        valid = len(samples) - errors
        if valid == 0:
            return BmShadowResult(
                benchmark=adapter.name, n_samples=len(samples),
                accuracy=0, coverage=0, avg_latency=0,
                applicable=True, error="all samples failed",
            )

        return BmShadowResult(
            benchmark=adapter.name,
            n_samples=valid,
            accuracy=round(correct / valid * 100, 1),
            coverage=round(coverage_count / valid * 100, 1),
            avg_latency=round(sum(elapsed_list) / len(elapsed_list), 1) if elapsed_list else 0,
            applicable=True,
        )
