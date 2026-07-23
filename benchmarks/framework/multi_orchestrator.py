"""framework/multi_orchestrator.py — MultiAdapterOrchestrator

多 benchmark 联合优化编排器。

替代 ResearchOrchestrator 的单 benchmark 串行优化模式，实现：
  1. 并行评估所有 benchmark → 多维指标向量
  2. 每个 benchmark 独立做 BadCase 分析
  3. 跨 benchmark 假设去重与合并
  4. Layer 0/1 的 Shadow 预评估（用 10% 样本估算跨 benchmark 影响）
  5. Pareto Dominance Gate：只接受不使任何 benchmark 退化的变更
  6. 人工确认（带 Pareto 影响矩阵展示）
  7. 收敛检测：Pareto frontier 停止扩张 → 建议停止

研究科学性保证：
  - 每次实验记录 git commit + global config hash
  - 回退检测：accuracy 降幅 > 2% 自动标记
  - Layer 2 (SPECIFIC) 变更无需联合评估，仍可高效单独应用
"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Optional

from .adapter import BenchmarkAdapter
from .analyzer import BadCaseAnalyzer
from .advisor import ImprovementAdvisor
from .confirm import HumanConfirmLoop
from .orchestrator import _compute_basic_metrics
from .pareto import ParetoTracker
from .runner import UnifiedExperimentRunner, _get_git_commit, _config_hash
from .schema import (
    BenchmarkSample,
    ChangeType,
    ConfigLayer,
    ImprovementHypothesis,
    PredictionResult,
)
from .shadow import ShadowEvaluator

logger = logging.getLogger(__name__)


class MultiAdapterOrchestrator:
    """多 benchmark 联合优化编排器。

    Usage::

        adapters = [
            FinanceBenchAdapter(".env.financebench"),
            HotpotQAAdapter(".env.hotpotqa"),
        ]
        orch = MultiAdapterOrchestrator(
            adapters=adapters,
            experiments_path="benchmarks/multi_experiments.jsonl",
        )
        await orch.run(max_iterations=5, limit_per_bm=50, shadow_fraction=0.10)
    """

    def __init__(
        self,
        adapters: List[BenchmarkAdapter],
        experiments_path: str = "benchmarks/multi_experiments.jsonl",
        dry_run: bool = False,
    ) -> None:
        """
        Args:
            adapters:          已注册的 BenchmarkAdapter 列表。
            experiments_path:  多维实验记录 JSONL 文件路径。
            dry_run:           True 时不实际写 .env 文件。
        """
        self._adapters = adapters
        self._pareto = ParetoTracker(experiments_path)
        self._confirm = HumanConfirmLoop(dry_run=dry_run)
        self._shadow = ShadowEvaluator()
        self._llm = None                         # 延迟获取

        # 为每个 adapter 创建独立的 runner
        self._runners: Dict[str, UnifiedExperimentRunner] = {
            a.name: UnifiedExperimentRunner(a) for a in adapters
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(
        self,
        max_iterations: int = 5,
        limit_per_bm: int = 0,
        seed: int = 42,
        shadow_fraction: float = 0.10,
        convergence_threshold: float = 1.0,
        convergence_window: int = 3,
    ) -> None:
        """启动多 benchmark 联合优化循环。

        Args:
            max_iterations:       最大迭代轮数。
            limit_per_bm:         每个 benchmark 每次评估的最大样本数（0=全量）。
            seed:                 随机种子。
            shadow_fraction:      Shadow eval 采样比例（默认 10%）。
            convergence_threshold:Pareto 收敛阈值（百分点）。
            convergence_window:   收敛判断所需连续轮数。
        """
        # 延迟获取 LLM
        if self._llm is None:
            try:
                self._llm = getattr(self._adapters[0].build_searcher(), "llm", None)
            except Exception:
                self._llm = None

        analyzer = BadCaseAnalyzer(llm=self._llm)
        advisor  = ImprovementAdvisor(llm=self._llm)

        bm_names = [a.name for a in self._adapters]
        prev_run_id: Optional[str] = None

        print(f"\n{'='*68}")
        print(f"  Multi-Benchmark Research Loop")
        print(f"  Benchmarks: {', '.join(bm_names)}")
        print(f"  Max iterations: {max_iterations}  |  Limit/bm: {limit_per_bm or 'ALL'}")
        print(f"{'='*68}\n")

        self._pareto.print_history()

        for iteration in range(1, max_iterations + 1):
            print(f"\n{'─'*68}")
            print(f"  Iteration {iteration}/{max_iterations}")
            print(f"{'─'*68}\n")

            # ── Step 1: 并行评估所有 benchmark ───────────────────────
            all_results, all_meta = await self._parallel_eval(limit_per_bm, seed)
            run_id = f"multi_{iteration}_{list(all_meta.values())[0].get('timestamp', '')[:15].replace(':', '')}"

            # ── Step 2: 计算多维指标向量 ──────────────────────────────
            metrics_vector = {
                bm: _compute_basic_metrics(results)
                for bm, results in all_results.items()
            }
            self._print_metrics_vector(metrics_vector)

            # ── Step 3: 记录 Pareto 点 ────────────────────────────────
            git_commit = _get_git_commit()
            global_cfg = {a.name: a.get_run_config() for a in self._adapters}
            cfg_hash = _config_hash(global_cfg)

            self._pareto.record_multi(
                run_id=run_id,
                metrics_vector=metrics_vector,
                git_commit=git_commit,
                config_hash=cfg_hash,
            )

            # ── Step 4: 打印 Pareto delta ─────────────────────────────
            if prev_run_id:
                delta = self._pareto.compare_runs(prev_run_id, run_id)
                if delta:
                    delta.print_summary()

            prev_run_id = run_id

            # ── Step 5: 各 benchmark BadCase 分析 ────────────────────
            samples_maps = {
                bm: self._build_samples_map(results)
                for bm, results in all_results.items()
            }
            reports = {}
            for bm, results in all_results.items():
                adapter = self._adapter_by_name(bm)
                analysis_schema = adapter.get_analysis_schema()
                report = await analyzer.analyze(
                    results,
                    samples_maps[bm],
                    question_type_key=analysis_schema.get("primary_group_key", "question_type"),
                )
                reports[bm] = report
                print(f"\n  ── {bm} BadCase Report ──")
                BadCaseAnalyzer.print_report(report)

            # ── Step 6: 生成假设 + 跨 bm 去重 ────────────────────────
            all_hypotheses: List[ImprovementHypothesis] = []
            for bm, report in reports.items():
                adapter = self._adapter_by_name(bm)
                hyps = await advisor.suggest(
                    report, adapter.get_run_config(), adapter.env_file
                )
                all_hypotheses.extend(hyps)

            merged = self._merge_hypotheses(all_hypotheses)
            print(f"\n  生成假设: {len(all_hypotheses)} 条 → 合并后 {len(merged)} 条")

            # ── Step 7: Shadow 预评估（Layer 0/1 CONFIG_CHANGE 优先）──
            shadow_matrices = {}
            for h in merged:
                if (h.config_layer in (ConfigLayer.GLOBAL, ConfigLayer.FAMILY)
                        and h.change_type == ChangeType.CONFIG_CHANGE):
                    print(f"\n  🔍 Shadow eval: [{h.hypothesis_id}] {h.title[:50]}")
                    print(f"     (sample_fraction={shadow_fraction:.0%}，正在评估...)")
                    try:
                        matrix = await self._shadow.evaluate(
                            h, self._adapters, metrics_vector,
                            sample_fraction=shadow_fraction, seed=seed,
                        )
                        shadow_matrices[h.hypothesis_id] = matrix
                        matrix.print_summary(h.title)
                        # 将 Pareto 状态注入 hypothesis（供 confirm 展示）
                        h._shadow_pareto = matrix.pareto_status
                    except Exception as exc:
                        logger.warning("[Multi] shadow eval failed for %s: %s",
                                       h.hypothesis_id, exc)

            # ── Step 8: 为假设附加 Pareto 标签 ──────────────────────
            self._annotate_pareto_gate(merged, shadow_matrices, metrics_vector)

            # ── Step 9: 人工确认 ──────────────────────────────────────
            self._print_multi_confirm_header(merged, shadow_matrices)
            chosen, applied = self._confirm.review(merged)

            if chosen is None:
                print("\n  [Multi] 用户退出，保留所有已记录实验。")
                break
            if not chosen:
                print("  [Multi] 用户跳过本轮。")

            # ── Step 10: 收敛检测 ────────────────────────────────────
            converged, msg = self._pareto.convergence_check(
                window=convergence_window, threshold=convergence_threshold
            )
            if converged:
                print(f"\n  ✅ {msg}")
                try:
                    stop = input("  是否继续? [y/N] > ").strip().lower()
                except (EOFError, KeyboardInterrupt):
                    stop = "n"
                if stop not in ("y", "yes"):
                    break

        # 最终 Pareto 历史
        print(f"\n{'='*68}")
        print("  联合优化结束，最终 Pareto 历史：")
        self._pareto.print_history()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _parallel_eval(
        self, limit: int, seed: int
    ) -> tuple[Dict[str, List[PredictionResult]], Dict]:
        """并发运行所有 benchmark。"""
        tasks = {
            adapter.name: asyncio.create_task(
                self._runners[adapter.name].run(limit=limit, seed=seed)
            )
            for adapter in self._adapters
        }
        all_results: Dict[str, List[PredictionResult]] = {}
        all_meta: Dict[str, dict] = {}

        for bm_name, task in tasks.items():
            results, meta = await task
            all_results[bm_name] = results
            all_meta[bm_name] = meta
            logger.info("[Multi] %s: %d samples done", bm_name, len(results))

        return all_results, all_meta

    def _adapter_by_name(self, name: str) -> BenchmarkAdapter:
        for a in self._adapters:
            if a.name == name:
                return a
        raise KeyError(f"Adapter '{name}' not registered")

    @staticmethod
    def _build_samples_map(results: List[PredictionResult]) -> Dict[str, BenchmarkSample]:
        from .schema import BenchmarkSample
        m: Dict[str, BenchmarkSample] = {}
        for r in results:
            raw = r.raw or {}
            sid = r.sample_id
            meta = {k: v for k, v in raw.items()
                    if k not in ("question", "gold_answer", "prediction",
                                 "judge_correct", "coverage", "elapsed",
                                 "telemetry", "error", "sample_id")}
            m[sid] = BenchmarkSample(
                sample_id=sid,
                question=raw.get("question", sid),
                gold_answer=raw.get("gold_answer", ""),
                metadata=meta,
            )
        return m

    @staticmethod
    def _merge_hypotheses(
        all_hyps: List[ImprovementHypothesis],
    ) -> List[ImprovementHypothesis]:
        """跨 benchmark 假设去重合并。

        合并规则：相同 config_changes 的 CONFIG_CHANGE 假设合并为一条（取高影响级别）；
        PIPELINE_PATCH / PROMPT_FIX 按 title 去重。
        """
        merged: Dict[str, ImprovementHypothesis] = {}
        others: List[ImprovementHypothesis] = []

        for h in all_hyps:
            if h.change_type == ChangeType.CONFIG_CHANGE and h.config_changes:
                key = json_key(h.config_changes)
                if key in merged:
                    # 取高影响级别的那条
                    existing = merged[key]
                    from .schema import ImpactLevel
                    order = {ImpactLevel.HIGH: 2, ImpactLevel.MEDIUM: 1, ImpactLevel.LOW: 0}
                    if order.get(h.estimated_impact, 0) > order.get(existing.estimated_impact, 0):
                        merged[key] = h
                else:
                    merged[key] = h
            else:
                # PIPELINE_PATCH / PROMPT_FIX：按 title 去重
                title_key = h.title.lower().strip()
                if not any(title_key == o.title.lower().strip() for o in others):
                    others.append(h)

        result = list(merged.values()) + others
        # 高影响优先排序
        from .schema import ImpactLevel
        _order = {ImpactLevel.HIGH: 0, ImpactLevel.MEDIUM: 1, ImpactLevel.LOW: 2}
        result.sort(key=lambda h: _order.get(h.estimated_impact, 3))
        return result

    def _annotate_pareto_gate(
        self,
        hypotheses: List[ImprovementHypothesis],
        shadow_matrices: dict,
        baseline: Dict[str, Dict],
    ) -> None:
        """为每条假设附加 Pareto 门控注解（用于 confirm 展示）。"""
        for h in hypotheses:
            if h.hypothesis_id in shadow_matrices:
                matrix = shadow_matrices[h.hypothesis_id]
                h._shadow_pareto = matrix.pareto_status
            elif h.config_layer == ConfigLayer.SPECIFIC:
                h._shadow_pareto = "specific"   # Layer 2，无需联合评估
            else:
                h._shadow_pareto = "unverified"  # 未做 shadow eval

    @staticmethod
    def _print_metrics_vector(metrics_vector: Dict[str, Dict]) -> None:
        """打印多维指标向量表格。"""
        print("\n  ┌── 本轮指标向量 ──────────────────────────────────────┐")
        print(f"  │ {'Benchmark':<25} {'Accuracy':>10} {'Coverage':>10} {'Latency':>10} │")
        print(f"  │ {'─'*25} {'─'*10} {'─'*10} {'─'*10} │")
        for bm, m in sorted(metrics_vector.items()):
            print(f"  │ {bm:<25} {m.get('accuracy', 0):>9.1f}% "
                  f"{m.get('coverage', 0):>9.1f}% "
                  f"{m.get('avg_latency', 0):>9.1f}s │")
        print("  └────────────────────────────────────────────────────┘\n")

    @staticmethod
    def _print_multi_confirm_header(
        hypotheses: List[ImprovementHypothesis],
        shadow_matrices: dict,
    ) -> None:
        """在人工确认前打印 Pareto 门控摘要。"""
        print("\n  ── Pareto 门控摘要 ──────────────────────────────────────")
        gate_icons = {
            "dominant":  "✅ SAFE     ",
            "trade_off": "⚖️  TRADE-OFF",
            "harmful":   "❌ HARMFUL  ",
            "neutral":   "➖ NEUTRAL  ",
            "specific":  "🔵 LAYER-2  ",
            "unverified": "❓ UNVERIFIED",
        }
        for h in hypotheses:
            pareto = getattr(h, "_shadow_pareto", "unknown")
            icon = gate_icons.get(pareto, "❓")
            print(f"  [{h.hypothesis_id}] {icon}  {h.title[:50]}")
        print()


def json_key(d: dict) -> str:
    """将 dict 序列化为稳定的 key 字符串。"""
    import json
    return json.dumps(d, sort_keys=True)
