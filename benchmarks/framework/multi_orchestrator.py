"""framework/multi_orchestrator.py — MultiAdapterOrchestrator

Orchestrator for multi-benchmark joint optimization.

It replaces the single-benchmark sequential optimization of ResearchOrchestrator and
provides:
  1. parallel evaluation of every benchmark -> multi-metric vector
  2. independent badcase analysis per benchmark
  3. cross-benchmark hypothesis deduplication and merging
  4. shadow pre-evaluation for Layer 0/1 (10% of samples estimate the cross-benchmark
     impact)
  5. Pareto dominance gate: accept only changes that do not regress any benchmark
  6. manual confirmation with the Pareto impact matrix on display
  7. convergence check: suggest stopping once the Pareto frontier stops expanding

Scientific guarantees:
  - every experiment records the git commit + global config hash
  - regression detection: an accuracy drop above 2% is flagged automatically
  - Layer 2 (SPECIFIC) changes need no joint evaluation and can still be applied
    efficiently on their own
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
    """Orchestrator for multi-benchmark joint optimization.

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
            adapters:          list of registered BenchmarkAdapter.
            experiments_path:  path of the multi-metric experiment JSONL.
            dry_run:           when True, no .env file is written.
        """
        self._adapters = adapters
        self._pareto = ParetoTracker(experiments_path)
        self._confirm = HumanConfirmLoop(dry_run=dry_run)
        self._shadow = ShadowEvaluator()
        self._llm = None                         # Lazy lookup

        # Create one runner per adapter
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
        """Start the multi-benchmark joint optimization loop.

        Args:
            max_iterations:       maximum number of iterations.
            limit_per_bm:         max samples per benchmark per evaluation (0 = full set).
            seed:                 random seed.
            shadow_fraction:      shadow eval sampling fraction (default 10%).
            convergence_threshold:Pareto convergence threshold in percentage points.
            convergence_window:   consecutive rounds required to declare convergence.
        """
        # Lazy LLM lookup
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

            # ── Step 1: evaluate all benchmarks in parallel ────
            all_results, all_meta = await self._parallel_eval(limit_per_bm, seed)
            run_id = f"multi_{iteration}_{list(all_meta.values())[0].get('timestamp', '')[:15].replace(':', '')}"

            # ── Step 2: compute the multi-metric vector ───────
            metrics_vector = {
                bm: _compute_basic_metrics(results)
                for bm, results in all_results.items()
            }
            self._print_metrics_vector(metrics_vector)

            # ── Step 3: record the Pareto point ────────────────────
            git_commit = _get_git_commit()
            global_cfg = {a.name: a.get_run_config() for a in self._adapters}
            cfg_hash = _config_hash(global_cfg)

            self._pareto.record_multi(
                run_id=run_id,
                metrics_vector=metrics_vector,
                git_commit=git_commit,
                config_hash=cfg_hash,
            )

            # ── Step 4: print the Pareto delta ──────────────────────
            if prev_run_id:
                delta = self._pareto.compare_runs(prev_run_id, run_id)
                if delta:
                    delta.print_summary()

            prev_run_id = run_id

            # ── Step 5: per-benchmark badcase analysis ────────────
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

            # ── Step 6: generate hypotheses and dedupe across benchmarks ──
            all_hypotheses: List[ImprovementHypothesis] = []
            for bm, report in reports.items():
                adapter = self._adapter_by_name(bm)
                hyps = await advisor.suggest(
                    report, adapter.get_run_config(), adapter.env_file
                )
                all_hypotheses.extend(hyps)

            merged = self._merge_hypotheses(all_hypotheses)
            print(f"\n  生成假设: {len(all_hypotheses)} 条 → 合并后 {len(merged)} 条")

            # ── Step 7: shadow pre-evaluation (Layer 0/1 CONFIG_CHANGE first) ──
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
                        # Inject the Pareto status into the hypothesis for the confirm view
                        h._shadow_pareto = matrix.pareto_status
                    except Exception as exc:
                        logger.warning("[Multi] shadow eval failed for %s: %s",
                                       h.hypothesis_id, exc)

            # ── Step 8: attach Pareto labels to hypotheses ───
            self._annotate_pareto_gate(merged, shadow_matrices, metrics_vector)

            # ── Step 9: manual confirmation ───────────────────────
            self._print_multi_confirm_header(merged, shadow_matrices)
            chosen, applied = self._confirm.review(merged)

            if chosen is None:
                print("\n  [Multi] 用户退出，保留所有已记录实验。")
                break
            if not chosen:
                print("  [Multi] 用户跳过本轮。")

            # ── Step 10: convergence check ───────────────────────
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

        # Final Pareto history
        print(f"\n{'='*68}")
        print("  联合优化结束，最终 Pareto 历史：")
        self._pareto.print_history()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _parallel_eval(
        self, limit: int, seed: int
    ) -> tuple[Dict[str, List[PredictionResult]], Dict]:
        """Run every benchmark concurrently."""
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
        """Deduplicate and merge hypotheses across benchmarks.

        Merge rules: CONFIG_CHANGE hypotheses with identical config_changes collapse into
        one entry keeping the higher impact level; PIPELINE_PATCH / PROMPT_FIX are
        deduplicated by title.
        """
        merged: Dict[str, ImprovementHypothesis] = {}
        others: List[ImprovementHypothesis] = []

        for h in all_hyps:
            if h.change_type == ChangeType.CONFIG_CHANGE and h.config_changes:
                key = json_key(h.config_changes)
                if key in merged:
                    # Keep the higher-impact entry
                    existing = merged[key]
                    from .schema import ImpactLevel
                    order = {ImpactLevel.HIGH: 2, ImpactLevel.MEDIUM: 1, ImpactLevel.LOW: 0}
                    if order.get(h.estimated_impact, 0) > order.get(existing.estimated_impact, 0):
                        merged[key] = h
                else:
                    merged[key] = h
            else:
                # PIPELINE_PATCH / PROMPT_FIX: deduplicate by title
                title_key = h.title.lower().strip()
                if not any(title_key == o.title.lower().strip() for o in others):
                    others.append(h)

        result = list(merged.values()) + others
        # Sort high-impact entries first
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
        """Attach the Pareto gate annotation to each hypothesis for the confirm view."""
        for h in hypotheses:
            if h.hypothesis_id in shadow_matrices:
                matrix = shadow_matrices[h.hypothesis_id]
                h._shadow_pareto = matrix.pareto_status
            elif h.config_layer == ConfigLayer.SPECIFIC:
                h._shadow_pareto = "specific"   # Layer 2, no joint evaluation needed
            else:
                h._shadow_pareto = "unverified"  # Shadow eval was not run

    @staticmethod
    def _print_metrics_vector(metrics_vector: Dict[str, Dict]) -> None:
        """Print the multi-metric vector table."""
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
        """Print the Pareto gate summary before manual confirmation."""
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
    """Serialize a dict into a stable key string."""
    import json
    return json.dumps(d, sort_keys=True)
