"""evaluation/table_generator.py — PaperTableGenerator

生成可直接放入论文的比较表格：
  - LaTeX (tabular 环境，可直接粘贴)
  - Markdown (可在 README/报告中预览)
  - JSON (机器可读，供进一步处理)

支持三种数据输入方式：
  1. add_system_results(system_name, results)    ← BaselineResult 列表 / PredictionResult 列表
  2. add_published_metrics(system_name, ...)     ← 只有发表数字（无需 Judge 重跑）
  3. set_ours(system_name)                       ← 标记本文系统（加粗 + 星号）

统计特性：
  - 自动对每个系统计算 Bootstrap 95% CI
  - 自动对每个 baseline 与 ours 运行 McNemar 检验（配对，需 correct 列表）
  - Bonferroni 校正（k = baseline 数量）
  - 显著性标记: *, **, *** 追加在 accuracy 后面
  - 最优值自动加粗（LaTeX: \\textbf{...}）
"""
from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from framework.metric_engine import collect_setup_metrics

from .golden_set import compute_sample_id_checksum
from .statistics import (
    bonferroni_correction,
    bootstrap_ci,
    mcnemar_test,
    significance_marker,
)

logger = logging.getLogger(__name__)


@dataclass
class SystemEntry:
    """论文表格中一行系统的聚合指标。"""
    system_name: str          # 表格展示名（citation_name）
    n: int = 0
    accuracy: float = 0.0
    official_em: float = 0.0
    official_f1: float = 0.0
    llm_assisted_accuracy: float = 0.0
    ci_lower: float = 0.0     # Bootstrap 95% CI lower
    ci_upper: float = 0.0     # Bootstrap 95% CI upper
    coverage: float = 0.0
    avg_latency: float = 0.0  # 秒
    avg_tokens: float = 0.0
    avg_oracle_calls: float = 0.0
    avg_llm_calls: float = 0.0
    avg_search_calls: float = 0.0
    avg_read_calls: float = 0.0
    evidence_trace_coverage: float = 0.0
    evidence_recall: float = 0.0
    supporting_fact_title_recall: float = 0.0
    source_grounding_accuracy: float = 0.0
    corpus_checksum: str = ""
    frozen_order_checksum: str = ""
    by_question_type: Dict[str, Dict] = field(default_factory=dict)
    is_ours: bool = False
    is_published_only: bool = False   # True = 只有发表数字，无 CI
    correct_list: List[bool] = field(default_factory=list)  # 用于 McNemar
    sample_ids: List[str] = field(default_factory=list)
    sample_id_checksum: str = ""
    setup_metrics: Dict[str, Any] = field(default_factory=dict)
    query_budget_summary: Dict[str, Any] = field(default_factory=dict)
    failure_counts: Dict[str, int] = field(default_factory=dict)
    failure_rate: float = 0.0
    imported_baseline: bool = False
    import_coverage: Optional[float] = None
    imported_samples: int = 0
    covered_samples: int = 0
    missing_samples: int = 0
    missing_sample_ids: List[str] = field(default_factory=list)
    sampling_method: str = ""
    population_size: int = 0
    sampled_n: int = 0
    sampling_protocol: Dict[str, Any] = field(default_factory=dict)
    sampling_manifest: Dict[str, Any] = field(default_factory=dict)
    strata_distribution: Dict[str, Any] = field(default_factory=dict)
    weighted_metric_available: bool = False
    # 显著性（由 finalize() 填入）
    p_value: Optional[float] = None
    is_significant: bool = False
    sig_marker: str = ""


class PaperTableGenerator:
    """论文比较表格生成器。

    Usage::

        gen = PaperTableGenerator(benchmark_name="FinanceBench", our_system_name="LENS")

        # 添加本文系统（从 Sirchmunk 的 results.jsonl 加载）
        gen.add_system_results(
            system_name="LENS (ours)",
            results=sirchmunk_results,   # List[PredictionResult 兼容]
            is_ours=True,
        )

        # 添加已运行的竞品（BaselineResult 列表）
        gen.add_system_results(
            system_name="GPT-4o (zero-shot)",
            results=gpt4o_results,        # List[BaselineResult]
        )

        # 添加只有发表数字的竞品（无需重跑）
        gen.add_published_metrics(
            system_name="Mafin 2.5",
            accuracy=98.7, coverage=100.0, avg_latency=0, citation="Gao et al., 2024"
        )

        # 生成表格
        paths = gen.generate(output_dir="output/paper_table/")
        # paths: {"latex": ..., "markdown": ..., "json": ...}
    """

    def __init__(
        self,
        benchmark_name: str = "Benchmark",
        our_system_name: Optional[str] = None,
    ) -> None:
        self._benchmark = benchmark_name
        self._our_name = our_system_name
        self._entries: List[SystemEntry] = []
        self._sampling_metadata: Dict[str, Any] = {}

    def set_sampling_metadata(self, sampling_metadata: Dict[str, Any]) -> None:
        """Attach auditable sampling metadata to the whole table and each row."""
        self._sampling_metadata = dict(sampling_metadata or {})

    # ------------------------------------------------------------------
    # 数据输入接口
    # ------------------------------------------------------------------

    def add_system_results(
        self,
        system_name: str,
        results: List[Any],             # BaselineResult 或 PredictionResult（鸭子类型）
        is_ours: bool = False,
        question_type_key: str = "question_type",
    ) -> None:
        """从结果列表添加一个系统。

        Args:
            system_name:       论文表格中的展示名称。
            results:           BaselineResult 或 PredictionResult 列表（duck typing）。
            is_ours:           是否为本文系统（用于加粗和显著性检验基准）。
            question_type_key: 从 metadata 中取 question_type 的 key（for breakdown）。
        """
        if not results:
            logger.warning("[TableGen] '%s': empty results, skipping.", system_name)
            return

        ordered_results = sorted(results, key=lambda r: str(getattr(r, "sample_id", "")))
        n = len(ordered_results)
        sample_ids = [str(getattr(r, "sample_id", "")) for r in ordered_results]
        correct_list = [bool(getattr(r, "judge_correct", False)) for r in ordered_results]
        coverage_list = [bool(getattr(r, "coverage", False)) for r in ordered_results]
        latencies = [float(getattr(r, "elapsed", 0)) for r in ordered_results]
        tokens = []
        for r in ordered_results:
            telemetry = getattr(r, "telemetry", {}) or {}
            if telemetry:
                tokens.append(int(telemetry.get("total_tokens", 0)) + int(telemetry.get("judge_tokens", 0)))
            else:
                tokens.append(int(getattr(r, "tokens_used", 0)) + int(getattr(r, "judge_tokens", 0)))

        query_budgets = [_query_budget_of(r) for r in ordered_results]
        query_budget_summary = _summarize_query_budgets(query_budgets)
        evidence_trace_count = sum(1 for r in ordered_results if _evidence_traces_of(r))

        accuracy, ci_lower, ci_upper = bootstrap_ci(correct_list)
        metric_payloads = [_metric_payload_of(r) for r in ordered_results]
        official_em_values = [float(p.get("official_em", p.get("em", 0.0)) or 0.0) for p in metric_payloads]
        official_f1_values = [float(p.get("official_f1", p.get("f1", 0.0)) or 0.0) for p in metric_payloads]
        evidence_recall_values = [float(p.get("evidence_recall", 0.0) or 0.0) for p in metric_payloads]
        title_recall_values = [float(p.get("supporting_fact_title_recall", 0.0) or 0.0) for p in metric_payloads]
        grounded_values = [1.0 if p.get("answer_source_grounded") else 0.0 for p in metric_payloads]
        failure_counts: Dict[str, int] = defaultdict(int)
        imported_baseline = False
        missing_sample_ids: List[str] = []
        for r in ordered_results:
            reason = _failure_reason_of(r)
            if reason:
                failure_counts[reason] += 1
            if _is_imported_baseline_result(r):
                imported_baseline = True
                metadata = _metadata_of(r)
                import_status = str(metadata.get("import_status") or _telemetry_of(r).get("import_status") or "")
                if reason == "import_missing" or import_status == "missing":
                    missing_sample_ids.append(str(getattr(r, "sample_id", "")))
        total_failures = sum(failure_counts.values())
        missing_samples = len(missing_sample_ids)
        covered_samples = n - missing_samples if imported_baseline else 0
        imported_samples = covered_samples if imported_baseline else 0
        import_coverage = round(covered_samples / n * 100, 1) if imported_baseline and n else None
        by_qt: Dict[str, Dict] = defaultdict(lambda: {"n": 0, "correct": 0, "coverage": 0})
        for r in ordered_results:
            metadata = _metadata_of(r)
            qt = (
                getattr(r, "question_type", "")
                or metadata.get(question_type_key, "unknown")
                or "unknown"
            )
            by_qt[qt]["n"] += 1
            if getattr(r, "judge_correct", False):
                by_qt[qt]["correct"] += 1
            if getattr(r, "coverage", False):
                by_qt[qt]["coverage"] += 1

        by_qt_final = {
            qt: {
                "accuracy": round(v["correct"] / v["n"] * 100, 1) if v["n"] else 0.0,
                "coverage": round(v["coverage"] / v["n"] * 100, 1) if v["n"] else 0.0,
                "n": v["n"],
            }
            for qt, v in by_qt.items()
        }

        sampling_protocol = self._sampling_metadata.get("sampling_protocol", {}) if isinstance(self._sampling_metadata, dict) else {}
        sampling_manifest = self._sampling_metadata.get("sampling_manifest", {}) if isinstance(self._sampling_metadata, dict) else {}
        corpus_metadata = self._sampling_metadata.get("corpus_snapshot", {}) if isinstance(self._sampling_metadata, dict) else {}
        result_corpus_checksum = _first_metric_value(ordered_results, "corpus_checksum")
        result_frozen_order_checksum = _first_metric_value(ordered_results, "frozen_order_checksum")
        entry = SystemEntry(
            system_name=system_name,
            n=n,
            accuracy=round(accuracy * 100, 1),
            official_em=round(sum(official_em_values) / n * 100, 1) if n else 0.0,
            official_f1=round(sum(official_f1_values) / n * 100, 1) if n else 0.0,
            llm_assisted_accuracy=round(accuracy * 100, 1),
            ci_lower=round(ci_lower * 100, 1),
            ci_upper=round(ci_upper * 100, 1),
            coverage=round(sum(coverage_list) / n * 100, 1),
            avg_latency=round(sum(latencies) / n, 1) if latencies else 0.0,
            avg_tokens=round(sum(tokens) / n, 1) if tokens else 0.0,
            avg_oracle_calls=round(query_budget_summary.get("avg_oracle_calls", 0.0), 3),
            avg_llm_calls=round(query_budget_summary.get("avg_llm_calls", 0.0), 3),
            avg_search_calls=round(query_budget_summary.get("avg_search_calls", 0.0), 3),
            avg_read_calls=round(query_budget_summary.get("avg_read_calls", 0.0), 3),
            evidence_trace_coverage=round(evidence_trace_count / n * 100, 1) if n else 0.0,
            evidence_recall=round(sum(evidence_recall_values) / n * 100, 1) if n else 0.0,
            supporting_fact_title_recall=round(sum(title_recall_values) / n * 100, 1) if n else 0.0,
            source_grounding_accuracy=round(sum(grounded_values) / n * 100, 1) if n else 0.0,
            corpus_checksum=str(result_corpus_checksum or corpus_metadata.get("corpus_checksum", "")),
            frozen_order_checksum=str(result_frozen_order_checksum or corpus_metadata.get("frozen_order_checksum", "")),
            by_question_type=by_qt_final,
            is_ours=bool(is_ours or (self._our_name and system_name == self._our_name)),
            is_published_only=False,
            correct_list=correct_list,
            sample_ids=sample_ids,
            sample_id_checksum=compute_sample_id_checksum(sample_ids),
            setup_metrics=collect_setup_metrics(ordered_results),
            query_budget_summary=query_budget_summary,
            failure_counts=dict(sorted(failure_counts.items())),
            failure_rate=round(total_failures / n * 100, 1) if n else 0.0,
            imported_baseline=imported_baseline,
            import_coverage=import_coverage,
            imported_samples=imported_samples,
            covered_samples=covered_samples,
            missing_samples=missing_samples,
            missing_sample_ids=missing_sample_ids[:50],
            sampling_method=str(sampling_protocol.get("method", "")),
            population_size=int(self._sampling_metadata.get("population_size", 0) or 0) if isinstance(self._sampling_metadata, dict) else 0,
            sampled_n=int(self._sampling_metadata.get("n_questions", n) or n) if isinstance(self._sampling_metadata, dict) else n,
            sampling_protocol=sampling_protocol,
            sampling_manifest=sampling_manifest,
            strata_distribution=sampling_manifest.get("distribution_after", {}).get("strata", {}) if isinstance(sampling_manifest.get("distribution_after", {}), dict) else {},
            weighted_metric_available=bool(sampling_manifest.get("weighted_metrics")),
        )
        self._entries.append(entry)

    def add_published_metrics(
        self,
        system_name: str,
        accuracy: float,
        coverage: float = 0.0,
        avg_latency: float = 0.0,
        avg_tokens: float = 0.0,
        n: int = 150,
        citation: str = "",
    ) -> None:
        """直接添加已发表数字（无法计算 CI 和 McNemar）。

        Args:
            system_name:  论文表格中的展示名。
            accuracy:     精度（百分比，如 98.7 表示 98.7%）。
            coverage:     覆盖率。
            avg_latency:  平均延迟（秒）。
            avg_tokens:   平均 token 数。
            n:            评估样本数。
            citation:     引用（可选）。
        """
        display = system_name
        if citation:
            display = f"{system_name}\\textsuperscript{{†}}"
        entry = SystemEntry(
            system_name=display,
            n=n,
            accuracy=float(accuracy),
            ci_lower=0.0, ci_upper=0.0,
            coverage=float(coverage),
            avg_latency=float(avg_latency),
            avg_tokens=float(avg_tokens),
            is_published_only=True,
            correct_list=[],
            setup_metrics={},
        )
        self._entries.append(entry)

    # ------------------------------------------------------------------
    # 生成表格
    # ------------------------------------------------------------------

    def generate(
        self,
        output_dir: str,
        caption: str = "",
        label: str = "tab:main_results",
        include_latency: bool = True,
        include_tokens: bool = True,
        include_breakdown: bool = True,
    ) -> Dict[str, str]:
        """生成 LaTeX + Markdown + JSON 表格。

        Args:
            output_dir:        输出目录。
            caption:           LaTeX 表格标题（为空时自动生成）。
            label:             LaTeX \\label{...} 标识符。
            include_latency:   是否包含延迟列。
            include_tokens:    是否包含 token 列。
            include_breakdown: 是否包含分题型列。

        Returns:
            {"latex": path, "markdown": path, "json": path}
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        self._finalize_statistics()

        latex_path = str(out / "paper_table.tex")
        md_path    = str(out / "paper_table.md")
        json_path  = str(out / "paper_table.json")

        _write(latex_path,
               self._to_latex(caption, label, include_latency, include_tokens, include_breakdown))
        _write(md_path,
               self._to_markdown(include_latency, include_tokens, include_breakdown))
        _write(json_path,
               self._to_json())

        logger.info("[TableGen] Generated: %s, %s, %s", latex_path, md_path, json_path)
        return {"latex": latex_path, "markdown": md_path, "json": json_path}

    def generate_feasibility_table(
        self,
        lifecycle_records: List[Any],
        output_dir: str,
        *,
        caption: str = "Full-corpus preprocessing feasibility.",
        label: str = "tab:full_corpus_feasibility",
    ) -> Dict[str, str]:
        """Generate lifecycle feasibility table with N/A failure reasons.

        Records can be ``BaselineLifecycleRecord`` instances or compatible dicts.
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        rows = [_lifecycle_row(r) for r in lifecycle_records]

        latex_path = str(out / "feasibility_table.tex")
        md_path = str(out / "feasibility_table.md")
        json_path = str(out / "feasibility_table.json")

        _write(latex_path, _feasibility_to_latex(rows, caption=caption, label=label))
        _write(md_path, _feasibility_to_markdown(rows))
        _write(json_path, json.dumps({"benchmark": self._benchmark, "systems": rows}, indent=2, ensure_ascii=False))
        return {"latex": latex_path, "markdown": md_path, "json": json_path}

    # ------------------------------------------------------------------
    # 统计计算
    # ------------------------------------------------------------------

    def _finalize_statistics(self) -> None:
        """计算 McNemar 检验 + Bonferroni 校正，写入各 entry。"""
        ours = next((e for e in self._entries if e.is_ours), None)
        baselines = [e for e in self._entries if not e.is_ours]

        if ours is None or not ours.correct_list:
            return

        p_values: List[float] = []
        baseline_with_data = []
        for b in baselines:
            if b.correct_list and len(b.correct_list) == len(ours.correct_list):
                _, p_val, _ = mcnemar_test(ours.correct_list, b.correct_list)
                p_values.append(p_val)
                baseline_with_data.append(b)
            else:
                p_values.append(1.0)
                baseline_with_data.append(b)

        # Bonferroni 校正
        corrected = bonferroni_correction(p_values)
        for b, p, sig in zip(baseline_with_data, p_values, corrected):
            b.p_value = p
            b.is_significant = sig
            b.sig_marker = significance_marker(p) if sig else ""

    # ------------------------------------------------------------------
    # LaTeX 输出
    # ------------------------------------------------------------------

    def _to_latex(
        self, caption, label, include_latency, include_tokens, include_breakdown
    ) -> str:
        entries = self._entries
        if not entries:
            return "% No entries\n"

        # 找出各列最优值（用于加粗）
        best_acc = max(e.accuracy for e in entries)
        best_cov = max(e.coverage for e in entries)

        # 列定义
        include_setup = any(e.setup_metrics for e in entries)
        include_import = any(e.imported_baseline for e in entries)
        include_failures = any(e.failure_counts for e in entries)
        cols = ["l", "r", "r", "r", "r", "r", "r"]
        headers = ["System", "Accuracy (\\%)", "EM (\\%)", "F1 (\\%)", "LLM Acc. (\\%)", "Evi. Rec. (\\%)", "Coverage (\\%)"]
        if include_import:
            cols.append("r")
            headers.append("Import Cov. (\\%)")
        if include_failures:
            cols.append("l")
            headers.append("Failures")
        if include_latency:
            cols.append("r")
            headers.append("Latency (s)")
        if include_tokens:
            cols.append("r")
            headers.append("Tokens")
        if include_setup:
            cols.append("r")
            headers.append("Setup (s)")

        col_spec = "".join(cols)

        def _fmt_acc(e: SystemEntry) -> str:
            val_str = f"{e.accuracy:.1f}"
            if not e.is_published_only and e.ci_lower:
                ci = f"{{\\scriptsize ±{(e.ci_upper - e.ci_lower) / 2:.1f}}}"
            else:
                ci = ""
            sig = e.sig_marker if not e.is_ours else ""
            full = f"{val_str}{ci}{sig}"
            if abs(e.accuracy - best_acc) < 0.05:
                full = f"\\textbf{{{full}}}"
            return full

        def _fmt_cov(e: SystemEntry) -> str:
            s = f"{e.coverage:.1f}"
            if abs(e.coverage - best_cov) < 0.05:
                s = f"\\textbf{{{s}}}"
            return s

        lines = [
            "\\begin{table}[t]",
            f"\\caption{{{caption or f'Results on {self._benchmark}'}}}"
            + f"\\label{{{label}}}",
            "\\centering",
            f"\\begin{{tabular}}{{{col_spec}}}",
            "\\hline",
            " & ".join(headers) + " \\\\",
            "\\hline",
        ]

        # 先写非 ours，再写 ours（中间加分隔线）
        non_ours = [e for e in entries if not e.is_ours]
        ours_entries = [e for e in entries if e.is_ours]

        for e in non_ours:
            row = [
                e.system_name,
                _fmt_acc(e),
                f"{e.official_em:.1f}",
                f"{e.official_f1:.1f}",
                f"{e.llm_assisted_accuracy:.1f}",
                f"{e.evidence_recall:.1f}",
                _fmt_cov(e),
            ]
            if include_import:
                row.append(_format_import_coverage(e))
            if include_failures:
                row.append(_format_failure_counts(e, latex=True))
            if include_latency:
                row.append(f"{e.avg_latency:.1f}")
            if include_tokens:
                row.append(f"{e.avg_tokens / 1000:.1f}K" if e.avg_tokens >= 1000
                           else f"{e.avg_tokens:.0f}")
            if include_setup:
                row.append(f"{_setup_seconds(e):.3f}")
            lines.append(" & ".join(row) + " \\\\")

        if ours_entries and non_ours:
            lines.append("\\hline")

        for e in ours_entries:
            row = [
                f"\\textbf{{{e.system_name}}}",
                _fmt_acc(e),
                f"{e.official_em:.1f}",
                f"{e.official_f1:.1f}",
                f"{e.llm_assisted_accuracy:.1f}",
                f"{e.evidence_recall:.1f}",
                _fmt_cov(e),
            ]
            if include_import:
                row.append(_format_import_coverage(e))
            if include_failures:
                row.append(_format_failure_counts(e, latex=True))
            if include_latency:
                row.append(f"{e.avg_latency:.1f}")
            if include_tokens:
                row.append(f"{e.avg_tokens / 1000:.1f}K" if e.avg_tokens >= 1000
                           else f"{e.avg_tokens:.0f}")
            if include_setup:
                row.append(f"{_setup_seconds(e):.3f}")
            lines.append(" & ".join(row) + " \\\\")

        lines += [
            "\\hline",
            "\\end{tabular}",
            "% † Published results. * p<0.05, ** p<0.01, *** p<0.001 (McNemar + Bonferroni)",
            "\\end{table}",
        ]

        # 分题型子表（可选）
        if include_breakdown:
            lines += self._breakdown_latex(entries)

        return "\n".join(lines) + "\n"

    def _breakdown_latex(self, entries: List[SystemEntry]) -> List[str]:
        """生成按 question_type 分类的子表格。"""
        all_qts = sorted(
            {qt for e in entries for qt in e.by_question_type}
        )
        if not all_qts:
            return []

        n_qt = len(all_qts)
        col_spec = "l" + "r" * n_qt
        qt_headers = " & ".join(f"{qt[:15]}" for qt in all_qts)

        lines = [
            "",
            "% --- Breakdown by Question Type (Accuracy %) ---",
            f"\\begin{{tabular}}{{{col_spec}}}",
            "\\hline",
            f"System & {qt_headers} \\\\",
            "\\hline",
        ]
        for e in entries:
            vals = [
                f"{e.by_question_type.get(qt, {}).get('accuracy', 0):.1f}"
                for qt in all_qts
            ]
            lines.append(f"{e.system_name} & {' & '.join(vals)} \\\\")
        lines += ["\\hline", "\\end{tabular}"]
        return lines

    # ------------------------------------------------------------------
    # Markdown 输出
    # ------------------------------------------------------------------

    def _to_markdown(
        self, include_latency, include_tokens, include_breakdown
    ) -> str:
        entries = self._entries
        if not entries:
            return "_No entries_\n"

        headers = ["System", "Accuracy (%)", "Official EM (%)", "Official F1 (%)", "LLM Acc (%)", "Evidence Recall (%)", "Coverage (%)"]
        include_setup = any(e.setup_metrics for e in entries)
        include_import = any(e.imported_baseline for e in entries)
        include_failures = any(e.failure_counts for e in entries)
        if include_import:
            headers.append("Import Cov (%)")
        if include_failures:
            headers.append("Failures")
        if include_latency:
            headers.append("Latency (s)")
        if include_tokens:
            headers.append("Tokens")
        if include_setup:
            headers.append("Setup (s)")

        rows = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
        ]

        best_acc = max(e.accuracy for e in entries)

        for e in entries:
            prefix = "**" if e.is_ours else ""
            suffix = "**" if e.is_ours else ""
            ci_str = ""
            if not e.is_published_only and e.ci_lower:
                half = (e.ci_upper - e.ci_lower) / 2
                ci_str = f" ±{half:.1f}"
            bold_marker = "🏆 " if abs(e.accuracy - best_acc) < 0.05 else ""
            acc_str = f"{bold_marker}{e.accuracy:.1f}{ci_str}{e.sig_marker}"
            row = [
                f"{prefix}{e.system_name}{suffix}",
                acc_str,
                f"{e.official_em:.1f}",
                f"{e.official_f1:.1f}",
                f"{e.llm_assisted_accuracy:.1f}",
                f"{e.evidence_recall:.1f}",
                f"{e.coverage:.1f}",
            ]
            if include_import:
                row.append(_format_import_coverage(e))
            if include_failures:
                row.append(_format_failure_counts(e))
            if include_latency:
                row.append(f"{e.avg_latency:.1f}")
            if include_tokens:
                row.append(f"{e.avg_tokens / 1000:.1f}K" if e.avg_tokens >= 1000
                           else f"{e.avg_tokens:.0f}")
            if include_setup:
                row.append(f"{_setup_seconds(e):.3f}")
            rows.append("| " + " | ".join(row) + " |")

        footer = "\n_* p<0.05, ** p<0.01, *** p<0.001 (McNemar + Bonferroni correction)_\n"

        # 分题型子表
        breakdown_md = ""
        if include_breakdown:
            all_qts = sorted({qt for e in entries for qt in e.by_question_type})
            if all_qts:
                bh = ["System"] + list(all_qts)
                breakdown_md = (
                    "\n\n### Accuracy by Question Type (%)\n\n"
                    "| " + " | ".join(bh) + " |\n"
                    "| " + " | ".join(["---"] * len(bh)) + " |\n"
                )
                for e in entries:
                    vals = [
                        f"{e.by_question_type.get(qt, {}).get('accuracy', 0):.1f}"
                        for qt in all_qts
                    ]
                    breakdown_md += "| " + " | ".join([e.system_name] + vals) + " |\n"

        return "\n".join(rows) + "\n" + footer + breakdown_md

    # ------------------------------------------------------------------
    # JSON 输出
    # ------------------------------------------------------------------

    def _to_json(self) -> str:
        data = {
            "benchmark": self._benchmark,
            "sampling": self._sampling_metadata,
            "systems": [
                {
                    "system_name":       e.system_name,
                    "n":                 e.n,
                    "accuracy":          e.accuracy,
                    "official_em":       e.official_em,
                    "official_f1":       e.official_f1,
                    "llm_assisted_accuracy": e.llm_assisted_accuracy,
                    "ci_lower":          e.ci_lower,
                    "ci_upper":          e.ci_upper,
                    "coverage":          e.coverage,
                    "evidence_recall":   e.evidence_recall,
                    "supporting_fact_title_recall": e.supporting_fact_title_recall,
                    "source_grounding_accuracy": e.source_grounding_accuracy,
                    "avg_latency":       e.avg_latency,
                    "avg_tokens":        e.avg_tokens,
                    "avg_oracle_calls":  e.avg_oracle_calls,
                    "avg_llm_calls":     e.avg_llm_calls,
                    "avg_search_calls":  e.avg_search_calls,
                    "avg_read_calls":    e.avg_read_calls,
                    "evidence_trace_coverage": e.evidence_trace_coverage,
                    "is_ours":           e.is_ours,
                    "is_published_only": e.is_published_only,
                    "p_value":           e.p_value,
                    "is_significant":    e.is_significant,
                    "sig_marker":        e.sig_marker,
                    "sample_id_checksum": e.sample_id_checksum,
                    "frozen_order_checksum": e.frozen_order_checksum,
                    "corpus_checksum":    e.corpus_checksum,
                    "sample_ids":         e.sample_ids,
                    "setup_metrics":      e.setup_metrics,
                    "query_budget_summary": e.query_budget_summary,
                    "failure_counts":     e.failure_counts,
                    "failure_rate":       e.failure_rate,
                    "imported_baseline":  e.imported_baseline,
                    "import_coverage":    e.import_coverage,
                    "imported_samples":   e.imported_samples,
                    "covered_samples":    e.covered_samples,
                    "missing_samples":    e.missing_samples,
                    "missing_sample_ids": e.missing_sample_ids,
                    "sampling_method":    e.sampling_method,
                    "population_size":    e.population_size,
                    "sampled_n":          e.sampled_n,
                    "sampling_protocol":  e.sampling_protocol,
                    "sampling_manifest":  e.sampling_manifest,
                    "strata_distribution": e.strata_distribution,
                    "weighted_metric_available": e.weighted_metric_available,
                    "by_question_type":  e.by_question_type,
                }
                for e in self._entries
            ],
        }
        return json.dumps(data, ensure_ascii=False, indent=2)


def _metadata_of(result: Any) -> Dict[str, Any]:
    metadata = getattr(result, "metadata", None)
    if isinstance(metadata, dict) and metadata:
        return metadata
    raw = getattr(result, "raw", None)
    if isinstance(raw, dict):
        nested = raw.get("metadata")
        if isinstance(nested, dict):
            return nested
        return raw
    return {}


def _telemetry_of(result: Any) -> Dict[str, Any]:
    telemetry = getattr(result, "telemetry", None)
    if isinstance(telemetry, dict):
        return telemetry
    raw = getattr(result, "raw", None)
    if isinstance(raw, dict) and isinstance(raw.get("telemetry"), dict):
        return raw["telemetry"]
    return {}


def _failure_reason_of(result: Any) -> str:
    reason = getattr(result, "failure_reason", "") or _metadata_of(result).get("failure_reason") or _telemetry_of(result).get("failure_reason")
    if reason:
        return str(reason)
    if getattr(result, "error", None):
        return "system_error"
    raw = getattr(result, "raw", None)
    if isinstance(raw, dict) and raw.get("error"):
        return "system_error"
    return ""


def _is_imported_baseline_result(result: Any) -> bool:
    metadata = _metadata_of(result)
    telemetry = _telemetry_of(result)
    return bool(
        metadata.get("imported_baseline")
        or metadata.get("import_adapter")
        or metadata.get("external_prediction_required")
        or metadata.get("imported_from")
        or telemetry.get("imported_baseline")
    )


def _format_import_coverage(entry: SystemEntry) -> str:
    if not entry.imported_baseline or entry.import_coverage is None:
        return "-"
    return f"{entry.import_coverage:.1f}"


def _format_failure_counts(entry: SystemEntry, *, latex: bool = False) -> str:
    if not entry.failure_counts:
        return "0"
    text = ", ".join(f"{name}={count}" for name, count in entry.failure_counts.items())
    return text.replace("_", "\\_") if latex else text


def _metric_payload_of(result: Any) -> Dict[str, Any]:
    payload: Dict[str, Any] = {}
    raw = getattr(result, "raw", None)
    if isinstance(raw, dict):
        per_sample = raw.get("per_sample_eval")
        if isinstance(per_sample, dict):
            payload.update(per_sample)
        raw_telemetry = raw.get("telemetry")
        if isinstance(raw_telemetry, dict):
            payload.update(raw_telemetry)
    metadata = getattr(result, "metadata", None)
    if isinstance(metadata, dict):
        judge_result = metadata.get("judge_result")
        if isinstance(judge_result, dict):
            payload.update(judge_result)
        for key in (
            "evidence_recall",
            "supporting_fact_title_recall",
            "answer_source_grounded",
            "official_em",
            "official_f1",
        ):
            if key in metadata:
                payload[key] = metadata[key]
    telemetry = getattr(result, "telemetry", None)
    if isinstance(telemetry, dict):
        payload.update(telemetry)
    return payload


def _first_metric_value(results: List[Any], key: str) -> Any:
    for result in results:
        payload = _metric_payload_of(result)
        value = payload.get(key)
        if value not in (None, ""):
            return value
    return ""


def _query_budget_of(result: Any) -> Dict[str, Any]:
    telemetry = _telemetry_of(result)
    metadata = _metadata_of(result)
    query_budget = telemetry.get("query_budget") if isinstance(telemetry.get("query_budget"), dict) else {}
    if not query_budget and isinstance(metadata.get("query_budget"), dict):
        query_budget = metadata.get("query_budget", {})
    return {
        "oracle_calls": _number(query_budget.get("oracle_calls", telemetry.get("oracle_calls", telemetry.get("loop_count", 0.0)))),
        "llm_calls": _number(query_budget.get("llm_calls", telemetry.get("llm_calls", telemetry.get("total_llm_calls", 0.0)))),
        "search_calls": _number(query_budget.get("search_calls", telemetry.get("search_calls", len(telemetry.get("search_history", []) or [])))),
        "read_calls": _number(query_budget.get("read_calls", telemetry.get("read_calls", len(telemetry.get("read_file_ids", []) or [])))),
        "total_tokens": _number(query_budget.get("total_tokens", telemetry.get("total_tokens", getattr(result, "tokens_used", 0)))),
        "judge_tokens": _number(query_budget.get("judge_tokens", telemetry.get("judge_tokens", getattr(result, "judge_tokens", 0)))),
        "latency_seconds": _number(query_budget.get("latency_seconds", getattr(result, "elapsed", 0.0))),
        "budget_exceeded": bool(query_budget.get("budget_exceeded") or telemetry.get("failure_reason") == "budget_exceeded"),
    }


def _summarize_query_budgets(query_budgets: List[Dict[str, Any]]) -> Dict[str, Any]:
    oracle_calls = [_number(row.get("oracle_calls")) for row in query_budgets]
    llm_calls = [_number(row.get("llm_calls")) for row in query_budgets]
    search_calls = [_number(row.get("search_calls")) for row in query_budgets]
    read_calls = [_number(row.get("read_calls")) for row in query_budgets]
    total_tokens = [_number(row.get("total_tokens")) for row in query_budgets]
    judge_tokens = [_number(row.get("judge_tokens")) for row in query_budgets]
    latency_seconds = [_number(row.get("latency_seconds")) for row in query_budgets]
    return {
        "avg_oracle_calls": _avg(oracle_calls),
        "std_oracle_calls": _std(oracle_calls),
        "max_oracle_calls": max(oracle_calls, default=0.0),
        "avg_llm_calls": _avg(llm_calls),
        "avg_search_calls": _avg(search_calls),
        "avg_read_calls": _avg(read_calls),
        "avg_total_tokens": _avg(total_tokens),
        "std_total_tokens": _std(total_tokens),
        "max_total_tokens": max(total_tokens, default=0.0),
        "avg_judge_tokens": _avg(judge_tokens),
        "avg_latency_seconds": _avg(latency_seconds),
        "std_latency_seconds": _std(latency_seconds),
        "max_latency_seconds": max(latency_seconds, default=0.0),
        "budget_exceeded_count": sum(1 for row in query_budgets if row.get("budget_exceeded")),
    }


def _avg(values: List[float]) -> float:
    return sum(values) / max(len(values), 1)


def _std(values: List[float]) -> float:
    if not values:
        return 0.0
    mean = _avg(values)
    return (sum((value - mean) ** 2 for value in values) / len(values)) ** 0.5


def _evidence_traces_of(result: Any) -> List[Any]:
    telemetry = _telemetry_of(result)
    metadata = _metadata_of(result)
    for source in (telemetry, metadata):
        traces = source.get("evidence_traces") if isinstance(source, dict) else None
        if isinstance(traces, list) and traces:
            return traces
    for source in (telemetry, metadata):
        if not isinstance(source, dict):
            continue
        for key in ("evidence_sources", "read_file_ids", "retrieval_logs"):
            values = source.get(key)
            if isinstance(values, list) and values:
                return values
    return []


def _number(value: Any, default: float = 0.0) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return default


def _setup_seconds(entry: SystemEntry) -> float:
    setup = entry.setup_metrics or {}
    try:
        return float(setup.get("setup_seconds", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _lifecycle_row(record: Any) -> Dict[str, Any]:
    data = record.to_dict() if hasattr(record, "to_dict") else dict(record)
    build_completed = bool(data.get("build_completed", False))
    index_ready = bool(data.get("index_ready", False))
    query_eligible = bool(data.get("query_eligible", index_ready))
    failure_reason = str(data.get("failure_reason") or "none")
    metadata = data.get("metadata") if isinstance(data.get("metadata"), dict) else {}
    index_required = bool(data.get("index_required", metadata.get("index_required", True)))
    return {
        "system": data.get("citation_name") or data.get("baseline_name") or data.get("method") or "unknown",
        "baseline_name": data.get("baseline_name", ""),
        "index_required": "Yes" if index_required else "No",
        "build_completed": "Yes" if build_completed else "No" if failure_reason != "none" else "N/A",
        "index_ready": "Yes" if index_ready else "No" if failure_reason != "none" else "N/A",
        "query_eligible": "Yes" if query_eligible else "N/A",
        "build_time_seconds": float(data.get("build_time_seconds", 0.0) or 0.0),
        "indexed_documents": int(data.get("indexed_documents", 0) or 0),
        "peak_ram_bytes": int(data.get("peak_ram_bytes", 0) or 0),
        "disk_bytes": int(data.get("disk_bytes", 0) or 0),
        "preprocess_llm_tokens": int(data.get("preprocess_llm_tokens", 0) or 0),
        "failure_reason": failure_reason if failure_reason != "none" else "-",
    }


def _feasibility_to_markdown(rows: List[Dict[str, Any]]) -> str:
    headers = [
        "System", "Index Req.", "Build", "Ready", "Query", "Build Time (s)",
        "Disk", "LLM Tokens", "Failure Reason",
    ]
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        out.append("| " + " | ".join([
            str(row["system"]),
            str(row["index_required"]),
            str(row["build_completed"]),
            str(row["index_ready"]),
            str(row["query_eligible"]),
            f"{row['build_time_seconds']:.1f}",
            _format_bytes(row["disk_bytes"]),
            _format_int(row["preprocess_llm_tokens"]),
            str(row["failure_reason"]),
        ]) + " |")
    return "\n".join(out) + "\n"


def _feasibility_to_latex(rows: List[Dict[str, Any]], *, caption: str, label: str) -> str:
    lines = [
        "\\begin{table}[t]",
        f"\\caption{{{caption}}}\\label{{{label}}}",
        "\\centering",
        "\\begin{tabular}{llllrrl}",
        "\\hline",
        "System & Build & Ready & Query & Build(s) & Storage & Failure " + "\\\\",
        "\\hline",
    ]
    for row in rows:
        failure = str(row["failure_reason"]).replace("_", "\\_")
        system = str(row["system"]).replace("_", "\\_")
        lines.append(
            f"{system} & {row['build_completed']} & {row['index_ready']} & "
            f"{row['query_eligible']} & {row['build_time_seconds']:.1f} & "
            f"{_format_bytes(row['disk_bytes'])} & {failure} \\\\"
        )
    lines += ["\\hline", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines) + "\n"


def _format_bytes(value: int) -> str:
    value = int(value or 0)
    if value <= 0:
        return "0"
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(value)
    idx = 0
    while size >= 1024 and idx < len(units) - 1:
        size /= 1024
        idx += 1
    return f"{size:.1f}{units[idx]}"


def _format_int(value: int) -> str:
    return f"{int(value):,}" if value else "0"


def _write(path: str, content: str) -> None:
    Path(path).write_text(content, encoding="utf-8")
    logger.debug("[TableGen] Written: %s", path)
