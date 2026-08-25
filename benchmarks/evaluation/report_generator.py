"""Metric-first academic report generator."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from .error_appendix import ErrorAppendixGenerator
from .figure_generator import FigureGenerator
from .report_validator import AcademicReportValidator, ValidationReport
from .reproducibility import ReproducibilityChecklist


class ReportGenerator:
    """Generate Markdown/LaTeX reports only from structured facts."""

    def generate(
        self,
        *,
        run_dir: str | Path | None = None,
        table_json: str | Path | None = None,
        output_dir: str | Path | None = None,
        title: str = "Sirchmunk ResearchOps Report",
    ) -> Dict[str, str]:
        run_path = Path(run_dir).resolve() if run_dir else None
        table_path = Path(table_json).resolve() if table_json else None
        if output_dir:
            out = Path(output_dir).resolve()
        elif run_path:
            out = run_path / "reports"
        elif table_path:
            out = table_path.parent / "report"
        else:
            raise ValueError("ReportGenerator requires run_dir or table_json")
        out.mkdir(parents=True, exist_ok=True)
        fig_dir = out / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)

        validator = AcademicReportValidator()
        validation = validator.validate(run_dir=run_path, table_json=table_path)
        table = _read_json(table_path) if table_path else {}
        manifest = _read_json(run_path / "manifest.json") if run_path else {}
        protocol = _read_json(run_path / "protocol.yaml") if run_path else {}
        metrics = _read_json(run_path / "results" / "metrics.json") if run_path else {}
        predictions_path = run_path / "results" / "predictions.jsonl" if run_path else None
        badcase_taxonomy = _read_json(run_path / "analysis" / "failure_taxonomy.json") if run_path else {}
        answer_type_consistency = _read_json(run_path / "analysis" / "answer_type_consistency.json") if run_path else {}
        quality_gate = _read_json(run_path / "analysis" / "quality_gate.json") if run_path else {}

        figures = FigureGenerator().generate_from_table(table_path, fig_dir) if table_path else {}
        repro_md = ReproducibilityChecklist().to_markdown(run_path) if run_path else ""
        error_md = ErrorAppendixGenerator().to_markdown(predictions_path) if predictions_path and predictions_path.exists() else ""

        report_md = self._to_markdown(
            title=title,
            validation=validation,
            table=table,
            manifest=manifest,
            protocol=protocol,
            metrics=metrics,
            figures=figures,
            repro_md=repro_md,
            error_md=error_md,
            badcase_taxonomy=badcase_taxonomy,
            answer_type_consistency=answer_type_consistency,
            quality_gate=quality_gate,
        )
        report_tex_fragment = self._to_latex_table_fragment(
            title=title,
            table=table,
            metrics=metrics,
            manifest=manifest,
            protocol=protocol,
        )
        report_tex = self._to_latex_standalone(title=title, body=report_tex_fragment)

        md_path = out / "report.md"
        tex_path = out / "report.tex"
        tex_fragment_path = out / "report_fragment.tex"
        validation_path = out / "validation.json"
        md_path.write_text(report_md, encoding="utf-8")
        tex_path.write_text(report_tex, encoding="utf-8")
        tex_fragment_path.write_text(report_tex_fragment, encoding="utf-8")
        validation_path.write_text(json.dumps(validation.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        return {
            "markdown": str(md_path),
            "latex": str(tex_path),
            "latex_fragment": str(tex_fragment_path),
            "validation": str(validation_path),
            **{f"figure_{k}": v for k, v in figures.items()},
        }

    def _to_markdown(
        self,
        *,
        title: str,
        validation: ValidationReport,
        table: Dict[str, Any],
        manifest: Dict[str, Any],
        protocol: Dict[str, Any],
        metrics: Dict[str, Any],
        figures: Dict[str, str],
        repro_md: str,
        error_md: str,
        badcase_taxonomy: Dict[str, Any] | None = None,
        answer_type_consistency: Dict[str, Any] | None = None,
        quality_gate: Dict[str, Any] | None = None,
    ) -> str:
        lines = [f"# {title}", ""]
        status = "PASS" if validation.passed else "BLOCKED"
        lines.append(f"## Validation Status: `{status}`")
        lines.append("")
        if validation.issues:
            lines.append("### Validation Issues")
            for issue in validation.issues:
                lines.append(f"- `{issue.severity}` / `{issue.check}`: {issue.message}")
            lines.append("")
        lines.append("## Experiment Summary")
        lines.append(f"- Benchmark: `{manifest.get('benchmark') or protocol.get('benchmark')}`")
        lines.append(f"- Run ID: `{manifest.get('run_id') or protocol.get('run_id')}`")
        lines.append(f"- Git commit: `{manifest.get('git_commit')}`")
        lines.append(f"- Config hash: `{manifest.get('config_hash')}`")
        lines.append(f"- Samples: `{metrics.get('n', 'unknown')}`")
        sampling = table.get("sampling", {}) if isinstance(table.get("sampling", {}), dict) else {}
        sampling_protocol = sampling.get("sampling_protocol", {}) if isinstance(sampling.get("sampling_protocol", {}), dict) else {}
        sampling_manifest = sampling.get("sampling_manifest", {}) if isinstance(sampling.get("sampling_manifest", {}), dict) else {}
        if sampling:
            lines.append(f"- Evaluation scope: `{sampling.get('evaluation_scope', 'unknown')}`")
            if sampling.get("benchmark_label"):
                lines.append(f"- Benchmark label: `{sampling.get('benchmark_label')}`")
            lines.append(f"- Sampling method: `{sampling_protocol.get('method', 'unknown')}`")
            lines.append(f"- Population size: `{sampling.get('population_size', sampling_manifest.get('population_size', 'unknown'))}`")
            lines.append(f"- Sampled questions: `{sampling.get('n_questions', sampling_manifest.get('actual_n', 'unknown'))}`")
            lines.append(f"- Sample ID checksum: `{sampling.get('sample_id_checksum', sampling_manifest.get('sample_id_checksum', ''))}`")
        if metrics:
            if "official_exact_match" in metrics:
                lines.append(f"- Official exact match: `{metrics.get('official_exact_match')}`")
            if "f1" in metrics:
                lines.append(f"- Official token F1: `{metrics.get('f1')}`")
            if "official_f1_correct" in metrics:
                lines.append(f"- Official F1-correct: `{metrics.get('official_f1_correct')}`")
            if "llm_assisted_accuracy" in metrics:
                lines.append(f"- LLM-assisted accuracy: `{metrics.get('llm_assisted_accuracy')}`")
            else:
                lines.append(f"- Accuracy: `{metrics.get('accuracy', 'n/a')}`")
            lines.append(f"- Coverage: `{metrics.get('coverage', 'n/a')}`")
            if "evidence_recall" in metrics:
                lines.append(f"- Evidence recall: `{metrics.get('evidence_recall')}`")
            if "target_slot_verification_rate" in metrics:
                lines.append(f"- Target-slot verification rate: `{metrics.get('target_slot_verification_rate')}`")
                lines.append(f"- Target-slot checked samples: `{metrics.get('target_slot_checked_samples', 'unknown')}`")
            if "latency" in metrics:
                lines.append(f"- Latency: `{json.dumps(metrics.get('latency'), ensure_ascii=False)}`")
            qgate = metrics.get("quality_gate") if isinstance(metrics.get("quality_gate"), dict) else quality_gate
            if isinstance(qgate, dict) and qgate:
                lines.append(f"- Quickstart pipeline gate: `{qgate.get('pipeline_ok', qgate.get('quality_ok'))}`")
                lines.append(f"- Quickstart quality gate: `{qgate.get('quality_ok')}`")
                if qgate.get("failed_pipeline_checks"):
                    lines.append(f"- Pipeline failed checks: `{qgate.get('failed_pipeline_checks')}`")
                if qgate.get("failed_quality_checks"):
                    lines.append(f"- Quality failed checks: `{qgate.get('failed_quality_checks')}`")
                elif qgate.get("failed_checks"):
                    lines.append(f"- Quality failed checks: `{qgate.get('failed_checks')}`")
        lines.append("")

        if badcase_taxonomy or answer_type_consistency:
            lines.append("## Structured Diagnostics")
            if badcase_taxonomy:
                lines.append(f"- Badcases: `{badcase_taxonomy.get('badcase_count', 0)}` / `{badcase_taxonomy.get('total_samples', 'unknown')}`")
                lines.append(f"- Failure taxonomy: `{json.dumps(badcase_taxonomy.get('failure_type_breakdown', {}), ensure_ascii=False)}`")
                lines.append(f"- Root causes: `{json.dumps(badcase_taxonomy.get('root_cause_breakdown', {}), ensure_ascii=False)}`")
                if badcase_taxonomy.get("fixability_breakdown"):
                    lines.append(f"- Fixability: `{json.dumps(badcase_taxonomy.get('fixability_breakdown', {}), ensure_ascii=False)}`")
            if answer_type_consistency:
                lines.append(f"- Answer type mismatches: `{answer_type_consistency.get('type_mismatch_count', 0)}`")
                lines.append(f"- Gold/question type conflicts: `{answer_type_consistency.get('gold_question_type_conflicts', 0)}`")
                lines.append(f"- Prediction type mismatches: `{answer_type_consistency.get('prediction_type_mismatches', 0)}`")
            lines.append("")

        if sampling:
            lines.append("## Sampling Protocol")
            lines.append(f"- Method: `{sampling_protocol.get('method', 'unknown')}`")
            lines.append(f"- Evaluation scope: `{sampling.get('evaluation_scope', 'unknown')}`")
            lines.append(f"- Seed: `{sampling_protocol.get('seed', 'unknown')}`")
            lines.append(f"- Target N: `{sampling_protocol.get('target_n', 'unknown')}`")
            lines.append(f"- Strata: `{', '.join(sampling_protocol.get('strata', []) or [])}`")
            lines.append(f"- Allocation: `{sampling_protocol.get('allocation', 'n/a')}`")
            deviation = sampling_manifest.get('deviation_report', {}) if isinstance(sampling_manifest, dict) else {}
            if deviation:
                lines.append(f"- Max absolute distribution delta: `{deviation.get('max_abs_proportion_delta', 'n/a')}`")
            lines.append("- Publication claim rule: sampled tables must be described as sampled evaluation unless `method=full` and actual N equals the population size.")
            lines.append("")

        if table.get("systems"):
            lines.append("## Main Result Table")
            lines.append("")
            systems = table["systems"]
            include_import = any(system.get("imported_baseline") for system in systems)
            include_failures = any(system.get("failure_counts") for system in systems)
            headers = ["System", "N", "Official EM", "Official F1", "LLM Acc", "Coverage"]
            if include_import:
                headers.append("Import Cov")
            if include_failures:
                headers.append("Failures")
            headers.extend(["Latency", "Setup(s)", "p-value"])
            lines.append("| " + " | ".join(headers) + " |")
            lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
            for system in systems:
                setup = system.get("setup_metrics") or {}
                row = [
                    str(system.get("system_name", "")),
                    str(system.get("n", "")),
                    str(system.get("official_em", system.get("accuracy", ""))),
                    str(system.get("official_f1", "")),
                    str(system.get("llm_assisted_accuracy", system.get("accuracy", ""))),
                    str(system.get("coverage", "")),
                ]
                if include_import:
                    row.append(_format_import_coverage(system))
                if include_failures:
                    row.append(_format_failure_counts(system))
                row.extend([
                    str(system.get("avg_latency", "")),
                    f"{float(setup.get('setup_seconds', 0) or 0):.3f}",
                    str(system.get("p_value")),
                ])
                lines.append("| " + " | ".join(row) + " |")
            lines.append("")

        if figures:
            lines.append("## Figures")
            for name, path in figures.items():
                rel = Path(path).name
                lines.append(f"- `{name}`: `{rel}`")
            lines.append("")

        if repro_md:
            lines.append(repro_md)
        if error_md:
            lines.append(error_md)

        lines.append("## Limitations")
        lines.append("- This report is metric-first: narrative claims must be derived from the structured metrics above.")
        lines.append("- A validation status of `BLOCKED` means the report should not be used as a publication-ready claim package.")
        return "\n".join(lines) + "\n"

    def _to_latex_table_fragment(
        self,
        *,
        title: str,
        table: Dict[str, Any],
        metrics: Dict[str, Any],
        manifest: Dict[str, Any],
        protocol: Dict[str, Any],
    ) -> str:
        caption = _latex_escape(_table_caption(title, table, manifest, protocol))
        label = "tab:main_experiment_results"
        rows = _table_rows(table, metrics, manifest, protocol)
        headers = ["System", "N", "EM", "F1", "F1-C", "LLM Acc.", "Cov.", "Evi. Rec.", "Lat.(s)"]
        latex_row_end = r" \\"
        lines = [
            "\\begin{table}[t]",
            "\\centering",
            "\\caption{" + caption + "}",
            "\\label{" + label + "}",
            "\\small",
            "\\resizebox{\\linewidth}{!}{%",
            "\\begin{tabular}{lrrrrrrrr}",
            "\\toprule",
            " & ".join(headers) + latex_row_end,
            "\\midrule",
        ]
        for row in rows:
            lines.append(" & ".join(_latex_escape(cell) for cell in row) + latex_row_end)
        lines.extend([
            "\\bottomrule",
            "\\end{tabular}%",
            "}",
            "\\end{table}",
            "",
        ])
        return "\n".join(lines)

    def _to_latex_standalone(self, *, title: str, body: str) -> str:
        return "\n".join([
            "\\documentclass[11pt]{article}",
            "\\usepackage[a4paper,margin=1in]{geometry}",
            "\\usepackage[T1]{fontenc}",
            "\\usepackage[utf8]{inputenc}",
            "\\usepackage{lmodern}",
            "\\usepackage{booktabs}",
            "\\usepackage{graphicx}",
            "\\begin{document}",
            body,
            "\\end{document}",
            "",
        ])


def _format_import_coverage(system: Dict[str, Any]) -> str:
    if not system.get("imported_baseline") or system.get("import_coverage") is None:
        return "-"
    try:
        return f"{float(system.get('import_coverage')):.1f}"
    except (TypeError, ValueError):
        return str(system.get("import_coverage"))


def _format_failure_counts(system: Dict[str, Any]) -> str:
    counts = system.get("failure_counts") or {}
    if not isinstance(counts, dict) or not counts:
        return "0"
    return ", ".join(f"{name}={count}" for name, count in counts.items())


def _read_json(path: Optional[Path]) -> Dict[str, Any]:
    if not path or not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _table_caption(title: str, table: Dict[str, Any], manifest: Dict[str, Any], protocol: Dict[str, Any]) -> str:
    sampling = table.get("sampling", {}) if isinstance(table.get("sampling", {}), dict) else {}
    if sampling.get("benchmark_label"):
        return str(sampling.get("benchmark_label"))
    benchmark = str(table.get("benchmark") or manifest.get("benchmark") or protocol.get("benchmark") or title or "Main experiment")
    return f"{benchmark} main experiment results"


def _table_rows(table: Dict[str, Any], metrics: Dict[str, Any], manifest: Dict[str, Any], protocol: Dict[str, Any]) -> list[list[str]]:
    systems = table.get("systems", []) if isinstance(table, dict) else []
    if systems:
        return [_system_row(system) for system in systems]
    system_name = str(manifest.get("system") or "Sirchmunk / LENS")
    return [[
        system_name,
        _fmt_cell(metrics.get("n", "")),
        _fmt_cell(metrics.get("official_exact_match", metrics.get("accuracy", ""))),
        _fmt_cell(metrics.get("f1", metrics.get("official_f1", ""))),
        _fmt_cell(metrics.get("official_f1_correct", "")),
        _fmt_cell(metrics.get("llm_assisted_accuracy", metrics.get("accuracy", ""))),
        _fmt_cell(metrics.get("coverage", "")),
        _fmt_cell(metrics.get("evidence_recall", "")),
        _fmt_cell((metrics.get("latency") or {}).get("avg") if isinstance(metrics.get("latency"), dict) else ""),
    ]]


def _system_row(system: Dict[str, Any]) -> list[str]:
    return [
        str(system.get("system_name", "")),
        _fmt_cell(system.get("n", "")),
        _fmt_cell(system.get("official_em", system.get("accuracy", ""))),
        _fmt_cell(system.get("official_f1", "")),
        _fmt_cell(system.get("official_f1_correct", "")),
        _fmt_cell(system.get("llm_assisted_accuracy", system.get("accuracy", ""))),
        _fmt_cell(system.get("coverage", "")),
        _fmt_cell(system.get("evidence_recall", "")),
        _fmt_cell(system.get("avg_latency", "")),
    ]


def _fmt_cell(value: Any) -> str:
    if value is None:
        return "--"
    if isinstance(value, float):
        return f"{value:.2f}" if abs(value) < 100 else f"{value:.1f}"
    if isinstance(value, int):
        return str(value)
    text = str(value)
    return text if text else "--"


def _latex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in str(text))
