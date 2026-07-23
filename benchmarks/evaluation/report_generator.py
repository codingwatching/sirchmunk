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
        )
        report_tex = self._to_latex(title=title, markdown=report_md)

        md_path = out / "report.md"
        tex_path = out / "report.tex"
        validation_path = out / "validation.json"
        md_path.write_text(report_md, encoding="utf-8")
        tex_path.write_text(report_tex, encoding="utf-8")
        validation_path.write_text(json.dumps(validation.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        return {
            "markdown": str(md_path),
            "latex": str(tex_path),
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
        if metrics:
            lines.append(f"- Accuracy: `{metrics.get('accuracy', 'n/a')}`")
            lines.append(f"- Coverage: `{metrics.get('coverage', 'n/a')}`")
            if "evidence_recall" in metrics:
                lines.append(f"- Evidence recall: `{metrics.get('evidence_recall')}`")
            if "latency" in metrics:
                lines.append(f"- Latency: `{json.dumps(metrics.get('latency'), ensure_ascii=False)}`")
        lines.append("")

        if table.get("systems"):
            lines.append("## Main Result Table")
            lines.append("")
            headers = ["System", "N", "Accuracy", "Coverage", "Latency", "Setup(s)", "p-value"]
            lines.append("| " + " | ".join(headers) + " |")
            lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
            for system in table["systems"]:
                setup = system.get("setup_metrics") or {}
                row = [
                    str(system.get("system_name", "")),
                    str(system.get("n", "")),
                    str(system.get("accuracy", "")),
                    str(system.get("coverage", "")),
                    str(system.get("avg_latency", "")),
                    f"{float(setup.get('setup_seconds', 0) or 0):.3f}",
                    str(system.get("p_value")),
                ]
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

    def _to_latex(self, *, title: str, markdown: str) -> str:
        escaped = markdown.replace("\\", "\\textbackslash{}")
        escaped = escaped.replace("_", "\\_").replace("%", "\\%")
        return "\n".join([
            "\\section*{" + _latex_escape(title) + "}",
            "\\begin{verbatim}",
            escaped[:50000],
            "\\end{verbatim}",
            "",
        ])


def _read_json(path: Optional[Path]) -> Dict[str, Any]:
    if not path or not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _latex_escape(text: str) -> str:
    return text.replace("_", "\\_").replace("%", "\\%")
