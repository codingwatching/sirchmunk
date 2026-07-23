"""Dependency-light SVG figure generation for ResearchOps reports."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


class FigureGenerator:
    def generate_from_table(self, table_json: str | Path, output_dir: str | Path) -> Dict[str, str]:
        table = _read_json(Path(table_json))
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        systems = table.get("systems", []) if isinstance(table, dict) else []
        paths: Dict[str, str] = {}
        if systems:
            paths["accuracy_latency"] = str(out / "accuracy_latency.svg")
            (out / "accuracy_latency.svg").write_text(_scatter_svg(systems), encoding="utf-8")
            paths["setup_cost"] = str(out / "setup_cost.svg")
            (out / "setup_cost.svg").write_text(_bar_svg(systems, "setup"), encoding="utf-8")
            paths["storage_overhead"] = str(out / "storage_overhead.svg")
            (out / "storage_overhead.svg").write_text(_bar_svg(systems, "storage"), encoding="utf-8")
        return paths


def _scatter_svg(systems: List[Dict[str, Any]]) -> str:
    width, height = 720, 420
    margin = 60
    max_latency = max([float(s.get("avg_latency", 0) or 0) for s in systems] + [1.0])
    max_acc = max([float(s.get("accuracy", 0) or 0) for s in systems] + [100.0])
    points = []
    for idx, s in enumerate(systems):
        x = margin + (float(s.get("avg_latency", 0) or 0) / max_latency) * (width - 2 * margin)
        y = height - margin - (float(s.get("accuracy", 0) or 0) / max_acc) * (height - 2 * margin)
        color = "#2563eb" if s.get("is_ours") else "#64748b"
        label = _escape(str(s.get("system_name", "system")))[:32]
        points.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6" fill="{color}" />')
        points.append(f'<text x="{x + 8:.1f}" y="{y - 8:.1f}" font-size="11">{label}</text>')
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white"/>
<line x1="{margin}" y1="{height-margin}" x2="{width-margin}" y2="{height-margin}" stroke="#334155"/>
<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height-margin}" stroke="#334155"/>
<text x="{width/2}" y="{height-15}" text-anchor="middle" font-size="13">Latency (s)</text>
<text x="18" y="{height/2}" transform="rotate(-90 18,{height/2})" text-anchor="middle" font-size="13">Accuracy (%)</text>
<text x="{width/2}" y="28" text-anchor="middle" font-size="16" font-weight="bold">Accuracy-Latency Trade-off</text>
{''.join(points)}
</svg>'''


def _bar_svg(systems: List[Dict[str, Any]], metric: str) -> str:
    width, height = 760, 420
    margin = 70
    values = []
    for s in systems:
        setup = s.get("setup_metrics", {}) or {}
        if metric == "setup":
            value = float(setup.get("setup_seconds", 0) or 0)
            title = "Setup Cost"
            ylabel = "Setup seconds"
        else:
            value = float(setup.get("storage_bytes", 0) or 0) / (1024 * 1024)
            title = "Storage Overhead"
            ylabel = "Storage MB"
        values.append((str(s.get("system_name", "system")), value, bool(s.get("is_ours"))))
    max_value = max([v for _, v, _ in values] + [1.0])
    bar_width = max(22, (width - 2 * margin) / max(len(values), 1) * 0.55)
    gap = (width - 2 * margin) / max(len(values), 1)
    bars = []
    for idx, (name, value, is_ours) in enumerate(values):
        x = margin + idx * gap + (gap - bar_width) / 2
        h = (value / max_value) * (height - 2 * margin)
        y = height - margin - h
        color = "#2563eb" if is_ours else "#64748b"
        bars.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" height="{h:.1f}" fill="{color}" />')
        bars.append(f'<text x="{x + bar_width/2:.1f}" y="{height-margin+14}" text-anchor="middle" font-size="9" transform="rotate(20 {x + bar_width/2:.1f},{height-margin+14})">{_escape(name)[:18]}</text>')
        bars.append(f'<text x="{x + bar_width/2:.1f}" y="{y-4:.1f}" text-anchor="middle" font-size="10">{value:.3g}</text>')
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white"/>
<line x1="{margin}" y1="{height-margin}" x2="{width-margin}" y2="{height-margin}" stroke="#334155"/>
<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height-margin}" stroke="#334155"/>
<text x="{width/2}" y="28" text-anchor="middle" font-size="16" font-weight="bold">{title}</text>
<text x="18" y="{height/2}" transform="rotate(-90 18,{height/2})" text-anchor="middle" font-size="13">{ylabel}</text>
{''.join(bars)}
</svg>'''


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
