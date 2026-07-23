"""Reproducibility checklist generation."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


class ReproducibilityChecklist:
    def build(self, run_dir: str | Path) -> Dict[str, Any]:
        root = Path(run_dir)
        manifest = _read_json(root / "manifest.json")
        protocol = _read_json(root / "protocol.yaml")
        dataset = _read_json(root / "dataset_manifest.json")
        metrics = _read_json(root / "results" / "metrics.json")
        cache_report = _read_json(root / "cache_report.json")
        return {
            "run_id": manifest.get("run_id") or protocol.get("run_id"),
            "benchmark": manifest.get("benchmark") or protocol.get("benchmark"),
            "git_commit": manifest.get("git_commit"),
            "git_branch": manifest.get("git_branch"),
            "git_dirty": manifest.get("git_dirty"),
            "git_diff_hash": manifest.get("git_diff_hash"),
            "python": manifest.get("python"),
            "platform": manifest.get("platform"),
            "systems": protocol.get("systems", []),
            "seeds": protocol.get("seeds", []),
            "cache_policy": protocol.get("cache_policy", {}),
            "cache_report": cache_report,
            "dataset_manifest": dataset or manifest.get("dataset_manifest", {}),
            "config_hash": manifest.get("config_hash"),
            "config": manifest.get("config", {}),
            "n": metrics.get("n"),
        }

    def to_markdown(self, run_dir: str | Path) -> str:
        data = self.build(run_dir)
        lines = ["## Reproducibility Checklist", ""]
        for key in (
            "run_id", "benchmark", "git_commit", "git_branch", "git_dirty",
            "git_diff_hash", "python", "platform", "config_hash", "n",
        ):
            lines.append(f"- `{key}`: `{data.get(key)}`")
        lines.append(f"- `systems`: `{', '.join(map(str, data.get('systems', [])))}`")
        lines.append(f"- `seeds`: `{data.get('seeds', [])}`")
        lines.append(f"- `cache_policy`: `{json.dumps(data.get('cache_policy', {}), ensure_ascii=False)}`")
        lines.append(f"- `cache_report`: `{json.dumps(data.get('cache_report', {}), ensure_ascii=False)}`")
        dataset = data.get("dataset_manifest", {}) or {}
        if dataset:
            lines.append("- `dataset_manifest`:")
            for key, value in dataset.items():
                if key.endswith("path") or key.endswith("dir") or key in ("corpus_paths", "wiki_dir", "dataset_dir"):
                    lines.append(f"  - `{key}`: `{value}`")
                elif "checksum" in key or key.endswith("bytes") or key.endswith("count"):
                    lines.append(f"  - `{key}`: `{value}`")
        return "\n".join(lines) + "\n"


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
