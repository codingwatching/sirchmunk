"""Stage helpers for v4 G_n/D_n dynamic evaluation artifacts."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List


@dataclass
class V4StageBinding:
    """One strict G_n/D_n binding used by v4 evaluation."""

    stage_name: str
    g_stage: str
    d_stage: str
    sample_ids_file: str
    corpus_snapshot_dir: str
    search_corpus_dir: str
    sample_id_checksum: str
    frozen_order_checksum: str
    corpus_checksum: str
    work_path: str = ""
    output_dir: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def fingerprint(self) -> Dict[str, str]:
        return {
            "sample_id_checksum": self.sample_id_checksum,
            "frozen_order_checksum": self.frozen_order_checksum,
            "corpus_checksum": self.corpus_checksum,
        }


@dataclass
class StageExecutionRecord:
    """A machine-readable execution record for one system on one v4 stage."""

    stage_name: str
    system_name: str
    sample_id_checksum: str
    frozen_order_checksum: str
    corpus_checksum: str
    system_config_hash: str = ""
    baseline_version: str = ""
    cache_mode: str = ""
    results_path: str = ""
    output_dir: str = ""
    work_path: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def reuse_fingerprint(self) -> Dict[str, str]:
        return {
            "sample_id_checksum": self.sample_id_checksum,
            "frozen_order_checksum": self.frozen_order_checksum,
            "corpus_checksum": self.corpus_checksum,
            "system_config_hash": self.system_config_hash,
            "baseline_version": self.baseline_version,
            "cache_mode": self.cache_mode,
        }


def build_stage_bindings(
    *,
    nested_sample_manifest: Dict[str, Any],
    corpus_manifests: Iterable[Dict[str, Any]],
    base_work_path: str | Path,
    base_output_dir: str | Path,
) -> List[V4StageBinding]:
    sample_by_n = {
        int(stage.get("stage_n", 0)): stage
        for stage in nested_sample_manifest.get("stages", [])
        if isinstance(stage, dict)
    }
    bindings: List[V4StageBinding] = []
    for corpus in corpus_manifests:
        stage_name = str(corpus.get("stage_name") or "")
        sample_count = int(corpus.get("sample_count") or 0)
        sample_stage = sample_by_n.get(sample_count)
        if not stage_name or not sample_stage:
            continue
        g_stage = str(sample_stage.get("stage_name") or f"G_{sample_count}")
        d_stage = stage_name if stage_name.startswith("D_") else f"D_{sample_count}"
        binding_name = f"{g_stage}_{d_stage}"
        bindings.append(V4StageBinding(
            stage_name=binding_name,
            g_stage=g_stage,
            d_stage=d_stage,
            sample_ids_file=str(sample_stage.get("sample_ids_file", "")),
            corpus_snapshot_dir=str(corpus.get("snapshot_dir", "")),
            search_corpus_dir=str(corpus.get("search_corpus_dir") or Path(str(corpus.get("snapshot_dir", ""))) / "documents"),
            sample_id_checksum=str(corpus.get("sample_id_checksum", "")),
            frozen_order_checksum=str(corpus.get("frozen_order_checksum", "")),
            corpus_checksum=str(corpus.get("corpus_checksum", "")),
            work_path=str(Path(base_work_path) / "evaluation" / "v4" / d_stage),
            output_dir=str(Path(base_output_dir) / "dynamic_eval_v4" / "runs" / binding_name),
            metadata={"corpus_manifest": corpus, "sample_stage": sample_stage},
        ))
    return bindings


def validate_result_reuse(expected: Any, record: Dict[str, Any]) -> Dict[str, Any]:
    """Validate whether a previous result can be reused for this stage.

    ``expected`` may be a StageExecutionRecord (preferred, six-field gate) or a
    V4StageBinding (binding-level three-field gate for artifact-only checks).
    """
    if hasattr(expected, "reuse_fingerprint"):
        required = expected.reuse_fingerprint()
    elif hasattr(expected, "fingerprint"):
        required = expected.fingerprint()
    else:
        required = dict(expected or {})
    observed_source = record.get("reuse_fingerprint") if isinstance(record.get("reuse_fingerprint"), dict) else record
    metadata = record.get("metadata", {}) if isinstance(record.get("metadata"), dict) else {}
    checks = {}
    for key, value in required.items():
        observed = str(observed_source.get(key) or metadata.get(key) or "")
        checks[key] = {"expected": str(value), "observed": observed, "matched": observed == str(value)}
    reusable = all(item["matched"] for item in checks.values())
    return {"reusable": reusable, "checks": checks}


def save_stage_bindings(bindings: Iterable[V4StageBinding], path: str | Path) -> str:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps([binding.to_dict() for binding in bindings], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return str(p)


def load_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


__all__ = [
    "StageExecutionRecord",
    "V4StageBinding",
    "build_stage_bindings",
    "load_json",
    "save_stage_bindings",
    "validate_result_reuse",
]
