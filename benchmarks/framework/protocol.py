"""Experiment protocol utilities for ResearchOps P0."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List


@dataclass
class ExperimentProtocol:
    """Machine-readable experiment declaration.

    JSON is emitted into a .yaml file because JSON is a strict subset of YAML,
    avoiding a hard dependency on PyYAML for P0.
    """

    run_id: str
    benchmark: str
    suite: List[str] = field(default_factory=list)
    systems: List[str] = field(default_factory=lambda: ["sirchmunk"])
    metrics: Dict[str, List[str]] = field(default_factory=dict)
    seeds: List[int] = field(default_factory=list)
    cache_policy: Dict[str, Any] = field(default_factory=dict)
    report: Dict[str, Any] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ProtocolLoader:
    """Load protocol files in JSON/YAML-subset format."""

    @staticmethod
    def load(path: str | Path) -> Dict[str, Any]:
        text = Path(path).read_text(encoding="utf-8")
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return _parse_simple_yaml(text)


class ProtocolValidator:
    """Minimal P0 protocol validator."""

    REQUIRED_KEYS = {"run_id", "benchmark", "systems", "metrics", "seeds"}

    @classmethod
    def validate(cls, protocol: Dict[str, Any]) -> tuple[bool, List[str]]:
        errors: List[str] = []
        missing = sorted(cls.REQUIRED_KEYS - set(protocol))
        if missing:
            errors.append(f"missing required keys: {missing}")
        if not isinstance(protocol.get("systems", []), list):
            errors.append("systems must be a list")
        if not isinstance(protocol.get("metrics", {}), dict):
            errors.append("metrics must be a mapping")
        if not isinstance(protocol.get("seeds", []), list):
            errors.append("seeds must be a list")
        return not errors, errors


def default_protocol(
    *,
    run_id: str,
    benchmark: str,
    config: Dict[str, Any],
    seed: int,
) -> ExperimentProtocol:
    return ExperimentProtocol(
        run_id=run_id,
        benchmark=benchmark,
        suite=[benchmark],
        systems=["sirchmunk"],
        metrics={
            "answer_quality": ["accuracy", "coverage", "em", "f1"],
            "retrieval": ["evidence_recall", "supporting_fact_hit_rate"],
            "efficiency": ["avg_latency", "latency_p50", "latency_p95", "tokens"],
            "researchops": ["time_to_first_query", "storage_overhead", "freshness_accuracy"],
        },
        seeds=[seed],
        cache_policy={
            "mode": config.get("cache_mode", "declared_by_adapter"),
            "reuse_knowledge": config.get("reuse_knowledge", False),
        },
        report={"format": ["markdown", "latex"]},
        config=config,
    )


def protocol_to_text(protocol: Dict[str, Any]) -> str:
    return json.dumps(protocol, indent=2, ensure_ascii=False) + "\n"


def _parse_simple_yaml(text: str) -> Dict[str, Any]:
    """Very small YAML subset parser used only as a fallback for flat files."""
    out: Dict[str, Any] = {}
    current_key: str | None = None
    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if not line.startswith(" ") and ":" in line:
            key, _, value = line.partition(":")
            key = key.strip()
            value = value.strip()
            current_key = key
            if not value:
                out[key] = []
            elif value.startswith("["):
                try:
                    out[key] = json.loads(value)
                except json.JSONDecodeError:
                    out[key] = [v.strip() for v in value.strip("[]").split(",") if v.strip()]
            else:
                out[key] = value
        elif current_key and line.strip().startswith("-"):
            out.setdefault(current_key, [])
            if isinstance(out[current_key], list):
                out[current_key].append(line.strip()[1:].strip())
    return out
