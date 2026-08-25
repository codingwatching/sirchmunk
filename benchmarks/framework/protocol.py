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
    sampling: Dict[str, Any] = field(default_factory=dict)
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
    """Protocol validator with frozen-evaluation publication gates."""

    REQUIRED_KEYS = {"run_id", "benchmark", "systems", "metrics", "seeds"}
    FROZEN_CACHE_MODES = {"cold", "compiled"}

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

        config = protocol.get("config", {}) if isinstance(protocol.get("config", {}), dict) else {}
        cache_policy = protocol.get("cache_policy", {}) if isinstance(protocol.get("cache_policy", {}), dict) else {}
        stage = str(protocol.get("stage") or config.get("stage") or "").lower()
        if stage and stage not in {"exploration", "frozen"}:
            errors.append(f"invalid stage: {stage}")
        if stage == "frozen":
            errors.extend(cls._validate_frozen(protocol, config, cache_policy))
        return not errors, errors

    @classmethod
    def _validate_frozen(
        cls,
        protocol: Dict[str, Any],
        config: Dict[str, Any],
        cache_policy: Dict[str, Any],
    ) -> List[str]:
        errors: List[str] = []
        if int(protocol.get("protocol_schema_version", 0) or 0) < 2:
            errors.append("frozen protocol requires protocol_schema_version >= 2")
        if not protocol.get("systems"):
            errors.append("frozen protocol must declare at least one fixed system")
        if not protocol.get("seeds"):
            errors.append("frozen protocol must declare fixed seeds")
        if not (protocol.get("sample_id_checksum") or config.get("sample_id_checksum")):
            errors.append("frozen protocol must record sample_id_checksum")

        cache_mode = str(
            cache_policy.get("mode")
            or config.get("cache_mode")
            or config.get("CACHE_MODE")
            or ""
        ).lower()
        if cache_mode not in cls.FROZEN_CACHE_MODES:
            errors.append("frozen protocol cache_policy.mode must be one of: cold, compiled")
        if _as_bool(cache_policy.get("dry_run") or config.get("cache_dry_run") or config.get("CACHE_DRY_RUN")):
            errors.append("frozen protocol cannot use cache dry-run mode")
        if _as_bool(config.get("enable_eval_feedback") or config.get("HOTPOT_ENABLE_EVAL_FEEDBACK")):
            errors.append("frozen protocol must disable eval feedback")

        memory_enabled = _as_bool(config.get("enable_memory") or config.get("SIRCHMUNK_ENABLE_MEMORY"))
        if memory_enabled:
            memory_state = (
                config.get("memory_state_version")
                or config.get("fixed_memory_state_version")
                or config.get("SIRCHMUNK_MEMORY_STATE_VERSION")
            )
            if not _has_value(memory_state):
                errors.append("frozen protocol with memory enabled must record a fixed memory_state_version")
            if not _as_bool(config.get("memory_read_only") or config.get("frozen_memory_read_only") or config.get("SIRCHMUNK_MEMORY_READ_ONLY")):
                errors.append("frozen protocol with memory enabled must mark memory as read-only")
            if _as_bool(config.get("enable_memory_updates") or config.get("memory_write_enabled") or config.get("SIRCHMUNK_ENABLE_MEMORY_UPDATES")):
                errors.append("frozen protocol cannot enable memory updates")

        llm_judge_enabled = _as_bool(config.get("enable_llm_judge") or config.get("HOTPOT_ENABLE_LLM_JUDGE"))
        llm_judge_allowed = _as_bool(
            config.get("allow_frozen_llm_judge_auxiliary")
            or config.get("ALLOW_FROZEN_LLM_JUDGE_AUXILIARY")
        )
        if llm_judge_enabled and not llm_judge_allowed:
            errors.append("frozen protocol cannot enable LLM judge unless it is explicitly auxiliary")
        return errors


def _as_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if value is None:
        return default
    text = str(value).strip().lower()
    if text in ("1", "true", "yes", "y", "on"):
        return True
    if text in ("0", "false", "no", "n", "off", ""):
        return False
    return default


def _has_value(value: Any) -> bool:
    return value is not None and str(value).strip() != ""


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
        sampling=config.get("sampling", {}) if isinstance(config.get("sampling", {}), dict) else {},
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
