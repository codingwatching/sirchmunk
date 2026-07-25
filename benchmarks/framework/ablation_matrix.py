"""Ablation matrix utilities for LENS/Sirchmunk mechanism studies."""
from __future__ import annotations

import hashlib
import itertools
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


@dataclass
class AblationAxis:
    """One experimental axis in an ablation matrix."""

    name: str
    values: List[Any]
    env_key: str = ""
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AblationVariant:
    """One concrete ablation variant with config overrides."""

    variant_id: str
    label: str
    assignments: Dict[str, Any]
    config_overrides: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AblationSpec:
    """A reproducible ablation design."""

    name: str
    axes: List[AblationAxis]
    design: str = "orthogonal"  # orthogonal | cartesian
    baseline: Dict[str, Any] = field(default_factory=dict)
    max_combinations: int = 32
    metadata: Dict[str, Any] = field(default_factory=dict)

    def generate(self) -> List[AblationVariant]:
        if self.design not in {"orthogonal", "cartesian"}:
            raise ValueError("design must be 'orthogonal' or 'cartesian'")
        variants = (
            _generate_orthogonal(self)
            if self.design == "orthogonal"
            else _generate_cartesian(self)
        )
        return variants[: self.max_combinations]

    def save(self, path: str | Path) -> str:
        variants = [v.to_dict() for v in self.generate()]
        payload = {
            "name": self.name,
            "design": self.design,
            "axes": [axis.to_dict() for axis in self.axes],
            "baseline": self.baseline,
            "max_combinations": self.max_combinations,
            "metadata": self.metadata,
            "variants": variants,
        }
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return str(p)


def default_lens_ablation_spec(*, benchmark_prefix: str = "HOTPOT") -> AblationSpec:
    """Return a conservative orthogonal LENS ablation spec.

    The default design varies one mechanism at a time around the declared
    baseline so fullwiki experiments remain tractable.
    """
    p = benchmark_prefix.upper().rstrip("_")
    baseline = {
        f"{p}_MODE": "DEEP",
        f"{p}_REUSE_KNOWLEDGE": "true",
        f"{p}_ENABLE_INTENT_MODULATION": "true",
        f"{p}_POSITION_PRIOR": "fuzz",
        f"{p}_MAX_LOOPS": "10",
    }
    return AblationSpec(
        name="lens_default_ablation",
        design="orthogonal",
        baseline=baseline,
        max_combinations=16,
        axes=[
            AblationAxis("search_mode", ["FAST", "DEEP"], env_key=f"{p}_MODE", description="FAST vs DEEP retrieval mode"),
            AblationAxis("knowledge_reuse", ["false", "true"], env_key=f"{p}_REUSE_KNOWLEDGE", description="Warm-start / cluster reuse"),
            AblationAxis("position_prior", ["uniform", "fuzz", "rga_position"], env_key=f"{p}_POSITION_PRIOR", description="Position-level prior strategy"),
            AblationAxis("intent_modulation", ["false", "true"], env_key=f"{p}_ENABLE_INTENT_MODULATION", description="Intent-aware stopping/prior modulation"),
            AblationAxis("loop_budget", ["1", "3", "10"], env_key=f"{p}_MAX_LOOPS", description="Sequential inference loop budget"),
        ],
        metadata={"purpose": "LENS mechanism ablation; one-axis-at-a-time by default"},
    )


def _generate_orthogonal(spec: AblationSpec) -> List[AblationVariant]:
    variants: List[AblationVariant] = []
    base_assignments = _baseline_assignments(spec)
    variants.append(_variant(spec, "baseline", base_assignments))
    for axis in spec.axes:
        base_value = base_assignments.get(axis.name)
        for value in axis.values:
            if value == base_value:
                continue
            assignments = dict(base_assignments)
            assignments[axis.name] = value
            variants.append(_variant(spec, f"{axis.name}={value}", assignments))
    return variants


def _generate_cartesian(spec: AblationSpec) -> List[AblationVariant]:
    variants: List[AblationVariant] = []
    axes = spec.axes
    for values in itertools.product(*(axis.values for axis in axes)):
        assignments = {axis.name: value for axis, value in zip(axes, values)}
        label = ",".join(f"{k}={v}" for k, v in assignments.items())
        variants.append(_variant(spec, label, assignments))
    return variants


def _baseline_assignments(spec: AblationSpec) -> Dict[str, Any]:
    assignments: Dict[str, Any] = {}
    for axis in spec.axes:
        if axis.env_key and axis.env_key in spec.baseline:
            assignments[axis.name] = spec.baseline[axis.env_key]
        elif axis.values:
            assignments[axis.name] = axis.values[-1]
    return assignments


def _variant(spec: AblationSpec, label: str, assignments: Dict[str, Any]) -> AblationVariant:
    overrides = dict(spec.baseline)
    axis_by_name = {axis.name: axis for axis in spec.axes}
    for name, value in assignments.items():
        axis = axis_by_name[name]
        if axis.env_key:
            overrides[axis.env_key] = value
    variant_id = _stable_id(spec.name, assignments)
    return AblationVariant(
        variant_id=variant_id,
        label=label,
        assignments=assignments,
        config_overrides=overrides,
        metadata={"spec": spec.name, "design": spec.design},
    )


def _stable_id(prefix: str, assignments: Dict[str, Any]) -> str:
    raw = json.dumps(assignments, sort_keys=True, ensure_ascii=False)
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:10]
    safe_prefix = "".join(ch if ch.isalnum() else "_" for ch in prefix.lower()).strip("_")
    return f"{safe_prefix}_{digest}"


__all__ = [
    "AblationAxis",
    "AblationVariant",
    "AblationSpec",
    "default_lens_ablation_spec",
]
