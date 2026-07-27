"""Unified control-layer phase and output-path contracts for ResearchOps P0.

This module is intentionally additive: it does not modify any existing
``run_*.py`` entry point or core algorithm code. It only defines the shared
vocabulary (``ControlBlock`` / phase names) and the canonical output-directory
layout that a future total-control entry point (``run_benchmark.py``) and its
sibling scripts (``run_baseline_assets.py`` / ``run_paper_experiment.py``)
will read and write against.

Design goals:
  1. Single source of truth for block/stage naming across future scripts.
  2. Deterministic, benchmark-scoped output directory layout.
  3. No dependency on any specific benchmark or baseline implementation.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict


class ControlBlock(str, Enum):
    """Top-level control-layer blocks (``run_benchmark.py <block>``)."""

    ASSETS = "assets"
    SMOKE_TUNE = "smoke-tune"
    MAIN = "main"
    ABLATION = "ablation"
    REPORT = "report"
    STATUS = "status"


class ExperimentStage(str, Enum):
    """Experiment stage semantics shared with ``framework.protocol``.

    This mirrors the existing ``exploration`` / ``frozen`` vocabulary already
    enforced by ``ProtocolValidator`` so the control layer does not introduce a
    second, inconsistent stage taxonomy.
    """

    EXPLORATION = "exploration"
    FROZEN = "frozen"


_ALLOWED_STAGES_BY_BLOCK: Dict[ControlBlock, tuple[ExperimentStage, ...]] = {
    ControlBlock.ASSETS: (ExperimentStage.EXPLORATION, ExperimentStage.FROZEN),
    ControlBlock.SMOKE_TUNE: (ExperimentStage.EXPLORATION,),
    ControlBlock.MAIN: (ExperimentStage.FROZEN,),
    ControlBlock.ABLATION: (ExperimentStage.FROZEN,),
    ControlBlock.REPORT: (ExperimentStage.EXPLORATION, ExperimentStage.FROZEN),
    ControlBlock.STATUS: (ExperimentStage.EXPLORATION, ExperimentStage.FROZEN),
}
"""Which stages each control block is allowed to declare.

``main`` and ``ablation`` are paper-claim blocks and therefore restricted to
``frozen`` only; ``smoke-tune`` is exploration-only by design so its artifacts
can never be mistaken for a frozen result.
"""


def allowed_stages(block: ControlBlock | str) -> tuple[ExperimentStage, ...]:
    """Return the stages a given control block is allowed to run under."""
    resolved = block if isinstance(block, ControlBlock) else ControlBlock(str(block))
    return _ALLOWED_STAGES_BY_BLOCK[resolved]


def validate_block_stage(block: ControlBlock | str, stage: ExperimentStage | str) -> None:
    """Raise ``ValueError`` if ``stage`` is not permitted for ``block``.

    This is a pure validation helper; it does not read or write anything.
    """
    resolved_block = block if isinstance(block, ControlBlock) else ControlBlock(str(block))
    resolved_stage = stage if isinstance(stage, ExperimentStage) else ExperimentStage(str(stage))
    allowed = allowed_stages(resolved_block)
    if resolved_stage not in allowed:
        allowed_names = ", ".join(s.value for s in allowed)
        raise ValueError(
            f"control block '{resolved_block.value}' does not allow stage="
            f"'{resolved_stage.value}'; allowed stages: {allowed_names}"
        )


@dataclass
class ControlOutputLayout:
    """Canonical output-directory layout for one benchmark's control-layer runs.

    All paths are derived from a single ``base_dir`` (typically
    ``benchmarks/{benchmark}/output``) so every future script agrees on where
    assets, exploration, main, ablation, scaling, and queue artifacts live.
    """

    base_dir: Path

    def __post_init__(self) -> None:
        self.base_dir = Path(self.base_dir).resolve()

    # -- top-level block directories -----------------------------------
    @property
    def assets_dir(self) -> Path:
        return self.base_dir / "assets"

    @property
    def exploration_dir(self) -> Path:
        return self.base_dir / "exploration"

    @property
    def main_dir(self) -> Path:
        return self.base_dir / "main"

    @property
    def ablation_dir(self) -> Path:
        return self.base_dir / "ablation"

    @property
    def scaling_dir(self) -> Path:
        return self.base_dir / "scaling"

    @property
    def queue_dir(self) -> Path:
        return self.base_dir / "queue"

    # -- assets sub-paths -------------------------------------------------
    @property
    def asset_registry_path(self) -> Path:
        return self.assets_dir / "asset_registry.jsonl"

    @property
    def asset_lifecycle_dir(self) -> Path:
        return self.assets_dir / "lifecycle"

    # -- exploration sub-paths --------------------------------------------
    @property
    def exploration_runs_dir(self) -> Path:
        return self.exploration_dir / "runs"

    @property
    def exploration_candidates_dir(self) -> Path:
        return self.exploration_dir / "candidates"

    @property
    def exploration_reports_dir(self) -> Path:
        return self.exploration_dir / "reports"

    # -- main sub-paths -----------------------------------------------------
    @property
    def main_sampling_dir(self) -> Path:
        return self.main_dir / "sampling"

    @property
    def main_runs_dir(self) -> Path:
        return self.main_dir / "runs"

    @property
    def main_evaluation_dir(self) -> Path:
        return self.main_dir / "evaluation"

    @property
    def main_report_dir(self) -> Path:
        return self.main_dir / "report"

    @property
    def main_summary_path(self) -> Path:
        return self.main_dir / "main_summary.json"

    # -- ablation sub-paths ---------------------------------------------
    @property
    def ablation_spec_path(self) -> Path:
        return self.ablation_dir / "ablation_spec.json"

    @property
    def ablation_variants_path(self) -> Path:
        return self.ablation_dir / "variants.json"

    @property
    def ablation_runs_dir(self) -> Path:
        return self.ablation_dir / "runs"

    # -- queue sub-paths --------------------------------------------------
    @property
    def experiment_queue_path(self) -> Path:
        return self.queue_dir / "experiment_queue.json"

    @property
    def experiment_registry_path(self) -> Path:
        return self.queue_dir / "experiment_registry.jsonl"

    def ensure(self, *, blocks: tuple[ControlBlock, ...] = ()) -> None:
        """Create the base directory and, optionally, specific block dirs.

        This never deletes or mutates existing content; it only calls
        ``mkdir(parents=True, exist_ok=True)`` so it is safe to call
        repeatedly and does not risk clobbering existing artifacts.
        """
        self.base_dir.mkdir(parents=True, exist_ok=True)
        block_dirs = {
            ControlBlock.ASSETS: self.assets_dir,
            ControlBlock.SMOKE_TUNE: self.exploration_dir,
            ControlBlock.MAIN: self.main_dir,
            ControlBlock.ABLATION: self.ablation_dir,
        }
        for block in blocks:
            resolved = block if isinstance(block, ControlBlock) else ControlBlock(str(block))
            target = block_dirs.get(resolved)
            if target is not None:
                target.mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> Dict[str, str]:
        """Return every canonical path as a plain string mapping.

        Useful for embedding into ``run_summary.json`` or debug logging
        without importing this module elsewhere.
        """
        return {
            "base_dir": str(self.base_dir),
            "assets_dir": str(self.assets_dir),
            "exploration_dir": str(self.exploration_dir),
            "main_dir": str(self.main_dir),
            "ablation_dir": str(self.ablation_dir),
            "scaling_dir": str(self.scaling_dir),
            "queue_dir": str(self.queue_dir),
            "asset_registry_path": str(self.asset_registry_path),
            "asset_lifecycle_dir": str(self.asset_lifecycle_dir),
            "exploration_runs_dir": str(self.exploration_runs_dir),
            "exploration_candidates_dir": str(self.exploration_candidates_dir),
            "exploration_reports_dir": str(self.exploration_reports_dir),
            "main_sampling_dir": str(self.main_sampling_dir),
            "main_runs_dir": str(self.main_runs_dir),
            "main_evaluation_dir": str(self.main_evaluation_dir),
            "main_report_dir": str(self.main_report_dir),
            "main_summary_path": str(self.main_summary_path),
            "ablation_spec_path": str(self.ablation_spec_path),
            "ablation_variants_path": str(self.ablation_variants_path),
            "ablation_runs_dir": str(self.ablation_runs_dir),
            "experiment_queue_path": str(self.experiment_queue_path),
            "experiment_registry_path": str(self.experiment_registry_path),
        }


def for_benchmark_output_dir(output_dir: str | Path) -> ControlOutputLayout:
    """Build a :class:`ControlOutputLayout` from a benchmark's output dir.

    ``output_dir`` is typically the value returned by
    ``BenchmarkAdapter.get_output_dir()`` (e.g. ``benchmarks/hotpotqa/output``).
    """
    return ControlOutputLayout(base_dir=Path(output_dir))


__all__ = [
    "ControlBlock",
    "ExperimentStage",
    "allowed_stages",
    "validate_block_stage",
    "ControlOutputLayout",
    "for_benchmark_output_dir",
]
