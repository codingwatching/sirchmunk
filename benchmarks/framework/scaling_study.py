"""Scaling study orchestration for full-corpus feasibility experiments."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

from baselines.base_adapter import BaselineAdapter
from framework.baseline_lifecycle import BaselineLifecycleManager
from framework.lifecycle_schema import BaselineLifecycleRecord, BaselinePhase, ResourceBudget
from framework.metric_engine import lifecycle_cost_curve, scaling_efficiency
from framework.time_utils import local_timestamp


@dataclass
class CorpusScaleSpec:
    """One corpus scale point in a scaling study."""

    name: str
    max_docs: int = 0
    corpus_dir: str = ""
    manifest: Dict[str, Any] = field(default_factory=dict)
    materialized: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ScalingStudyResult:
    run_id: str
    benchmark: str
    records: List[BaselineLifecycleRecord]
    scaling_metrics: Dict[str, Any]
    cost_curves: List[Dict[str, Any]]
    output_dir: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "benchmark": self.benchmark,
            "records": [r.to_dict() for r in self.records],
            "scaling_metrics": self.scaling_metrics,
            "cost_curves": self.cost_curves,
            "output_dir": self.output_dir,
        }


class ScalingStudyManager:
    """Run baseline lifecycle evaluation over multiple corpus scales."""

    def __init__(
        self,
        bm_adapter: Any,
        output_dir: str | Path,
        *,
        resource_budget: Optional[ResourceBudget] = None,
        q_values: Iterable[int] = (1, 10, 100, 1000),
    ) -> None:
        self.bm_adapter = bm_adapter
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.resource_budget = resource_budget or ResourceBudget()
        self.q_values = tuple(q_values)

    async def run(
        self,
        *,
        baseline_factories: Iterable[Callable[[], BaselineAdapter]],
        scales: Iterable[CorpusScaleSpec],
        golden_set: Any = None,
        run_id: str = "",
        include_sirchmunk: bool = True,
    ) -> ScalingStudyResult:
        run_id = run_id or f"scaling_{self.bm_adapter.name}_{_timestamp()}"
        records: List[BaselineLifecycleRecord] = []
        cost_curves: List[Dict[str, Any]] = []

        for scale in scales:
            scale_dir = self.output_dir / scale.name
            scale_dir.mkdir(parents=True, exist_ok=True)
            scale_adapter = _CorpusOverrideAdapter(self.bm_adapter, scale)
            corpus_manifest = scale_adapter.get_dataset_manifest()
            manager = BaselineLifecycleManager(scale_dir, resource_budget=self.resource_budget)

            if include_sirchmunk:
                sirchmunk_record = _sirchmunk_no_index_record(
                    run_id=run_id,
                    benchmark=self.bm_adapter.name,
                    scale=scale,
                    corpus_manifest=corpus_manifest,
                )
                manager.save_record(sirchmunk_record)
                records.append(sirchmunk_record)
                cost_curves.append(lifecycle_cost_curve(sirchmunk_record, q_values=self.q_values))

            for factory in baseline_factories:
                baseline = factory()
                record = await manager.run_build(
                    baseline,
                    run_id=run_id,
                    benchmark=self.bm_adapter.name,
                    corpus_manifest=corpus_manifest,
                    golden_set=golden_set,
                    bm_adapter=scale_adapter,
                    corpus_scale=scale.name,
                )
                records.append(record)
                cost_curves.append(lifecycle_cost_curve(record, q_values=self.q_values))

        scaling_metrics = scaling_efficiency(records)
        result = ScalingStudyResult(
            run_id=run_id,
            benchmark=self.bm_adapter.name,
            records=records,
            scaling_metrics=scaling_metrics,
            cost_curves=cost_curves,
            output_dir=str(self.output_dir),
        )
        self.save_result(result)
        return result

    def save_result(self, result: ScalingStudyResult) -> str:
        summary_path = self.output_dir / "scaling_study_summary.json"
        summary_path.write_text(
            json.dumps(result.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        records_path = self.output_dir / "scaling_lifecycle_records.jsonl"
        with records_path.open("w", encoding="utf-8") as fp:
            for record in result.records:
                fp.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
        cost_path = self.output_dir / "amortized_cost_curves.json"
        cost_path.write_text(json.dumps(result.cost_curves, indent=2, ensure_ascii=False), encoding="utf-8")
        metrics_path = self.output_dir / "scaling_metrics.json"
        metrics_path.write_text(json.dumps(result.scaling_metrics, indent=2, ensure_ascii=False), encoding="utf-8")
        return str(summary_path)


class _CorpusOverrideAdapter:
    """Proxy adapter that overrides corpus path/manifest for a scale point."""

    def __init__(self, base_adapter: Any, scale: CorpusScaleSpec) -> None:
        self._base = base_adapter
        self._scale = scale

    def __getattr__(self, name: str) -> Any:
        return getattr(self._base, name)

    @property
    def name(self) -> str:
        return getattr(self._base, "name", "benchmark")

    def get_search_paths(self, sample: Any) -> List[str]:
        if self._scale.corpus_dir:
            return [self._scale.corpus_dir]
        return self._base.get_search_paths(sample)

    def validate_corpus(self):
        if self._scale.corpus_dir:
            path = Path(self._scale.corpus_dir)
            return (1, []) if path.exists() else (0, [str(path)])
        return self._base.validate_corpus()

    def get_dataset_manifest(self) -> Dict[str, Any]:
        base_manifest = {}
        try:
            base_manifest = dict(self._base.get_dataset_manifest())
        except Exception:
            base_manifest = {}
        manifest = {**base_manifest, **dict(self._scale.manifest or {})}
        if self._scale.corpus_dir:
            manifest["wiki_dir"] = self._scale.corpus_dir
        manifest["corpus_scale"] = self._scale.name
        manifest["max_docs"] = self._scale.max_docs
        manifest["doc_count"] = int(
            manifest.get("selected_documents")
            or manifest.get("doc_count")
            or manifest.get("total_documents")
            or self._scale.max_docs
            or 0
        )
        manifest.setdefault("corpus_id", f"{self.name}_{self._scale.name}")
        return manifest


def _sirchmunk_no_index_record(
    *,
    run_id: str,
    benchmark: str,
    scale: CorpusScaleSpec,
    corpus_manifest: Dict[str, Any],
) -> BaselineLifecycleRecord:
    return BaselineLifecycleRecord(
        run_id=run_id,
        benchmark=benchmark,
        baseline_name="sirchmunk",
        citation_name="Sirchmunk / LENS (no index required)",
        corpus_id=str(corpus_manifest.get("corpus_id") or ""),
        corpus_scale=scale.name,
        corpus_size_docs=int(corpus_manifest.get("doc_count") or scale.max_docs or 0),
        index_required=False,
        phase=BaselinePhase.READY,
        build_completed=True,
        index_ready=True,
        query_eligible=True,
        build_time_seconds=0.0,
        preprocessing_seconds=0.0,
        index_build_seconds=0.0,
        disk_bytes=0,
        preprocess_llm_tokens=0,
        metadata={"index_required": False, "scale": scale.to_dict()},
    )


def _timestamp() -> str:
    return local_timestamp()


__all__ = ["CorpusScaleSpec", "ScalingStudyManager", "ScalingStudyResult"]
