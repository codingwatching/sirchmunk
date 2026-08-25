"""External graph-RAG style baseline adapters.

For reproducible papers, LightRAG/GraphRAG are often run outside this process
because they require separate services, graph stores, or long preprocessing.
These adapters make such systems first-class baselines by importing their
per-sample predictions while still accounting for setup metrics.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base_adapter import BaselineAdapter, BaselinePrediction, BaselineSetupResult


class ExternalPredictionBaseline(BaselineAdapter):
    def __init__(
        self,
        *,
        name: str,
        citation_name: str,
        predictions_path: str = "",
        setup_metrics_path: str = "",
    ) -> None:
        self._name = name
        self._citation = citation_name
        self._predictions_path = predictions_path
        self._setup_metrics_path = setup_metrics_path
        self._predictions: Dict[str, Dict[str, Any]] = {}
        self._setup = BaselineSetupResult()

    @property
    def name(self) -> str:
        return self._name

    @property
    def citation_name(self) -> str:
        return self._citation

    def is_available(self) -> bool:
        return bool(self._predictions_path and Path(self._predictions_path).exists())

    async def prepare(self, golden_set: Any = None, bm_adapter: Any = None) -> BaselineSetupResult:
        self._predictions = _load_predictions(self._predictions_path)
        self._setup = _load_setup_metrics(self._setup_metrics_path)
        self._setup.metadata.update({
            "predictions_path": self._predictions_path,
            "setup_metrics_path": self._setup_metrics_path,
            "external_system": self._name,
        })
        return self._setup

    async def predict(self, question: str, context_paths: List[str]) -> BaselinePrediction:
        return BaselinePrediction(
            answer="",
            elapsed=0.0,
            metadata={
                "external_prediction_required": True,
                "imported_baseline": True,
                "import_status": "lookup_required",
                "import_source_path": self._predictions_path,
            },
        )

    def predict_by_id(self, sample_id: str) -> Optional[BaselinePrediction]:
        row = self._predictions.get(str(sample_id))
        if not row:
            return None
        return BaselinePrediction(
            answer=str(row.get("prediction") or row.get("answer") or row.get("raw_prediction") or ""),
            elapsed=float(row.get("elapsed", 0.0) or 0.0),
            tokens_used=int(row.get("tokens_used", 0) or 0),
            metadata={
                "external_system": self._name,
                "imported_baseline": True,
                "import_status": "imported",
                "import_source_path": self._predictions_path,
                "setup_metrics": self.collect_setup_metrics(),
                **(row.get("metadata", {}) if isinstance(row.get("metadata"), dict) else {}),
            },
        )

    def collect_setup_metrics(self) -> Dict[str, Any]:
        return {
            "setup_seconds": self._setup.setup_seconds,
            "preprocessing_seconds": self._setup.preprocessing_seconds,
            "index_build_seconds": self._setup.index_build_seconds,
            "storage_bytes": self._setup.storage_bytes,
            "indexed_documents": self._setup.indexed_documents,
            "metadata": self._setup.metadata,
        }

    def requires_import_coverage(self) -> bool:
        return True

    def extra_metadata(self) -> Dict[str, Any]:
        return {
            "external_system": self._name,
            "imported_baseline": True,
            "import_source_path": self._predictions_path,
            "setup_metrics_path": self._setup_metrics_path,
            "loaded_count": len(self._predictions),
        }


class LightRAGV1Baseline(ExternalPredictionBaseline):
    def __init__(self, predictions_path: str = "", setup_metrics_path: str = "") -> None:
        super().__init__(
            name="lightrag_v1",
            citation_name="LightRAG v1 (imported)",
            predictions_path=predictions_path,
            setup_metrics_path=setup_metrics_path,
        )


class GraphRAGBaseline(ExternalPredictionBaseline):
    def __init__(self, predictions_path: str = "", setup_metrics_path: str = "") -> None:
        super().__init__(
            name="graphrag",
            citation_name="GraphRAG (imported)",
            predictions_path=predictions_path,
            setup_metrics_path=setup_metrics_path,
        )


def _load_predictions(path: str) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    p = Path(path)
    if not p.exists():
        return out
    with p.open(encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            sid = row.get("sample_id") or row.get("hotpot_id") or row.get("id")
            if sid:
                out[str(sid)] = row
    return out


def _load_setup_metrics(path: str) -> BaselineSetupResult:
    p = Path(path) if path else None
    if not p or not p.exists():
        return BaselineSetupResult(metadata={"setup_metrics_missing": True})
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return BaselineSetupResult(metadata={"setup_metrics_parse_error": True, "path": str(p)})
    return BaselineSetupResult(
        setup_seconds=float(data.get("setup_seconds", data.get("total_setup_seconds", 0.0)) or 0.0),
        preprocessing_seconds=float(data.get("preprocessing_seconds", 0.0) or 0.0),
        index_build_seconds=float(data.get("index_build_seconds", 0.0) or 0.0),
        storage_bytes=int(data.get("storage_bytes", 0) or 0),
        indexed_documents=int(data.get("indexed_documents", 0) or 0),
        metadata={k: v for k, v in data.items() if k not in {"setup_seconds", "preprocessing_seconds", "index_build_seconds", "storage_bytes", "indexed_documents"}},
    )
