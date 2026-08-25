"""baselines/sdk_baseline.py — generic SDK wrapper + published-result import

SdkBaseline:
    Wraps any Python SDK or framework; a REST API is not required.
    The competitor call logic is injected through the predict_fn callback, while the
    framework only handles scheduling and timing.

ManualImportAdapter:
    Imports competitor predictions from a precomputed JSONL file.
    Two scenarios:
    a) the competitor provides raw predictions (re-scored with our judge to keep judging
       consistent)
    b) the competitor only has published aggregate numbers (pass metrics_dict to skip
       judging)

Fastest path to onboard a new competitor:
    1. If the competitor ships a Python package, wrap it with SdkBaseline
    2. If only published numbers exist, write them straight into the table via
       PaperTableGenerator.add_published_metrics()
    3. If a raw predictions JSONL exists, load it with ManualImportAdapter and re-judge
"""
from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .base_adapter import BaselineAdapter, BaselinePrediction


class SdkBaseline(BaselineAdapter):
    """Generic wrapper for a Python SDK / framework competitor.

    Usage, with a hypothetical NaiveRAG::

        # 1. Initialize the competitor system (instantiate once)
        from some_rag_package import NaiveRAGSystem
        system = NaiveRAGSystem(
            model="gpt-4o-mini",
            top_k=5,
        )

        # 2. Define predict_fn: accepts (system, question, context_paths) -> str
        def naive_rag_predict(sys, question, paths):
            return sys.retrieve_and_answer(question, document_paths=paths)

        # 3. Wrap it as an SdkBaseline
        baseline = SdkBaseline(
            name="naive_rag_v1",
            citation_name="Naive RAG (Gao et al., 2023)",
            system=system,
            predict_fn=naive_rag_predict,
            is_async=False,      # set True when predict_fn is async
            max_concurrent=2,
            metadata={"model": "gpt-4o-mini", "top_k": 5},
        )

    When the competitor predict is asynchronous::

        async def async_predict(sys, question, paths):
            return await sys.apredict(question, paths)

        baseline = SdkBaseline(..., predict_fn=async_predict, is_async=True)
    """

    def __init__(
        self,
        name: str,
        citation_name: str,
        system: Any,
        predict_fn: Callable,
        is_async: bool = False,
        max_concurrent: int = 1,
        request_delay: float = 0.5,
        tokens_fn: Optional[Callable] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Args:
            name:           internal system ID without spaces, used for file naming.
            citation_name:  display name in paper tables.
            system:         competitor system instance of any type.
            predict_fn:     call signature (system, question: str, paths: List[str]) -> str;
                            an async callable when is_async=True.
            is_async:       whether predict_fn is a coroutine function.
            max_concurrent: maximum concurrent requests.
            request_delay:  delay between requests in seconds.
            tokens_fn:      optional function extracting the token count, with signature
                            (system, question, result) -> int
            metadata:       system metadata written into BaselineResult.metadata.
        """
        self._name = name
        self._citation = citation_name
        self._system = system
        self._predict_fn = predict_fn
        self._is_async = is_async
        self._max_concurrent = max_concurrent
        self._delay = request_delay
        self._tokens_fn = tokens_fn
        self._metadata = metadata or {}

    @property
    def name(self) -> str:
        return self._name

    @property
    def citation_name(self) -> str:
        return self._citation

    async def predict(self, question: str, context_paths: List[str]) -> BaselinePrediction:
        t0 = time.monotonic()
        try:
            if self._is_async:
                answer = await self._predict_fn(self._system, question, context_paths)
            else:
                loop = asyncio.get_event_loop()
                answer = await loop.run_in_executor(
                    None,
                    lambda: self._predict_fn(self._system, question, context_paths)
                )
        except Exception as exc:
            answer = f"[SdkBaseline error: {exc}]"

        elapsed = time.monotonic() - t0

        tokens = 0
        if self._tokens_fn:
            try:
                tokens = int(self._tokens_fn(self._system, question, answer))
            except Exception:
                pass

        return BaselinePrediction(
            answer=str(answer),
            elapsed=elapsed,
            tokens_used=tokens,
            metadata=dict(self._metadata),
        )

    def get_max_concurrent(self) -> int:
        return self._max_concurrent

    def get_request_delay(self) -> float:
        return self._delay

    def extra_metadata(self) -> Dict[str, Any]:
        return dict(self._metadata)


class ManualImportAdapter(BaselineAdapter):
    """Import competitor predictions from a precomputed JSONL and re-score with our judge.

    Use case: the competitor system has no API or SDK but has produced a JSONL file where
    each line holds {"sample_id": "...", "prediction": "...", "elapsed": 3.2}

    This keeps every system scored by the same judge, which the paper fairness
    requirement demands.

    JSONL format, one record per line::

        {"sample_id": "hotpotqa_id_001", "prediction": "Paris", "elapsed": 5.2}

    Note: sample_id must match the sample_id in the GoldenSet, otherwise the sample is
    skipped.
    """

    def __init__(
        self,
        name: str,
        citation_name: str,
        predictions_path: str,
        default_elapsed: float = 0.0,
        setup_metrics_path: str = "",
    ) -> None:
        """
        Args:
            name:              internal system ID.
            citation_name:     display name in paper tables.
            predictions_path:  JSONL path where each line holds sample_id + prediction.
            default_elapsed:   fallback used when the JSONL has no elapsed field.
            setup_metrics_path: optional setup metrics JSON, used to report preprocessing
                                and indexing cost fairly.
        """
        self._name = name
        self._citation = citation_name
        self._default_elapsed = default_elapsed
        self._predictions_path = str(Path(predictions_path).resolve())
        self._setup_metrics_path = str(Path(setup_metrics_path).resolve()) if setup_metrics_path else ""
        self._setup_metrics = _load_setup_metrics(self._setup_metrics_path)
        self._predictions: Dict[str, Dict[str, Any]] = {}
        self._load(predictions_path)

    def _load(self, path: str) -> None:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(
                f"ManualImportAdapter: predictions file not found: {path}"
            )
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                    sid = (
                        row.get("sample_id")
                        or row.get("hotpot_id")
                        or row.get("id")
                        or ""
                    )
                    if sid:
                        self._predictions[str(sid)] = row
                except (json.JSONDecodeError, KeyError):
                    pass

    @property
    def name(self) -> str:
        return self._name

    @property
    def citation_name(self) -> str:
        return self._citation

    async def predict(self, question: str, context_paths: List[str]) -> BaselinePrediction:
        # Cannot be resolved from the question text, so this path is unused
        # BaselineEvaluationSuite drives evaluation through predict_by_id()
        return BaselinePrediction(
            answer="",
            elapsed=self._default_elapsed,
            metadata={
                "import_adapter": True,
                "imported_baseline": True,
                "import_status": "lookup_required",
                "import_source_path": self._predictions_path,
            },
        )

    def predict_by_id(self, sample_id: str) -> Optional[BaselinePrediction]:
        """Look up a preloaded prediction by sample_id.

        Returns:
            A BaselinePrediction, or None when the sample_id is unknown.
        """
        row = self._predictions.get(str(sample_id))
        if row is None:
            return None
        return BaselinePrediction(
            answer=str(row.get("prediction") or row.get("raw_prediction") or ""),
            elapsed=float(row.get("elapsed", self._default_elapsed)),
            tokens_used=int(row.get("tokens_used", 0)),
            metadata={
                "imported_baseline": True,
                "imported_from": "jsonl",
                "import_status": "imported",
                "import_source_path": self._predictions_path,
            },
        )

    @property
    def loaded_count(self) -> int:
        """Number of predictions successfully loaded."""
        return len(self._predictions)

    def requires_import_coverage(self) -> bool:
        return True

    def extra_metadata(self) -> Dict[str, Any]:
        return {
            "import_adapter": True,
            "imported_baseline": True,
            "import_source_path": self._predictions_path,
            "setup_metrics_path": self._setup_metrics_path,
            "loaded_count": self.loaded_count,
        }

    def collect_setup_metrics(self) -> Dict[str, Any]:
        return dict(self._setup_metrics)

    def get_request_delay(self) -> float:
        return 0.0   # Pure in-memory lookup, no latency to report


def _load_setup_metrics(path: str) -> Dict[str, Any]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {"metadata": {"setup_metrics_missing": True, "path": str(p)}}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"metadata": {"setup_metrics_parse_error": True, "path": str(p)}}
    return {
        "setup_seconds": float(data.get("setup_seconds", data.get("total_setup_seconds", 0.0)) or 0.0),
        "preprocessing_seconds": float(data.get("preprocessing_seconds", 0.0) or 0.0),
        "index_build_seconds": float(data.get("index_build_seconds", 0.0) or 0.0),
        "storage_bytes": int(data.get("storage_bytes", 0) or 0),
        "indexed_documents": int(data.get("indexed_documents", 0) or 0),
        "metadata": {
            k: v
            for k, v in data.items()
            if k not in {"setup_seconds", "total_setup_seconds", "preprocessing_seconds", "index_build_seconds", "storage_bytes", "indexed_documents"}
        },
    }
