"""baselines — competitor baseline evaluation adapter package

Onboarding cheat sheet::

    # Live / external competitors
    from baselines import SdkBaseline, LocalBM25Baseline, NaiveRAGBaseline
    from baselines import LightRAGV1Baseline, GraphRAGBaseline

    # Import from precomputed JSONL
    from baselines import ManualImportAdapter

Evaluation protocol for a new baseline
--------------------------------------
Every baseline is compared under the same accounting, so a new adapter must satisfy
the three rules below. The first two are checked automatically by
``BaselineAdapter.validate_prediction_contract`` on the first prediction: a violation
raises instead of silently emitting numbers that are not comparable.

1. Declare ``retrieval_mode``. The default is ``"retrieval_based"`` (reads the
   corpus); the other value is ``"retrieval_free"`` (answers from model parameters
   only). The strict default is intentional: forgetting to declare lands on the
   stricter obligation and is caught by validation instead of passing quietly.

2. Report evidence according to that declaration. ``retrieval_based`` must expose
   ``read_file_ids`` or ``evidence_sources`` in the ``predict()`` metadata; an empty
   list means the question genuinely retrieved nothing, whereas a missing key cannot
   be distinguished from "never wired up". ``retrieval_free`` must not emit either
   key at all.

   This is not a formality. A missing evidence field is invisible in the scores: the
   run completes normally, evidence metrics read as 0, and the system is recorded as
   having retrieved nothing, which makes any comparison against evidence-reporting
   systems meaningless.

3. Run against the closed-book reference arm on the same samples.
   ``ClosedBookBaseline`` reads no corpus, so its score is the share of the benchmark
   answerable from model memory alone (measured EM 0.355 on HotpotQA G_500). The part
   of an open-book score that sits below that floor is not evidence of retrieval
   ability, so a new baseline should be reported with: total EM, the net retrieval
   gain over closed-book, the grounded rate, and grounded EM. Closed-book costs about
   127 tokens / 2.6s per question and can serve as a permanent reference row.
"""
from .base_adapter import BaselineAdapter, BaselinePrediction, BaselineResult, BaselineSetupResult
from .bm25_rag import BM25RAGBaseline
from .closed_book import ClosedBookBaseline
from .external import ExternalPredictionBaseline, GraphRAGBaseline, LightRAGV1Baseline
from .lexical import LocalBM25Baseline, NaiveRAGBaseline
from .hybrid_rag import HybridRAGBaseline
from .lightrag_v136 import LightRAGV136Baseline
from .react_search import ReActSearchBaseline
from .sdk_baseline import ManualImportAdapter, SdkBaseline
from .indexing_sdk_baseline import IndexingSdkBaseline

__all__ = [
    # ABC + data structures
    "BaselineAdapter",
    "BaselinePrediction",
    "BaselineResult",
    "BaselineSetupResult",
    # Local / external baselines
    "LocalBM25Baseline",
    "BM25RAGBaseline",
    "ClosedBookBaseline",
    "HybridRAGBaseline",
    "NaiveRAGBaseline",
    "ReActSearchBaseline",
    "ExternalPredictionBaseline",
    "LightRAGV1Baseline",
    "LightRAGV136Baseline",
    "GraphRAGBaseline",
    # SDK + import helpers
    "SdkBaseline",
    "IndexingSdkBaseline",
    "ManualImportAdapter",
]
