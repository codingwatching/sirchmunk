"""baselines — 竞品基线评估适配器包

接入方式速查：

    # 真实/外部竞品
    from baselines import SdkBaseline, LocalBM25Baseline, NaiveRAGBaseline
    from baselines import LightRAGV1Baseline, GraphRAGBaseline

    # 从预计算 JSONL 导入
    from baselines import ManualImportAdapter
"""
from .base_adapter import BaselineAdapter, BaselinePrediction, BaselineResult, BaselineSetupResult
from .external import ExternalPredictionBaseline, GraphRAGBaseline, LightRAGV1Baseline
from .lexical import LocalBM25Baseline, NaiveRAGBaseline
from .sdk_baseline import ManualImportAdapter, SdkBaseline
from .indexing_sdk_baseline import IndexingSdkBaseline

__all__ = [
    # ABC + 数据结构
    "BaselineAdapter",
    "BaselinePrediction",
    "BaselineResult",
    "BaselineSetupResult",
    # 本地/外部基线
    "LocalBM25Baseline",
    "NaiveRAGBaseline",
    "ExternalPredictionBaseline",
    "LightRAGV1Baseline",
    "GraphRAGBaseline",
    # SDK + 导入
    "SdkBaseline",
    "IndexingSdkBaseline",
    "ManualImportAdapter",
]
