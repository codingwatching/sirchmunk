"""baselines — 竞品基线评估适配器包

接入方式速查：

    # 真实/外部竞品
    from baselines import SdkBaseline, LocalBM25Baseline, NaiveRAGBaseline
    from baselines import LightRAGV1Baseline, GraphRAGBaseline

    # 从预计算 JSONL 导入
    from baselines import ManualImportAdapter

新增 baseline 的评估协议
------------------------
每个 baseline 都按同一口径比较，因此新增适配器必须满足以下三条。前两条由
``BaselineAdapter.validate_prediction_contract`` 在首个预测上自动校验，违约直接
报错而非静默产出不可比的数字。

1. 声明 ``retrieval_mode``。默认 ``"retrieval_based"``（读语料），另一取值为
   ``"retrieval_free"``（仅用模型参数作答）。默认取严是有意的：忘记声明会落到
   更严的义务上并被校验拦下，而不是悄悄通过。

2. 按声明上报证据。``retrieval_based`` 必须在 ``predict()`` 的 metadata 里给出
   ``read_file_ids`` 或 ``evidence_sources``；空列表表示这道题确实没检索到，键
   缺失则无法与「未接线」区分。``retrieval_free`` 必须完全不发这两个键。

   这条不是形式要求。缺失证据字段在分数上是看不见的：运行照常完成，证据类指标
   读作 0，该系统被记录为「什么都没检索到」，与其他上报证据的系统并列比较即失去
   意义。

3. 与 closed-book 参照臂同样本对跑。``ClosedBookBaseline`` 不读任何语料，其分数
   即该基准可被模型记忆直接答对的比例（HotpotQA G_500 实测 EM 0.355）。任何
   open-book 分数中低于该地板的部分都不构成检索能力的证据，因此报告新 baseline
   时应同时给出：总 EM、相对 closed-book 的检索净增益、grounded 率与 grounded EM。
   closed-book 每题约 127 token / 2.6s，可长期作为标准参照行。
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
    # ABC + 数据结构
    "BaselineAdapter",
    "BaselinePrediction",
    "BaselineResult",
    "BaselineSetupResult",
    # 本地/外部基线
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
    # SDK + 导入
    "SdkBaseline",
    "IndexingSdkBaseline",
    "ManualImportAdapter",
]
