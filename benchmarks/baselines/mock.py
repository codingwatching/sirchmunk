"""baselines/mock.py — Mock Baselines（测试与占位用）

提供 4 种 Mock 实现，覆盖评估管道的不同测试场景：

  ConstantMockBaseline     → 始终拒绝回答（测试 refusal 检测）
  GoldCopyMockBaseline     → 直接返回金答案（验证 Judge upper bound）
  RandomAnswerMockBaseline → 随机编造金融数据（测试 Judge 鲁棒性）
  FixedAccuracyMockBaseline→ 以指定准确率概率性答对（集成测试）

所有 Mock 均支持 seed 以保证复现性。
真实竞品接入时，将对应的 Mock 替换为真实 adapter 即可，其余流水线不变。
"""
from __future__ import annotations

import random
import time
from typing import List

from .base_adapter import BaselineAdapter, BaselinePrediction


class ConstantMockBaseline(BaselineAdapter):
    """始终返回固定拒绝文本，用于测试 Judge 的 refusal 检测分支。

    场景：验证 refusal_rate 指标的计算逻辑。
    """

    _DEFAULT_REFUSAL = "I cannot find the answer in the provided documents."

    def __init__(self, refusal_text: str = _DEFAULT_REFUSAL) -> None:
        self._refusal = refusal_text

    @property
    def name(self) -> str:
        return "constant_mock"

    @property
    def citation_name(self) -> str:
        return "ConstantRefusal (mock)"

    async def predict(self, question: str, context_paths: List[str]) -> BaselinePrediction:
        return BaselinePrediction(
            answer=self._refusal,
            elapsed=0.001,
            tokens_used=0,
            metadata={"mock_type": "constant"},
        )


class GoldCopyMockBaseline(BaselineAdapter):
    """直接复制金答案，验证 Judge 评分接近 100%（上界测试）。

    场景：若此 mock 的 accuracy < 98%，说明 Judge 本身有问题。
    此 mock 需要从外部传入 {sample_id: gold_answer} 映射。
    """

    def __init__(self, gold_map: dict) -> None:
        """
        Args:
            gold_map: {sample_id: gold_answer} 字典。
                      可通过 GoldenSet.to_gold_map() 获取。
        """
        self._gold_map = gold_map

    @property
    def name(self) -> str:
        return "gold_copy_mock"

    @property
    def citation_name(self) -> str:
        return "GoldCopy (mock upper-bound)"

    async def predict(self, question: str, context_paths: List[str]) -> BaselinePrediction:
        # 无法通过 question 反查 sample_id，返回空 answer；
        # 实际使用时由 BaselineEvaluationSuite 传入 sample_id 并调用 predict_by_id()
        return BaselinePrediction(
            answer="(gold answer unavailable — use predict_by_id)",
            elapsed=0.001,
            tokens_used=0,
            metadata={"mock_type": "gold_copy"},
        )

    def predict_by_id(self, sample_id: str) -> BaselinePrediction:
        """通过 sample_id 直接查取金答案（供 BaselineEvaluationSuite 调用）。"""
        gold = self._gold_map.get(sample_id, "unknown")
        return BaselinePrediction(
            answer=gold,
            elapsed=0.001,
            tokens_used=0,
            metadata={"mock_type": "gold_copy", "sample_id": sample_id},
        )


class RandomAnswerMockBaseline(BaselineAdapter):
    """随机编造金融数据，验证 Judge 的 wrong_value 检测能力。

    场景：accuracy 应该接近 0，若显著高于 0 则 Judge 可能存在误判。
    """

    _FAKE_TEMPLATES = [
        "${val:.1f} million",
        "${val:.2f} billion",
        "{val:.1f}%",
        "${val:,.0f}",
        "({val:.2f})",
        "Increased by {val:.1f}%",
        "Decreased by {val:.1f}%",
        "${val:.1f}K",
    ]

    def __init__(self, seed: int = 42) -> None:
        self._rng = random.Random(seed)

    @property
    def name(self) -> str:
        return "random_mock"

    @property
    def citation_name(self) -> str:
        return "Random (mock lower-bound)"

    async def predict(self, question: str, context_paths: List[str]) -> BaselinePrediction:
        template = self._rng.choice(self._FAKE_TEMPLATES)
        val = self._rng.uniform(1.0, 999.9)
        fake_answer = template.format(val=val)
        return BaselinePrediction(
            answer=fake_answer,
            elapsed=0.001,
            tokens_used=0,
            metadata={"mock_type": "random"},
        )


class FixedAccuracyMockBaseline(BaselineAdapter):
    """以指定概率答对，用于集成测试 pipeline 的 accuracy 统计逻辑。

    场景：设定 target_accuracy=0.3，运行后 accuracy 应在 [0.2, 0.4] 范围内。
    需要从外部传入 gold_map 以供答对时复制金答案。
    """

    def __init__(
        self,
        gold_map: dict,
        target_accuracy: float = 0.3,
        seed: int = 42,
    ) -> None:
        """
        Args:
            gold_map:        {sample_id: gold_answer}，答对时从此取答案。
            target_accuracy: 目标准确率 [0, 1]。
            seed:            随机种子，保证复现性。
        """
        self._gold_map = gold_map
        self._target = target_accuracy
        self._rng = random.Random(seed)

    @property
    def name(self) -> str:
        return f"fixed_acc_{int(self._target * 100)}_mock"

    @property
    def citation_name(self) -> str:
        return f"FixedAccuracy-{self._target:.0%} (mock)"

    async def predict(self, question: str, context_paths: List[str]) -> BaselinePrediction:
        # 无法通过 question 反查 sample_id，此方法不使用
        return BaselinePrediction(
            answer="N/A — use predict_by_id",
            elapsed=0.001,
            tokens_used=0,
            metadata={"mock_type": "fixed_accuracy"},
        )

    def predict_by_id(self, sample_id: str) -> BaselinePrediction:
        """通过 sample_id 按概率决定是否答对（供 BaselineEvaluationSuite 调用）。"""
        if self._rng.random() < self._target:
            # 答对：复制金答案
            answer = self._gold_map.get(sample_id, "correct answer")
        else:
            # 答错：返回拒绝文本
            answer = "I cannot find the answer in the provided documents."
        return BaselinePrediction(
            answer=answer,
            elapsed=0.001,
            tokens_used=0,
            metadata={"mock_type": "fixed_accuracy", "sample_id": sample_id},
        )

    def get_request_delay(self) -> float:
        return 0.0   # mock 无需延迟
