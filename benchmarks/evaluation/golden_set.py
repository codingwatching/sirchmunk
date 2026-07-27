"""evaluation/golden_set.py — GoldenSet 固定测试集管理

GoldenSet 的核心职责：
  保证所有系统（Sirchmunk 和所有竞品）在完全相同的问题集上评估，
  避免不同采样导致的测试集差异污染对比结果。

科学严谨性设计：
  1. GoldenSet 由 (benchmark, seed, n) 唯一确定
  2. checksum（SHA-256）覆盖 sample_id/question/gold/metadata，防止测试集被意外修改
  3. 持久化为 JSON 文件，跨运行复现
  4. 与自改进循环（run_research_loop.py）完全隔离：
     自改进循环可以用任意 limit/seed；
     GoldenSet 专用于 run_evaluation.py 的横向对比

文件命名约定：
  benchmarks/{benchmark}/golden_set_{seed}_{n}.json
"""
from __future__ import annotations

import hashlib
import json
import logging
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List

from framework.time_utils import now_local_iso

logger = logging.getLogger(__name__)


@dataclass
class GoldenSet:
    """固定测试集，所有系统使用相同的问题。"""
    benchmark: str
    seed: int
    n_questions: int
    created_at: str                      # ISO 8601
    checksum: str                        # SHA-256 of sample content 防篡改
    samples: List[Dict[str, Any]] = field(default_factory=list)
    """samples 是字典列表（非 BenchmarkSample），以避免对 framework/ 的硬依赖。
    
    每条记录含：
      sample_id:    str
      question:     str
      gold_answer:  str
      metadata:     dict
    """

    def to_gold_map(self) -> Dict[str, str]:
        """返回 {sample_id: gold_answer} 映射，供 GoldCopyMockBaseline 使用。"""
        return {s["sample_id"]: s["gold_answer"] for s in self.samples}

    def to_question_map(self) -> Dict[str, str]:
        """返回 {sample_id: question} 映射。"""
        return {s["sample_id"]: s["question"] for s in self.samples}

    def sample_ids(self) -> List[str]:
        """Return sample ids in GoldenSet order."""
        return [str(s["sample_id"]) for s in self.samples]

    def sample_id_checksum(self) -> str:
        """Return order-insensitive checksum of sample ids."""
        return compute_sample_id_checksum(self.sample_ids())

    def verify_results_sample_ids(self, results: List[Any], *, system_name: str = "system") -> None:
        """Ensure result sample ids match this GoldenSet exactly."""
        expected = set(self.sample_ids())
        observed = {str(getattr(r, "sample_id", "")) for r in results}
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        if missing or extra:
            raise ValueError(
                f"Sample id mismatch for {system_name}: "
                f"missing={missing[:10]} extra={extra[:10]} "
                f"expected_n={len(expected)} observed_n={len(observed)}"
            )

    def to_benchmark_samples(self):
        """转换为 framework/schema.py 的 BenchmarkSample 列表（按需 import）。"""
        try:
            from framework.schema import BenchmarkSample
            return [
                BenchmarkSample(
                    sample_id=s["sample_id"],
                    question=s["question"],
                    gold_answer=s["gold_answer"],
                    metadata=s.get("metadata", {}),
                )
                for s in self.samples
            ]
        except ImportError:
            raise ImportError(
                "GoldenSet.to_benchmark_samples() requires benchmarks/framework/. "
                "Make sure benchmarks/ is in sys.path."
            )

    def verify_checksum(self) -> bool:
        """验证当前样本列表的 checksum 与存储值一致。"""
        return _compute_checksum(self.samples) == self.checksum

    def save(self, path: str) -> None:
        """持久化到 JSON 文件。"""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "benchmark":   self.benchmark,
            "seed":        self.seed,
            "n_questions": self.n_questions,
            "created_at":  self.created_at,
            "checksum":    self.checksum,
            "samples":     self.samples,
        }
        p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("[GoldenSet] Saved to %s (%d questions)", path, self.n_questions)

    @classmethod
    def load(cls, path: str) -> "GoldenSet":
        """从 JSON 文件加载，并验证 checksum。"""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"GoldenSet file not found: {path}")
        data = json.loads(p.read_text(encoding="utf-8"))
        gs = cls(
            benchmark=data["benchmark"],
            seed=data["seed"],
            n_questions=data["n_questions"],
            created_at=data["created_at"],
            checksum=data["checksum"],
            samples=data["samples"],
        )
        if not gs.verify_checksum():
            raise ValueError(
                f"GoldenSet checksum mismatch at {path}. "
                "The test set may have been modified. "
                "Delete the file and regenerate to reset."
            )
        logger.info("[GoldenSet] Loaded %s (%d questions)", path, gs.n_questions)
        return gs


class GoldenSetManager:
    """GoldenSet 的工厂与持久化管理器。

    Usage::

        # 生成并保存（只需一次）
        manager = GoldenSetManager("benchmarks/financebench")
        gs = manager.get_or_create(
            adapter=fb_adapter,    # BenchmarkAdapter，提供 load_samples()
            seed=42,
            n=150,
        )

        # 后续直接加载（checksum 验证）
        gs = manager.get_or_create(adapter, seed=42, n=150)
    """

    def __init__(self, benchmark_dir: str) -> None:
        """
        Args:
            benchmark_dir: 存放 golden_set_*.json 的目录，
                           通常为 benchmarks/{benchmark_name}/
        """
        self._dir = Path(benchmark_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

    def get_path(self, seed: int, n: int) -> str:
        """返回 GoldenSet 文件路径（不保证文件存在）。"""
        return str(self._dir / f"golden_set_{seed}_{n}.json")

    def exists(self, seed: int, n: int) -> bool:
        """检查 GoldenSet 文件是否已存在。"""
        return Path(self.get_path(seed, n)).exists()

    def get_or_create(
        self,
        adapter,           # BenchmarkAdapter
        seed: int = 42,
        n: int = 150,
        force_recreate: bool = False,
    ) -> GoldenSet:
        """加载已有 GoldenSet 或新建并保存。

        Args:
            adapter:        BenchmarkAdapter 实例，提供 load_samples() 和 name。
            seed:           随机种子。
            n:              测试集大小（0 = 全量）。
            force_recreate: 强制重新生成（即使文件已存在）。

        Returns:
            GoldenSet 实例（checksum 已验证）。
        """
        path = self.get_path(seed, n)

        if not force_recreate and Path(path).exists():
            try:
                return GoldenSet.load(path)
            except (ValueError, KeyError) as exc:
                logger.warning(
                    "[GoldenSet] Load failed (%s), recreating...", exc
                )

        # 从 adapter 加载样本。n=0 表示全量，其他值表示固定seed采样。
        samples_obj = adapter.load_samples(limit=n, seed=seed)
        samples = [
            {
                "sample_id":  s.sample_id,
                "question":   s.question,
                "gold_answer": s.gold_answer,
                "metadata":   s.metadata,
            }
            for s in samples_obj
        ]

        checksum = _compute_checksum(samples)
        gs = GoldenSet(
            benchmark=adapter.name,
            seed=seed,
            n_questions=len(samples),
            created_at=now_local_iso(),
            checksum=checksum,
            samples=samples,
        )
        gs.save(path)
        return gs


def compute_sample_id_checksum(sample_ids: List[str]) -> str:
    canonical = sorted(str(sample_id) for sample_id in sample_ids)
    raw = json.dumps(canonical, ensure_ascii=False)
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------

def _compute_checksum(samples: List[Dict[str, Any]]) -> str:
    """计算样本列表的 SHA-256 checksum。

    覆盖 sample_id、question、gold_answer 和 metadata，避免数据内容变更
    但 sample_id 不变时被误认为同一 GoldenSet。
    """
    canonical = []
    for sample in samples:
        canonical.append({
            "sample_id": sample.get("sample_id", ""),
            "question": sample.get("question", ""),
            "gold_answer": sample.get("gold_answer", ""),
            "metadata": sample.get("metadata", {}),
        })
    canonical.sort(key=lambda x: x["sample_id"])
    raw = json.dumps(canonical, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()[:32]
