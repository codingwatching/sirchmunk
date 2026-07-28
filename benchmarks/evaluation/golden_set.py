"""evaluation/golden_set.py — GoldenSet 固定测试集管理

GoldenSet 的核心职责：
  保证所有系统（Sirchmunk 和所有竞品）在完全相同的问题集上评估，
  避免不同采样导致的测试集差异污染对比结果。

科学严谨性设计：
  1. GoldenSet 由 sampling protocol / seed / sample IDs 唯一确定
  2. checksum（SHA-256）覆盖 sample_id/question/gold/metadata，防止测试集被意外修改
  3. sample_id_checksum 独立记录，用于跨系统配对检验
  4. sampling_protocol / sampling_manifest 持久化，支持 sampled evaluation 审计
  5. 与自改进循环（run_research_loop.py）完全隔离

文件命名约定：
  sampled: benchmarks/{benchmark}/golden_set_{method}_{seed}_{n}_{checksum8}.json
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from evaluation.sampling_protocol import (
    SamplingProtocol,
    compute_sample_id_checksum,
    create_sample,
)
from framework.time_utils import now_local_iso

logger = logging.getLogger(__name__)


@dataclass
class GoldenSet:
    """固定测试集，所有系统使用相同的问题。"""
    benchmark: str
    seed: int
    n_questions: int
    created_at: str
    checksum: str
    samples: List[Dict[str, Any]] = field(default_factory=list)
    sample_id_checksum_value: str = ""
    population_size: int = 0
    sampling_protocol: Dict[str, Any] = field(default_factory=dict)
    sampling_manifest: Dict[str, Any] = field(default_factory=dict)
    schema_version: int = 2

    def to_gold_map(self) -> Dict[str, str]:
        """返回 {sample_id: gold_answer} 映射。"""
        return {s["sample_id"]: s["gold_answer"] for s in self.samples}

    def to_question_map(self) -> Dict[str, str]:
        """返回 {sample_id: question} 映射。"""
        return {s["sample_id"]: s["question"] for s in self.samples}

    def sample_ids(self) -> List[str]:
        """Return sample ids in GoldenSet order."""
        return [str(s["sample_id"]) for s in self.samples]

    def sample_id_checksum(self) -> str:
        """Return order-insensitive checksum of sample ids."""
        return self.sample_id_checksum_value or compute_sample_id_checksum(self.sample_ids())

    def verify_results_sample_ids(self, results: List[Any], *, system_name: str = "system") -> None:
        """Ensure result sample ids match this GoldenSet exactly."""
        expected_ids = self.sample_ids()
        expected = set(expected_ids)
        observed_ids = [str(getattr(r, "sample_id", "")) for r in results]
        observed = set(observed_ids)
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        duplicates = sorted({sample_id for sample_id in observed_ids if observed_ids.count(sample_id) > 1})
        if missing or extra or duplicates or len(observed_ids) != len(expected_ids):
            raise ValueError(
                f"Sample id mismatch for {system_name}: "
                f"missing={missing[:10]} extra={extra[:10]} duplicates={duplicates[:10]} "
                f"expected_n={len(expected_ids)} observed_n={len(observed_ids)}"
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

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "benchmark": self.benchmark,
            "seed": self.seed,
            "n_questions": self.n_questions,
            "created_at": self.created_at,
            "checksum": self.checksum,
            "sample_id_checksum": self.sample_id_checksum(),
            "population_size": self.population_size,
            "sampling_protocol": self.sampling_protocol,
            "sampling_manifest": self.sampling_manifest,
            "samples": self.samples,
        }

    def save(self, path: str) -> None:
        """持久化到 JSON 文件。"""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
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
            seed=int(data["seed"]),
            n_questions=int(data["n_questions"]),
            created_at=data["created_at"],
            checksum=data["checksum"],
            samples=data["samples"],
            sample_id_checksum_value=data.get("sample_id_checksum", ""),
            population_size=int(data.get("population_size", data.get("n_questions", 0)) or 0),
            sampling_protocol=data.get("sampling_protocol", {}) if isinstance(data.get("sampling_protocol", {}), dict) else {},
            sampling_manifest=data.get("sampling_manifest", {}) if isinstance(data.get("sampling_manifest", {}), dict) else {},
            schema_version=int(data.get("schema_version", 1) or 1),
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
    """GoldenSet 的工厂与持久化管理器。"""

    def __init__(self, benchmark_dir: str) -> None:
        self._dir = Path(benchmark_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

    def get_path(
        self,
        seed: int,
        n: int,
        sampling_protocol: Optional[SamplingProtocol | Dict[str, Any]] = None,
    ) -> str:
        """返回 GoldenSet 文件路径（不保证文件存在）。"""
        protocol = _coerce_protocol(sampling_protocol)
        if protocol is None:
            return str(self._dir / f"golden_set_{seed}_{n}.json")
        method = _safe_name(protocol.method)
        checksum = _protocol_checksum(protocol)[:8]
        target_n = protocol.target_n if protocol.method != "full" else 0
        return str(self._dir / f"golden_set_{method}_{protocol.seed}_{target_n}_{checksum}.json")

    def exists(self, seed: int, n: int, sampling_protocol: Optional[SamplingProtocol | Dict[str, Any]] = None) -> bool:
        """检查 GoldenSet 文件是否已存在。"""
        return Path(self.get_path(seed, n, sampling_protocol=sampling_protocol)).exists()

    def get_or_create(
        self,
        adapter,
        seed: int = 42,
        n: int = 150,
        force_recreate: bool = False,
        sampling_protocol: Optional[SamplingProtocol | Dict[str, Any]] = None,
    ) -> GoldenSet:
        """加载已有 GoldenSet 或新建并保存。

        Args:
            adapter: BenchmarkAdapter 实例，提供 load_samples() 和 name。
            seed: 随机种子。
            n: 测试集大小（0 = 全量）。
            force_recreate: 强制重新生成（即使文件已存在）。
            sampling_protocol: 可选 SamplingProtocol；提供时使用协议生成 GoldenSet。
        """
        protocol = _coerce_protocol(sampling_protocol)
        path = self.get_path(seed, n, sampling_protocol=protocol)

        if not force_recreate and Path(path).exists():
            try:
                return GoldenSet.load(path)
            except (ValueError, KeyError) as exc:
                logger.warning("[GoldenSet] Load failed (%s), recreating...", exc)

        if protocol is not None:
            population_loader = getattr(adapter, "load_sampling_population", None)
            if callable(population_loader):
                all_samples_obj = population_loader(seed=protocol.seed)
            else:
                all_samples_obj = adapter.load_samples(limit=0, seed=protocol.seed)
            selected_obj, manifest = create_sample(all_samples_obj, protocol)
            samples = [_sample_to_dict(sample) for sample in selected_obj]
            sampling_protocol_dict = manifest.protocol
            sampling_manifest_dict = manifest.to_dict()
            population_size = manifest.population_size
            seed = int(sampling_protocol_dict.get("seed", seed))
        else:
            samples_obj = adapter.load_samples(limit=n, seed=seed)
            samples = [_sample_to_dict(sample) for sample in samples_obj]
            sampling_protocol_dict = {
                "benchmark": getattr(adapter, "name", ""),
                "method": "simple_random" if n else "full",
                "seed": seed,
                "target_n": n,
                "created_at": now_local_iso(),
                "protocol_version": 0,
            }
            sampling_manifest_dict = {
                "protocol": sampling_protocol_dict,
                "sample_ids": [s["sample_id"] for s in samples],
                "sample_id_checksum": compute_sample_id_checksum([s["sample_id"] for s in samples]),
                "population_size": len(samples),
                "target_n": n,
                "actual_n": len(samples),
                "distribution_before": {},
                "distribution_after": {},
                "deviation_report": {},
            }
            population_size = len(samples)

        checksum = _compute_checksum(samples)
        gs = GoldenSet(
            benchmark=adapter.name,
            seed=seed,
            n_questions=len(samples),
            created_at=now_local_iso(),
            checksum=checksum,
            samples=samples,
            sample_id_checksum_value=compute_sample_id_checksum([s["sample_id"] for s in samples]),
            population_size=population_size,
            sampling_protocol=sampling_protocol_dict,
            sampling_manifest=sampling_manifest_dict,
        )
        gs.save(path)
        return gs


def _sample_to_dict(sample: Any) -> Dict[str, Any]:
    return {
        "sample_id": str(getattr(sample, "sample_id", "")),
        "question": str(getattr(sample, "question", "")),
        "gold_answer": str(getattr(sample, "gold_answer", "")),
        "metadata": getattr(sample, "metadata", {}) or {},
    }


def _coerce_protocol(value: Optional[SamplingProtocol | Dict[str, Any]]) -> Optional[SamplingProtocol]:
    if value is None:
        return None
    if isinstance(value, SamplingProtocol):
        return value
    if isinstance(value, dict):
        return SamplingProtocol.from_dict(value)
    raise TypeError(f"Unsupported sampling_protocol type: {type(value)!r}")


def _protocol_checksum(protocol: SamplingProtocol) -> str:
    payload = protocol.to_dict()
    payload.pop("created_at", None)
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(value).lower()) or "sampling"


def _compute_checksum(samples: List[Dict[str, Any]]) -> str:
    """计算样本列表的 SHA-256 checksum。"""
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


__all__ = ["GoldenSet", "GoldenSetManager", "compute_sample_id_checksum"]
