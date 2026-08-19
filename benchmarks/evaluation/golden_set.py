"""evaluation/golden_set.py — frozen test set management

Core responsibility of GoldenSet:
  guarantee that every system (Sirchmunk and all competitors) is evaluated on exactly
  the same question set, so differing sampling cannot contaminate the comparison.

Scientific rigor by design:
  1. A GoldenSet is uniquely determined by sampling protocol / seed / sample IDs
  2. The SHA-256 checksum covers sample_id/question/gold/metadata to prevent accidental
     modification of the test set
  3. sample_id_checksum is recorded separately for paired cross-system testing
  4. sampling_protocol / sampling_manifest are persisted to support sampled-evaluation
     audits
  5. Fully isolated from the self-improvement loop (run_research_loop.py)

File naming convention:
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
    extract_sample_ids,
)
from framework.time_utils import now_local_iso

logger = logging.getLogger(__name__)


@dataclass
class GoldenSet:
    """Frozen test set; every system uses the same questions."""
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
        """Return the {sample_id: gold_answer} mapping."""
        return {s["sample_id"]: s["gold_answer"] for s in self.samples}

    def to_question_map(self) -> Dict[str, str]:
        """Return the {sample_id: question} mapping."""
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
        """Convert to a list of BenchmarkSample from framework/schema.py (imported lazily)."""
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
        """Verify that the checksum of the current sample list matches the stored value."""
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
        """Persist to a JSON file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("[GoldenSet] Saved to %s (%d questions)", path, self.n_questions)

    @classmethod
    def load(cls, path: str) -> "GoldenSet":
        """Load from a JSON file and verify the checksum."""
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
    """Factory and persistence manager for GoldenSet."""

    def __init__(self, benchmark_dir: str) -> None:
        self._dir = Path(benchmark_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

    def get_path(
        self,
        seed: int,
        n: int,
        sampling_protocol: Optional[SamplingProtocol | Dict[str, Any]] = None,
    ) -> str:
        """Return the GoldenSet file path; existence is not guaranteed."""
        protocol = _coerce_protocol(sampling_protocol)
        if protocol is None:
            return str(self._dir / f"golden_set_{seed}_{n}.json")
        method = _safe_name(protocol.method)
        checksum = _protocol_checksum(protocol)[:8]
        target_n = protocol.target_n if protocol.method != "full" else 0
        return str(self._dir / f"golden_set_{method}_{protocol.seed}_{target_n}_{checksum}.json")

    def exists(self, seed: int, n: int, sampling_protocol: Optional[SamplingProtocol | Dict[str, Any]] = None) -> bool:
        """Check whether the GoldenSet file already exists."""
        return Path(self.get_path(seed, n, sampling_protocol=sampling_protocol)).exists()

    def get_or_create(
        self,
        adapter,
        seed: int = 42,
        n: int = 150,
        force_recreate: bool = False,
        sampling_protocol: Optional[SamplingProtocol | Dict[str, Any]] = None,
    ) -> GoldenSet:
        """Load an existing GoldenSet, or create and save a new one.

        Args:
            adapter: BenchmarkAdapter instance providing load_samples() and name.
            seed: random seed.
            n: test set size (0 = full set).
            force_recreate: regenerate even when the file already exists.
            sampling_protocol: optional SamplingProtocol; when provided the GoldenSet is
                built through the protocol.
        """
        protocol = _coerce_protocol(sampling_protocol)
        path = self.get_path(seed, n, sampling_protocol=protocol)

        if not force_recreate and Path(path).exists():
            try:
                loaded = GoldenSet.load(path)
            except (ValueError, KeyError) as exc:
                logger.warning("[GoldenSet] Load failed (%s), recreating...", exc)
            else:
                if _matches_fixed_ids(loaded, protocol):
                    return loaded
                logger.warning(
                    "[GoldenSet] Cached set at %s no longer matches the fixed sample IDs content, recreating...",
                    path,
                )

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
    fingerprint = _sample_ids_fingerprint(protocol)
    if fingerprint:
        payload["sample_ids_fingerprint"] = fingerprint
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _sample_ids_fingerprint(protocol: Optional[SamplingProtocol]) -> str:
    """Content checksum of a fixed_ids sample list.

    fixed_ids protocols often reuse one file path across runs (e.g. quickstart
    rewrites quickstart_sample_ids.json), so the path alone cannot distinguish
    different frozen question sets. Hashing the ids keeps the GoldenSet cache
    key bound to the actual sample content.
    """
    if protocol is None or getattr(protocol, "method", "") != "fixed_ids":
        return ""
    ids_file = str(getattr(protocol, "sample_ids_file", "") or "")
    if not ids_file or not Path(ids_file).exists():
        return ""
    try:
        return compute_sample_id_checksum(extract_sample_ids(ids_file))
    except (OSError, ValueError, json.JSONDecodeError):
        return ""


def _matches_fixed_ids(golden_set: GoldenSet, protocol: Optional[SamplingProtocol]) -> bool:
    """Verify a cached fixed_ids GoldenSet still matches the ids file content."""
    fingerprint = _sample_ids_fingerprint(protocol)
    if not fingerprint:
        return True
    return golden_set.sample_id_checksum() == fingerprint


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(value).lower()) or "sampling"


def _compute_checksum(samples: List[Dict[str, Any]]) -> str:
    """Compute the SHA-256 checksum of a sample list."""
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
