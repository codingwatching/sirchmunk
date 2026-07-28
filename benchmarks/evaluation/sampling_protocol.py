"""Sampling protocol utilities for paper-grade benchmark subsets.

The module is intentionally dependency-light. It operates on BenchmarkSample-like
objects or dictionaries and records enough metadata for later reproducibility and
validator checks.
"""
from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from framework.time_utils import now_local_iso


DEFAULT_HOTPOTQA_POPULATION_SIZE = 7405
DEFAULT_HOTPOTQA_STRATA = ["type", "supporting_fact_bucket"]
DEFAULT_MONITORED_FIELDS = ["answer_type"]


@dataclass
class SamplingProtocol:
    benchmark: str
    split: str = "validation"
    population_size: int = 0
    method: str = "simple_random"
    seed: int = 42
    target_n: int = 0
    strata: List[str] = field(default_factory=list)
    allocation: str = "proportional"
    min_per_stratum: int = 1
    expected_population_size: int = 0
    sample_ids_file: str = ""
    monitored_fields: List[str] = field(default_factory=lambda: list(DEFAULT_MONITORED_FIELDS))
    created_at: str = field(default_factory=now_local_iso)
    protocol_version: int = 1
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SamplingProtocol":
        payload = dict(data or {})
        if isinstance(payload.get("strata"), str):
            payload["strata"] = _split_csv(payload["strata"])
        if isinstance(payload.get("monitored_fields"), str):
            payload["monitored_fields"] = _split_csv(payload["monitored_fields"])
        allowed = set(cls.__dataclass_fields__)  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in payload.items() if k in allowed})


@dataclass
class SamplingManifest:
    protocol: Dict[str, Any]
    sample_ids: List[str]
    sample_id_checksum: str
    population_size: int
    target_n: int
    actual_n: int
    distribution_before: Dict[str, Any]
    distribution_after: Dict[str, Any]
    deviation_report: Dict[str, Any]
    generated_at: str = field(default_factory=now_local_iso)
    manifest_version: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> str:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        return str(p)


@dataclass
class StratumSpec:
    key: str
    population_count: int
    target_count: int
    actual_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def create_sampling_protocol(
    *,
    benchmark: str,
    split: str = "validation",
    population_size: int = 0,
    method: str = "simple_random",
    seed: int = 42,
    target_n: int = 0,
    strata: Sequence[str] | str | None = None,
    allocation: str = "proportional",
    min_per_stratum: int = 1,
    expected_population_size: int = 0,
    monitored_fields: Sequence[str] | str | None = None,
    sample_ids_file: str = "",
) -> SamplingProtocol:
    return SamplingProtocol(
        benchmark=benchmark,
        split=split,
        population_size=population_size,
        method=(method or "simple_random").strip().lower(),
        seed=int(seed),
        target_n=max(int(target_n or 0), 0),
        strata=_split_csv(strata) if isinstance(strata, str) else list(strata or []),
        allocation=(allocation or "proportional").strip().lower(),
        min_per_stratum=max(int(min_per_stratum or 0), 0),
        expected_population_size=max(int(expected_population_size or 0), 0),
        sample_ids_file=str(sample_ids_file or ""),
        monitored_fields=_split_csv(monitored_fields) if isinstance(monitored_fields, str) else list(monitored_fields or DEFAULT_MONITORED_FIELDS),
    )


def load_sampling_protocol(path: str | Path) -> SamplingProtocol:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict) and "protocol" in data and isinstance(data["protocol"], dict):
        data = data["protocol"]
    if isinstance(data, dict) and "sampling_protocol" in data and isinstance(data["sampling_protocol"], dict):
        data = data["sampling_protocol"]
    if not isinstance(data, dict):
        raise ValueError(f"Invalid sampling protocol file: {path}")
    return SamplingProtocol.from_dict(data)


def describe_population(samples: Sequence[Any], strata: Sequence[str] | None = None) -> Dict[str, Any]:
    rows = [_sample_to_dict(sample) for sample in samples]
    out: Dict[str, Any] = {
        "population_size": len(rows),
        "type": _value_counts(rows, "type"),
        "level": _value_counts(rows, "level"),
        "answer_type": _value_counts(rows, "answer_type"),
        "supporting_fact_bucket": _value_counts(rows, "supporting_fact_bucket"),
    }
    for key in strata or []:
        if key not in out:
            out[key] = _value_counts(rows, key)
    if strata:
        out["strata"] = _stratum_distribution(rows, strata)
    return out


def create_sample(
    samples: Sequence[Any],
    protocol: SamplingProtocol,
) -> tuple[List[Any], SamplingManifest]:
    rows = list(samples)
    population_size = len(rows)
    protocol = SamplingProtocol.from_dict({**protocol.to_dict(), "population_size": population_size})
    target_n = protocol.target_n
    if protocol.method == "fixed_ids":
        selected = _fixed_ids(rows, protocol.sample_ids_file)
    elif protocol.method == "diagnostic_rare":
        selected = _diagnostic_rare(rows)
    elif protocol.method == "full" or target_n <= 0 or target_n >= population_size:
        selected = list(rows)
    elif protocol.method in {"simple_random", "random"}:
        selected = _simple_random(rows, target_n=target_n, seed=protocol.seed)
    elif protocol.method in {"stratified", "stratified_proportional"}:
        strata = protocol.strata or DEFAULT_HOTPOTQA_STRATA
        selected = _stratified(rows, protocol=protocol, strata=strata)
    else:
        raise ValueError(f"Unsupported sampling method: {protocol.method}")

    sample_ids = [_sample_id(sample) for sample in selected]
    before = describe_population(rows, protocol.strata)
    after = describe_population(selected, protocol.strata)
    manifest = SamplingManifest(
        protocol=protocol.to_dict(),
        sample_ids=sample_ids,
        sample_id_checksum=compute_sample_id_checksum(sample_ids),
        population_size=population_size,
        target_n=target_n,
        actual_n=len(selected),
        distribution_before=before,
        distribution_after=after,
        deviation_report=_deviation_report(before, after, protocol),
    )
    return selected, manifest


def validate_sampling_manifest(manifest: SamplingManifest | Dict[str, Any]) -> Dict[str, Any]:
    data = manifest.to_dict() if isinstance(manifest, SamplingManifest) else dict(manifest or {})
    protocol = data.get("protocol", {}) if isinstance(data.get("protocol"), dict) else {}
    errors: List[str] = []
    warnings: List[str] = []
    sample_ids = data.get("sample_ids") or []
    if not sample_ids:
        errors.append("sampling manifest has no sample_ids")
    checksum = data.get("sample_id_checksum")
    if checksum and checksum != compute_sample_id_checksum([str(s) for s in sample_ids]):
        errors.append("sample_id_checksum does not match sample_ids")
    expected = int(protocol.get("expected_population_size") or 0)
    population = int(data.get("population_size") or protocol.get("population_size") or 0)
    if expected and population != expected:
        errors.append(f"population_size mismatch: expected={expected}, actual={population}")
    method = str(protocol.get("method") or "")
    target_n = int(data.get("target_n") or protocol.get("target_n") or 0)
    actual_n = int(data.get("actual_n") or len(sample_ids) or 0)
    population_n = int(data.get("population_size") or protocol.get("population_size") or 0)
    if not method:
        errors.append("sampling method is missing")
    if method.startswith("stratified") and not protocol.get("strata"):
        errors.append("stratified sampling requires strata keys")
    if method not in {"full", "diagnostic_rare", "fixed_ids"} and target_n > 0 and population_n > 0:
        expected_n = min(target_n, population_n)
        if actual_n != expected_n:
            errors.append(f"actual_n mismatch: expected={expected_n}, actual={actual_n}")
    if method == "full" and population_n and actual_n != population_n:
        errors.append(f"full sampling requires actual_n=population_size: actual={actual_n}, population={population_n}")
    if method.startswith("stratified"):
        before = data.get("distribution_before") or {}
        after = data.get("distribution_after") or {}
        if not (isinstance(before, dict) and before.get("strata") and isinstance(after, dict) and after.get("strata")):
            errors.append("stratified sampling requires before/after strata distributions")
    deviation = data.get("deviation_report") or {}
    max_abs = _safe_float(deviation.get("max_abs_proportion_delta"), 0.0)
    if max_abs > 0.05:
        warnings.append(f"sampling distribution drift is above 5%: {max_abs:.4f}")
    monitored = deviation.get("monitored_distribution_delta", {}) if isinstance(deviation, dict) else {}
    answer_type_delta = monitored.get("answer_type", {}) if isinstance(monitored, dict) else {}
    max_answer_type_delta = max((abs(_safe_float(v)) for v in answer_type_delta.values()), default=0.0) if isinstance(answer_type_delta, dict) else 0.0
    if max_answer_type_delta > 0.05:
        warnings.append(f"answer_type distribution drift is above 5%: {max_answer_type_delta:.4f}")
    return {"passed": not errors, "errors": errors, "warnings": warnings}


def compute_sample_id_checksum(sample_ids: Iterable[str]) -> str:
    canonical = sorted(str(sample_id) for sample_id in sample_ids)
    raw = json.dumps(canonical, ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


def stratum_key_for(sample: Any, strata: Sequence[str]) -> str:
    """Return the stratum key of one sample for the given strata keys."""
    return _stratum_key(_sample_to_dict(sample), strata)


def stratum_distribution_for(samples: Sequence[Any], strata: Sequence[str]) -> Dict[str, int]:
    """Return stratum counts for a sample set, using the same keys as manifests."""
    return _stratum_distribution([_sample_to_dict(sample) for sample in samples], strata)


def proportion_deltas(
    stage_counts: Dict[str, int],
    reference_counts: Dict[str, int],
) -> Dict[str, float]:
    """Per-stratum proportion difference between a stage and a reference set.

    Positive values mean the stage over-represents that stratum. Keys come from
    the reference so a stratum missing from the stage is reported rather than
    silently dropped.
    """
    stage_total = sum(stage_counts.values()) or 1
    reference_total = sum(reference_counts.values()) or 1
    return {
        key: round(stage_counts.get(key, 0) / stage_total - value / reference_total, 6)
        for key, value in sorted(reference_counts.items())
    }


def extract_sample_ids(path: str | Path) -> List[str]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, list):
        return [str(item) for item in data]
    if isinstance(data, dict):
        for key in ("sample_ids", "ids"):
            if isinstance(data.get(key), list):
                return [str(item) for item in data[key]]
        if isinstance(data.get("sampling_manifest"), dict):
            return [str(item) for item in data["sampling_manifest"].get("sample_ids", [])]
        if isinstance(data.get("samples"), list):
            ids = [str(item.get("sample_id", "")) for item in data["samples"] if isinstance(item, dict) and item.get("sample_id")]
            if ids:
                return ids
    raise ValueError(f"Cannot extract sample ids from {path}")


def write_sample_ids(path: str | Path, sample_ids: Sequence[str], *, metadata: Dict[str, Any] | None = None) -> str:
    payload = {
        "sample_ids": [str(sample_id) for sample_id in sample_ids],
        "sample_id_checksum": compute_sample_id_checksum(sample_ids),
        "metadata": metadata or {},
    }
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return str(p)


def _simple_random(samples: Sequence[Any], *, target_n: int, seed: int) -> List[Any]:
    rng = random.Random(seed)
    return [samples[idx] for idx in sorted(rng.sample(range(len(samples)), target_n))]


def _diagnostic_rare(samples: Sequence[Any]) -> List[Any]:
    return [sample for sample in samples if int(_sample_to_dict(sample).get("supporting_fact_count", 0) or 0) >= 5]


def _fixed_ids(samples: Sequence[Any], sample_ids_file: str) -> List[Any]:
    if not sample_ids_file:
        raise ValueError("fixed_ids sampling requires sample_ids_file")
    sample_ids = extract_sample_ids(sample_ids_file)
    by_id = {_sample_id(sample): sample for sample in samples}
    missing = [sample_id for sample_id in sample_ids if sample_id not in by_id]
    if missing:
        raise ValueError(f"fixed_ids sample_ids_file contains ids outside population: missing={missing[:10]} total_missing={len(missing)}")
    return [by_id[sample_id] for sample_id in sample_ids]


def _stratified(samples: Sequence[Any], *, protocol: SamplingProtocol, strata: Sequence[str]) -> List[Any]:
    groups: Dict[str, List[Any]] = defaultdict(list)
    for sample in samples:
        groups[_stratum_key(_sample_to_dict(sample), strata)].append(sample)
    allocation = _allocate_counts(
        {key: len(value) for key, value in groups.items()},
        target_n=protocol.target_n,
        strategy=protocol.allocation,
        min_per_stratum=protocol.min_per_stratum,
    )
    selected: List[Any] = []
    for key in sorted(groups):
        count = min(allocation.get(key, 0), len(groups[key]))
        rng = random.Random(_stable_seed(protocol.seed, key))
        indices = sorted(rng.sample(range(len(groups[key])), count)) if count < len(groups[key]) else list(range(len(groups[key])))
        selected.extend(groups[key][idx] for idx in indices)
    rng = random.Random(protocol.seed)
    rng.shuffle(selected)
    return selected


def _allocate_counts(
    group_sizes: Dict[str, int],
    *,
    target_n: int,
    strategy: str,
    min_per_stratum: int,
) -> Dict[str, int]:
    if target_n <= 0:
        return dict(group_sizes)
    non_empty = {key: value for key, value in group_sizes.items() if value > 0}
    target_n = min(target_n, sum(non_empty.values()))
    if not non_empty:
        return {}
    if strategy in {"equal", "uniform"}:
        base = target_n // len(non_empty)
        allocation = {key: min(value, base) for key, value in non_empty.items()}
    else:
        total = sum(non_empty.values())
        raw = {key: target_n * value / total for key, value in non_empty.items()}
        allocation = {key: min(non_empty[key], int(raw[key])) for key in non_empty}
        remainders = sorted(
            ((raw[key] - int(raw[key]), key) for key in non_empty),
            reverse=True,
        )
        remaining = target_n - sum(allocation.values())
        idx = 0
        while remaining > 0 and remainders:
            key = remainders[idx % len(remainders)][1]
            if allocation[key] < non_empty[key]:
                allocation[key] += 1
                remaining -= 1
            idx += 1
            if idx > len(remainders) * 3 and all(allocation[k] >= non_empty[k] for k in non_empty):
                break
    if min_per_stratum > 0 and target_n >= len(non_empty) * min_per_stratum:
        for key, size in non_empty.items():
            allocation[key] = max(allocation.get(key, 0), min(min_per_stratum, size))
        while sum(allocation.values()) > target_n:
            candidates = sorted(
                (count, key) for key, count in allocation.items() if count > min_per_stratum
            )
            if not candidates:
                break
            _, key = candidates[-1]
            allocation[key] -= 1
    while sum(allocation.values()) < target_n:
        candidates = sorted(
            (non_empty[key] - allocation.get(key, 0), key)
            for key in non_empty
            if allocation.get(key, 0) < non_empty[key]
        )
        if not candidates:
            break
        _, key = candidates[-1]
        allocation[key] = allocation.get(key, 0) + 1
    return allocation


def _sample_to_dict(sample: Any) -> Dict[str, Any]:
    if isinstance(sample, dict):
        metadata = sample.get("metadata", {}) if isinstance(sample.get("metadata", {}), dict) else {}
        row = {**metadata, **sample}
    else:
        metadata = getattr(sample, "metadata", {}) or {}
        row = {**metadata, "sample_id": getattr(sample, "sample_id", ""), "gold_answer": getattr(sample, "gold_answer", "")}
    row.setdefault("answer_type", _answer_type(row.get("gold_answer") or row.get("answer")))
    row.setdefault("supporting_fact_count", _supporting_fact_count(row.get("supporting_facts")))
    row.setdefault("supporting_fact_bucket", _supporting_fact_bucket(row["supporting_fact_count"]))
    row.setdefault("type", "unknown")
    row.setdefault("level", "unknown")
    return row


def _sample_id(sample: Any) -> str:
    if isinstance(sample, dict):
        return str(sample.get("sample_id", sample.get("id", "")))
    return str(getattr(sample, "sample_id", ""))


def _stratum_key(row: Dict[str, Any], strata: Sequence[str]) -> str:
    return "|".join(f"{key}={row.get(key, 'unknown')}" for key in strata)


def _stratum_distribution(rows: Sequence[Dict[str, Any]], strata: Sequence[str]) -> Dict[str, int]:
    counter = Counter(_stratum_key(row, strata) for row in rows)
    return dict(sorted(counter.items()))


def _value_counts(rows: Sequence[Dict[str, Any]], key: str) -> Dict[str, int]:
    counter = Counter(str(row.get(key, "unknown")) for row in rows)
    return dict(sorted(counter.items()))


def _deviation_report(before: Dict[str, Any], after: Dict[str, Any], protocol: SamplingProtocol) -> Dict[str, Any]:
    before_dist = before.get("strata", {}) if isinstance(before.get("strata"), dict) else {}
    after_dist = after.get("strata", {}) if isinstance(after.get("strata"), dict) else {}
    before_total = max(int(before.get("population_size", 0) or 0), 1)
    after_total = max(int(after.get("population_size", 0) or 0), 1)
    deltas: Dict[str, float] = {}
    for key in sorted(set(before_dist) | set(after_dist)):
        before_p = int(before_dist.get(key, 0)) / before_total
        after_p = int(after_dist.get(key, 0)) / after_total
        deltas[key] = round(after_p - before_p, 6)
    max_abs = max((abs(v) for v in deltas.values()), default=0.0)
    monitored_distribution_delta = {
        field: _count_delta(before.get(field, {}), after.get(field, {}), before_total, after_total)
        for field in protocol.monitored_fields
        if isinstance(before.get(field), dict) and isinstance(after.get(field), dict)
    }
    return {
        "method": protocol.method,
        "allocation": protocol.allocation,
        "strata": protocol.strata,
        "proportion_delta_by_stratum": deltas,
        "max_abs_proportion_delta": round(max_abs, 6),
        "monitored_fields": protocol.monitored_fields,
        "monitored_distribution_delta": monitored_distribution_delta,
    }


def _count_delta(before_counts: Dict[str, int], after_counts: Dict[str, int], before_total: int, after_total: int) -> Dict[str, float]:
    deltas: Dict[str, float] = {}
    for key in sorted(set(before_counts) | set(after_counts)):
        before_p = int(before_counts.get(key, 0)) / max(before_total, 1)
        after_p = int(after_counts.get(key, 0)) / max(after_total, 1)
        deltas[str(key)] = round(after_p - before_p, 6)
    return deltas


def _supporting_fact_count(value: Any) -> int:
    if isinstance(value, dict):
        titles = value.get("title") or value.get("titles") or []
        if isinstance(titles, str):
            return 1
        try:
            return len(titles)
        except TypeError:
            return 0
    if isinstance(value, (list, tuple, set)):
        return len(value)
    return 0


def _supporting_fact_bucket(count: int) -> str:
    if count <= 2:
        return "2"
    if count == 3:
        return "3"
    if count == 4:
        return "4"
    return "5_plus"


def _answer_type(answer: Any) -> str:
    text = str(answer or "").strip().lower()
    return "yes_no" if text in {"yes", "no"} else "span"


def _stable_seed(seed: int, key: str) -> int:
    digest = hashlib.sha256(f"{seed}:{key}".encode("utf-8")).hexdigest()
    return int(digest[:12], 16)


def _split_csv(value: Sequence[str] | str | None) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    return [str(part).strip() for part in value if str(part).strip()]


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


__all__ = [
    "DEFAULT_HOTPOTQA_POPULATION_SIZE",
    "DEFAULT_HOTPOTQA_STRATA",
    "DEFAULT_MONITORED_FIELDS",
    "SamplingProtocol",
    "SamplingManifest",
    "StratumSpec",
    "create_sampling_protocol",
    "load_sampling_protocol",
    "describe_population",
    "create_sample",
    "validate_sampling_manifest",
    "compute_sample_id_checksum",
    "extract_sample_ids",
    "write_sample_ids",
]
