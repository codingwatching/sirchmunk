"""HotpotQA loader and corpus validation helpers."""
from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

from framework.schema import BenchmarkSample


def load_hotpotqa_samples(
    dataset_dir: Path,
    *,
    setting: str = "fullwiki",
    split: str = "validation",
    limit: int = 0,
    seed: int = 42,
) -> List[BenchmarkSample]:
    df = load_hotpotqa_dataframe(dataset_dir, setting=setting, split=split)

    samples: List[BenchmarkSample] = []
    for _, row in df.iterrows():
        qid = str(row.get("id", ""))
        question = str(row.get("question", ""))
        gold = str(row.get("answer", ""))
        supporting_facts = _json_safe(row.get("supporting_facts", []))
        sf_count = _supporting_fact_count(supporting_facts)
        question_type = str(row.get("type", "")) or "unknown"
        level = str(row.get("level", "")) or "unknown"
        samples.append(BenchmarkSample(
            sample_id=qid,
            question=question,
            gold_answer=gold,
            metadata={
                "type": question_type,
                "level": level,
                "answer_type": _answer_type(gold),
                "supporting_facts": supporting_facts,
                "supporting_fact_count": sf_count,
                "supporting_fact_bucket": _supporting_fact_bucket(sf_count),
                "type_level": f"{question_type}:{level}",
                "context": _json_safe(row.get("context", None)),
                "setting": setting,
                "split": split,
            },
        ))

    if limit > 0 and limit < len(samples):
        random.seed(seed)
        samples = random.sample(samples, limit)
    return samples


def load_hotpotqa_dataframe(
    dataset_dir: Path,
    *,
    setting: str = "fullwiki",
    split: str = "validation",
):
    parquet_dir = _resolve_parquet_dir(dataset_dir, setting)
    if not parquet_dir.exists():
        raise FileNotFoundError(f"HotpotQA dataset directory not found: {parquet_dir}")
    parquet_files = sorted(parquet_dir.glob(f"{split}*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {parquet_dir} matching '{split}*.parquet'")

    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError(
            "HotpotQAAdapter requires pandas. Install project dependencies with: "
            "pip install -r requirements/core.txt -r requirements/benchmarks.txt"
        ) from exc

    try:
        import pyarrow  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "HotpotQAAdapter requires pyarrow for parquet support. Install benchmark "
            "dependencies with: pip install -r requirements/benchmarks.txt"
        ) from exc

    frames = [pd.read_parquet(path, engine="pyarrow") for path in parquet_files]
    return pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]


def describe_hotpotqa_split(
    dataset_dir: Path,
    *,
    setting: str = "fullwiki",
    split: str = "validation",
) -> Dict[str, Any]:
    samples = load_hotpotqa_samples(dataset_dir, setting=setting, split=split, limit=0)
    rows = [s.metadata for s in samples]
    type_by_bucket: Dict[str, Dict[str, int]] = {}
    for sample in samples:
        bucket = str(sample.metadata.get("supporting_fact_bucket", "unknown"))
        qtype = str(sample.metadata.get("type", "unknown"))
        type_by_bucket.setdefault(bucket, {})[qtype] = type_by_bucket.setdefault(bucket, {}).get(qtype, 0) + 1
    return {
        "setting": setting,
        "split": split,
        "population_size": len(samples),
        "type": _counter(rows, "type"),
        "level": _counter(rows, "level"),
        "answer_type": _counter(rows, "answer_type"),
        "supporting_fact_count": _counter(rows, "supporting_fact_count"),
        "supporting_fact_bucket": _counter(rows, "supporting_fact_bucket"),
        "type_by_supporting_fact_bucket": {k: dict(sorted(v.items())) for k, v in sorted(type_by_bucket.items())},
    }


def validate_hotpotqa_corpus(wiki_dir: Path) -> Tuple[int, List[str]]:
    if wiki_dir.exists():
        return 1, []
    return 0, [str(wiki_dir)]


def build_dataset_manifest(dataset_dir: Path, wiki_dir: Path, *, setting: str, split: str) -> Dict[str, Any]:
    parquet_dir = _resolve_parquet_dir(dataset_dir, setting)
    parquet_files = sorted(parquet_dir.glob(f"{split}*.parquet")) if parquet_dir.exists() else []
    manifest = {
        "dataset_dir": str(dataset_dir),
        "wiki_dir": str(wiki_dir),
        "setting": setting,
        "split": split,
        "parquet_files": [str(p) for p in parquet_files],
        "parquet_file_count": len(parquet_files),
        "parquet_checksums": {str(p): _hash_file(p) for p in parquet_files},
        "parquet_total_bytes": sum(_file_size(p) for p in parquet_files),
        "wiki_exists": wiki_dir.exists(),
        "parquet_checksum": _hash_file(parquet_files[0]) if parquet_files else "",
        "manifest_version": 3,
    }
    if parquet_files:
        try:
            manifest["dataset_distribution"] = describe_hotpotqa_split(dataset_dir, setting=setting, split=split)
        except Exception as exc:
            manifest["dataset_distribution_error"] = str(exc)
    if wiki_dir.exists():
        count, size = _summarize_dir(wiki_dir, max_files=5000)
        manifest.update({
            "wiki_file_count_sampled": count,
            "wiki_size_bytes_sampled": size,
            "wiki_scan_truncated": count >= 5000,
            "wiki_summary_sample_max_files": 5000,
            "corpus_validation_level": "directory_exists_with_sampled_stats",
        })
    return manifest


def _resolve_parquet_dir(dataset_dir: Path, setting: str) -> Path:
    """Resolve parquet directory from either dataset root or setting directory.

    Supported layouts:
    - HOTPOT_DATASET_DIR=/path/hotpotqa_dataset with /fullwiki/*.parquet
    - HOTPOT_DATASET_DIR=/path/hotpotqa_dataset/fullwiki directly
    """
    direct = dataset_dir / setting
    if direct.exists():
        return direct
    if dataset_dir.name == setting and list(dataset_dir.glob("*.parquet")):
        return dataset_dir
    return direct


def _json_safe(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "tolist"):
        try:
            return _json_safe(value.tolist())
        except Exception:
            pass
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


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


def _counter(rows: List[Dict[str, Any]], key: str) -> Dict[str, int]:
    return dict(sorted(Counter(str(row.get(key, "unknown")) for row in rows).items()))


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    try:
        with path.open("rb") as fp:
            for chunk in iter(lambda: fp.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return ""


def _file_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


def _summarize_dir(path: Path, max_files: int = 5000) -> tuple[int, int]:
    count = 0
    size = 0
    for child in path.rglob("*"):
        if child.is_file():
            count += 1
            try:
                size += child.stat().st_size
            except OSError:
                pass
            if count >= max_files:
                break
    return count, size
