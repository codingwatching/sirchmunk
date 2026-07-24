"""HotpotQA loader and corpus validation helpers."""
from __future__ import annotations

import hashlib
import json
import random
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
    parquet_dir = dataset_dir / setting
    if not parquet_dir.exists():
        raise FileNotFoundError(f"HotpotQA dataset directory not found: {parquet_dir}")
    parquet_files = sorted(parquet_dir.glob(f"{split}*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {parquet_dir} matching '{split}*.parquet'")

    try:
        import pandas as pd
        frames = [pd.read_parquet(path) for path in parquet_files]
        df = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
    except ImportError as exc:
        raise ImportError("HotpotQAAdapter requires 'pyarrow' and 'pandas'.") from exc

    samples: List[BenchmarkSample] = []
    for _, row in df.iterrows():
        qid = str(row.get("id", ""))
        question = str(row.get("question", ""))
        gold = str(row.get("answer", ""))
        samples.append(BenchmarkSample(
            sample_id=qid,
            question=question,
            gold_answer=gold,
            metadata={
                "type": str(row.get("type", "")),
                "level": str(row.get("level", "")),
                "supporting_facts": _json_safe(row.get("supporting_facts", [])),
                "context": _json_safe(row.get("context", None)),
                "setting": setting,
                "split": split,
            },
        ))

    if limit > 0 and limit < len(samples):
        random.seed(seed)
        samples = random.sample(samples, limit)
    return samples


def validate_hotpotqa_corpus(wiki_dir: Path) -> Tuple[int, List[str]]:
    if wiki_dir.exists():
        return 1, []
    return 0, [str(wiki_dir)]


def build_dataset_manifest(dataset_dir: Path, wiki_dir: Path, *, setting: str, split: str) -> Dict[str, Any]:
    parquet_dir = dataset_dir / setting
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
        "manifest_version": 2,
    }
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
