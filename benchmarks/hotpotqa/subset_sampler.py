"""HotpotQA corpus subset helpers for scaling studies.

The default mode writes a manifest only.  When a physical subset is needed for
index-heavy baselines, ``materialize='symlink'`` creates a lightweight directory
of symlinks that preserves relative paths without copying fullwiki content.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


@dataclass
class CorpusSubsetManifest:
    corpus_id: str
    source_dir: str
    subset_dir: str
    scale_name: str
    max_docs: int
    selected_documents: int
    selected_bytes: int
    seed: int
    strategy: str = "random_shard"
    materialized: bool = False
    materialize_mode: str = "manifest"
    checksum: str = ""
    document_paths: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> str:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        return str(p)


def create_corpus_subset(
    source_dir: str | Path,
    *,
    output_dir: str | Path,
    scale_name: str,
    max_docs: int,
    seed: int = 42,
    strategy: str = "random_shard",
    materialize: str = "manifest",
    allowed_suffixes: Optional[Iterable[str]] = None,
) -> CorpusSubsetManifest:
    """Create a deterministic corpus subset manifest and optional symlink dir.

    Args:
        source_dir: Full corpus directory.
        output_dir: Directory where manifests/subsets are written.
        scale_name: Human-readable scale label, e.g. ``10k`` or ``fullwiki``.
        max_docs: Max document count. ``0`` means all documents.
        seed: Deterministic sampling seed.
        strategy: ``random_shard`` or ``prefix``.
        materialize: ``manifest`` (default), ``symlink``, or ``copy``.
        allowed_suffixes: Optional file suffix filter. Empty means all files.
    """
    src = Path(source_dir).expanduser().resolve()
    if not src.exists():
        raise FileNotFoundError(f"Corpus source directory not found: {src}")
    out = Path(output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    docs = _list_documents(src, allowed_suffixes=allowed_suffixes)
    selected = _select_documents(docs, max_docs=max_docs, seed=seed, strategy=strategy)
    selected_rel = [str(path.relative_to(src)) for path in selected]
    selected_bytes = sum(_safe_size(path) for path in selected)
    checksum = _checksum(selected_rel)
    subset_dir = out / f"subset_{scale_name}_{checksum[:8]}"
    materialized = False

    if materialize == "symlink":
        _materialize_symlinks(src, subset_dir, selected)
        materialized = True
    elif materialize == "copy":
        _materialize_copies(src, subset_dir, selected)
        materialized = True
    elif materialize != "manifest":
        raise ValueError("materialize must be one of: manifest, symlink, copy")

    manifest = CorpusSubsetManifest(
        corpus_id=f"{src.name}_{scale_name}_{checksum[:12]}",
        source_dir=str(src),
        subset_dir=str(subset_dir if materialized else src),
        scale_name=scale_name,
        max_docs=max_docs,
        selected_documents=len(selected),
        selected_bytes=selected_bytes,
        seed=seed,
        strategy=strategy,
        materialized=materialized,
        materialize_mode=materialize,
        checksum=checksum,
        document_paths=selected_rel,
        metadata={
            "source_document_count": len(docs),
            "scan_truncated": False,
        },
    )
    manifest.save(out / f"corpus_subset_{scale_name}_{checksum[:8]}.json")
    return manifest


def _list_documents(source_dir: Path, *, allowed_suffixes: Optional[Iterable[str]] = None) -> List[Path]:
    suffixes = {s.lower() for s in (allowed_suffixes or []) if s}
    docs: List[Path] = []
    for path in source_dir.rglob("*"):
        if not path.is_file():
            continue
        if suffixes and path.suffix.lower() not in suffixes:
            continue
        docs.append(path)
    return sorted(docs, key=lambda p: str(p.relative_to(source_dir)))


def _select_documents(docs: List[Path], *, max_docs: int, seed: int, strategy: str) -> List[Path]:
    if max_docs <= 0 or max_docs >= len(docs):
        return list(docs)
    if strategy == "prefix":
        return list(docs[:max_docs])
    if strategy != "random_shard":
        raise ValueError("strategy must be one of: random_shard, prefix")
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(docs)), max_docs))
    return [docs[i] for i in indices]


def _materialize_symlinks(source_dir: Path, subset_dir: Path, selected: List[Path]) -> None:
    subset_dir.mkdir(parents=True, exist_ok=True)
    for path in selected:
        rel = path.relative_to(source_dir)
        target = subset_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() or target.is_symlink():
            continue
        os.symlink(path, target)


def _materialize_copies(source_dir: Path, subset_dir: Path, selected: List[Path]) -> None:
    import shutil

    subset_dir.mkdir(parents=True, exist_ok=True)
    for path in selected:
        rel = path.relative_to(source_dir)
        target = subset_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            continue
        shutil.copy2(path, target)


def _checksum(paths: List[str]) -> str:
    raw = json.dumps(paths, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _safe_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


__all__ = ["CorpusSubsetManifest", "create_corpus_subset"]
