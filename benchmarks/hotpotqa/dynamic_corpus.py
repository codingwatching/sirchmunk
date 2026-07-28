"""Dynamic G_n/D_n corpus snapshot utilities for HotpotQA experiments."""
from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Set

from evaluation.sampling_protocol import compute_sample_id_checksum, write_sample_ids
from framework.time_utils import now_local_iso
from hotpotqa.title_resolver import HotpotQATitleResolver, normalize_title, safe_title_filename


@dataclass
class NestedSampleManifest:
    """Lineage manifest for G_500/G_1000/G_2000 derived from one parent set."""

    parent_sample_count: int
    parent_sample_id_checksum: str
    parent_frozen_order_checksum: str
    stages: List[Dict[str, Any]] = field(default_factory=list)
    created_at: str = field(default_factory=now_local_iso)
    manifest_version: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CorpusSnapshotManifest:
    """Auditable manifest for one D_n raw-corpus snapshot."""

    stage_name: str
    sample_set_id: str
    sample_count: int
    sample_ids: List[str]
    sample_id_checksum: str
    frozen_order_checksum: str
    source_wiki_corpus_dir: str
    snapshot_dir: str
    search_corpus_dir: str
    materialize_mode: str
    article_count: int
    evidence_article_count: int
    context_distractor_count: int
    background_article_count: int
    selected_article_titles: List[str]
    selected_document_paths: List[str]
    missing_evidence_titles: List[str]
    corpus_checksum: str
    title_resolution_report: Dict[str, Any] = field(default_factory=dict)
    background_selection_manifest: Dict[str, Any] = field(default_factory=dict)
    validation: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=now_local_iso)
    manifest_version: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self, snapshot_dir: str | Path) -> str:
        out = Path(snapshot_dir)
        out.mkdir(parents=True, exist_ok=True)
        path = out / "corpus_snapshot_manifest.json"
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        return str(path)


def compute_frozen_order_checksum(sample_ids: Iterable[str]) -> str:
    raw = json.dumps([str(sample_id) for sample_id in sample_ids], ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


def derive_nested_sample_sets(
    golden_set: Any,
    *,
    stages: Sequence[int] = (500, 1000, 2000),
    output_dir: str | Path,
) -> NestedSampleManifest:
    """Derive nested G_n sample-id files from a parent GoldenSet order."""
    samples = _samples_from(golden_set)
    sample_ids = [str(_sample_id(sample)) for sample in samples]
    out = Path(output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    parent_checksum = compute_sample_id_checksum(sample_ids)
    parent_order_checksum = compute_frozen_order_checksum(sample_ids)
    manifest = NestedSampleManifest(
        parent_sample_count=len(sample_ids),
        parent_sample_id_checksum=parent_checksum,
        parent_frozen_order_checksum=parent_order_checksum,
    )

    for stage_n in stages:
        stage_n = int(stage_n)
        if stage_n <= 0 or stage_n > len(sample_ids):
            raise ValueError(f"Invalid stage size {stage_n}; parent has {len(sample_ids)} samples")
        stage_ids = sample_ids[:stage_n]
        stage_name = f"G_{stage_n}"
        path = write_sample_ids(
            out / f"{stage_name}_sample_ids.json",
            stage_ids,
            metadata={
                "stage_name": stage_name,
                "parent_sample_id_checksum": parent_checksum,
                "parent_frozen_order_checksum": parent_order_checksum,
                "frozen_order_checksum": compute_frozen_order_checksum(stage_ids),
                "prefix_of_parent": True,
            },
        )
        manifest.stages.append({
            "stage_name": stage_name,
            "stage_n": stage_n,
            "sample_ids_file": path,
            "sample_id_checksum": compute_sample_id_checksum(stage_ids),
            "frozen_order_checksum": compute_frozen_order_checksum(stage_ids),
            "prefix_of_parent": True,
        })

    (out / "nested_sample_manifest.json").write_text(
        json.dumps(manifest.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return manifest


def build_dynamic_corpus_snapshot(
    samples: Sequence[Any],
    *,
    sample_ids: Sequence[str],
    wiki_dir: str | Path,
    output_dir: str | Path,
    stage_name: str,
    materialize_mode: str = "symlink",
    background_ratio: float = 3.0,
    background_seed: int = 42,
    resolver: HotpotQATitleResolver | None = None,
    strict_evidence: bool = True,
) -> CorpusSnapshotManifest:
    """Build one D_n snapshot aligned with a frozen G_n sample-id set."""
    selected = _select_samples_by_ids(samples, sample_ids)
    selected_ids = [str(_sample_id(sample)) for sample in selected]
    snapshot_dir = Path(output_dir).expanduser().resolve()
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    evidence_titles: Set[str] = set()
    context_titles: Set[str] = set()
    for sample in selected:
        metadata = _metadata(sample)
        evidence_titles.update(_supporting_titles(metadata.get("supporting_facts", [])))
        context_titles.update(_context_titles(metadata.get("context")))
    context_distractors = context_titles - evidence_titles

    target_titles = sorted(evidence_titles | context_distractors)
    wiki_path = Path(wiki_dir).expanduser().resolve()
    title_resolver = resolver or HotpotQATitleResolver(wiki_path)
    resolution_dir = snapshot_dir / "_resolved_articles" if materialize_mode != "manifest" else None
    resolution = title_resolver.resolve_many(target_titles, materialized_dir=resolution_dir)

    evidence_resolved = [resolution.get(title) for title in sorted(evidence_titles)]
    context_resolved = [resolution.get(title) for title in sorted(context_distractors)]
    missing_evidence = [r.title for r in evidence_resolved if r is not None and not r.resolved]

    selected_sources: List[tuple[str, Path, str]] = []
    resolved_source_paths: Set[str] = set()
    for title in sorted(evidence_titles):
        result = resolution.get(title)
        if result and result.resolved and result.path_for_snapshot():
            selected_sources.append((result.title, Path(result.path_for_snapshot()), "evidence"))
            if result.source_path:
                resolved_source_paths.add(str(Path(result.source_path).expanduser().resolve()))
    for title in sorted(context_distractors):
        result = resolution.get(title)
        if result and result.resolved and result.path_for_snapshot():
            selected_sources.append((result.title, Path(result.path_for_snapshot()), "context_distractor"))
            if result.source_path:
                resolved_source_paths.add(str(Path(result.source_path).expanduser().resolve()))

    existing_paths = {str(path.resolve()) for _, path, _ in selected_sources if path.exists()}
    existing_paths.update(resolved_source_paths)
    background_count = max(0, int(round(len(selected_sources) * max(background_ratio, 0.0))))
    background_sources = _select_background_documents(
        title_resolver.candidate_files(),
        exclude_paths=existing_paths,
        count=background_count,
        seed=background_seed,
    )
    for path in background_sources:
        selected_sources.append((path.stem or path.name, path, "background"))

    materialized_paths: List[str] = []
    document_records: List[Dict[str, Any]] = []
    selected_titles: List[str] = []
    docs_dir = snapshot_dir / "documents"
    search_corpus_dir = docs_dir if materialize_mode != "manifest" else snapshot_dir
    if materialize_mode != "manifest":
        docs_dir.mkdir(parents=True, exist_ok=True)
    for title, source, role in selected_sources:
        if not source.exists():
            continue
        selected_titles.append(title)
        materialized = _materialize_source(source, docs_dir, title, role, materialize_mode)
        materialized_paths.append(str(materialized))
        document_records.append(_document_record(
            title=title,
            role=role,
            source=source,
            materialized=materialized,
            snapshot_dir=snapshot_dir,
            wiki_dir=wiki_path,
        ))

    corpus_checksum = _checksum_payload({"documents": _checksum_documents(document_records)})

    title_report = {
        "resolved": [r.to_dict() for r in resolution.values() if r.resolved],
        "missing": [r.to_dict() for r in resolution.values() if not r.resolved],
        "requested_title_count": len(target_titles),
        "resolved_title_count": sum(1 for r in resolution.values() if r.resolved),
    }
    background_manifest = {
        "seed": background_seed,
        "background_ratio": background_ratio,
        "requested_background_count": background_count,
        "selected_background_count": len(background_sources),
        "selected_background_paths": [str(p) for p in background_sources],
        "selection_strategy": "deterministic_shuffled_prefix",
    }
    validation = {
        "evidence_articles_fully_resolved": not missing_evidence,
        "missing_evidence_titles": missing_evidence,
        "accepted_for_main_table": not missing_evidence,
        "blocking_errors": ["missing_evidence_titles"] if missing_evidence else [],
    }

    (snapshot_dir / "title_resolution_report.json").write_text(
        json.dumps(title_report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (snapshot_dir / "background_selection_manifest.json").write_text(
        json.dumps(background_manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (snapshot_dir / "selected_article_titles.json").write_text(
        json.dumps(selected_titles, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (snapshot_dir / "selected_document_paths.json").write_text(
        json.dumps(materialized_paths, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (snapshot_dir / "document_checksum_records.json").write_text(
        json.dumps(document_records, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (snapshot_dir / "validation_manifest.json").write_text(
        json.dumps(validation, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    if strict_evidence and missing_evidence:
        raise ValueError(
            f"Evidence articles not fully resolved for {stage_name}: "
            f"{missing_evidence[:10]} total_missing={len(missing_evidence)}"
        )
    write_sample_ids(
        snapshot_dir / "sample_ids.json",
        selected_ids,
        metadata={"stage_name": stage_name, "corpus_checksum": corpus_checksum},
    )

    manifest = CorpusSnapshotManifest(
        stage_name=stage_name,
        sample_set_id=stage_name.replace("D_", "G_"),
        sample_count=len(selected_ids),
        sample_ids=selected_ids,
        sample_id_checksum=compute_sample_id_checksum(selected_ids),
        frozen_order_checksum=compute_frozen_order_checksum(selected_ids),
        source_wiki_corpus_dir=str(wiki_path),
        snapshot_dir=str(snapshot_dir),
        search_corpus_dir=str(search_corpus_dir),
        materialize_mode=materialize_mode,
        article_count=len(materialized_paths),
        evidence_article_count=sum(1 for _, _, role in selected_sources if role == "evidence"),
        context_distractor_count=sum(1 for _, _, role in selected_sources if role == "context_distractor"),
        background_article_count=len(background_sources),
        selected_article_titles=selected_titles,
        selected_document_paths=materialized_paths,
        missing_evidence_titles=missing_evidence,
        corpus_checksum=corpus_checksum,
        title_resolution_report=title_report,
        background_selection_manifest=background_manifest,
        validation=validation,
    )
    manifest.save(snapshot_dir)
    (snapshot_dir / "stage_manifest.json").write_text(
        json.dumps({
            "stage_name": stage_name,
            "sample_set_id": manifest.sample_set_id,
            "sample_id_checksum": manifest.sample_id_checksum,
            "frozen_order_checksum": manifest.frozen_order_checksum,
            "corpus_checksum": manifest.corpus_checksum,
            "snapshot_dir": manifest.snapshot_dir,
            "search_corpus_dir": manifest.search_corpus_dir,
            "validation": manifest.validation,
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return manifest


def selected_document_paths_for_checksum(paths: Sequence[str]) -> List[str]:
    return sorted(str(path) for path in paths)


def _samples_from(golden_set: Any) -> List[Any]:
    if hasattr(golden_set, "to_benchmark_samples"):
        return list(golden_set.to_benchmark_samples())
    if hasattr(golden_set, "samples"):
        return list(getattr(golden_set, "samples") or [])
    return list(golden_set or [])


def _select_samples_by_ids(samples: Sequence[Any], sample_ids: Sequence[str]) -> List[Any]:
    by_id = {str(_sample_id(sample)): sample for sample in samples}
    missing = [str(sample_id) for sample_id in sample_ids if str(sample_id) not in by_id]
    if missing:
        raise ValueError(f"sample_ids not found in parent samples: {missing[:10]} total_missing={len(missing)}")
    return [by_id[str(sample_id)] for sample_id in sample_ids]


def _sample_id(sample: Any) -> str:
    if isinstance(sample, dict):
        return str(sample.get("sample_id", sample.get("id", "")))
    return str(getattr(sample, "sample_id", ""))


def _metadata(sample: Any) -> Dict[str, Any]:
    if isinstance(sample, dict):
        return dict(sample.get("metadata", {}) or {})
    return dict(getattr(sample, "metadata", {}) or {})


def _supporting_titles(supporting_facts: Any) -> Set[str]:
    titles: Set[str] = set()
    if supporting_facts is None:
        return titles
    if isinstance(supporting_facts, dict):
        values = supporting_facts.get("title") or supporting_facts.get("titles") or []
        if isinstance(values, str):
            values = [values]
        for value in values or []:
            normalized = normalize_title(str(value))
            if normalized:
                titles.add(normalized)
        return titles
    if hasattr(supporting_facts, "tolist"):
        try:
            supporting_facts = supporting_facts.tolist()
        except Exception:
            pass
    if isinstance(supporting_facts, (list, tuple, set)):
        for item in supporting_facts:
            title = None
            if isinstance(item, dict):
                title = item.get("title") or item.get("doc_title") or item.get("page")
            elif isinstance(item, (list, tuple)) and item:
                title = item[0]
            else:
                title = item
            normalized = normalize_title(str(title)) if title is not None else ""
            if normalized:
                titles.add(normalized)
    return titles


def _context_titles(context: Any) -> Set[str]:
    titles: Set[str] = set()
    if context is None:
        return titles
    if hasattr(context, "tolist"):
        try:
            context = context.tolist()
        except Exception:
            pass
    if isinstance(context, dict):
        values = context.get("title") or context.get("titles") or []
        if isinstance(values, str):
            values = [values]
        for value in values or []:
            normalized = normalize_title(str(value))
            if normalized:
                titles.add(normalized)
        return titles
    if isinstance(context, (list, tuple)):
        for item in context:
            title = None
            if isinstance(item, dict):
                title = item.get("title") or item.get("doc_title") or item.get("page")
            elif isinstance(item, (list, tuple)) and item:
                title = item[0]
            normalized = normalize_title(str(title)) if title is not None else ""
            if normalized:
                titles.add(normalized)
    return titles


def _select_background_documents(
    candidates: Sequence[Path],
    *,
    exclude_paths: Set[str],
    count: int,
    seed: int,
) -> List[Path]:
    pool = [p for p in candidates if str(p.resolve()) not in exclude_paths]
    if count <= 0 or not pool:
        return []
    rng = random.Random(seed)
    shuffled = list(pool)
    rng.shuffle(shuffled)
    # Taking a deterministic shuffled prefix makes larger stage selections
    # monotonic for the same candidate universe and seed.
    return shuffled[: min(count, len(shuffled))]


def _materialize_source(source: Path, docs_dir: Path, title: str, role: str, materialize_mode: str) -> Path:
    if materialize_mode == "manifest":
        return source
    suffix = source.suffix or ".txt"
    digest = hashlib.sha256(str(source.resolve()).encode("utf-8")).hexdigest()[:8]
    target = docs_dir / role / f"{safe_title_filename(title)}_{digest}{suffix}"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        return target
    if materialize_mode == "copy":
        shutil.copy2(source, target)
    elif materialize_mode == "symlink":
        os.symlink(source, target)
    else:
        raise ValueError("materialize_mode must be one of: symlink, copy, manifest")
    return target


def _document_record(
    *,
    title: str,
    role: str,
    source: Path,
    materialized: Path,
    snapshot_dir: Path,
    wiki_dir: Path,
) -> Dict[str, Any]:
    return {
        "title": title,
        "role": role,
        "source_path": str(source),
        "source_relative_path": _relative_to(source, wiki_dir),
        "materialized_path": str(materialized),
        "snapshot_relative_path": _relative_to(materialized, snapshot_dir),
        "content_sha256": _file_sha256(materialized),
    }


def _checksum_documents(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(
        [
            {
                "snapshot_relative_path": str(record.get("snapshot_relative_path", "")),
                "source_relative_path": str(record.get("source_relative_path", "")),
                "content_sha256": str(record.get("content_sha256", "")),
            }
            for record in records
        ],
        key=lambda item: (item["snapshot_relative_path"], item["source_relative_path"]),
    )


def _relative_to(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except Exception:
        return path.name


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    try:
        with path.open("rb") as fp:
            for chunk in iter(lambda: fp.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return ""


def _checksum_payload(payload: Dict[str, Any]) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


__all__ = [
    "CorpusSnapshotManifest",
    "NestedSampleManifest",
    "build_dynamic_corpus_snapshot",
    "compute_frozen_order_checksum",
    "derive_nested_sample_sets",
]
