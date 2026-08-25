"""HotpotQA raw-corpus synchronization: fingerprint, title index, closure report.

The fullwiki protocol depends on two artifacts staying in sync: the parquet split
that defines the questions and their ``supporting_facts`` titles, and the raw
enwiki dump that must actually contain those articles. Nothing in the parquet
files records which dump they belong to, so a swapped or partial dump would
otherwise only surface late, when a corpus snapshot build fails on an
unresolvable evidence title after the sample IDs were already frozen.

This module makes that dependency explicit and cheap to check:

- ``compute_wiki_fingerprint`` derives a stable identity for a dump from its
  shard inventory, so a swapped dump changes the recorded fingerprint.
- ``build_title_index`` resolves the titles a split actually references to their
  shard files and caches the result under that fingerprint. The index is scoped
  to the referenced title universe (tens of thousands of titles) rather than the
  whole dump (millions of articles), which keeps it small enough to serialize
  and reuse across stage builds.
- ``evaluate_corpus_sync`` turns the index into a blocking closure report that
  sampling and dynamic-stage entry points can gate on before freezing anything.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Set, Tuple

from framework.time_utils import now_local_iso
from hotpotqa.title_resolver import normalize_title

# Article records in the enwiki abstracts dump are one JSON object per line with
# the title field ahead of the body, so the first match is the record title.
_TITLE_PATTERN = re.compile(rb'"title"\s*:\s*"((?:[^"\\]|\\.)*)"')
_INDEX_SCHEMA_VERSION = 2
_INDEX_FILENAME_PREFIX = "hotpotqa_title_index"


@dataclass
class WikiCorpusFingerprint:
    """Stable identity of a raw wiki dump, derived from its shard inventory.

    The fingerprint hashes relative shard paths and sizes rather than file
    contents: it is computed from ``stat`` calls only, so it stays cheap enough
    to run on every entry point while still changing when the dump is replaced,
    truncated, or extended.
    """

    wiki_dir: str
    shard_count: int
    total_bytes: int
    fingerprint: str
    computed_at: str = field(default_factory=now_local_iso)
    fingerprint_scope: str = "shard_inventory"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def short(self) -> str:
        return self.fingerprint[:12]


@dataclass
class WikiTitleIndex:
    """Mapping from referenced article titles to the shard that contains them.

    ``missing_titles`` records the in-scope titles the dump does not contain. It
    is what makes the index authoritative for subset queries: a consumer asking
    about fewer titles can reuse a wider index because an absent title is known
    to be genuinely absent rather than merely out of scope.
    """

    wiki_dir: str
    wiki_fingerprint: str
    title_scope_checksum: str
    title_to_shard: Dict[str, str] = field(default_factory=dict)
    missing_titles: List[str] = field(default_factory=list)
    requested_title_count: int = 0
    scanned_record_count: int = 0
    scanned_shard_count: int = 0
    schema_version: int = _INDEX_SCHEMA_VERSION
    built_at: str = field(default_factory=now_local_iso)
    cache_path: str = ""

    def contains(self, title: str) -> bool:
        return normalize_title(title) in self.title_to_shard

    def shard_for(self, title: str) -> str:
        return self.title_to_shard.get(normalize_title(title), "")

    def absolute_shard_for(self, title: str) -> Path | None:
        relative = self.shard_for(title)
        return (Path(self.wiki_dir) / relative) if relative else None

    def covers(self, titles: Iterable[str]) -> bool:
        """Whether this index can answer every requested title authoritatively."""
        known = set(self.title_to_shard) | set(self.missing_titles)
        for title in titles:
            normalized = normalize_title(str(title))
            if normalized and normalized not in known:
                return False
        return True

    def missing(self, titles: Iterable[str]) -> List[str]:
        """Return the input titles that this index could not locate."""
        missing: List[str] = []
        seen: Set[str] = set()
        for title in titles:
            normalized = normalize_title(str(title))
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            if normalized not in self.title_to_shard:
                missing.append(str(title))
        return missing

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> str:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path = str(p)
        p.write_text(json.dumps(self.to_dict(), ensure_ascii=False), encoding="utf-8")
        return str(p)

    @classmethod
    def load(cls, path: str | Path) -> "WikiTitleIndex":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if int(data.get("schema_version", 0)) != _INDEX_SCHEMA_VERSION:
            raise ValueError(f"Unsupported title index schema at {path}")
        return cls(
            wiki_dir=str(data["wiki_dir"]),
            wiki_fingerprint=str(data["wiki_fingerprint"]),
            title_scope_checksum=str(data["title_scope_checksum"]),
            title_to_shard={str(k): str(v) for k, v in (data.get("title_to_shard") or {}).items()},
            missing_titles=[str(item) for item in (data.get("missing_titles") or [])],
            requested_title_count=int(data.get("requested_title_count", 0) or 0),
            scanned_record_count=int(data.get("scanned_record_count", 0) or 0),
            scanned_shard_count=int(data.get("scanned_shard_count", 0) or 0),
            schema_version=int(data.get("schema_version", _INDEX_SCHEMA_VERSION)),
            built_at=str(data.get("built_at", "")),
            cache_path=str(path),
        )


@dataclass
class CorpusSyncReport:
    """Closure of a split's referenced titles against a raw wiki dump."""

    wiki_dir: str
    wiki_fingerprint: str
    shard_count: int
    setting: str = ""
    split: str = ""
    dataset_checksum: str = ""
    question_count: int = 0
    evidence_title_count: int = 0
    resolved_evidence_title_count: int = 0
    missing_evidence_titles: List[str] = field(default_factory=list)
    context_title_count: int = 0
    resolved_context_title_count: int = 0
    missing_context_titles: List[str] = field(default_factory=list)
    unresolvable_sample_ids: List[str] = field(default_factory=list)
    title_index_path: str = ""
    title_index_cache_hit: bool = False
    checked_at: str = field(default_factory=now_local_iso)

    @property
    def evidence_title_closure(self) -> float:
        if not self.evidence_title_count:
            return 0.0
        return round(self.resolved_evidence_title_count / self.evidence_title_count * 100, 4)

    @property
    def context_title_closure(self) -> float:
        if not self.context_title_count:
            return 0.0
        return round(self.resolved_context_title_count / self.context_title_count * 100, 4)

    @property
    def question_closure(self) -> float:
        if not self.question_count:
            return 0.0
        resolvable = self.question_count - len(self.unresolvable_sample_ids)
        return round(resolvable / self.question_count * 100, 4)

    @property
    def passed(self) -> bool:
        """Evidence closure is blocking; distractor context closure is not.

        A question whose supporting-fact article is absent cannot be answered
        from the snapshot at all, so it invalidates the protocol. A missing
        context distractor only makes the snapshot slightly easier and is
        reported without blocking.
        """
        return not self.missing_evidence_titles and self.question_count > 0

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data.update({
            "evidence_title_closure": self.evidence_title_closure,
            "context_title_closure": self.context_title_closure,
            "question_closure": self.question_closure,
            "unresolvable_question_count": len(self.unresolvable_sample_ids),
            "passed": self.passed,
        })
        return data

    def summary_line(self) -> str:
        return (
            f"evidence {self.resolved_evidence_title_count}/{self.evidence_title_count} "
            f"({self.evidence_title_closure}%), questions {self.question_closure}% resolvable, "
            f"wiki={self.wiki_fingerprint[:12]} shards={self.shard_count}"
        )


def compute_wiki_fingerprint(wiki_dir: str | Path) -> WikiCorpusFingerprint:
    """Fingerprint a raw wiki dump from its shard inventory."""
    root = Path(wiki_dir).expanduser().resolve()
    shards = list_wiki_shards(root)
    hasher = hashlib.sha256()
    total_bytes = 0
    for path in shards:
        try:
            size = path.stat().st_size
        except OSError:
            size = -1
        total_bytes += max(size, 0)
        hasher.update(f"{path.relative_to(root)}:{size}\n".encode("utf-8"))
    return WikiCorpusFingerprint(
        wiki_dir=str(root),
        shard_count=len(shards),
        total_bytes=total_bytes,
        fingerprint=hasher.hexdigest()[:32] if shards else "",
    )


def list_wiki_shards(wiki_dir: str | Path) -> List[Path]:
    """Return dump shard files in a deterministic order, skipping dotfiles.

    macOS metadata files such as ``.DS_Store`` and AppleDouble ``._*`` siblings
    are excluded so the inventory stays identical across machines.
    """
    root = Path(wiki_dir).expanduser().resolve()
    if not root.exists():
        return []
    shards = [
        path for path in root.rglob("*")
        if path.is_file() and not path.name.startswith(".")
    ]
    return sorted(shards, key=lambda p: str(p.relative_to(root)))


def title_scope_checksum(titles: Iterable[str]) -> str:
    """Order-insensitive checksum of a referenced title universe."""
    normalized = sorted({normalize_title(str(title)) for title in titles} - {""})
    raw = json.dumps(normalized, ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


def _index_cache_path(cache_dir: str | Path, fingerprint: WikiCorpusFingerprint, scope: str) -> Path:
    return Path(cache_dir).expanduser().resolve() / (
        f"{_INDEX_FILENAME_PREFIX}_{fingerprint.short()}_{scope[:12]}.json"
    )


def load_covering_index(
    wiki_dir: str | Path,
    titles: Iterable[str],
    *,
    cache_dir: str | Path | None,
    fingerprint: WikiCorpusFingerprint | None = None,
) -> WikiTitleIndex | None:
    """Return a cached index that authoritatively answers ``titles``, if any.

    Read-only: callers that merely consume the mapping (such as snapshot title
    resolution) use this instead of ``build_title_index`` so a narrow query can
    reuse a wider cached index rather than triggering a fresh dump scan.
    """
    if not cache_dir:
        return None
    root = Path(cache_dir).expanduser().resolve()
    if not root.exists():
        return None
    dump_fingerprint = fingerprint or compute_wiki_fingerprint(wiki_dir)
    wanted = [normalize_title(str(title)) for title in titles]
    wanted = [title for title in wanted if title]
    if not wanted:
        return None

    exact = _index_cache_path(root, dump_fingerprint, title_scope_checksum(wanted))
    candidates: List[Path] = [exact] if exact.exists() else []
    candidates.extend(
        path for path in sorted(root.glob(f"{_INDEX_FILENAME_PREFIX}_{dump_fingerprint.short()}_*.json"))
        if path != exact
    )
    best: WikiTitleIndex | None = None
    for path in candidates:
        try:
            index = WikiTitleIndex.load(path)
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            continue
        if index.wiki_fingerprint != dump_fingerprint.fingerprint:
            continue
        if not index.covers(wanted):
            continue
        if best is None or index.requested_title_count > best.requested_title_count:
            best = index
    return best


def build_title_index(
    wiki_dir: str | Path,
    titles: Iterable[str],
    *,
    cache_dir: str | Path | None = None,
    fingerprint: WikiCorpusFingerprint | None = None,
    force_rebuild: bool = False,
) -> Tuple[WikiTitleIndex, bool]:
    """Locate each referenced title in the dump, caching by dump fingerprint.

    Returns ``(index, cache_hit)``. The cache key combines the dump fingerprint
    with the requested title scope, so replacing the dump or changing the split
    invalidates the cache instead of silently reusing a stale mapping. A cached
    index built for a wider scope is reused when it already covers the request.
    """
    root = Path(wiki_dir).expanduser().resolve()
    dump_fingerprint = fingerprint or compute_wiki_fingerprint(root)
    wanted: Dict[str, str] = {}
    for title in titles:
        normalized = normalize_title(str(title))
        if normalized:
            wanted.setdefault(normalized, str(title))
    scope = title_scope_checksum(wanted.keys())

    if cache_dir and not force_rebuild:
        cached = load_covering_index(
            root, wanted.keys(), cache_dir=cache_dir, fingerprint=dump_fingerprint,
        )
        if cached is not None:
            return cached, True

    shards = list_wiki_shards(root)
    remaining = set(wanted)
    title_to_shard: Dict[str, str] = {}
    scanned_records = 0
    for path in shards:
        if not remaining:
            break
        relative = str(path.relative_to(root))
        try:
            with path.open("rb") as handle:
                for line in handle:
                    match = _TITLE_PATTERN.search(line)
                    if match is None:
                        continue
                    scanned_records += 1
                    normalized = normalize_title(_decode_title(match.group(1)))
                    if normalized in remaining:
                        title_to_shard[normalized] = relative
                        remaining.discard(normalized)
        except OSError:
            continue

    index = WikiTitleIndex(
        wiki_dir=str(root),
        wiki_fingerprint=dump_fingerprint.fingerprint,
        title_scope_checksum=scope,
        title_to_shard=title_to_shard,
        missing_titles=sorted(remaining),
        requested_title_count=len(wanted),
        scanned_record_count=scanned_records,
        scanned_shard_count=len(shards),
    )
    if cache_dir:
        try:
            index.save(_index_cache_path(cache_dir, dump_fingerprint, scope))
        except OSError:
            pass
    return index, False


def collect_referenced_titles(samples: Sequence[Any]) -> Tuple[List[str], List[str]]:
    """Return ``(evidence_titles, context_titles)`` referenced by the samples."""
    evidence: Dict[str, str] = {}
    context: Dict[str, str] = {}
    for sample in samples:
        metadata = _metadata_of(sample)
        for raw in _iter_titles(metadata.get("supporting_facts")):
            normalized = normalize_title(raw)
            if normalized:
                evidence.setdefault(normalized, raw)
        for raw in _iter_titles(metadata.get("context")):
            normalized = normalize_title(raw)
            if normalized:
                context.setdefault(normalized, raw)
    context_only = {key: value for key, value in context.items() if key not in evidence}
    return sorted(evidence.values()), sorted(context_only.values())


def evaluate_corpus_sync(
    samples: Sequence[Any],
    wiki_dir: str | Path,
    *,
    cache_dir: str | Path | None = None,
    setting: str = "",
    split: str = "",
    dataset_checksum: str = "",
    force_rebuild: bool = False,
) -> CorpusSyncReport:
    """Check that every referenced article of ``samples`` exists in the dump."""
    fingerprint = compute_wiki_fingerprint(wiki_dir)
    evidence_titles, context_titles = collect_referenced_titles(samples)
    index, cache_hit = build_title_index(
        wiki_dir,
        list(evidence_titles) + list(context_titles),
        cache_dir=cache_dir,
        fingerprint=fingerprint,
        force_rebuild=force_rebuild,
    )

    missing_evidence = index.missing(evidence_titles)
    missing_context = index.missing(context_titles)
    missing_evidence_keys = {normalize_title(title) for title in missing_evidence}
    unresolvable: List[str] = []
    for sample in samples:
        metadata = _metadata_of(sample)
        titles = {normalize_title(raw) for raw in _iter_titles(metadata.get("supporting_facts"))}
        if titles & missing_evidence_keys:
            unresolvable.append(str(_sample_id_of(sample)))

    return CorpusSyncReport(
        wiki_dir=str(Path(wiki_dir).expanduser().resolve()),
        wiki_fingerprint=fingerprint.fingerprint,
        shard_count=fingerprint.shard_count,
        setting=setting,
        split=split,
        dataset_checksum=dataset_checksum,
        question_count=len(samples),
        evidence_title_count=len(evidence_titles),
        resolved_evidence_title_count=len(evidence_titles) - len(missing_evidence),
        missing_evidence_titles=missing_evidence[:200],
        context_title_count=len(context_titles),
        resolved_context_title_count=len(context_titles) - len(missing_context),
        missing_context_titles=missing_context[:200],
        unresolvable_sample_ids=unresolvable[:200],
        title_index_path=index.cache_path,
        title_index_cache_hit=cache_hit,
    )


def _decode_title(raw: bytes) -> str:
    text = raw.decode("utf-8", "ignore")
    if "\\" not in text:
        return text
    try:
        return json.loads(f'"{text}"')
    except json.JSONDecodeError:
        return text


def _metadata_of(sample: Any) -> Dict[str, Any]:
    if isinstance(sample, dict):
        metadata = sample.get("metadata")
        return dict(metadata) if isinstance(metadata, dict) else dict(sample)
    metadata = getattr(sample, "metadata", None)
    return dict(metadata) if isinstance(metadata, dict) else {}


def _sample_id_of(sample: Any) -> str:
    if isinstance(sample, dict):
        return str(sample.get("sample_id", sample.get("id", "")))
    return str(getattr(sample, "sample_id", ""))


def _iter_titles(value: Any) -> List[str]:
    """Extract article titles from HotpotQA supporting_facts or context fields."""
    if value is None:
        return []
    if hasattr(value, "tolist"):
        try:
            value = value.tolist()
        except Exception:
            pass
    if isinstance(value, dict):
        titles = value.get("title") or value.get("titles") or []
        if isinstance(titles, str):
            return [titles]
        if hasattr(titles, "tolist"):
            try:
                titles = titles.tolist()
            except Exception:
                titles = []
        return [str(item) for item in titles or []]
    if isinstance(value, (list, tuple)):
        out: List[str] = []
        for item in value:
            if isinstance(item, dict):
                title = item.get("title") or item.get("doc_title") or item.get("page")
            elif isinstance(item, (list, tuple)) and item:
                title = item[0]
            else:
                title = item
            if title is not None:
                out.append(str(title))
        return out
    return []


__all__ = [
    "CorpusSyncReport",
    "WikiCorpusFingerprint",
    "WikiTitleIndex",
    "build_title_index",
    "collect_referenced_titles",
    "compute_wiki_fingerprint",
    "evaluate_corpus_sync",
    "list_wiki_shards",
    "load_covering_index",
    "title_scope_checksum",
]
