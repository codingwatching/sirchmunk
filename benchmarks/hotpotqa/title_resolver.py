"""HotpotQA raw wiki title resolver for dynamic corpus snapshots.

The resolver is intentionally conservative: it first tries article-level file
names, then scans JSON/JSONL or extensionless shard files for article records.
When a title is found inside a shard, the caller may ask the resolver to
materialize a standalone article text file so downstream systems can search a
bounded D_n snapshot instead of the whole shard.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set


_TEXT_KEYS = ("text", "content", "abstract", "sentences", "paragraphs")
_TITLE_KEYS = ("title", "doc_title", "page", "name")
_SCAN_SUFFIXES = {"", ".jsonl", ".json", ".txt", ".tsv"}


@dataclass
class TitleResolutionResult:
    """Resolution outcome for one HotpotQA article title."""

    title: str
    normalized_title: str
    resolved: bool = False
    method: str = "failed"
    confidence: float = 0.0
    source_path: str = ""
    materialized_path: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def path_for_snapshot(self) -> str:
        return self.materialized_path or self.source_path

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class HotpotQATitleResolver:
    """Resolve HotpotQA supporting/context titles against a raw wiki corpus."""

    def __init__(
        self,
        wiki_dir: str | Path,
        *,
        max_json_line_chars: int = 2_000_000,
        index_cache_dir: str | Path | None = None,
    ) -> None:
        self.wiki_dir = Path(wiki_dir).expanduser().resolve()
        self.max_json_line_chars = max_json_line_chars
        self.index_cache_dir = str(index_cache_dir) if index_cache_dir else ""
        self._file_index: Optional[Dict[str, Path]] = None
        self._candidate_files: Optional[List[Path]] = None

    def resolve(self, title: str, *, materialized_dir: str | Path | None = None) -> TitleResolutionResult:
        return self.resolve_many([title], materialized_dir=materialized_dir).get(
            normalize_title(title),
            TitleResolutionResult(title=title, normalized_title=normalize_title(title)),
        )

    def resolve_many(
        self,
        titles: Iterable[str],
        *,
        materialized_dir: str | Path | None = None,
    ) -> Dict[str, TitleResolutionResult]:
        wanted: Dict[str, str] = {}
        for title in titles:
            normalized = normalize_title(str(title))
            if normalized:
                wanted.setdefault(normalized, str(title))

        results: Dict[str, TitleResolutionResult] = {}
        if not wanted:
            return results

        direct_index = self._build_file_index()
        unresolved: Set[str] = set(wanted)
        for normalized in list(unresolved):
            source = direct_index.get(normalized)
            if source is not None:
                results[normalized] = TitleResolutionResult(
                    title=wanted[normalized],
                    normalized_title=normalized,
                    resolved=True,
                    method="file_name_exact",
                    confidence=1.0,
                    source_path=str(source),
                )
                unresolved.remove(normalized)

        if unresolved:
            self._scan_shards_for_titles(
                wanted=wanted,
                unresolved=unresolved,
                results=results,
                materialized_dir=Path(materialized_dir).expanduser().resolve() if materialized_dir else None,
                shard_hint=self._shard_hint(unresolved),
            )

        for normalized in unresolved:
            results.setdefault(
                normalized,
                TitleResolutionResult(title=wanted[normalized], normalized_title=normalized),
            )
        return results

    def candidate_files(self) -> List[Path]:
        if self._candidate_files is not None:
            return self._candidate_files
        if not self.wiki_dir.exists():
            self._candidate_files = []
            return self._candidate_files
        self._candidate_files = sorted(
            [p for p in self.wiki_dir.rglob("*") if p.is_file() and not p.name.startswith(".")],
            key=lambda p: str(p.relative_to(self.wiki_dir)),
        )
        return self._candidate_files

    def _shard_hint(self, unresolved: Set[str]) -> Dict[str, Path]:
        """Map still-unresolved titles to their shard using a cached corpus index.

        Without a hint the resolver has to stream every shard until each title is
        found, which dominates snapshot build time on a full dump. This only
        consumes an index that some earlier sync check already produced; it never
        builds one, so a narrow resolve call cannot trigger a full dump scan.
        """
        if not self.index_cache_dir or not unresolved:
            return {}
        try:
            from hotpotqa.corpus_index import load_covering_index

            index = load_covering_index(
                self.wiki_dir,
                sorted(unresolved),
                cache_dir=self.index_cache_dir,
            )
        except Exception:
            return {}
        if index is None:
            return {}
        hint: Dict[str, Path] = {}
        for normalized in unresolved:
            shard = index.absolute_shard_for(normalized)
            if shard is not None:
                hint[normalized] = shard
        return hint

    def _build_file_index(self) -> Dict[str, Path]:
        if self._file_index is not None:
            return self._file_index
        index: Dict[str, Path] = {}
        for path in self.candidate_files():
            candidates = {path.stem, path.name}
            if path.suffix:
                candidates.add(path.name[: -len(path.suffix)])
            for candidate in candidates:
                normalized = normalize_title(candidate)
                if normalized and normalized not in index:
                    index[normalized] = path
        self._file_index = index
        return index

    def _scan_shards_for_titles(
        self,
        *,
        wanted: Dict[str, str],
        unresolved: Set[str],
        results: Dict[str, TitleResolutionResult],
        materialized_dir: Optional[Path],
        shard_hint: Optional[Dict[str, Path]] = None,
    ) -> None:
        if materialized_dir is not None:
            materialized_dir.mkdir(parents=True, exist_ok=True)
        hinted_shards: List[Path] = []
        if shard_hint:
            seen: Set[str] = set()
            for path in shard_hint.values():
                key = str(path)
                if key not in seen:
                    seen.add(key)
                    hinted_shards.append(path)
            # shard_hint is assembled from set-ordered lookups, so its value
            # order is hash-randomized per process. Titles that occur in more
            # than one shard resolve to whichever shard is scanned first, so
            # the scan order must be deterministic for snapshots (and their
            # checksums) to be reproducible across runs and machines.
            hinted_shards.sort(key=str)
        scan_order = hinted_shards + [
            path for path in self.candidate_files() if path not in set(hinted_shards)
        ] if hinted_shards else self.candidate_files()
        for path in scan_order:
            if not unresolved:
                return
            if path.suffix.lower() not in _SCAN_SUFFIXES:
                continue
            for record_idx, record in enumerate(_iter_article_records(path, self.max_json_line_chars)):
                raw_title = _record_title(record)
                normalized = normalize_title(raw_title)
                if normalized not in unresolved:
                    continue
                materialized_path = ""
                if materialized_dir is not None:
                    materialized_path = str(_materialize_record(record, raw_title, path, record_idx, materialized_dir))
                results[normalized] = TitleResolutionResult(
                    title=wanted[normalized],
                    normalized_title=normalized,
                    resolved=True,
                    method="indexed_shard_record" if shard_hint and normalized in shard_hint else "shard_record",
                    confidence=0.95,
                    source_path=str(path),
                    materialized_path=materialized_path,
                    metadata={"record_idx": record_idx},
                )
                unresolved.remove(normalized)
                if not unresolved:
                    return


def normalize_title(title: str) -> str:
    text = str(title or "").replace("_", " ")
    text = re.sub(r"\.[A-Za-z0-9]+$", "", text)
    text = re.sub(r"[^a-zA-Z0-9]+", " ", text).strip().lower()
    return " ".join(text.split())


def safe_title_filename(title: str, *, fallback: str = "article") -> str:
    normalized = re.sub(r"[^A-Za-z0-9]+", "_", str(title or "")).strip("_")
    return (normalized[:140] or fallback)


def _iter_article_records(path: Path, max_line_chars: int) -> Iterable[Dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        yield from _iter_json_records(path)
        return
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as fp:
            for line_idx, raw in enumerate(fp):
                line = raw.strip()
                if not line or len(line) > max_line_chars:
                    continue
                if line.startswith("{"):
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(obj, dict) and _record_title(obj):
                        yield obj
                elif line_idx == 0 and _looks_like_title_line(line):
                    text = path.read_text(encoding="utf-8", errors="ignore")
                    yield {"title": line.lstrip("# ").strip(), "text": text}
                    return
    except OSError:
        return


def _iter_json_records(path: Path) -> Iterable[Dict[str, Any]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    except (OSError, json.JSONDecodeError):
        return
    if isinstance(data, dict):
        if _record_title(data):
            yield data
        for key in ("data", "documents", "articles", "records"):
            rows = data.get(key)
            if isinstance(rows, list):
                for row in rows:
                    if isinstance(row, dict) and _record_title(row):
                        yield row
    elif isinstance(data, list):
        for row in data:
            if isinstance(row, dict) and _record_title(row):
                yield row


def _record_title(record: Dict[str, Any]) -> str:
    for key in _TITLE_KEYS:
        value = record.get(key)
        if value:
            return str(value)
    return ""


def _record_text(record: Dict[str, Any]) -> str:
    for key in _TEXT_KEYS:
        value = record.get(key)
        if value is None:
            continue
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            return "\n".join(_flatten_text(value))
        return str(value)
    return json.dumps(record, ensure_ascii=False)


def _flatten_text(value: Any) -> List[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        out: List[str] = []
        for item in value:
            out.extend(_flatten_text(item))
        return out
    if value is None:
        return []
    return [str(value)]


def _materialize_record(record: Dict[str, Any], title: str, source: Path, record_idx: int, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = safe_title_filename(title, fallback=f"record_{record_idx}")
    # Use the shard's stable relative identity (parent dir + filename) instead
    # of its absolute path so materialized article filenames — and every
    # checksum derived from them — are reproducible across machines and
    # dataset mount points.
    shard_key = f"{source.parent.name}/{source.name}"
    digest = hashlib.sha256(f"{shard_key}:{record_idx}:{title}".encode("utf-8")).hexdigest()[:8]
    path = out_dir / f"{stem}_{digest}.txt"
    if path.exists():
        return path
    body = {
        "title": title,
        "source_path": shard_key,
        "record_idx": record_idx,
    }
    text = _record_text(record)
    path.write_text(
        "# " + title + "\n\n"
        + "<!-- " + json.dumps(body, ensure_ascii=False, sort_keys=True) + " -->\n\n"
        + text + "\n",
        encoding="utf-8",
    )
    return path


def _looks_like_title_line(line: str) -> bool:
    if line.startswith("#") and len(line) <= 240:
        return True
    return False


__all__ = [
    "HotpotQATitleResolver",
    "TitleResolutionResult",
    "normalize_title",
    "safe_title_filename",
]
