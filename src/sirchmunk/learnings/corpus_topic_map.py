# Copyright (c) ModelScope Contributors. All rights reserved.
"""Lightweight cross-document topic map built from compiled tree titles.

This artifact contains no embeddings and requires no external index service.
It is a compact projection of already-compiled document structure, allowing
query entities and concepts to discover related files through section titles.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


_TOPIC_TOKEN_RE = re.compile(
    r"[a-zA-Z0-9][a-zA-Z0-9_\-]{2,}|[\u4e00-\u9fff]{2,}"
)
_TOPIC_STOP_WORDS = {
    "the", "and", "for", "with", "from", "this", "that", "section",
    "chapter", "document", "introduction", "summary", "overview",
}


@dataclass(frozen=True)
class TopicReference:
    """One structural topic occurrence in a compiled document."""

    file_path: str
    section_title: str

    def to_dict(self) -> Dict[str, str]:
        return {
            "file_path": self.file_path,
            "section_title": self.section_title,
        }


@dataclass
class CorpusTopicMap:
    """Token-to-document map derived from tree node titles."""

    topic_to_files: Dict[str, List[TopicReference]] = field(default_factory=dict)
    version: str = "1.0"

    @staticmethod
    def _tokens(text: str) -> Set[str]:
        tokens: Set[str] = set()
        for match in _TOPIC_TOKEN_RE.finditer(text or ""):
            token = match.group(0).lower()
            if token in _TOPIC_STOP_WORDS:
                continue
            tokens.add(token)
            if re.fullmatch(r"[\u4e00-\u9fff]{3,}", token):
                for width in (2, 3):
                    tokens.update(
                        token[index:index + width]
                        for index in range(len(token) - width + 1)
                    )
        return tokens

    @classmethod
    def build_from_indexer(
        cls,
        tree_indexer: Any,
        file_paths: Iterable[str],
        *,
        max_postings_per_topic: int = 500,
    ) -> "CorpusTopicMap":
        """Build a topic map from cached document trees."""
        mapping: Dict[str, List[TopicReference]] = {}
        seen: Dict[str, Set[Tuple[str, str]]] = {}

        for file_path in file_paths:
            try:
                tree = tree_indexer.load_tree(file_path)
            except Exception:
                continue
            if tree is None or tree.root is None:
                continue
            stack = [tree.root]
            while stack:
                node = stack.pop()
                stack.extend(getattr(node, "children", []) or [])
                title = str(getattr(node, "title", "") or "").strip()
                if not title or title == "Document":
                    continue
                reference = TopicReference(file_path=str(file_path), section_title=title)
                for token in cls._tokens(title):
                    postings = mapping.setdefault(token, [])
                    topic_seen = seen.setdefault(token, set())
                    marker = (reference.file_path, reference.section_title)
                    if marker in topic_seen or len(postings) >= max_postings_per_topic:
                        continue
                    topic_seen.add(marker)
                    postings.append(reference)

        return cls(topic_to_files=mapping)

    def search(
        self,
        terms: Iterable[str],
        *,
        allowed_paths: Optional[Set[str]] = None,
        top_k: int = 30,
    ) -> List[Tuple[str, float]]:
        """Return files ranked by structural-topic overlap."""
        query_tokens: Set[str] = set()
        normalized_terms: List[str] = []
        for term in terms:
            normalized = str(term).strip().lower()
            if not normalized:
                continue
            normalized_terms.append(normalized)
            query_tokens.update(self._tokens(normalized))
        if not query_tokens:
            return []

        scores: Dict[str, float] = {}
        matched_sections: Dict[str, Set[str]] = {}
        for token in query_tokens:
            for reference in self.topic_to_files.get(token, []):
                if allowed_paths is not None and reference.file_path not in allowed_paths:
                    continue
                title_lower = reference.section_title.lower()
                exact_phrase = any(term in title_lower for term in normalized_terms)
                increment = 2.0 if exact_phrase else 1.0
                scores[reference.file_path] = scores.get(reference.file_path, 0.0) + increment
                matched_sections.setdefault(reference.file_path, set()).add(
                    reference.section_title
                )

        ranked = sorted(
            scores,
            key=lambda path: (
                -scores[path],
                -len(matched_sections.get(path, set())),
                path,
            ),
        )
        denominator = max(len(query_tokens), 1)
        return [
            (path, round(scores[path] / denominator, 4))
            for path in ranked[:max(1, top_k)]
        ]

    def save(self, path: Path) -> None:
        """Persist the map as deterministic JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": self.version,
            "topic_to_files": {
                topic: [reference.to_dict() for reference in references]
                for topic, references in sorted(self.topic_to_files.items())
            },
        }
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )
        temporary.replace(path)

    @classmethod
    def load(cls, path: Path) -> Optional["CorpusTopicMap"]:
        """Load a persisted map, returning ``None`` on invalid artifacts."""
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            raw_mapping = payload.get("topic_to_files", {})
            mapping = {
                str(topic): [
                    TopicReference(
                        file_path=str(item["file_path"]),
                        section_title=str(item["section_title"]),
                    )
                    for item in items
                    if isinstance(item, dict)
                    and item.get("file_path")
                    and item.get("section_title")
                ]
                for topic, items in raw_mapping.items()
                if isinstance(items, list)
            }
            return cls(
                topic_to_files=mapping,
                version=str(payload.get("version") or "1.0"),
            )
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            return None
