"""HotpotQA evidence utilities."""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set


def evaluate_supporting_facts(
    supporting_facts: Any,
    read_file_ids: Iterable[str] | None = None,
    prediction: str = "",
    retrieval_logs: Iterable[Dict[str, Any]] | None = None,
    evidence_sources: Iterable[str] | None = None,
    evidence_texts: Iterable[str] | None = None,
    context: Any = None,
    resolved_title_paths: Dict[str, str] | None = None,
) -> Dict[str, Any]:
    """Compute lightweight supporting-fact recall from retrieved file paths.

    Title-level coverage is derived from three complementary signals so it stays
    correct across both corpus layouts used by the protocol:

    - filename-title match: dynamic ``G_n/D_n`` snapshots materialize one file
      per article with the title in the filename, so the article title can be
      recovered from the read path directly.
    - resolved shard-path match: the raw enwiki dump packs many articles into a
      single multi-article shard whose filename carries no title. ``resolved_
      title_paths`` maps each gold title to the absolute shard that contains it
      (via the cached corpus title index), so a gold title counts as retrieved
      when the system read the shard holding it.
    - sentence-level match: when the system exposes evidence snippets, the gold
      supporting sentence is matched against snippet content directly.
    """
    gold_facts = _attach_supporting_sentences(_extract_facts(supporting_facts), context)
    gold_titles = {fact["title"] for fact in gold_facts}
    retrieved_titles = _titles_from_paths(read_file_ids or [])
    retrieved_titles.update(_titles_from_paths(evidence_sources or []))
    retrieved_titles.update(_titles_from_retrieval_logs(retrieval_logs or []))

    # Resolve which gold titles were retrieved by shard path. In the raw enwiki
    # dump the filename carries no title, so this is the only reliable title
    # signal for the frozen main experiment.
    read_path_set = _normalized_path_set(read_file_ids or [])
    read_path_set |= _normalized_path_set(evidence_sources or [])
    read_path_set |= _normalized_paths_from_retrieval_logs(retrieval_logs or [])
    shard_hit_titles: Set[str] = set()
    for norm_title, shard_path in (resolved_title_paths or {}).items():
        if shard_path and _normalize_path(shard_path) in read_path_set:
            shard_hit_titles.add(_normalize_title(norm_title))

    if not gold_titles:
        return {
            "supporting_facts": [],
            "supporting_fact_titles": [],
            "retrieved_titles": sorted(retrieved_titles),
            "supporting_fact_hit": False,
            "supporting_fact_count": 0,
            "supporting_sentence_count": 0,
            "matched_supporting_sentence_count": 0,
            "missing_supporting_fact_titles": [],
            "missing_supporting_facts": [],
            "missing_supporting_sentences": [],
            "supporting_sentence_completion_rate": None,
            "supporting_sentence_completion_complete": False,
            "evidence_completion_needed": False,
            "evidence_recall": 0.0,
            "answer_source_grounded": False,
        }

    title_hits = {
        title for title in gold_titles
        if any(_title_matches(title, retrieved) for retrieved in retrieved_titles)
    }
    title_hits |= (gold_titles & shard_hit_titles)
    fact_hits = [fact for fact in gold_facts if fact["title"] in title_hits]
    sentence_facts = [fact for fact in gold_facts if fact.get("sentence")]
    sentence_hits = [
        fact for fact in sentence_facts
        if any(_sentence_matches(fact["sentence"], text) for text in (evidence_texts or []))
    ]
    title_recall = len(title_hits) / max(len(gold_titles), 1)
    fact_recall = len(fact_hits) / max(len(gold_facts), 1)
    sentence_recall = len(sentence_hits) / max(len(sentence_facts), 1) if sentence_facts else None
    effective_recall = sentence_recall if sentence_recall is not None and evidence_texts else fact_recall
    answer_source_grounded = (bool(title_hits) or bool(sentence_hits)) and bool((prediction or "").strip())
    missing_titles = sorted(title for title in gold_titles if title not in title_hits)
    missing_facts = [fact for fact in gold_facts if fact not in fact_hits]
    missing_sentences = [fact for fact in sentence_facts if fact not in sentence_hits]
    completion_rate = round(sentence_recall, 4) if sentence_recall is not None else None
    completion_complete = bool(sentence_facts) and len(sentence_hits) == len(sentence_facts)

    return {
        "supporting_facts": gold_facts,
        "supporting_fact_titles": sorted(gold_titles),
        "retrieved_titles": sorted(retrieved_titles),
        "matched_supporting_fact_titles": sorted(title_hits),
        "matched_supporting_facts": fact_hits,
        "matched_supporting_sentences": sentence_hits,
        "missing_supporting_fact_titles": missing_titles,
        "missing_supporting_facts": missing_facts,
        "missing_supporting_sentences": missing_sentences,
        "supporting_fact_hit": bool(title_hits) or bool(sentence_hits),
        "supporting_fact_count": len(gold_facts),
        "supporting_sentence_count": len(sentence_facts),
        "matched_supporting_sentence_count": len(sentence_hits),
        "supporting_fact_title_recall": round(title_recall, 4),
        "supporting_sentence_recall": completion_rate,
        "supporting_sentence_completion_rate": completion_rate,
        "supporting_sentence_completion_complete": completion_complete,
        "evidence_completion_needed": bool(missing_titles or missing_sentences),
        "evidence_recall": round(effective_recall, 4),
        "answer_source_grounded": answer_source_grounded,
    }


def _extract_facts(supporting_facts: Any) -> List[Dict[str, Any]]:
    """Normalize HotpotQA supporting_facts structures into title/sent_id facts."""
    facts: List[Dict[str, Any]] = []
    if supporting_facts is None:
        return facts

    if isinstance(supporting_facts, dict):
        titles = supporting_facts.get("title") or supporting_facts.get("titles") or []
        sent_ids = supporting_facts.get("sent_id") or supporting_facts.get("sent_ids") or []
        if isinstance(titles, str):
            titles = [titles]
        if not isinstance(sent_ids, list):
            sent_ids = [sent_ids] * len(titles)
        for idx, title in enumerate(titles):
            normalized = _normalize_title(str(title))
            if normalized:
                facts.append({"title": normalized, "sent_id": _safe_sent_id(sent_ids, idx)})
        return facts

    if hasattr(supporting_facts, "tolist"):
        try:
            supporting_facts = supporting_facts.tolist()
        except Exception:
            supporting_facts = list(supporting_facts)

    if isinstance(supporting_facts, (list, tuple, set)):
        for item in supporting_facts:
            title = None
            sent_id = None
            if isinstance(item, dict):
                title = item.get("title") or item.get("doc_title") or item.get("page")
                sent_id = item.get("sent_id") or item.get("sentence_id")
            elif isinstance(item, (list, tuple)) and item:
                title = item[0]
                sent_id = item[1] if len(item) > 1 else None
            else:
                title = item
            normalized = _normalize_title(str(title)) if title is not None else ""
            if normalized:
                facts.append({"title": normalized, "sent_id": sent_id})
    return facts


def _extract_titles(supporting_facts: Any) -> Set[str]:
    return {fact["title"] for fact in _extract_facts(supporting_facts)}


def _titles_from_paths(paths: Iterable[str]) -> Set[str]:
    titles: Set[str] = set()
    for raw in paths:
        if not raw:
            continue
        path = Path(str(raw))
        candidates = [path.stem, path.name]
        # Some wiki corpora store article titles in parent dirs.
        candidates.extend(part for part in path.parts[-3:] if part)
        for candidate in candidates:
            normalized = _normalize_title(candidate)
            if normalized:
                titles.add(normalized)
    return titles


def _attach_supporting_sentences(facts: List[Dict[str, Any]], context: Any) -> List[Dict[str, Any]]:
    if not facts or context is None:
        return facts
    context_map = _context_sentence_map(context)
    enriched: List[Dict[str, Any]] = []
    for fact in facts:
        item = dict(fact)
        sentence = context_map.get((fact.get("title"), fact.get("sent_id")))
        if sentence:
            item["sentence"] = sentence
        enriched.append(item)
    return enriched


def _context_sentence_map(context: Any) -> Dict[tuple[str, Any], str]:
    mapping: Dict[tuple[str, Any], str] = {}
    if hasattr(context, "tolist"):
        try:
            context = context.tolist()
        except Exception:
            pass
    if isinstance(context, dict):
        titles = context.get("title") or context.get("titles") or []
        sentences = context.get("sentences") or context.get("sentence") or []
        if isinstance(titles, str):
            titles = [titles]
        for title, sent_list in zip(titles, sentences):
            norm_title = _normalize_title(str(title))
            if hasattr(sent_list, "tolist"):
                sent_list = sent_list.tolist()
            if isinstance(sent_list, (list, tuple)):
                for idx, sentence in enumerate(sent_list):
                    mapping[(norm_title, idx)] = str(sentence)
        return mapping
    if isinstance(context, (list, tuple)):
        for item in context:
            if isinstance(item, dict):
                title = item.get("title")
                sentences = item.get("sentences") or item.get("sentence") or []
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                title, sentences = item[0], item[1]
            else:
                continue
            norm_title = _normalize_title(str(title))
            if hasattr(sentences, "tolist"):
                sentences = sentences.tolist()
            if isinstance(sentences, (list, tuple)):
                for idx, sentence in enumerate(sentences):
                    mapping[(norm_title, idx)] = str(sentence)
    return mapping


def _sentence_matches(gold_sentence: str, evidence_text: str) -> bool:
    gold_tokens = set(_normalize_title(gold_sentence).split())
    evidence_tokens = set(_normalize_title(evidence_text).split())
    if not gold_tokens or not evidence_tokens:
        return False
    overlap = len(gold_tokens & evidence_tokens) / max(len(gold_tokens), 1)
    return overlap >= 0.6


def _normalize_path(raw: Any) -> str:
    """Return a canonical absolute path for read/evidence overlap checks."""
    if not raw:
        return ""
    try:
        return os.path.realpath(os.path.abspath(str(raw)))
    except (OSError, ValueError):
        return str(raw)


def _normalized_path_set(paths: Iterable[Any]) -> Set[str]:
    result: Set[str] = set()
    for raw in paths:
        normalized = _normalize_path(raw)
        if normalized:
            result.add(normalized)
    return result


def _normalized_paths_from_retrieval_logs(logs: Iterable[Dict[str, Any]]) -> Set[str]:
    result: Set[str] = set()
    for log in logs:
        metadata = log.get("metadata", {}) if isinstance(log, dict) else {}
        for key in ("path", "file", "file_path", "files", "file_paths"):
            value = metadata.get(key) if isinstance(metadata, dict) else None
            if isinstance(value, list):
                result |= _normalized_path_set(value)
            elif value:
                normalized = _normalize_path(value)
                if normalized:
                    result.add(normalized)
    return result


def _titles_from_retrieval_logs(logs: Iterable[Dict[str, Any]]) -> Set[str]:
    titles: Set[str] = set()
    for log in logs:
        metadata = log.get("metadata", {}) if isinstance(log, dict) else {}
        candidates: List[Any] = []
        for key in ("path", "file", "file_path", "files", "file_paths", "results", "candidates"):
            value = metadata.get(key)
            if isinstance(value, list):
                candidates.extend(value)
            elif value:
                candidates.append(value)
        for candidate in candidates:
            if isinstance(candidate, dict):
                candidate = candidate.get("path") or candidate.get("file") or candidate.get("title")
            normalized = _normalize_title(str(candidate)) if candidate is not None else ""
            if normalized:
                titles.add(normalized)
    return titles


def _safe_sent_id(values: Any, idx: int) -> Any:
    try:
        return values[idx]
    except Exception:
        return None


def _title_matches(gold: str, retrieved: str) -> bool:
    if not gold or not retrieved:
        return False
    if gold == retrieved:
        return True
    return gold in retrieved or retrieved in gold


def _normalize_title(title: str) -> str:
    title = title.replace("_", " ")
    title = re.sub(r"\.[A-Za-z0-9]+$", "", title)
    title = re.sub(r"[^a-zA-Z0-9]+", " ", title).strip().lower()
    return " ".join(title.split())
