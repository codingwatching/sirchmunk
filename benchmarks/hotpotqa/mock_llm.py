"""Deterministic mock LLM for HotpotQA pipeline smoke tests.

The goal is not answer quality; it is to exercise the full retrieval / runner /
judge / artifact chain without calling external APIs.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from sirchmunk.llm.openai_chat import OpenAIChatResponse


class MockHotpotLLM:
    """Small OpenAIChat-compatible mock implementing ``achat``."""

    def __init__(self, model: str = "mock-hotpot-llm") -> None:
        self.model = model
        self.calls: List[Dict[str, Any]] = []

    async def achat(
        self,
        messages: Optional[List[Dict[str, str]]] = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> OpenAIChatResponse:
        prompt = _last_prompt(messages)
        content = self._respond(prompt)
        usage = {
            "prompt_tokens": max(1, len(prompt) // 4),
            "completion_tokens": max(1, len(content) // 4),
            "total_tokens": max(2, (len(prompt) + len(content)) // 4),
        }
        self.calls.append({"prompt": prompt[:500], "stream": stream, "usage": usage})
        return OpenAIChatResponse(content=content, usage=usage, model=self.model)

    def _respond(self, prompt: str) -> str:
        lower = prompt.lower()
        query = _extract_query(prompt)
        keywords = _keywords_from_query(query)

        if "doc_level" in lower and "classify" in lower:
            return json.dumps({"doc_level": False, "op": None})

        if "return json only" in lower and "primary" in lower and "fallback" in lower:
            primary = " ".join(keywords[:2]) if len(keywords) >= 2 else (keywords[0] if keywords else "hotpotqa")
            fallback = keywords[:3] or ["hotpotqa", "answer"]
            idf = {primary: 8.0, **{kw: 6.0 for kw in fallback}}
            return json.dumps({
                "type": "search",
                "primary": [primary],
                "fallback": fallback,
                "idf": idf,
                "primary_alt": [],
                "fallback_alt": [],
                "file_hints": [],
                "intent": "mock smoke search",
                "selected_docs": [],
                "doc_confidence": "low",
            })

        if "<keywords_level_1>" in lower or "multi-level keywords" in lower:
            level1 = {" ".join(keywords[:2]) or "hotpotqa": 8.0}
            level2 = {kw: 6.0 for kw in (keywords[:5] or ["hotpotqa", "answer"])}
            return (
                f"<KEYWORDS_LEVEL_1>\n{json.dumps(level1)}\n</KEYWORDS_LEVEL_1>\n"
                f"<KEYWORDS_LEVEL_2>\n{json.dumps(level2)}\n</KEYWORDS_LEVEL_2>\n"
                "<KEYWORDS_ALT>\n{}\n</KEYWORDS_ALT>\n"
                "<MULTI_SOURCE_INTENT>\n0.2\n</MULTI_SOURCE_INTENT>"
            )

        if "complexity" in lower and "intent" in lower and "json" in lower:
            return json.dumps({"complexity": "simple", "intent": "lookup"})

        if "data_points" in lower or "likely_sources" in lower:
            return json.dumps({
                "data_points": [query or "answer"],
                "likely_sources": keywords[:3],
                "formula": None,
                "time_period": None,
                "intent": "lookup",
            })

        if "score" in lower and "reasoning" in lower and "text snippet" in lower:
            return json.dumps({"score": 8.0, "reasoning": "Mock relevant evidence."})

        if "select" in lower and "page" in lower and "json" in lower:
            return "[1, 2, 3]"

        if "equivalent" in lower and "gold answer" in lower:
            gold = _field_after(prompt, "Gold answer:")
            pred = _field_after(prompt, "Prediction:")
            equivalent = bool(gold and pred and gold.lower() in pred.lower())
            return json.dumps({
                "equivalent": equivalent,
                "confidence": 0.9 if equivalent else 0.2,
                "reasoning": "Mock lexical equivalence check.",
            })

        if "focused_evidence" in lower:
            return "<FOCUSED_EVIDENCE>Mock focused evidence.</FOCUSED_EVIDENCE>"

        if "should_answer" in lower or "precise_answer" in lower or "summary" in lower:
            answer = keywords[0] if keywords else "mock answer"
            return (
                f"<PRECISE_ANSWER>{answer}</PRECISE_ANSWER>\n"
                f"<SUMMARY>Mock answer generated for pipeline smoke test. Query: {query}</SUMMARY>\n"
                "<SHOULD_ANSWER>true</SHOULD_ANSWER>\n"
                "<SHOULD_SAVE>false</SHOULD_SAVE>"
            )

        return (
            "<SUMMARY>Mock response for HotpotQA smoke test.</SUMMARY>\n"
            "<SHOULD_ANSWER>true</SHOULD_ANSWER>\n"
            "<SHOULD_SAVE>false</SHOULD_SAVE>"
        )


def _last_prompt(messages: Optional[List[Dict[str, str]]]) -> str:
    if not messages:
        return ""
    return str(messages[-1].get("content", ""))


def _extract_query(prompt: str) -> str:
    patterns = [
        r"### User Query:\s*(.+?)(?:\n###|\Z)",
        r"### User Query\s*\n\s*(.+?)(?:\n###|\Z)",
        r"Query:\s*\"?(.+?)\"?(?:\n|\Z)",
        r"Question:\s*(.+?)(?:\n|\Z)",
        r"User Input\s*[:：]\s*(.+?)(?:\n|\Z)",
    ]
    for pattern in patterns:
        match = re.search(pattern, prompt, re.IGNORECASE | re.DOTALL)
        if match:
            return " ".join(match.group(1).strip().split())[:300]
    return ""


def _keywords_from_query(query: str) -> List[str]:
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", query or "")
    stop = {
        "what", "when", "where", "which", "this", "that", "with", "from",
        "were", "was", "does", "have", "answer", "year", "university",
        "professor", "actor", "actress", "writer", "heritage", "between",
        "and", "the", "genus", "contains", "more", "species", "member",
        "upper", "house", "legislature", "many", "both", "publish",
        "than", "bestselling", "novels", "born", "ruined", "castle",
        "characterized", "isolated", "fortifications",
    }
    proper = [t.lower() for t in re.findall(r"\b[A-Z][A-Za-z0-9_-]{2,}\b", query or "")]
    out: List[str] = []
    for token in proper + [t.lower() for t in tokens]:
        low = token.lower()
        if low in stop or low in out:
            continue
        out.append(low)
    return out[:8]


def _field_after(prompt: str, label: str) -> str:
    pattern = re.escape(label) + r"\s*(.+?)(?:\n[A-Z][A-Za-z ]+:|\Z)"
    match = re.search(pattern, prompt, re.DOTALL)
    return " ".join(match.group(1).strip().split()) if match else ""


__all__ = ["MockHotpotLLM"]
