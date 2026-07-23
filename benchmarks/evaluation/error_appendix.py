"""Error appendix generation for paper reports."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List


class ErrorAppendixGenerator:
    def generate(self, predictions_path: str | Path, *, max_cases: int = 20) -> Dict[str, Any]:
        path = Path(predictions_path)
        rows = _load_jsonl(path)
        badcases = [row for row in rows if not row.get("judge_correct")]
        buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for row in badcases:
            failure_type = _failure_type(row)
            buckets[failure_type].append(row)
        selected = []
        for failure_type, items in sorted(buckets.items()):
            for row in items[: max(1, max_cases // max(len(buckets), 1))]:
                selected.append(_case_summary(row, failure_type))
                if len(selected) >= max_cases:
                    break
        return {
            "total": len(rows),
            "badcase_count": len(badcases),
            "failure_type_breakdown": {key: len(value) for key, value in sorted(buckets.items())},
            "cases": selected,
        }

    def to_markdown(self, predictions_path: str | Path, *, max_cases: int = 20) -> str:
        data = self.generate(predictions_path, max_cases=max_cases)
        lines = ["## Error Appendix", ""]
        lines.append(f"- Total samples: `{data['total']}`")
        lines.append(f"- Badcases: `{data['badcase_count']}`")
        if data["failure_type_breakdown"]:
            lines.append("- Failure breakdown:")
            for key, count in data["failure_type_breakdown"].items():
                lines.append(f"  - `{key}`: `{count}`")
        if data["cases"]:
            lines.append("")
            lines.append("### Representative Cases")
            for case in data["cases"]:
                lines.append("")
                lines.append(f"#### {case['sample_id']} ({case['failure_type']})")
                lines.append(f"- Question: {case['question']}")
                lines.append(f"- Gold: {case['gold_answer']}")
                lines.append(f"- Prediction: {case['prediction']}")
                lines.append(f"- Evidence recall: `{case.get('evidence_recall')}`")
                lines.append(f"- Files read: `{case.get('num_files_read')}`")
        return "\n".join(lines) + "\n"


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _failure_type(row: Dict[str, Any]) -> str:
    prediction = str(row.get("prediction") or "").lower()
    telemetry = row.get("telemetry", {}) or {}
    if not prediction or "cannot" in prediction or "not found" in prediction or "unknown" in prediction:
        return "refusal_or_empty"
    if telemetry.get("num_files_read", 0) == 0:
        return "retrieval_failure"
    if telemetry.get("evidence_recall", 0.0) == 0:
        return "evidence_miss"
    return "answer_mismatch"


def _case_summary(row: Dict[str, Any], failure_type: str) -> Dict[str, Any]:
    telemetry = row.get("telemetry", {}) or {}
    return {
        "sample_id": row.get("sample_id", ""),
        "failure_type": failure_type,
        "question": str(row.get("question", ""))[:300],
        "gold_answer": str(row.get("gold_answer", ""))[:200],
        "prediction": str(row.get("prediction", ""))[:300],
        "evidence_recall": telemetry.get("evidence_recall"),
        "num_files_read": telemetry.get("num_files_read"),
    }
