"""framework/analyzer.py — BadCaseAnalyzer

Responsibilities:
1. Fast LLM-free classification of every failing sample (refusal / wrong_value /
   no_coverage / partial_answer)
2. Map rule signals to a root cause (retrieval_failure / evidence_partial /
   synthesis_error / judge_suspect)
3. One LLM call to induce the shared failure patterns of the top-30 badcases
4. Return a structured BadCaseReport and support Markdown printing

Single Responsibility: this module only analyzes and never modifies config or code.
"""
from __future__ import annotations

import logging
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional

from .schema import BadCase, BadCaseReport, PredictionResult, RootCause

logger = logging.getLogger(__name__)

# Refusal markers, kept aligned with search.py AgenticSearch._REFUSAL_PATTERN
_REFUSAL_PHRASES = frozenset({
    "i cannot", "i can't", "unable to", "not able to",
    "i don't know", "i do not know", "unknown",
    "no results found", "cannot determine", "insufficient data",
    "data not found", "could not find", "couldn't find",
    "unable to determine", "unable to find", "not found",
    "no relevant", "no information",
})

# Regexes that detect numeric answers
_NUMERIC_RE = re.compile(
    r"(?:"
    r"[\$€£¥]\s*[\d,.]+|"      # Currency amount
    r"[\d,.]+\s*%|"             # Percentage
    r"\b\d{1,3}(?:,\d{3})+\b|" # Thousands-separated number
    r"\b\d+\.\d+\b"             # Decimal
    r")"
)

# Prompt used for LLM failure-pattern induction
_PATTERN_SUMMARY_PROMPT = """\
You are a research assistant analyzing failure cases of a document QA system.

Below are {n} failed QA samples. Each contains:
- Question
- Gold Answer (correct)
- System Prediction (wrong)

Your task: identify 3-5 concise, generalizable failure patterns that explain WHY the system failed.
Focus on retrieval and reasoning failure modes, not surface-level differences.

Failed samples:
{samples_text}

Return ONLY a numbered list (1-5 items), each ≤ 30 words. No preamble, no JSON.
"""


def _is_refusal(text: str) -> bool:
    """Detect whether the prediction is a refusal, reusing the heuristics of search.py."""
    if not text or not text.strip():
        return True
    lower = text.strip().lower()
    if lower in ("unknown", "n/a", "none", ""):
        return True
    # Inspect the content behind the **Answer:** marker
    answer_match = re.search(r'\*\*answer:\s*(.+?)\*\*', lower)
    if answer_match:
        val = answer_match.group(1).strip()
        return any(p in val for p in _REFUSAL_PHRASES)
    # Inspect the first 300 characters
    check = lower[:300]
    return any(p in check for p in _REFUSAL_PHRASES)


def _has_numeric_in_gold(gold: str) -> bool:
    """Whether the gold answer contains a number, used to decide if wrong_value applies."""
    return bool(_NUMERIC_RE.search(gold))


def _classify_failure(result: PredictionResult, gold_answer: str) -> str:
    """Rule-classify one failing sample and return the failure_type string."""
    pred = result.prediction or ""

    if _is_refusal(pred):
        return "refusal"

    if not result.coverage:
        return "no_coverage"

    # coverage=True but judge_correct=False -> refine the classification
    if _has_numeric_in_gold(gold_answer):
        return "wrong_value"

    return "partial_answer"


def _infer_root_cause(result: PredictionResult, failure_type: str) -> RootCause:
    """Infer the root cause from telemetry signals, without an LLM."""
    telemetry = result.telemetry or {}
    num_files_read: int = telemetry.get("num_files_read", 0)
    loop_count: int = telemetry.get("loop_count", 0)
    max_loops: int = int(telemetry.get("max_loops", telemetry.get("configured_max_loops", 10)) or 10)
    loop_exhausted = loop_count >= max(max_loops - 1, 1)

    if failure_type in ("refusal", "no_coverage"):
        # No file was read -> retrieval failure
        if num_files_read == 0:
            return RootCause.RETRIEVAL_FAILURE
        return RootCause.EVIDENCE_PARTIAL

    if failure_type == "partial_answer":
        # Content was found but the answer was not synthesized correctly
        if loop_exhausted:
            return RootCause.EVIDENCE_PARTIAL
        return RootCause.SYNTHESIS_ERROR

    if failure_type == "wrong_value":
        # Coverage present with a wrong number: most likely a synthesis/computation error
        return RootCause.SYNTHESIS_ERROR

    return RootCause.UNKNOWN


def _build_evidence_note(result: PredictionResult, root_cause: RootCause) -> str:
    """Attach a short supporting-evidence note to the root-cause decision."""
    telemetry = result.telemetry or {}
    num_files = telemetry.get("num_files_read", "?")
    loop = telemetry.get("loop_count", "?")
    tokens = telemetry.get("total_tokens", "?")

    if root_cause == RootCause.RETRIEVAL_FAILURE:
        return f"files_read={num_files}, loop={loop}"
    if root_cause == RootCause.EVIDENCE_PARTIAL:
        return f"files_read={num_files}, loop={loop} (loop may be exhausted)"
    if root_cause == RootCause.SYNTHESIS_ERROR:
        return f"coverage=True, tokens={tokens}, loop={loop}"
    return f"files_read={num_files}, loop={loop}, tokens={tokens}"


class BadCaseAnalyzer:
    """Badcase analyzer.

    Usage::

        analyzer = BadCaseAnalyzer(llm=llm)
        report = await analyzer.analyze(results, samples_map)
        analyzer.print_report(report)
    """

    # Maximum badcases sent to the LLM (bounds token cost)
    _MAX_SAMPLES_FOR_LLM = 30
    # Maximum characters per badcase summary
    _SAMPLE_SUMMARY_MAX = 200

    def __init__(self, llm: Optional[Any] = None) -> None:
        """
        Args:
            llm: OpenAIChat instance; when None, LLM pattern induction is skipped and only
                the rule-based analysis is provided.
        """
        self._llm = llm

    async def analyze(
        self,
        results: List[PredictionResult],
        samples_map: Dict[str, Any],          # sample_id -> BenchmarkSample
        question_type_key: str = "question_type",
    ) -> BadCaseReport:
        """Analyze the experiment results and produce a BadCaseReport.

        Args:
            results:           list of PredictionResult.
            samples_map:       {sample_id: BenchmarkSample}, used to read gold_answer and
                               metadata.
            question_type_key: metadata field name holding the question type.

        Returns:
            BadCaseReport
        """
        total = len(results)
        if total == 0:
            return BadCaseReport(total_samples=0, total_badcases=0,
                                 accuracy=0.0, coverage=0.0)

        judge_correct_count = sum(1 for r in results if r.judge_correct)
        coverage_count = sum(1 for r in results if r.coverage)
        accuracy = judge_correct_count / total * 100
        coverage = coverage_count / total * 100

        # Badcase = judge_correct is False
        bad_results = [r for r in results if not r.judge_correct]

        badcases: List[BadCase] = []
        failure_type_counts: Dict[str, int] = defaultdict(int)
        root_cause_counts: Dict[str, int] = defaultdict(int)
        judge_suspect_ids: List[str] = []
        by_qt: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "n": 0, "correct": 0, "coverage": 0
        })

        # Aggregate all results by question_type
        for r in results:
            sample = samples_map.get(r.sample_id)
            qt = (sample.metadata.get(question_type_key, "unknown")
                  if sample else "unknown")
            by_qt[qt]["n"] += 1
            if r.judge_correct:
                by_qt[qt]["correct"] += 1
            if r.coverage:
                by_qt[qt]["coverage"] += 1

        # Classify the badcases
        for r in bad_results:
            sample = samples_map.get(r.sample_id)
            gold = sample.gold_answer if sample else ""
            qt = (sample.metadata.get(question_type_key, "unknown")
                  if sample else "unknown")

            failure_type = _classify_failure(r, gold)
            root_cause = _infer_root_cause(r, failure_type)
            evidence_note = _build_evidence_note(r, root_cause)

            # Suspect judge detection: prediction holds the correct number but the judge scored it wrong
            if (failure_type == "wrong_value"
                    and gold
                    and _numeric_overlap(r.prediction, gold)):
                root_cause = RootCause.JUDGE_SUSPECT
                judge_suspect_ids.append(r.sample_id)

            failure_type_counts[failure_type] += 1
            root_cause_counts[root_cause.value] += 1

            badcases.append(BadCase(
                sample_id=r.sample_id,
                question=sample.question if sample else r.sample_id,
                gold_answer=gold,
                prediction=r.prediction[:300],
                failure_type=failure_type,
                root_cause=root_cause,
                evidence=evidence_note,
                metadata={"group": qt},
            ))

        # LLM failure-pattern induction
        pattern_summary = ""
        if self._llm and bad_results:
            pattern_summary = await self._summarize_patterns(badcases)

        # Compute per-question-type accuracy
        by_qt_final: Dict[str, Dict[str, Any]] = {}
        for qt, stats in by_qt.items():
            n = stats["n"]
            by_qt_final[qt] = {
                "n": n,
                "accuracy": round(stats["correct"] / n * 100, 1) if n else 0.0,
                "coverage": round(stats["coverage"] / n * 100, 1) if n else 0.0,
            }

        return BadCaseReport(
            total_samples=total,
            total_badcases=len(bad_results),
            accuracy=round(accuracy, 2),
            coverage=round(coverage, 2),
            badcases=badcases,
            failure_type_breakdown=dict(failure_type_counts),
            root_cause_breakdown=dict(root_cause_counts),
            by_question_type=by_qt_final,
            pattern_summary=pattern_summary,
            judge_suspect_ids=judge_suspect_ids,
        )

    async def _summarize_patterns(self, badcases: List[BadCase]) -> str:
        """One LLM call that induces the shared failure patterns of the top-N badcases."""
        top = badcases[: self._MAX_SAMPLES_FOR_LLM]
        lines = []
        for i, bc in enumerate(top, 1):
            q = bc.question[:100]
            g = bc.gold_answer[:80]
            p = bc.prediction[:80]
            lines.append(
                f"{i}. Q: {q}\n   Gold: {g}\n   Pred: {p}\n   Type: {bc.failure_type}"
            )
        samples_text = "\n\n".join(lines)
        prompt = _PATTERN_SUMMARY_PROMPT.format(
            n=len(top), samples_text=samples_text
        )
        try:
            resp = await self._llm.achat(
                messages=[{"role": "user", "content": prompt}],
                stream=False,
            )
            return (resp.content or "").strip()
        except Exception as exc:
            logger.warning("[Analyzer] LLM pattern summarization failed: %s", exc)
            return ""

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    @staticmethod
    def print_report(report: BadCaseReport) -> None:
        """Print the report to stdout in structured Markdown."""
        sep = "=" * 64
        print(f"\n{sep}")
        print("  BadCase Analysis Report")
        print(f"  Total: {report.total_samples}  |  "
              f"Accuracy: {report.accuracy:.1f}%  |  "
              f"Coverage: {report.coverage:.1f}%")
        print(f"  Badcases: {report.total_badcases} "
              f"({report.total_badcases / max(report.total_samples, 1) * 100:.1f}%)")
        print(sep)

        # Failure-type distribution
        print("\n### Failure Type Breakdown")
        for ft, cnt in sorted(report.failure_type_breakdown.items(),
                              key=lambda x: -x[1]):
            pct = cnt / max(report.total_badcases, 1) * 100
            print(f"  {ft:<20}  {cnt:>4}  ({pct:.1f}%)")

        # Root-cause distribution
        print("\n### Root Cause Breakdown")
        for rc, cnt in sorted(report.root_cause_breakdown.items(),
                              key=lambda x: -x[1]):
            pct = cnt / max(report.total_badcases, 1) * 100
            print(f"  {rc:<25}  {cnt:>4}  ({pct:.1f}%)")

        # Per-type statistics
        if report.by_question_type:
            print("\n### By Question Type")
            print(f"  {'Type':<30} {'Acc%':>6} {'Cov%':>6} {'N':>4}")
            print("  " + "-" * 48)
            for qt, stats in sorted(report.by_question_type.items()):
                print(f"  {qt:<30} {stats['accuracy']:>5.1f} "
                      f"{stats['coverage']:>5.1f} {stats['n']:>4}")

        # LLM-induced patterns
        if report.pattern_summary:
            print("\n### LLM Pattern Summary")
            for line in report.pattern_summary.splitlines():
                print(f"  {line}")

        # Suspect judge list
        if report.judge_suspect_ids:
            print(f"\n### Judge Suspect Cases ({len(report.judge_suspect_ids)})")
            print("  (Coverage=True, numeric overlap detected — may be false negative)")
            for sid in report.judge_suspect_ids[:10]:
                print(f"  - {sid}")

        # Details of the first 10 badcases
        if report.badcases:
            print(f"\n### Top Badcases (showing up to 10 of {len(report.badcases)})")
            for bc in report.badcases[:10]:
                print(f"\n  [{bc.sample_id}] [{bc.failure_type}] root={bc.root_cause.value}")
                print(f"    Q:    {bc.question[:100]}")
                print(f"    Gold: {bc.gold_answer[:80]}")
                print(f"    Pred: {bc.prediction[:80]}")
                print(f"    Hint: {bc.evidence}")

        print(f"\n{sep}\n")


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _numeric_overlap(prediction: str, gold: str) -> bool:
    """Detect whether the prediction contains the main numbers of gold (simple overlap)."""
    gold_nums = set(_NUMERIC_RE.findall(gold.lower()))
    if not gold_nums:
        return False
    pred_lower = prediction.lower()
    return any(n in pred_lower for n in gold_nums)
