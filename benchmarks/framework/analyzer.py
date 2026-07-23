"""framework/analyzer.py — BadCaseAnalyzer

职责：
1. 对所有失败样本进行无 LLM 快速分类（refusal / wrong_value / no_coverage / partial_answer）
2. 通过规则信号映射根因（retrieval_failure / evidence_partial / synthesis_error / judge_suspect）
3. 单次 LLM call 归纳 top-30 badcase 的共性失败模式
4. 返回结构化 BadCaseReport，并支持 Markdown 格式打印

遵循 Single Responsibility：本模块只做分析，不修改任何配置或代码。
"""
from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional

from .schema import BadCase, BadCaseReport, PredictionResult, RootCause

logger = logging.getLogger(__name__)

# 拒绝回答检测词列表（与 search.py AgenticSearch._REFUSAL_PATTERN 保持一致方向）
_REFUSAL_PHRASES = frozenset({
    "i cannot", "i can't", "unable to", "not able to",
    "i don't know", "i do not know", "unknown",
    "no results found", "cannot determine", "insufficient data",
    "data not found", "could not find", "couldn't find",
    "unable to determine", "unable to find", "not found",
    "no relevant", "no information",
})

# 数值型回答检测正则
_NUMERIC_RE = re.compile(
    r"(?:"
    r"[\$€£¥]\s*[\d,.]+|"      # 货币金额
    r"[\d,.]+\s*%|"             # 百分比
    r"\b\d{1,3}(?:,\d{3})+\b|" # 千分位数字
    r"\b\d+\.\d+\b"             # 小数
    r")"
)

# LLM 模式归纳 Prompt
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
    """检测 prediction 是否为拒绝回答。复用与 search.py 相同的启发式逻辑。"""
    if not text or not text.strip():
        return True
    lower = text.strip().lower()
    if lower in ("unknown", "n/a", "none", ""):
        return True
    # 检查 **Answer:** 标记内容
    answer_match = re.search(r'\*\*answer:\s*(.+?)\*\*', lower)
    if answer_match:
        val = answer_match.group(1).strip()
        return any(p in val for p in _REFUSAL_PHRASES)
    # 检查前 300 字符
    check = lower[:300]
    return any(p in check for p in _REFUSAL_PHRASES)


def _has_numeric_in_gold(gold: str) -> bool:
    """判断 gold answer 是否包含数值（用于判断是否应检测 wrong_value）。"""
    return bool(_NUMERIC_RE.search(gold))


def _classify_failure(result: PredictionResult, gold_answer: str) -> str:
    """对单个失败样本进行规则分类，返回 failure_type 字符串。"""
    pred = result.prediction or ""

    if _is_refusal(pred):
        return "refusal"

    if not result.coverage:
        return "no_coverage"

    # coverage=True 但 judge_correct=False → 进一步区分
    if _has_numeric_in_gold(gold_answer):
        return "wrong_value"

    return "partial_answer"


def _infer_root_cause(result: PredictionResult, failure_type: str) -> RootCause:
    """通过 telemetry 信号推断根因（无 LLM）。"""
    telemetry = result.telemetry or {}
    num_files_read: int = telemetry.get("num_files_read", 0)
    loop_count: int = telemetry.get("loop_count", 0)
    max_loops: int = int(telemetry.get("max_loops", telemetry.get("configured_max_loops", 10)) or 10)
    loop_exhausted = loop_count >= max(max_loops - 1, 1)

    if failure_type in ("refusal", "no_coverage"):
        # 文件没读到 → 检索失败
        if num_files_read == 0:
            return RootCause.RETRIEVAL_FAILURE
        return RootCause.EVIDENCE_PARTIAL

    if failure_type == "partial_answer":
        # 找到了内容但没合成出正确答案
        if loop_exhausted:
            return RootCause.EVIDENCE_PARTIAL
        return RootCause.SYNTHESIS_ERROR

    if failure_type == "wrong_value":
        # 有 coverage 且数值错误：最可能是合成/计算错误
        return RootCause.SYNTHESIS_ERROR

    return RootCause.UNKNOWN


def _build_evidence_note(result: PredictionResult, root_cause: RootCause) -> str:
    """为根因判断附加简短的支持性证据描述。"""
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
    """BadCase 分析器。

    Usage::

        analyzer = BadCaseAnalyzer(llm=llm)
        report = await analyzer.analyze(results, samples_map)
        analyzer.print_report(report)
    """

    # 送给 LLM 的最多 badcase 数（控制 token 消耗）
    _MAX_SAMPLES_FOR_LLM = 30
    # 每个 badcase 摘要的最大字符数
    _SAMPLE_SUMMARY_MAX = 200

    def __init__(self, llm: Optional[Any] = None) -> None:
        """
        Args:
            llm: OpenAIChat 实例；为 None 时跳过 LLM 模式归纳，仍提供规则分析。
        """
        self._llm = llm

    async def analyze(
        self,
        results: List[PredictionResult],
        samples_map: Dict[str, Any],          # sample_id -> BenchmarkSample
        question_type_key: str = "question_type",
    ) -> BadCaseReport:
        """分析实验结果，生成 BadCaseReport。

        Args:
            results:           PredictionResult 列表。
            samples_map:       {sample_id: BenchmarkSample}，用于取 gold_answer / metadata。
            question_type_key: metadata 中代表题目类型的字段名。

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

        # 坏案例 = judge_correct=False
        bad_results = [r for r in results if not r.judge_correct]

        badcases: List[BadCase] = []
        failure_type_counts: Dict[str, int] = defaultdict(int)
        root_cause_counts: Dict[str, int] = defaultdict(int)
        judge_suspect_ids: List[str] = []
        by_qt: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "n": 0, "correct": 0, "coverage": 0
        })

        # 按 question_type 统计全量结果
        for r in results:
            sample = samples_map.get(r.sample_id)
            qt = (sample.metadata.get(question_type_key, "unknown")
                  if sample else "unknown")
            by_qt[qt]["n"] += 1
            if r.judge_correct:
                by_qt[qt]["correct"] += 1
            if r.coverage:
                by_qt[qt]["coverage"] += 1

        # 对坏案例分类
        for r in bad_results:
            sample = samples_map.get(r.sample_id)
            gold = sample.gold_answer if sample else ""
            qt = (sample.metadata.get(question_type_key, "unknown")
                  if sample else "unknown")

            failure_type = _classify_failure(r, gold)
            root_cause = _infer_root_cause(r, failure_type)
            evidence_note = _build_evidence_note(r, root_cause)

            # Judge 可疑检测：预测里含有正确数字但 judge 判为错
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

        # LLM 模式归纳
        pattern_summary = ""
        if self._llm and bad_results:
            pattern_summary = await self._summarize_patterns(badcases)

        # 计算 by_qt accuracy
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
        """单次 LLM call，归纳 top-N badcase 的共性失败模式。"""
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
    # 打印
    # ------------------------------------------------------------------

    @staticmethod
    def print_report(report: BadCaseReport) -> None:
        """以结构化 Markdown 格式打印报告到 stdout。"""
        sep = "=" * 64
        print(f"\n{sep}")
        print(f"  BadCase Analysis Report")
        print(f"  Total: {report.total_samples}  |  "
              f"Accuracy: {report.accuracy:.1f}%  |  "
              f"Coverage: {report.coverage:.1f}%")
        print(f"  Badcases: {report.total_badcases} "
              f"({report.total_badcases / max(report.total_samples, 1) * 100:.1f}%)")
        print(sep)

        # 失败类型分布
        print("\n### Failure Type Breakdown")
        for ft, cnt in sorted(report.failure_type_breakdown.items(),
                              key=lambda x: -x[1]):
            pct = cnt / max(report.total_badcases, 1) * 100
            print(f"  {ft:<20}  {cnt:>4}  ({pct:.1f}%)")

        # 根因分布
        print("\n### Root Cause Breakdown")
        for rc, cnt in sorted(report.root_cause_breakdown.items(),
                              key=lambda x: -x[1]):
            pct = cnt / max(report.total_badcases, 1) * 100
            print(f"  {rc:<25}  {cnt:>4}  ({pct:.1f}%)")

        # 分类型统计
        if report.by_question_type:
            print("\n### By Question Type")
            print(f"  {'Type':<30} {'Acc%':>6} {'Cov%':>6} {'N':>4}")
            print("  " + "-" * 48)
            for qt, stats in sorted(report.by_question_type.items()):
                print(f"  {qt:<30} {stats['accuracy']:>5.1f} "
                      f"{stats['coverage']:>5.1f} {stats['n']:>4}")

        # LLM 归纳模式
        if report.pattern_summary:
            print("\n### LLM Pattern Summary")
            for line in report.pattern_summary.splitlines():
                print(f"  {line}")

        # Judge 可疑列表
        if report.judge_suspect_ids:
            print(f"\n### Judge Suspect Cases ({len(report.judge_suspect_ids)})")
            print("  (Coverage=True, numeric overlap detected — may be false negative)")
            for sid in report.judge_suspect_ids[:10]:
                print(f"  - {sid}")

        # 前 10 个坏案例详情
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
    """检测 prediction 中是否包含 gold 里的主要数值（简单重叠检测）。"""
    gold_nums = set(_NUMERIC_RE.findall(gold.lower()))
    if not gold_nums:
        return False
    pred_lower = prediction.lower()
    return any(n in pred_lower for n in gold_nums)
