"""framework/advisor.py — ImprovementAdvisor

职责：接收 BadCaseReport + 当前 config，输出 List[ImprovementHypothesis]。

分两层：
1. 信号驱动快路径（无 LLM）：基于 root_cause_breakdown 和 metrics 直接生成规则建议
2. LLM 深度分析（1次 LLM call）：提供上下文，请 LLM 补充更深层假设

每条 hypothesis 只标注定性影响（low/medium/high），严禁承诺具体数字。
PIPELINE_PATCH / PROMPT_FIX 类只给文字描述，不自动修改代码。
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from .schema import BadCaseReport, ChangeType, ConfigLayer, ImpactLevel, ImprovementHypothesis, RootCause

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config 层级分类辅助
# ---------------------------------------------------------------------------

# Layer 0 全局配置键集合：修改这些键将影响所有 benchmark
_GLOBAL_CONFIG_KEYS: frozenset = frozenset({
    "LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL_NAME", "LLM_TIMEOUT",
    "EMBEDDING_MODEL_ID", "EMBEDDING_CACHE_DIR",
    "SIRCHMUNK_WORK_PATH", "GREP_CONCURRENT_LIMIT",
})


def _classify_config_layer(hypothesis: ImprovementHypothesis) -> ConfigLayer:
    """Classify a hypothesis into Config 三层隔离 layer.

    分类规则：
    - PIPELINE_PATCH / PROMPT_FIX 改的是 src/sirchmunk/ 中的共享代码 → Layer 0 GLOBAL
    - CONFIG_CHANGE 改的是 _GLOBAL_CONFIG_KEYS 中的全局 key → Layer 0 GLOBAL
    - CONFIG_CHANGE 改的是 benchmark 专属配置（如 HOTPOT_*）→ Layer 2 SPECIFIC
    - 默认：Layer 2 SPECIFIC（保守策略）
    """
    if hypothesis.change_type in (ChangeType.PIPELINE_PATCH, ChangeType.PROMPT_FIX):
        # 代码和 prompt 修改均影响全局
        return ConfigLayer.GLOBAL

    if hypothesis.change_type == ChangeType.CONFIG_CHANGE:
        keys = set(hypothesis.config_changes.keys())
        if keys & _GLOBAL_CONFIG_KEYS:
            # 包含全局 key → Layer 0
            return ConfigLayer.GLOBAL
        # 全是 benchmark 专属配置 → Layer 2
        return ConfigLayer.SPECIFIC

    return ConfigLayer.SPECIFIC


def _benchmark_env_key(env_file: str, suffix: str) -> str:
    """Infer a benchmark-specific env key from the env file path."""
    lower = (env_file or "").lower()
    if "hotpot" in lower:
        return f"HOTPOT_{suffix}"
    if "setup_cost" in lower or "freshness" in lower or "storage_overhead" in lower or "source_fidelity" in lower or "warm_reuse" in lower:
        return f"MECHANISM_{suffix}"
    return suffix


# ---------------------------------------------------------------------------

_DEEP_ANALYSIS_PROMPT = """\
You are a research engineer helping diagnose and improve a document QA retrieval system (Sirchmunk).

## Current Experiment Metrics
{metrics_summary}

## Failure Analysis Summary
{failure_summary}

## LLM-Identified Failure Patterns
{pattern_summary}

## Current Key Configuration
{config_summary}

## Task
Generate 2-4 concrete improvement hypotheses beyond the rule-based suggestions already made.
Each hypothesis must address a specific root cause observed in the failure analysis.

RULES:
- Do NOT promise specific accuracy numbers
- Clearly label risk (low/medium/high)
- For CONFIG_CHANGE: provide exact env var key and new value
- For PIPELINE_PATCH / PROMPT_FIX: describe the code change location and intent only (no code)
- Each hypothesis must be independent (can be applied without others)

Return ONLY a valid JSON array. Each element:
{{
  "title": "<short title ≤ 10 words>",
  "root_cause": "<retrieval_failure|evidence_partial|synthesis_error|judge_suspect|unknown>",
  "change_type": "<config_change|prompt_fix|pipeline_patch>",
  "description": "<what to change and why, ≤ 50 words>",
  "estimated_impact": "<low|medium|high>",
  "risk_level": "<low|medium|high>",
  "config_changes": {{<env_key>: "<new_value>"}},
  "env_file": "<relative path to .env file or empty>",
  "code_guidance": "<description of code change location and intent, or empty>"
}}
"""


class ImprovementAdvisor:
    """改进建议引擎。

    Usage::

        advisor = ImprovementAdvisor(llm=llm)
        hypotheses = await advisor.suggest(report, config, env_file)
    """

    def __init__(self, llm: Optional[Any] = None) -> None:
        """
        Args:
            llm: OpenAIChat 实例；为 None 时只给出规则驱动的建议。
        """
        self._llm = llm
        self._id_counter = 0

    def _next_id(self) -> str:
        self._id_counter += 1
        return f"H{self._id_counter:03d}"

    async def suggest(
        self,
        report: BadCaseReport,
        config: Dict[str, Any],
        env_file: str = "",
    ) -> List[ImprovementHypothesis]:
        """生成改进假设列表。

        Args:
            report:   BadCaseReport（来自 BadCaseAnalyzer.analyze()）。
            config:   当前运行配置字典（来自 adapter.get_run_config()）。
            env_file: 受影响的 .env 文件相对路径（供 CONFIG_CHANGE 使用）。

        Returns:
            排序好的 ImprovementHypothesis 列表（高影响优先）。
        """
        hypotheses: List[ImprovementHypothesis] = []

        # ---- Layer 1: 信号驱动规则建议 ----
        hypotheses.extend(self._rule_based_suggestions(report, config, env_file))

        # ---- Layer 2: LLM 深度分析 ----
        if self._llm:
            llm_hypotheses = await self._llm_deep_analysis(report, config, env_file)
            # 去重：标题相似的不重复添加
            existing_titles = {h.title.lower() for h in hypotheses}
            for h in llm_hypotheses:
                if h.title.lower() not in existing_titles:
                    hypotheses.append(h)
                    existing_titles.add(h.title.lower())

        # 按 estimated_impact 降序排列
        _impact_order = {ImpactLevel.HIGH: 0, ImpactLevel.MEDIUM: 1, ImpactLevel.LOW: 2}
        hypotheses.sort(key=lambda h: _impact_order.get(h.estimated_impact, 3))

        return hypotheses

    # ------------------------------------------------------------------
    # Layer 1: 规则驱动
    # ------------------------------------------------------------------

    def _rule_based_suggestions(
        self,
        report: BadCaseReport,
        config: Dict[str, Any],
        env_file: str,
    ) -> List[ImprovementHypothesis]:
        """基于 root_cause_breakdown 和 metrics 生成规则建议。"""
        results: List[ImprovementHypothesis] = []
        rc = report.root_cause_breakdown
        total_bad = max(report.total_badcases, 1)

        retrieval_pct = rc.get(RootCause.RETRIEVAL_FAILURE.value, 0) / total_bad
        evidence_partial_pct = rc.get(RootCause.EVIDENCE_PARTIAL.value, 0) / total_bad
        synthesis_pct = rc.get(RootCause.SYNTHESIS_ERROR.value, 0) / total_bad
        no_cov_pct = (report.failure_type_breakdown.get("no_coverage", 0)
                      + report.failure_type_breakdown.get("refusal", 0)) / total_bad

        current_top_k = int(config.get("top_k_files", 5))
        current_mode = config.get("mode", "FAST")
        top_k_key = str(config.get("top_k_env_key") or _benchmark_env_key(env_file, "TOP_K_FILES"))
        mode_key = str(config.get("mode_env_key") or _benchmark_env_key(env_file, "MODE"))

        # --- 规则 1: 检索失败率高 → 提升 top_k ---
        if retrieval_pct > 0.35 or no_cov_pct > 0.40:
            new_top_k = min(current_top_k + 5, 20)
            if new_top_k > current_top_k:
                h = ImprovementHypothesis(
                    hypothesis_id=self._next_id(),
                    title=f"Increase top_k_files {current_top_k} → {new_top_k}",
                    root_cause=RootCause.RETRIEVAL_FAILURE.value,
                    change_type=ChangeType.CONFIG_CHANGE,
                    description=(
                        f"Retrieval failure accounts for {retrieval_pct:.0%} of badcases. "
                        f"Increasing top_k_files from {current_top_k} to {new_top_k} "
                        f"exposes more candidate files to the search agent."
                    ),
                    estimated_impact=ImpactLevel.MEDIUM,
                    risk_level="low",
                    config_changes={top_k_key: str(new_top_k)},
                    env_file=env_file,
                    estimated_coverage_fraction=retrieval_pct,
                )
                h.config_layer = _classify_config_layer(h)  # Layer 2 (SPECIFIC)
                results.append(h)

        # --- 规则 2: 证据不完整率高 → 切换 DEEP 模式 ---
        if evidence_partial_pct > 0.25 and current_mode == "FAST":
            h = ImprovementHypothesis(
                hypothesis_id=self._next_id(),
                title="Switch search mode FAST → DEEP",
                root_cause=RootCause.EVIDENCE_PARTIAL.value,
                change_type=ChangeType.CONFIG_CHANGE,
                description=(
                    f"Evidence partial failure accounts for {evidence_partial_pct:.0%} of badcases. "
                    "DEEP mode uses multi-phase retrieval and Monte Carlo evidence sampling "
                    "which can surface evidence missed by FAST mode. Note: higher latency and cost."
                ),
                estimated_impact=ImpactLevel.HIGH,
                risk_level="medium",
                config_changes={mode_key: "DEEP"},
                env_file=env_file,
                estimated_coverage_fraction=evidence_partial_pct,
            )
            h.config_layer = _classify_config_layer(h)  # Layer 2 (SPECIFIC)
            results.append(h)

        # --- 规则 3: 合成错误率高 → 强化答案/证据校验 ---
        if synthesis_pct > 0.30:
            h = ImprovementHypothesis(
                hypothesis_id=self._next_id(),
                title="Strengthen synthesis verification",
                root_cause=RootCause.SYNTHESIS_ERROR.value,
                change_type=ChangeType.PIPELINE_PATCH,
                description=(
                    f"Synthesis errors account for {synthesis_pct:.0%} of badcases. "
                    "Consider strengthening final-answer extraction and evidence-grounding checks."
                ),
                estimated_impact=ImpactLevel.MEDIUM,
                risk_level="medium",
                code_guidance=(
                    "Files: src/sirchmunk/search.py and benchmark-specific judge/evidence modules\n"
                    "Change: ensure final synthesis returns a concise answer span and verify it "
                    "against retrieved evidence before marking coverage as true."
                ),
                estimated_coverage_fraction=synthesis_pct,
            )
            h.config_layer = _classify_config_layer(h)
            results.append(h)

        # --- 规则 4: judge_suspect 超过阈值 → 提醒人工审查 ---
        if report.judge_suspect_ids:
            pct = len(report.judge_suspect_ids) / total_bad
            if pct > 0.05:
                h = ImprovementHypothesis(
                    hypothesis_id=self._next_id(),
                    title="Review suspected judge false-negatives",
                    root_cause=RootCause.JUDGE_SUSPECT.value,
                    change_type=ChangeType.PIPELINE_PATCH,
                    description=(
                        f"{len(report.judge_suspect_ids)} cases ({pct:.0%} of badcases) "
                        "have numeric overlap between prediction and gold but judge=False. "
                        "Manual review recommended; consider adjusting judge confidence threshold."
                    ),
                    estimated_impact=ImpactLevel.LOW,
                    risk_level="low",
                    code_guidance=(
                        "File: benchmarks/hotpotqa/judge.py or the active benchmark judge module\n"
                        "Class: HotpotQAJudge (or equivalent)\n"
                        "Change: Review the semantic judge confidence threshold and F1 fallback "
                        "policy to reduce false negatives without introducing false positives."
                    ),
                    estimated_coverage_fraction=pct,
                )
                # Benchmark-specific judge modules are treated as SPECIFIC unless the adapter marks otherwise.
                h.config_layer = ConfigLayer.SPECIFIC
                results.append(h)

        return results

    # ------------------------------------------------------------------
    # Layer 2: LLM 深度分析
    # ------------------------------------------------------------------

    async def _llm_deep_analysis(
        self,
        report: BadCaseReport,
        config: Dict[str, Any],
        env_file: str,
    ) -> List[ImprovementHypothesis]:
        """单次 LLM call，生成补充假设。"""
        metrics_summary = (
            f"Accuracy: {report.accuracy:.1f}%  Coverage: {report.coverage:.1f}%  "
            f"Total: {report.total_samples}  Badcases: {report.total_badcases}"
        )
        failure_summary = "\n".join(
            f"  {k}: {v} ({v / max(report.total_badcases, 1) * 100:.1f}%)"
            for k, v in sorted(report.root_cause_breakdown.items(), key=lambda x: -x[1])
        ) or "  (no badcases)"

        pattern_summary = report.pattern_summary or "(not available)"

        # 只展示关键 config 字段
        key_config = {
            k: v for k, v in config.items()
            if any(s in k.lower() for s in (
                "mode", "top_k", "budget", "judge", "eval_mode", "model"
            ))
        }
        config_summary = json.dumps(key_config, indent=2, ensure_ascii=False)

        prompt = _DEEP_ANALYSIS_PROMPT.format(
            metrics_summary=metrics_summary,
            failure_summary=failure_summary,
            pattern_summary=pattern_summary,
            config_summary=config_summary,
        )

        try:
            resp = await self._llm.achat(
                messages=[{"role": "user", "content": prompt}],
                stream=False,
            )
            raw = (resp.content or "").strip()
            items = self._parse_json_array(raw)
            return [self._dict_to_hypothesis(item, env_file) for item in items if item]
        except Exception as exc:
            logger.warning("[Advisor] LLM deep analysis failed: %s", exc)
            return []

    def _dict_to_hypothesis(self, item: Dict[str, Any], env_file: str) -> ImprovementHypothesis:
        """将 LLM 输出 dict 转为 ImprovementHypothesis，并自动标注 config_layer。"""
        try:
            ct = ChangeType(item.get("change_type", "config_change"))
        except ValueError:
            ct = ChangeType.CONFIG_CHANGE

        try:
            impact = ImpactLevel(item.get("estimated_impact", "medium"))
        except ValueError:
            impact = ImpactLevel.MEDIUM

        h = ImprovementHypothesis(
            hypothesis_id=self._next_id(),
            title=str(item.get("title", "LLM suggestion"))[:80],
            root_cause=str(item.get("root_cause", "unknown")),
            change_type=ct,
            description=str(item.get("description", ""))[:500],
            estimated_impact=impact,
            risk_level=str(item.get("risk_level", "low")),
            config_changes=dict(item.get("config_changes") or {}),
            env_file=str(item.get("env_file") or env_file),
            code_guidance=str(item.get("code_guidance", "")),
        )
        # 自动标注层级
        h.config_layer = _classify_config_layer(h)
        return h

    @staticmethod
    def _parse_json_array(raw: str) -> List[Dict[str, Any]]:
        """从 LLM 响应中提取 JSON 数组。"""
        import re
        # 去除 markdown code fences
        cleaned = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`").strip()
        # 尝试直接解析
        try:
            result = json.loads(cleaned)
            if isinstance(result, list):
                return result
        except (json.JSONDecodeError, ValueError):
            pass
        # 提取第一个 [...] 块
        m = re.search(r"\[.*\]", cleaned, re.DOTALL)
        if m:
            try:
                result = json.loads(m.group())
                if isinstance(result, list):
                    return result
            except (json.JSONDecodeError, ValueError):
                pass
        return []
