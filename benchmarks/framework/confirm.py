"""framework/confirm.py — HumanConfirmLoop

CLI 人工确认层：展示改进建议 → 收集用户选择 → 执行 CONFIG_CHANGE。

设计原则：
- CONFIG_CHANGE：框架自动写入 .env 文件，执行前展示完整 diff，**不静默修改**
- PIPELINE_PATCH / PROMPT_FIX：只打印建议，提示"可将建议粘贴给 Qoder 执行"
- 选 q：退出循环，保留所有已记录实验
"""
from __future__ import annotations

import logging
import re
import sys
from pathlib import Path
from typing import List, Optional, Tuple

from .schema import ChangeType, ConfigLayer, ImprovementHypothesis

logger = logging.getLogger(__name__)

_SEP = "=" * 64
_SUBSEP = "─" * 64


class HumanConfirmLoop:
    """人工确认层。

    Usage::

        confirm = HumanConfirmLoop()
        chosen, applied = confirm.review(hypotheses)
    """

    def __init__(self, dry_run: bool = False) -> None:
        """
        Args:
            dry_run: 若为 True，则不实际写 .env，只打印 diff。
        """
        self._dry_run = dry_run

    def review(
        self, hypotheses: List[ImprovementHypothesis]
    ) -> Tuple[List[ImprovementHypothesis], List[ImprovementHypothesis]]:
        """展示建议并收集用户确认。

        Returns:
            (chosen, applied)
            - chosen:  用户选择的所有 hypothesis（含 PIPELINE_PATCH）
            - applied: 实际执行（写 .env）的 hypothesis（仅 CONFIG_CHANGE）
        """
        if not hypotheses:
            print("\n  (No improvement hypotheses generated.)\n")
            return [], []

        self._print_hypotheses(hypotheses)
        chosen_indices = self._prompt_user(len(hypotheses))

        if chosen_indices is None:
            print("  [Confirm] Quit — no changes applied.")
            return [], []

        if not chosen_indices:
            print("  [Confirm] Skipped — no changes applied.")
            return [], []

        chosen = [hypotheses[i] for i in chosen_indices]
        applied: List[ImprovementHypothesis] = []

        for h in chosen:
            if h.change_type == ChangeType.CONFIG_CHANGE and h.config_changes:
                success = self._apply_config_change(h)
                if success:
                    applied.append(h)
            elif h.change_type in (ChangeType.PIPELINE_PATCH, ChangeType.PROMPT_FIX):
                self._print_code_guidance(h)
            # SKIP 类不需要处理

        return chosen, applied

    # ------------------------------------------------------------------
    # Printing
    # ------------------------------------------------------------------

    @staticmethod
    def _print_hypotheses(hypotheses: List[ImprovementHypothesis]) -> None:
        print(f"\n{_SEP}")
        print("  改进建议 (Improvement Hypotheses)")
        print(_SEP)

        for i, h in enumerate(hypotheses, 1):
            impact_icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(
                h.estimated_impact.value if hasattr(h.estimated_impact, "value")
                else str(h.estimated_impact), "⚪"
            )
            change_label = {
                ChangeType.CONFIG_CHANGE:  "CONFIG  ",
                ChangeType.PROMPT_FIX:     "PROMPT  ",
                ChangeType.PIPELINE_PATCH: "PIPELINE",
                ChangeType.SKIP:           "SKIP    ",
            }.get(h.change_type, "UNKNOWN ")

            # Config 层级图标
            layer_val = getattr(h, "config_layer", None)
            if layer_val == ConfigLayer.GLOBAL or layer_val == 0:
                layer_icon = "🌐 Layer-0 GLOBAL"
                layer_warn = "  ⚠໳  全局变更！将影响所有 benchmark"
            elif layer_val == ConfigLayer.FAMILY or layer_val == 1:
                layer_icon = "🟡 Layer-1 FAMILY"
                layer_warn = "  ⚠️  家族变更，影响同类 benchmark"
            else:
                layer_icon = "✅ Layer-2 SPECIFIC"
                layer_warn = ""

            print(f"\n[{i}] {change_label}  {impact_icon} impact={h.estimated_impact}  "
                  f"risk={h.risk_level}")
            print(f"    {layer_icon}")
            if layer_warn:
                print(f"   {layer_warn}")
            print(f"    {h.title}")
            print(f"    Root cause : {h.root_cause}")
            print(f"    Description: {h.description[:160]}")

            if h.change_type == ChangeType.CONFIG_CHANGE and h.config_changes:
                print(f"    Changes:")
                for k, v in h.config_changes.items():
                    print(f"      {k} = {v}")
                if h.env_file:
                    print(f"    File: {h.env_file}")

            if h.change_type in (ChangeType.PIPELINE_PATCH, ChangeType.PROMPT_FIX):
                print(f"    Guidance: (see details below after selection)")

        print(f"\n{_SUBSEP}")
        print("  请选择操作：")
        print("    数字 (如 1 2 3)  — 选择指定建议")
        print("    all              — 选择所有建议")
        print("    skip             — 跳过本轮，不做改动")
        print("    q / quit         — 退出研究循环")
        print(_SUBSEP)

    @staticmethod
    def _prompt_user(total: int) -> Optional[List[int]]:
        """收集用户输入，返回 0-based 索引列表；None 表示退出；[] 表示 skip。"""
        while True:
            try:
                raw = input("  选择 > ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print()
                return None

            if raw in ("q", "quit", "exit"):
                return None

            if raw in ("skip", "s", ""):
                return []

            if raw == "all":
                return list(range(total))

            # 解析数字（空格或逗号分隔）
            tokens = re.split(r"[\s,]+", raw)
            indices = []
            valid = True
            for t in tokens:
                if not t:
                    continue
                try:
                    idx = int(t) - 1   # 1-based → 0-based
                    if 0 <= idx < total:
                        indices.append(idx)
                    else:
                        print(f"  ❌ 选项 {t} 超出范围（1-{total}），请重新输入。")
                        valid = False
                        break
                except ValueError:
                    print(f"  ❌ 无法识别 '{t}'，请输入数字、all、skip 或 q。")
                    valid = False
                    break

            if valid and indices:
                # 去重且保序
                seen = set()
                unique = []
                for i in indices:
                    if i not in seen:
                        seen.add(i)
                        unique.append(i)
                return unique

    # ------------------------------------------------------------------
    # CONFIG_CHANGE executor
    # ------------------------------------------------------------------

    def _apply_config_change(self, h: ImprovementHypothesis) -> bool:
        """将 config_changes 写入 .env 文件。

        步骤：
        1. 打印 diff
        2. 用户再次确认（y/n）
        3. 写文件

        Returns:
            True 表示成功应用。
        """
        env_path = Path(h.env_file) if h.env_file else None
        if not env_path or not env_path.exists():
            print(f"\n  ⚠️  .env 文件不存在: {h.env_file}，跳过自动写入。")
            print(f"     请手动应用以下变更：")
            for k, v in h.config_changes.items():
                print(f"       {k}={v}")
            return False

        # Layer 0 全局变更额外警告
        layer_val = getattr(h, "config_layer", None)
        if layer_val == ConfigLayer.GLOBAL or layer_val == 0:
            print(f"\n  🌐 注意：此 CONFIG_CHANGE 包含全局配置键（Layer 0）")
            print(f"     将影响所有已注册的 benchmark！")
            print(f"     建议先在所有 benchmark 上做联合评估（将来 Pareto Gate）后再执行。")
            print()

        # 读取当前内容
        current_lines = env_path.read_text(encoding="utf-8").splitlines(keepends=True)
        new_lines, applied_keys = _apply_env_changes(current_lines, h.config_changes)

        # 打印 diff
        print(f"\n  📝 将修改 {h.env_file}：")
        print("  " + _SUBSEP)
        for k, v in h.config_changes.items():
            if k in applied_keys:
                old_val = applied_keys[k]
                print(f"  - {k}={old_val}")
                print(f"  + {k}={v}")
            else:
                print(f"  + {k}={v}  (new entry)")
        print("  " + _SUBSEP)

        if self._dry_run:
            print("  [DRY RUN] 未实际写入文件。\n")
            return False

        # 二次确认
        try:
            confirm = input("  确认写入? [y/N] > ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return False

        if confirm not in ("y", "yes"):
            print("  已取消。\n")
            return False

        env_path.write_text("".join(new_lines), encoding="utf-8")
        print(f"  ✅ 已更新 {h.env_file}\n")
        return True

    @staticmethod
    def _print_code_guidance(h: ImprovementHypothesis) -> None:
        """打印代码修改建议，提示用户手动或通过 Qoder 执行。"""
        print(f"\n  📌 [{h.change_type.value.upper()}] {h.title}")
        print("  " + _SUBSEP)
        if h.code_guidance:
            for line in h.code_guidance.splitlines():
                print(f"    {line}")
        else:
            print(f"    {h.description}")
        print("  " + _SUBSEP)
        print("  ℹ️  此修改需手动执行或粘贴给 Qoder 执行，框架不自动修改代码。\n")


# ---------------------------------------------------------------------------
# .env file manipulation helpers
# ---------------------------------------------------------------------------

def _apply_env_changes(
    lines: List[str],
    changes: dict,
) -> Tuple[List[str], dict]:
    """将 changes 应用到 .env 行列表。

    - 若 key 已存在（含注释版本），覆盖其值
    - 若 key 不存在，追加到文件末尾

    Returns:
        (new_lines, old_values_map)   old_values_map: {key: old_value}
    """
    new_lines = list(lines)
    old_values: dict = {}
    pending_keys = set(changes.keys())

    for i, line in enumerate(new_lines):
        stripped = line.strip()
        # 跳过注释
        if stripped.startswith("#"):
            continue
        if "=" not in stripped:
            continue
        key, _, _ = stripped.partition("=")
        key = key.strip()
        if key in pending_keys:
            # 保留行尾注释（如 KEY=VALUE  # comment）
            _, _, rest = stripped.partition("=")
            comment_match = re.search(r"\s+#.*$", rest)
            comment = comment_match.group(0) if comment_match else ""
            old_val = rest.split("#")[0].strip()
            old_values[key] = old_val
            new_lines[i] = f"{key}={changes[key]}{comment}\n"
            pending_keys.discard(key)

    # 追加未找到的 key
    if pending_keys:
        if new_lines and not new_lines[-1].endswith("\n"):
            new_lines.append("\n")
        for key in sorted(pending_keys):
            new_lines.append(f"{key}={changes[key]}\n")

    return new_lines, old_values
