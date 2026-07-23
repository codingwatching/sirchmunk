"""benchmarks/financebench/adapter.py — FinanceBenchAdapter

将现有 financebench 代码（config.py / data_loader.py / judge.py）包装为
BenchmarkAdapter 接口，不修改任何原始文件。

依赖倒置：外层 framework 只依赖 BenchmarkAdapter，不直接 import 具体代码。

Work Path 隔离原则：
  所有缓存（KnowledgeStorage / tree index / rga cache）均存放在
  benchmarks/financebench/.work/ 下，与其他 benchmark 完全隔离。
  相对路径 FB_WORK_PATH 以本文件所在目录（benchmarks/financebench/）为基准，
  而非调用者的 CWD，避免从项目根目录运行时路径错误。
"""
from __future__ import annotations

import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ── sys.path 注入：允许从任意目录运行 ────────────────────────────────
_HERE = Path(__file__).parent.resolve()          # benchmarks/financebench/
_BENCHMARKS_ROOT = _HERE.parent                  # benchmarks/
_PROJECT_ROOT = _BENCHMARKS_ROOT.parent          # sirchmunk/
_SRC = _PROJECT_ROOT / "src"

# Layer 0 全局共享配置文件路径（可选）
_GLOBAL_ENV = _BENCHMARKS_ROOT / ".env.global"

for _p in (_HERE, str(_SRC)):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
# ─────────────────────────────────────────────────────────────────────

from config import FinanceBenchConfig          # noqa: E402  (financebench/config.py)
from data_loader import FinanceBenchLoader     # noqa: E402

# framework 导入必须用绝对或相对路径
sys.path.insert(0, str(_BENCHMARKS_ROOT))
from framework.adapter import BenchmarkAdapter  # noqa: E402
from framework.schema import BenchmarkSample    # noqa: E402


def _load_global_env_as_defaults() -> None:
    """将 benchmarks/.env.global 中的键值注入 os.environ（最低优先级）。

    只有 os.environ 中尚未设置的 key 才会被注入，确保:
      benchmarks/.env.global  <  .env.financebench  <  os.environ

    调用时机：在 FinanceBenchConfig.from_env() 之前，作为全局默认层。
    """
    if not _GLOBAL_ENV.exists():
        return
    for line in _GLOBAL_ENV.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        # 仅当 os.environ 中未设置时才注入（最低优先级）
        os.environ.setdefault(k, v)


def _resolve_work_path(raw: str) -> str:
    """将 work_path 以 benchmarks/financebench/ 为基准解析为绝对路径。

    若 raw 已是绝对路径则直接返回（保持向后兼容）；
    若为相对路径（如 './.work'），以本文件所在目录（_HERE）为基准解析，
    而非调用者的 CWD。这确保从项目根目录运行时缓存仍隔离在 benchmark 目录下。
    """
    p = Path(raw)
    if p.is_absolute():
        return str(p.resolve())
    return str((_HERE / p).resolve())


class FinanceBenchAdapter(BenchmarkAdapter):
    """FinanceBench 适配器。

    Usage::

        adapter = FinanceBenchAdapter(
            env_file="benchmarks/financebench/.env.financebench"
        )
    """

    def __init__(self, env_file: str) -> None:
        """
        Args:
            env_file: .env.financebench 的绝对或相对路径。
        """
        # Step 1: 注入全局 Layer 0 默认配置（最低优先级）
        _load_global_env_as_defaults()

        self._env_file = str(Path(env_file).resolve())
        self._cfg = FinanceBenchConfig.from_env(self._env_file)
        self._loader = FinanceBenchLoader(
            data_dir=self._cfg.data_dir,
            pdf_dir=self._cfg.pdf_dir,
        )
        self._searcher = None   # 惰性初始化
        self._judge = None      # 惰性初始化

        # Work Path 隔离：相对路径以本文件目录为基准（非 CWD）
        self._resolved_work_path = _resolve_work_path(self._cfg.work_path)

    # ------------------------------------------------------------------
    # BenchmarkAdapter 接口实现
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "financebench"

    @property
    def env_file(self) -> str:
        return self._env_file

    def load_samples(self, limit: int = 0, seed: int = 42) -> List[BenchmarkSample]:
        """加载 FinanceBench 问题，转为 BenchmarkSample。"""
        questions = self._loader.load_questions()
        if limit > 0 and limit < len(questions):
            random.seed(seed)
            questions = random.sample(questions, limit)

        samples = []
        for q in questions:
            samples.append(BenchmarkSample(
                sample_id=q.get("financebench_id", ""),
                question=q["question"],
                gold_answer=q.get("answer", ""),
                metadata={
                    "doc_name":          q.get("doc_name", ""),
                    "company":           q.get("company", ""),
                    "question_type":     q.get("question_type", ""),
                    "question_reasoning": q.get("question_reasoning", ""),
                },
            ))
        return samples

    def validate_corpus(self) -> Tuple[int, List[str]]:
        questions = self._loader.load_questions()
        return self._loader.validate_corpus(questions)

    def get_search_paths(self, sample: BenchmarkSample) -> List[str]:
        """根据 eval_mode 返回搜索路径。"""
        if self._cfg.eval_mode == "singleDoc":
            doc_name = sample.metadata.get("doc_name", "")
            pdf_path = self._loader.get_pdf_path(doc_name)
            if pdf_path:
                return [pdf_path]
        return [self._cfg.pdf_dir]

    def get_run_config(self) -> Dict[str, Any]:
        """返回可序列化的配置字典。"""
        return {
            "mode":              self._cfg.mode,
            "eval_mode":         self._cfg.eval_mode,
            "top_k_files":       self._cfg.top_k_files,
            "max_token_budget":  self._cfg.max_token_budget,
            "enable_dir_scan":   self._cfg.enable_dir_scan,
            "enable_llm_judge":  self._cfg.enable_llm_judge,
            "llm_model":         self._cfg.llm_model,
            "llm_base_url":      self._cfg.llm_base_url,
            "max_concurrent":    self._cfg.max_concurrent,
        }

    def build_searcher(self) -> Any:
        """构建并缓存 AgenticSearch 实例。

        使用 self.get_work_path()（基于 benchmark 目录的绝对路径），
        确保 KnowledgeStorage / rga cache / tree index 等缓存完全隔离。
        """
        if self._searcher is None:
            from sirchmunk.llm.openai_chat import OpenAIChat
            from sirchmunk.search import AgenticSearch

            llm = OpenAIChat(
                api_key=self._cfg.llm_api_key,
                base_url=self._cfg.llm_base_url,
                model=self._cfg.llm_model,
            )
            self._searcher = AgenticSearch(
                llm=llm,
                work_path=self.get_work_path(),  # 使用隔离后的绝对路径
                reuse_knowledge=False,
                verbose=False,
            )
        return self._searcher

    def build_judge(self) -> Optional[Any]:
        """构建并缓存 FinanceBenchLLMJudge 实例。"""
        if not self._cfg.enable_llm_judge:
            return None
        if self._judge is None:
            from judge import FinanceBenchLLMJudge   # financebench/judge.py
            searcher = self.build_searcher()
            self._judge = FinanceBenchLLMJudge(llm=searcher.llm)
        return self._judge

    def get_output_dir(self) -> str:
        """输出目录：相对路径以 benchmark 目录为基准。"""
        raw = self._cfg.output_dir
        p = Path(raw)
        if p.is_absolute():
            return str(p.resolve())
        return str((_HERE / p).resolve())

    def get_work_path(self) -> str:
        """返回隔离后的 work_path 绝对路径（基于 benchmarks/financebench/）。"""
        return self._resolved_work_path

    def get_max_concurrent(self) -> int:
        return self._cfg.max_concurrent

    def get_request_delay(self) -> float:
        return self._cfg.request_delay

    def get_search_kwargs(self) -> Dict[str, Any]:
        return {
            "mode":             self._cfg.mode,
            "top_k_files":      self._cfg.top_k_files,
            "max_token_budget": self._cfg.max_token_budget,
            "enable_dir_scan":  self._cfg.enable_dir_scan,
        }

    def extra_result_fields(self, sample: BenchmarkSample) -> Dict[str, Any]:
        """FinanceBench 特有字段直接透传。"""
        return {
            "financebench_id":    sample.sample_id,
            "company":            sample.metadata.get("company", ""),
            "doc_name":           sample.metadata.get("doc_name", ""),
            "question_type":      sample.metadata.get("question_type", ""),
            "question_reasoning": sample.metadata.get("question_reasoning", ""),
        }
