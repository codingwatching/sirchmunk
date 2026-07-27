"""benchmarks/hotpotqa/adapter.py — HotpotQAAdapter

HotpotQA benchmark adapter for the ResearchOps framework.
It delegates loading, judging, evidence evaluation, and metrics to dedicated
modules so fullwiki runs can be tracked with reproducible artifacts.
"""
from __future__ import annotations

import os
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ── sys.path 注入 ─────────────────────────────────────────────────────
_HERE = Path(__file__).parent.resolve()      # benchmarks/hotpotqa/
_BENCHMARKS_ROOT = _HERE.parent              # benchmarks/
_PROJECT_ROOT = _BENCHMARKS_ROOT.parent      # sirchmunk/
_SRC = _PROJECT_ROOT / "src"

# Layer 0 全局共享配置文件路径（可选）
_GLOBAL_ENV = _BENCHMARKS_ROOT / ".env.global"

# Layer 1 HotpotQA 共享配置文件路径（可选，profile 可覆盖）
_HOTPOT_BASE_ENV = _HERE / ".env.hotpotqa.base"

# HotpotQA 专属 work_path（固定，不受 CWD 影响）
_HOTPOT_WORK_PATH = str(_HERE / ".work")

for _p in (str(_SRC),):
    if _p not in sys.path:
        sys.path.insert(0, _p)

sys.path.insert(0, str(_BENCHMARKS_ROOT))
from framework.adapter import BenchmarkAdapter  # noqa: E402
from framework.protocol import default_protocol  # noqa: E402
from framework.schema import BenchmarkSample    # noqa: E402
from evaluation.sampling_protocol import extract_sample_ids  # noqa: E402
from hotpotqa.evidence import evaluate_supporting_facts  # noqa: E402
from hotpotqa.judge import HotpotQAJudge  # noqa: E402
from hotpotqa.loader import (  # noqa: E402
    build_dataset_manifest,
    describe_hotpotqa_split,
    load_hotpotqa_samples,
    validate_hotpotqa_corpus,
)
from hotpotqa.metrics import compute_hotpotqa_metrics  # noqa: E402
# ─────────────────────────────────────────────────────────────────────


def _load_env_file(path: str | Path) -> Dict[str, str]:
    """简单解析 .env 文件为 dict（不依赖 python-dotenv）。"""
    result: Dict[str, str] = {}
    p = Path(path)
    if not p.exists():
        return result
    for raw_line in p.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        result[key.strip()] = value.strip().strip('"').strip("'")
    return result


def _default_base_env_path() -> Path:
    override = os.environ.get("HOTPOT_BASE_ENV_FILE", "").strip()
    return Path(override).expanduser().resolve() if override else _HOTPOT_BASE_ENV


def _load_env_layers(profile_env_file: str) -> tuple[Dict[str, str], List[str]]:
    """加载 HotpotQA 多层 env，优先级为 global < base < profile < os.environ。"""
    env: Dict[str, str] = {}
    sources: List[str] = []
    for path in (_GLOBAL_ENV, _default_base_env_path(), Path(profile_env_file)):
        layer = _load_env_file(path)
        if layer:
            env.update(layer)
            sources.append(str(Path(path).resolve()))
    return env, sources


class HotpotQAAdapter(BenchmarkAdapter):
    """HotpotQA 适配器（占位实现）。

    当前状态：接口已完整实现，数据加载使用 parquet 文件。
    如需完整运行，确保 HOTPOT_DATASET_DIR 正确配置，
    并安装 pyarrow / pandas 依赖。

    Usage::

        adapter = HotpotQAAdapter(
            env_file="benchmarks/hotpotqa/.env.hotpotqa.frozen"
        )
    """

    def __init__(self, env_file: str) -> None:
        # HotpotQA keeps an isolated work_path to avoid cache pollution.
        self._env_file = str(Path(env_file).resolve())
        self._env, self._env_sources = _load_env_layers(self._env_file)
        self._searcher = None
        self._judge = None

    # ------------------------------------------------------------------
    # Config helpers
    # ------------------------------------------------------------------

    def _get(self, key: str, default: str = "") -> str:
        value = os.environ.get(key)
        if value is not None and value != "":
            return value
        value = self._env.get(key)
        if value is not None and value != "":
            return value
        return default

    def _get_int(self, key: str, default: int = 0) -> int:
        try:
            return int(self._get(key, str(default)))
        except (ValueError, TypeError):
            return default

    def _get_bool(self, key: str, default: bool = False) -> bool:
        return self._get(key, str(default)).lower() in ("true", "1", "yes")

    def _judge_model_name(self, *, required: bool = True) -> str:
        """Return the explicitly configured HotpotQA judge model."""
        model = self._get("HOTPOT_JUDGE_MODEL_NAME", "")
        if model or not required:
            return model
        raise RuntimeError(
            "HOTPOT_JUDGE_MODEL_NAME is required when HOTPOT_ENABLE_LLM_JUDGE=true. "
            "It configures the HotpotQA judge model separately from LLM_MODEL_NAME."
        )

    def get_profile_limit(self, default: int = 0) -> int:
        return self._get_int("HOTPOT_LIMIT", default)

    @property
    def name(self) -> str:
        return "hotpotqa"

    @property
    def env_file(self) -> str:
        return self._env_file

    # ------------------------------------------------------------------
    # BenchmarkAdapter 接口
    # ------------------------------------------------------------------

    def load_samples(self, limit: int = 0, seed: int = 42) -> List[BenchmarkSample]:
        """Load HotpotQA samples from parquet files via HotpotQALoader.

        If HOTPOT_SAMPLE_IDS_FILE is configured, the adapter returns exactly
        those sample IDs in file order.  This lets Sirchmunk frozen runs execute
        the same sampled GoldenSet used later by run_evaluation.py.
        """
        samples = self.load_sampling_population(seed=seed)
        fixed_sample_ids_file = self._sample_ids_file()
        if self._require_context_answerable():
            samples = [s for s in samples if self._is_context_answerable(s)]
        if fixed_sample_ids_file:
            samples = self._filter_samples_by_ids(samples, fixed_sample_ids_file)
        elif limit > 0 and limit < len(samples):
            random.seed(seed)
            samples = random.sample(samples, limit)
        return samples

    def load_sampling_population(self, seed: int = 42) -> List[BenchmarkSample]:
        """Load the full declared split for sampling protocol generation.

        This intentionally ignores HOTPOT_SAMPLE_IDS_FILE and context-answerable
        smoke filters so GoldenSet creation always sees the 7405-example
        fullwiki validation population.
        """
        dataset_dir = Path(self._get("HOTPOT_DATASET_DIR", ""))
        return load_hotpotqa_samples(
            dataset_dir,
            setting=self._get("HOTPOT_SETTING", "fullwiki"),
            split=self._get("HOTPOT_SPLIT", "validation"),
            limit=0,
            seed=seed,
        )

    def validate_corpus(self) -> Tuple[int, List[str]]:
        """Validate HotpotQA wiki corpus availability."""
        return validate_hotpotqa_corpus(Path(self._wiki_dir()))

    def get_search_paths(self, sample: BenchmarkSample) -> List[str]:
        """Return search paths for this sample.

        Exploration smoke runs may use the parquet-provided HotpotQA context as
        a per-sample raw-text corpus.  Frozen/fullwiki runs should keep the
        default ``wiki`` mode and search the configured Wikipedia corpus.
        """
        mode = self._context_corpus_mode()
        if mode == "sample":
            return [str(self._materialize_sample_context(sample))]
        if mode == "hybrid":
            return [str(self._materialize_sample_context(sample)), self._wiki_dir()]
        return [self._wiki_dir()]

    def get_run_config(self) -> Dict[str, Any]:
        return {
            "mode":             self._get("HOTPOT_MODE", "DEEP"),
            "top_k_files":      self._get_int("HOTPOT_TOP_K_FILES", 10),
            "max_token_budget": self._get_int("HOTPOT_MAX_TOKEN_BUDGET", 128000),
            "enable_dir_scan":  self._get_bool("HOTPOT_ENABLE_DIR_SCAN", True),
            "enable_memory":    self._get_bool("SIRCHMUNK_ENABLE_MEMORY", False),
            "enable_eval_feedback": self._get_bool("HOTPOT_ENABLE_EVAL_FEEDBACK", False),
            "enable_llm_judge": self._get_bool("HOTPOT_ENABLE_LLM_JUDGE", True),
            "allow_frozen_llm_judge_auxiliary": self._get_bool("ALLOW_FROZEN_LLM_JUDGE_AUXILIARY", False),
            "enable_gpt_eval":  self._get_bool("HOTPOT_ENABLE_GPT_EVAL", False),
            "llm_model":        self._get("LLM_MODEL_NAME", ""),
            "judge_model":      self._judge_model_name(required=False),
            "llm_base_url":     self._get("LLM_BASE_URL", ""),
            "max_concurrent":   self._get_int("HOTPOT_MAX_CONCURRENT", 3),
            "setting":          self._get("HOTPOT_SETTING", "fullwiki"),
            "split":            self._get("HOTPOT_SPLIT", "validation"),
            "limit":            self.get_profile_limit(0),
            "sample_ids_file":   self._sample_ids_file(),
            "context_corpus_mode": self._context_corpus_mode(),
            "require_context_answerable": self._require_context_answerable(),
            "reuse_knowledge":  self._get_bool("HOTPOT_REUSE_KNOWLEDGE", False),
            "cache_mode":       self._get("HOTPOT_CACHE_MODE", "cold"),
            "sample_timeout_seconds": self._get_int("SAMPLE_TIMEOUT_SECONDS", 0),
            "system_timeout_seconds": self._get_int("SYSTEM_TIMEOUT_SECONDS", 0),
            "benchmark_timeout_seconds": self._get_int("BENCHMARK_TIMEOUT_SECONDS", 0),
            "global_timeout_seconds": self._get_int("GLOBAL_TIMEOUT_SECONDS", 0),
            "max_runtime_seconds": self._get_int("MAX_RUNTIME_SECONDS", 0),
            "max_total_tokens": self._get_int("MAX_TOTAL_TOKENS", 0),
            "max_api_cost_usd": float(self._get("MAX_API_COST_USD", "0") or 0),
            "max_disk_usage_bytes": self._get_int("MAX_DISK_USAGE_BYTES", 0),
            "min_free_disk_bytes": self._get_int("MIN_FREE_DISK_BYTES", 0),
            "env_sources": list(self._env_sources),
            "top_k_env_key":    "HOTPOT_TOP_K_FILES",
            "mode_env_key":     "HOTPOT_MODE",
            "judge_model_env_key": "HOTPOT_JUDGE_MODEL_NAME",
            "judge_threshold_env_key": "HOTPOT_JUDGE_F1_THRESHOLD",
        }

    def build_searcher(self) -> Any:
        """构建并缓存 AgenticSearch 实例。

        work_path 使用 self.get_work_path()（benchmarks/hotpotqa/.work），
        与 FinanceBenchAdapter 的 .work 完全隔离。
        """
        if self._searcher is None:
            from sirchmunk.llm.openai_chat import OpenAIChat
            from sirchmunk.search import AgenticSearch

            llm = OpenAIChat(
                api_key=self._get("LLM_API_KEY", ""),
                base_url=self._get("LLM_BASE_URL", "https://api.openai.com/v1"),
                model=self._get("LLM_MODEL_NAME", "gpt-4o-mini"),
            )
            self._searcher = AgenticSearch(
                llm=llm,
                work_path=self.get_work_path(),  # 使用隔离后的绝对路径
                reuse_knowledge=self._get_bool("HOTPOT_REUSE_KNOWLEDGE", False),
                verbose=False,
            )
        return self._searcher

    def build_judge(self) -> Optional[Any]:
        """Build HotpotQA EM/F1 judge with optional LLM semantic fallback."""
        if self._judge is None:
            llm = None
            enable_llm_judge = self._get_bool("HOTPOT_ENABLE_LLM_JUDGE", True)
            if enable_llm_judge:
                judge_model = self._judge_model_name()
                try:
                    from sirchmunk.llm.openai_chat import OpenAIChat

                    llm = OpenAIChat(
                        api_key=self._get("LLM_API_KEY", ""),
                        base_url=self._get("LLM_BASE_URL", "https://api.openai.com/v1"),
                        model=judge_model,
                    )
                except Exception as exc:
                    raise RuntimeError(
                        "Failed to build HotpotQA judge LLM. "
                        "Check LLM_BASE_URL, LLM_API_KEY, HOTPOT_JUDGE_MODEL_NAME, and benchmark env layering."
                    ) from exc
            self._judge = HotpotQAJudge(
                llm=llm,
                enable_llm_judge=enable_llm_judge,
                llm_fallback_f1_threshold=float(self._get("HOTPOT_JUDGE_F1_THRESHOLD", "0.3")),
            )
        return self._judge

    def get_output_dir(self) -> str:
        """HotpotQA 输出目录：相对路径以 benchmark 目录为基准。"""
        raw = self._get("HOTPOT_OUTPUT_DIR", "./output")
        p = Path(raw)
        if p.is_absolute():
            return str(p.resolve())
        return str((_HERE / p).resolve())

    def get_work_path(self) -> str:
        """返回 HotpotQA 专属 work_path（固定为 benchmarks/hotpotqa/.work）。

        不受 HOTPOT_OUTPUT_DIR 或 CWD 影响，与 FinanceBenchAdapter 缓存完全隔离。
        """
        return _HOTPOT_WORK_PATH

    def get_max_concurrent(self) -> int:
        return self._get_int("HOTPOT_MAX_CONCURRENT", 3)

    def get_request_delay(self) -> float:
        try:
            return float(self._get("HOTPOT_REQUEST_DELAY", "0.5"))
        except ValueError:
            return 0.5

    def get_search_kwargs(self) -> Dict[str, Any]:
        return {
            "mode":             self._get("HOTPOT_MODE", "DEEP"),
            "top_k_files":      self._get_int("HOTPOT_TOP_K_FILES", 10),
            "max_token_budget": self._get_int("HOTPOT_MAX_TOKEN_BUDGET", 128000),
            "enable_dir_scan":  self._get_bool("HOTPOT_ENABLE_DIR_SCAN", True),
        }

    def extra_result_fields(self, sample: BenchmarkSample) -> Dict[str, Any]:
        return {
            "hotpot_id": sample.sample_id,
            "type":      sample.metadata.get("type", ""),
            "level":     sample.metadata.get("level", ""),
            "answer_type": sample.metadata.get("answer_type", ""),
            "supporting_fact_count": sample.metadata.get("supporting_fact_count", 0),
            "supporting_fact_bucket": sample.metadata.get("supporting_fact_bucket", ""),
            "supporting_facts": sample.metadata.get("supporting_facts", []),
        }

    def get_protocol_spec(self, run_id: str, seed: int, limit: int) -> Dict[str, Any]:
        protocol = default_protocol(
            run_id=run_id,
            benchmark=self.name,
            config=self.get_run_config(),
            seed=seed,
        ).to_dict()
        protocol["suite"] = [f"hotpotqa_{self._get('HOTPOT_SETTING', 'fullwiki')}"]
        protocol["metrics"]["answer_quality"] = [
            "official_exact_match",
            "official_f1",
            "llm_assisted_accuracy",
            "coverage",
        ]
        protocol["metrics"]["retrieval"] = ["evidence_recall", "supporting_fact_hit_rate", "source_grounding_accuracy"]
        sample_ids_file = self._sample_ids_file()
        if sample_ids_file:
            protocol["sampling"] = {
                "sample_ids_file": sample_ids_file,
                "method": "fixed_sample_ids",
            }
        protocol["limit"] = limit
        return protocol

    def get_dataset_manifest(self) -> Dict[str, Any]:
        dataset_dir = Path(self._get("HOTPOT_DATASET_DIR", ""))
        manifest = build_dataset_manifest(
            dataset_dir,
            Path(self._wiki_dir()),
            setting=self._get("HOTPOT_SETTING", "fullwiki"),
            split=self._get("HOTPOT_SPLIT", "validation"),
        )
        sample_ids_file = self._sample_ids_file()
        if sample_ids_file:
            manifest["sample_ids_file"] = sample_ids_file
            try:
                ids = extract_sample_ids(sample_ids_file)
                manifest["sample_ids_count"] = len(ids)
            except Exception as exc:
                manifest["sample_ids_error"] = str(exc)
        return manifest

    def get_title_resolver(self):
        """Return a HotpotQA raw wiki title resolver for v4 dynamic snapshots."""
        from hotpotqa.title_resolver import HotpotQATitleResolver
        return HotpotQATitleResolver(self._wiki_dir())

    def derive_v4_sample_sets(self, golden_set: Any, *, stages: List[int], output_dir: str | Path):
        """Derive nested G_n sample-id artifacts from a parent GoldenSet."""
        from hotpotqa.dynamic_corpus import derive_nested_sample_sets
        return derive_nested_sample_sets(golden_set, stages=stages, output_dir=output_dir)

    def build_v4_corpus_snapshot(
        self,
        samples: List[BenchmarkSample],
        *,
        sample_ids: List[str],
        output_dir: str | Path,
        stage_name: str,
        materialize_mode: str = "symlink",
        background_ratio: float = 3.0,
        background_seed: int = 42,
    ):
        """Build a D_n snapshot aligned with one frozen G_n sample-id set."""
        from hotpotqa.dynamic_corpus import build_dynamic_corpus_snapshot
        return build_dynamic_corpus_snapshot(
            samples,
            sample_ids=sample_ids,
            wiki_dir=self._wiki_dir(),
            output_dir=output_dir,
            stage_name=stage_name,
            materialize_mode=materialize_mode,
            background_ratio=background_ratio,
            background_seed=background_seed,
            resolver=self.get_title_resolver(),
        )

    def enrich_telemetry(self, sample, prediction, telemetry, **kwargs) -> Dict[str, Any]:
        return evaluate_supporting_facts(
            sample.metadata.get("supporting_facts", []),
            read_file_ids=telemetry.get("read_file_ids", []),
            prediction=prediction,
            retrieval_logs=telemetry.get("retrieval_logs", []),
            evidence_sources=telemetry.get("evidence_sources", []),
            evidence_texts=telemetry.get("evidence_snippets", []),
            context=sample.metadata.get("context"),
        )

    def get_analysis_schema(self) -> Dict[str, Any]:
        return {
            "primary_group_key": "type",
            "secondary_group_key": "level",
            "evidence_key": "supporting_facts",
            "multi_hop": True,
            "numeric_sensitive": False,
        }

    def get_config_schema(self) -> Dict[str, Any]:
        return {
            "global_keys": [
                "LLM_BASE_URL",
                "LLM_API_KEY",
                "LLM_MODEL_NAME",
                "LLM_TIMEOUT",
                "EMBEDDING_MODEL_ID",
                "EMBEDDING_CACHE_DIR",
                "SIRCHMUNK_WORK_PATH",
                "GREP_CONCURRENT_LIMIT",
            ],
            "top_k_env_key": "HOTPOT_TOP_K_FILES",
            "mode_env_key": "HOTPOT_MODE",
            "judge_model_env_key": "HOTPOT_JUDGE_MODEL_NAME",
            "judge_threshold_env_key": "HOTPOT_JUDGE_F1_THRESHOLD",
        }

    def get_metric_aggregator(self):
        return compute_hotpotqa_metrics

    def describe_split(self) -> Dict[str, Any]:
        """Return HotpotQA split distribution for sampling protocol design."""
        return describe_hotpotqa_split(
            Path(self._get("HOTPOT_DATASET_DIR", "")),
            setting=self._get("HOTPOT_SETTING", "fullwiki"),
            split=self._get("HOTPOT_SPLIT", "validation"),
        )

    def _sample_ids_file(self) -> str:
        raw = self._get("HOTPOT_SAMPLE_IDS_FILE", "").strip()
        if not raw:
            return ""
        path = Path(raw).expanduser()
        if path.is_absolute():
            return str(path.resolve())
        candidates = [
            (_HERE / path),
            (_PROJECT_ROOT / path),
            (_BENCHMARKS_ROOT / path),
            (Path.cwd() / path),
        ]
        for candidate in candidates:
            if candidate.exists():
                return str(candidate.resolve())
        return str((_HERE / path).resolve())

    def _filter_samples_by_ids(self, samples: List[BenchmarkSample], sample_ids_file: str) -> List[BenchmarkSample]:
        sample_ids = extract_sample_ids(sample_ids_file)
        by_id = {str(sample.sample_id): sample for sample in samples}
        missing = [sample_id for sample_id in sample_ids if sample_id not in by_id]
        if missing:
            raise ValueError(
                f"HOTPOT_SAMPLE_IDS_FILE references ids not present in loaded split: "
                f"missing={missing[:10]} total_missing={len(missing)}"
            )
        return [by_id[sample_id] for sample_id in sample_ids]

    def _wiki_dir(self) -> str:
        dataset_dir = Path(self._get("HOTPOT_DATASET_DIR", ""))
        setting = self._get("HOTPOT_SETTING", "fullwiki")
        wiki_dir_override = self._get("HOTPOT_WIKI_CORPUS_DIR", "")
        if wiki_dir_override:
            return wiki_dir_override
        wiki_dirname = self._get(
            "HOTPOT_WIKI_CORPUS_DIRNAME",
            "enwiki-20171001-pages-meta-current-withlinks-abstracts",
        )
        # Support HOTPOT_DATASET_DIR pointing either to the dataset root or
        # directly to the setting directory (e.g. .../hotpotqa_dataset/fullwiki).
        if dataset_dir.name == setting and (dataset_dir.parent / wiki_dirname).exists():
            return str(dataset_dir.parent / wiki_dirname)
        return str(dataset_dir / wiki_dirname)

    def _context_corpus_mode(self) -> str:
        raw = self._get("HOTPOT_CONTEXT_CORPUS_MODE", "wiki").strip().lower()
        if raw in {"sample", "context", "sample_context"}:
            return "sample"
        if raw in {"hybrid", "sample_plus_wiki", "context_plus_wiki"}:
            return "hybrid"
        return "wiki"

    def _require_context_answerable(self) -> bool:
        return self._get_bool("HOTPOT_REQUIRE_CONTEXT_ANSWERABLE", False)

    def _is_context_answerable(self, sample: BenchmarkSample) -> bool:
        context = sample.metadata.get("context") or {}
        if not isinstance(context, dict):
            return False
        titles = context.get("title") or context.get("titles") or []
        sentences = context.get("sentences") or context.get("sentence") or []
        if isinstance(titles, str):
            titles = [titles]
        title_set = {self._normalize_text(str(t)) for t in titles}
        support = sample.metadata.get("supporting_facts") or {}
        support_titles = support.get("title") or support.get("titles") or [] if isinstance(support, dict) else []
        if isinstance(support_titles, str):
            support_titles = [support_titles]
        support_set = {self._normalize_text(str(t)) for t in support_titles if str(t).strip()}
        if support_set and not support_set.issubset(title_set):
            return False

        if isinstance(sentences, list):
            flat_sentences: List[str] = []
            for item in sentences:
                if isinstance(item, list):
                    flat_sentences.extend(str(s) for s in item)
                else:
                    flat_sentences.append(str(item))
            context_text = " ".join(flat_sentences)
        else:
            context_text = str(sentences)
        answer = str(sample.gold_answer or "").strip().lower()
        return bool(answer and answer in context_text.lower())

    @staticmethod
    def _normalize_text(text: str) -> str:
        return " ".join(re.sub(r"[^a-zA-Z0-9]+", " ", text).lower().split())

    def _materialize_sample_context(self, sample: BenchmarkSample) -> Path:
        """Materialize parquet context as per-sample raw text files."""
        out_dir = Path(self.get_work_path()) / "context_corpus" / sample.sample_id
        marker = out_dir / ".complete"
        if marker.exists():
            return out_dir

        out_dir.mkdir(parents=True, exist_ok=True)
        for old in out_dir.glob("*.txt"):
            try:
                old.unlink()
            except OSError:
                pass

        context = sample.metadata.get("context") or {}
        titles = context.get("title") or context.get("titles") or [] if isinstance(context, dict) else []
        sentences = context.get("sentences") or context.get("sentence") or [] if isinstance(context, dict) else []
        if isinstance(titles, str):
            titles = [titles]
        if not isinstance(titles, list):
            titles = list(titles) if titles is not None else []
        if not isinstance(sentences, list):
            sentences = list(sentences) if sentences is not None else []

        for idx, title in enumerate(titles):
            raw_sentences = sentences[idx] if idx < len(sentences) else []
            if isinstance(raw_sentences, str):
                raw_sentences = [raw_sentences]
            elif not isinstance(raw_sentences, list):
                try:
                    raw_sentences = list(raw_sentences)
                except TypeError:
                    raw_sentences = [str(raw_sentences)]
            text = "\n".join(str(s) for s in raw_sentences if str(s).strip())
            safe_title = self._safe_context_filename(str(title), idx)
            body = f"# {title}\n\n{text}\n" if text else f"# {title}\n"
            (out_dir / f"{idx:02d}_{safe_title}.txt").write_text(
                body,
                encoding="utf-8",
            )

        marker.write_text("ok\n", encoding="utf-8")
        return out_dir

    @staticmethod
    def _safe_context_filename(title: str, idx: int) -> str:
        normalized = re.sub(r"[^A-Za-z0-9]+", "_", title).strip("_")
        return (normalized[:120] or f"doc_{idx}")
