"""Lightweight mechanism benchmark adapters for P0 ResearchOps.

These adapters provide P0 prototypes for setup cost, freshness, storage
overhead, source fidelity, and warm reuse experiments. They intentionally avoid
requiring external baseline systems; the goal is to produce structured metrics
and artifact-compatible results early.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .adapter import BenchmarkAdapter
from .schema import BenchmarkSample


@dataclass
class _ProbeSearchResult:
    answer: str
    telemetry: Dict[str, Any]
    read_file_ids: set[str]
    total_llm_tokens: int = 0
    loop_count: int = 0
    llm_usages: list = None

    def __post_init__(self) -> None:
        if self.llm_usages is None:
            self.llm_usages = []


class MechanismProbeSearcher:
    def __init__(self, benchmark_name: str, env: Dict[str, str]) -> None:
        self.benchmark_name = benchmark_name
        self.env = env

    async def search(self, query: str, paths=None, return_context: bool = True, **kwargs):
        start = time.time()
        search_paths = [str(p) for p in (paths or [])]
        raw_size = sum(_path_size(Path(p)) for p in search_paths)
        file_count = sum(_file_count(Path(p), max_files=5000) for p in search_paths)
        telemetry: Dict[str, Any] = {
            "mechanism_benchmark": self.benchmark_name,
            "raw_corpus_size_bytes": raw_size,
            "raw_corpus_file_count_sampled": file_count,
            "preprocessing_time_seconds": 0.0,
            "index_build_time_seconds": 0.0,
            "llm_calls": 0,
            "total_tokens": 0,
        }

        if self.benchmark_name == "setup_cost":
            telemetry["time_to_first_query_seconds"] = round(time.time() - start, 6)
            answer = "Setup cost probe completed without preprocessing."
        elif self.benchmark_name == "freshness":
            expected = self.env.get("FRESHNESS_EXPECTED_ANSWER", "").strip()
            found = _contains_text(search_paths, expected) if expected else False
            telemetry["freshness_accuracy"] = 1.0 if found else 0.0
            telemetry["update_latency_seconds"] = round(time.time() - start, 6)
            answer = "Freshness probe found expected updated answer." if found else "Freshness probe did not find expected updated answer."
        elif self.benchmark_name == "storage_overhead":
            artifact_paths = _split_paths(self.env.get("STORAGE_ARTIFACT_PATHS", ""))
            artifact_size = sum(_path_size(Path(p)) for p in artifact_paths)
            telemetry["artifact_size_bytes"] = artifact_size
            telemetry["storage_overhead_ratio"] = round(artifact_size / raw_size, 6) if raw_size else 0.0
            answer = "Storage overhead probe completed."
        elif self.benchmark_name == "source_fidelity":
            expected = self.env.get("SOURCE_FIDELITY_EVIDENCE", "").strip()
            found = _contains_text(search_paths, expected) if expected else False
            telemetry["evidence_traceability_rate"] = 1.0 if found else 0.0
            telemetry["answer_source_grounded"] = found
            answer = "Source fidelity evidence located." if found else "Source fidelity evidence not located."
        elif self.benchmark_name == "warm_reuse":
            telemetry["warm_reuse_probe"] = True
            telemetry["warm_query_reuse_gain"] = 0.0
            answer = "Warm reuse probe completed; repeated-query gain requires paired runs."
        else:
            answer = f"Mechanism probe completed: {self.benchmark_name}."

        telemetry["probe_elapsed_seconds"] = round(time.time() - start, 6)
        return _ProbeSearchResult(answer=answer, telemetry=telemetry, read_file_ids=set(search_paths))


class MechanismJudge:
    async def judge(self, prediction: str, gold_answer: str, question: str = "") -> Dict[str, Any]:
        ok = bool((prediction or "").strip())
        return {"equivalent": ok, "confidence": 1.0, "reasoning": "Mechanism probe completed." if ok else "Empty probe result.", "tokens_used": 0}

    async def judge_coverage(self, prediction: str, question: str) -> Dict[str, Any]:
        ok = bool((prediction or "").strip())
        return {"has_coverage": ok, "confidence": 1.0, "reasoning": "Non-empty probe result." if ok else "Empty probe result.", "tokens_used": 0}


class MechanismProbeAdapter(BenchmarkAdapter):
    def __init__(self, env_file: str, benchmark_name: str) -> None:
        self._env_file = str(Path(env_file).resolve())
        self._env = _load_env(self._env_file)
        self._name = benchmark_name
        self._searcher = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def env_file(self) -> str:
        return self._env_file

    def _get(self, key: str, default: str = "") -> str:
        return self._env.get(key, os.environ.get(key, default))

    def load_samples(self, limit: int = 0, seed: int = 42) -> List[BenchmarkSample]:
        query = self._get("MECHANISM_QUERY", f"Run {self._name} probe")
        gold = self._get("MECHANISM_GOLD", "completed")
        return [BenchmarkSample(
            sample_id=f"{self._name}_probe_001",
            question=query,
            gold_answer=gold,
            metadata={"type": "mechanism", "level": "p0", "mechanism": self._name},
        )]

    def validate_corpus(self) -> Tuple[int, List[str]]:
        paths = self._corpus_paths()
        missing = [p for p in paths if not Path(p).exists()]
        return len(paths) - len(missing), missing

    def get_search_paths(self, sample: BenchmarkSample) -> List[str]:
        return self._corpus_paths()

    def get_run_config(self) -> Dict[str, Any]:
        return {
            "benchmark": self._name,
            "corpus_paths": self._corpus_paths(),
            "cache_mode": self._get("MECHANISM_CACHE_MODE", "cold"),
            "top_k_env_key": "MECHANISM_TOP_K_FILES",
            "mode_env_key": "MECHANISM_MODE",
        }

    def build_searcher(self) -> Any:
        if self._searcher is None:
            self._searcher = MechanismProbeSearcher(self._name, self._env)
        return self._searcher

    def build_judge(self) -> Any:
        return MechanismJudge()

    def get_output_dir(self) -> str:
        raw = self._get("MECHANISM_OUTPUT_DIR", f"./benchmarks/{self._name}/output")
        p = Path(raw)
        return str(p.resolve()) if p.is_absolute() else str((Path.cwd() / p).resolve())

    def get_work_path(self) -> str:
        raw = self._get("MECHANISM_WORK_PATH", f"./benchmarks/{self._name}/.work")
        p = Path(raw)
        return str(p.resolve()) if p.is_absolute() else str((Path.cwd() / p).resolve())

    def get_protocol_spec(self, run_id: str, seed: int, limit: int) -> Dict[str, Any]:
        return {
            "run_id": run_id,
            "benchmark": self._name,
            "suite": [self._name],
            "systems": ["sirchmunk_probe"],
            "metrics": {
                "mechanism": [
                    "time_to_first_query_seconds",
                    "freshness_accuracy",
                    "storage_overhead_ratio",
                    "evidence_traceability_rate",
                    "warm_query_reuse_gain",
                ]
            },
            "seeds": [seed],
            "cache_policy": {"mode": self._get("MECHANISM_CACHE_MODE", "cold")},
            "config": self.get_run_config(),
        }

    def get_dataset_manifest(self) -> Dict[str, Any]:
        paths = self._corpus_paths()
        return {
            "benchmark": self._name,
            "corpus_paths": paths,
            "raw_corpus_size_bytes": sum(_path_size(Path(p)) for p in paths),
            "raw_corpus_file_count_sampled": sum(_file_count(Path(p), max_files=5000) for p in paths),
        }

    def enrich_telemetry(self, sample, prediction, telemetry, **kwargs) -> Dict[str, Any]:
        return {"mechanism": self._name}

    def get_analysis_schema(self) -> Dict[str, Any]:
        return {"primary_group_key": "mechanism", "secondary_group_key": "level"}

    def get_config_schema(self) -> Dict[str, Any]:
        return {
            "global_keys": [
                "LLM_BASE_URL",
                "LLM_API_KEY",
                "LLM_MODEL_NAME",
                "LLM_TIMEOUT",
                "SIRCHMUNK_WORK_PATH",
            ],
            "top_k_env_key": "MECHANISM_TOP_K_FILES",
            "mode_env_key": "MECHANISM_MODE",
        }

    def _corpus_paths(self) -> List[str]:
        return _split_paths(self._get("MECHANISM_CORPUS_PATH", str(Path.cwd())))


def _load_env(path: str) -> Dict[str, str]:
    env: Dict[str, str] = {}
    p = Path(path)
    if not p.exists():
        return env
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        env[key.strip()] = value.strip().strip('"').strip("'")
    return env


def _split_paths(raw: str) -> List[str]:
    return [p.strip() for p in raw.split(",") if p.strip()]


def _path_size(path: Path) -> int:
    try:
        if path.is_file():
            return path.stat().st_size
        if path.is_dir():
            return sum(child.stat().st_size for child in path.rglob("*") if child.is_file())
    except OSError:
        return 0
    return 0


def _file_count(path: Path, max_files: int = 5000) -> int:
    if path.is_file():
        return 1
    if not path.is_dir():
        return 0
    count = 0
    for child in path.rglob("*"):
        if child.is_file():
            count += 1
            if count >= max_files:
                break
    return count


def _contains_text(paths: List[str], needle: str) -> bool:
    if not needle:
        return False
    target = needle.lower()
    for raw in paths:
        path = Path(raw)
        files = [path] if path.is_file() else list(path.rglob("*"))[:5000] if path.is_dir() else []
        for file_path in files:
            if not file_path.is_file():
                continue
            try:
                text = file_path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            if target in text.lower():
                return True
    return False
