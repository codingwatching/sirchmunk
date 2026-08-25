"""Benchmark adapter registry."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable, Dict

_BENCHMARKS_ROOT = Path(__file__).resolve().parents[1]
if str(_BENCHMARKS_ROOT) not in sys.path:
    sys.path.insert(0, str(_BENCHMARKS_ROOT))


def _load_hotpotqa(env_file: str) -> Any:
    from hotpotqa.adapter import HotpotQAAdapter
    return HotpotQAAdapter(env_file=env_file)


def _load_setup_cost(env_file: str) -> Any:
    from setup_cost.adapter import SetupCostAdapter
    return SetupCostAdapter(env_file=env_file)


def _load_freshness(env_file: str) -> Any:
    from freshness.adapter import FreshnessAdapter
    return FreshnessAdapter(env_file=env_file)


def _load_storage_overhead(env_file: str) -> Any:
    from storage_overhead.adapter import StorageOverheadAdapter
    return StorageOverheadAdapter(env_file=env_file)


def _load_source_fidelity(env_file: str) -> Any:
    from source_fidelity.adapter import SourceFidelityAdapter
    return SourceFidelityAdapter(env_file=env_file)


def _load_warm_reuse(env_file: str) -> Any:
    from warm_reuse.adapter import WarmReuseAdapter
    return WarmReuseAdapter(env_file=env_file)


_ADAPTER_LOADERS: Dict[str, Callable[[str], Any]] = {
    "hotpotqa": _load_hotpotqa,
    "setup_cost": _load_setup_cost,
    "freshness": _load_freshness,
    "storage_overhead": _load_storage_overhead,
    "source_fidelity": _load_source_fidelity,
    "warm_reuse": _load_warm_reuse,
}


def supported_benchmarks() -> list[str]:
    return sorted(_ADAPTER_LOADERS)


def load_benchmark_adapter(benchmark: str, env_file: str) -> Any:
    key = benchmark.strip().lower()
    if key not in _ADAPTER_LOADERS:
        raise ValueError(
            f"Unknown benchmark: '{benchmark}'. Supported: {', '.join(supported_benchmarks())}"
        )
    return _ADAPTER_LOADERS[key](env_file)
