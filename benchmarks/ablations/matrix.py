"""ablations/matrix.py — core LENS ablation matrix

The dynamic raw-corpus main ablation keeps only the two core mechanism ablations
and avoids fine-grained engineering ablations:
  - w/o Multi-signal Prior
  - w/o Sequential Exploration

`w/o Knowledge Reuse` remains an appendix-only profile and can still be built via
build_single_lens_ablation(..., "lens_no_reuse").

This module exposes factories that turn those profiles into BaselineAdapter lists
ready to be handed to BaselineEvaluationSuite.
"""
from __future__ import annotations

from typing import Any, List

from baselines.base_adapter import BaselineAdapter
from .lens_ablation_adapter import LensAblationAdapter
from .lens_profile import LensSearchProfile, core_lens_profiles


def build_lens_ablation_baselines(
    bm_adapter: Any,
    *,
    include_full: bool = True,
    max_loops: int = 10,
    max_token_budget: int = 128_000,
    top_k_files: int = 10,
) -> List[BaselineAdapter]:
    """Build BaselineAdapter instances for the core LENS ablation matrix.

    Args:
        bm_adapter:        BenchmarkAdapter used to build AgenticSearch and paths.
        include_full:      Include full LENS row in ablation table.
        max_loops:         Max DEEP retrieval loops.
        max_token_budget:  Token budget.
        top_k_files:       Target files.

    Returns:
        List[BaselineAdapter], all compatible with BaselineEvaluationSuite.
    """
    profiles = core_lens_profiles()
    if not include_full:
        profiles = [p for p in profiles if p.name != "lens_full"]
    return [
        LensAblationAdapter(
            bm_adapter=bm_adapter,
            profile=profile,
            max_loops=max_loops,
            max_token_budget=max_token_budget,
            top_k_files=top_k_files,
        )
        for profile in profiles
    ]


def build_single_lens_ablation(
    bm_adapter: Any,
    profile_name: str,
    *,
    max_loops: int = 10,
    max_token_budget: int = 128_000,
    top_k_files: int = 10,
) -> BaselineAdapter:
    """Build one ablation baseline by canonical profile name.

    Supported names:
      lens_full
      lens_no_prior
      lens_no_seq
      lens_no_reuse (appendix-only)
    """
    return LensAblationAdapter(
        bm_adapter=bm_adapter,
        profile=LensSearchProfile.by_name(profile_name),
        max_loops=max_loops,
        max_token_budget=max_token_budget,
        top_k_files=top_k_files,
    )
