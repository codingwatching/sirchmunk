"""ablations/lens_profile.py — core LENS ablation profiles

The minimal set of three profiles used by the dynamic raw-corpus main ablation:
  1. Full LENS
  2. w/o Multi-signal Prior
  3. w/o Sequential Exploration

`w/o Knowledge Reuse` stays an appendix-only profile for follow-up / warm-start
amortization studies and does not enter the main ablation table.

Note: a profile is an experiment abstraction of the benchmark/evaluation layer and
does not change the self-improvement loop. Execution is handled by
LensAblationAdapter through wrapper / instance patching, and search.py is left
unmodified by default.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass(frozen=True)
class LensSearchProfile:
    """LENS ablation profile.

    These switches map to the three high-level mechanism blocks of the LENS core
    algorithm A0-A7.
    """
    name: str
    citation_name: str
    description: str

    # A0-A2: prior formation
    enable_multi_signal_prior: bool = True
    enable_cluster_reuse: bool = True
    enable_knowledge_probe: bool = True
    enable_spec_cache: bool = True
    enable_tree_probe: bool = True
    enable_compile_hints: bool = True
    enable_summary_index: bool = True
    enable_catalog_probe: bool = True
    enable_dir_scan: bool = True

    # A3-A4: sequential exploration
    enable_sequential_exploration: bool = True
    one_shot_max_files: int = 3
    one_shot_max_chars_per_file: int = 18_000

    # A0/A7: reuse and persistence
    enable_knowledge_reuse: bool = True
    enable_persistence: bool = True

    # Search kwargs override
    search_kwargs: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def full(cls) -> "LensSearchProfile":
        return cls(
            name="lens_full",
            citation_name="LENS",
            description="Full LENS: multi-signal prior + sequential exploration + knowledge reuse.",
            search_kwargs={"mode": "DEEP", "enable_dir_scan": True},
        )

    @classmethod
    def without_multi_signal_prior(cls) -> "LensSearchProfile":
        return cls(
            name="lens_no_prior",
            citation_name="LENS w/o Multi-signal Prior",
            description=(
                "Disable A0-A2 multi-signal prior channels: cluster reuse, knowledge probe, "
                "spec cache, tree probe, compile hints, summary index, catalog probe, dir scan. "
                "Keep keyword extraction + rga retrieval + DEEP agentic retrieval."
            ),
            enable_multi_signal_prior=False,
            enable_cluster_reuse=False,
            enable_knowledge_probe=False,
            enable_spec_cache=False,
            enable_tree_probe=False,
            enable_compile_hints=False,
            enable_summary_index=False,
            enable_catalog_probe=False,
            enable_dir_scan=False,
            search_kwargs={"mode": "DEEP", "enable_dir_scan": False},
        )

    @classmethod
    def without_sequential_exploration(cls) -> "LensSearchProfile":
        return cls(
            name="lens_no_seq",
            citation_name="LENS w/o Sequential Exploration",
            description=(
                "Keep A0-A2 prior formation and file selection, but replace A3-A4 iterative "
                "agentic retrieval with one-shot evidence extraction from selected files."
            ),
            enable_sequential_exploration=False,
            search_kwargs={"mode": "DEEP", "enable_dir_scan": True},
        )

    @classmethod
    def without_knowledge_reuse(cls) -> "LensSearchProfile":
        return cls(
            name="lens_no_reuse",
            citation_name="LENS w/o Knowledge Reuse",
            description=(
                "Disable hard/soft cluster reuse, knowledge probe, spec cache, and persistence. "
                "Keep multi-signal retrieval and sequential exploration."
            ),
            enable_cluster_reuse=False,
            enable_knowledge_probe=False,
            enable_spec_cache=False,
            enable_knowledge_reuse=False,
            enable_persistence=False,
            search_kwargs={"mode": "DEEP", "enable_dir_scan": True},
        )

    @classmethod
    def by_name(cls, name: str) -> "LensSearchProfile":
        normalized = name.strip().lower()
        profiles = {
            "lens_full": cls.full,
            "lens_no_prior": cls.without_multi_signal_prior,
            "lens_no_seq": cls.without_sequential_exploration,
            "lens_no_reuse": cls.without_knowledge_reuse,
        }
        if normalized not in profiles:
            raise ValueError(f"Unknown LENS profile: {name}")
        return profiles[normalized]()


def core_lens_profiles() -> List[LensSearchProfile]:
    """Return the main ablation: full plus the two core mechanism ablations."""
    return [
        LensSearchProfile.full(),
        LensSearchProfile.without_multi_signal_prior(),
        LensSearchProfile.without_sequential_exploration(),
    ]


def appendix_lens_profiles() -> List[LensSearchProfile]:
    """Return appendix-only profiles that stay out of the main ablation table."""
    return [LensSearchProfile.without_knowledge_reuse()]


def all_lens_profiles() -> List[LensSearchProfile]:
    """Return every buildable profile, including the appendix-only variants."""
    return core_lens_profiles() + appendix_lens_profiles()
