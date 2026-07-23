"""Explicit cache policy management for ResearchOps runs."""
from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


class CacheMode(str, Enum):
    COLD = "cold"
    WARM = "warm"
    COMPILED = "compiled"
    NONE = "none"


@dataclass
class CacheActionReport:
    mode: CacheMode
    work_path: str
    cleared_paths: List[str] = field(default_factory=list)
    preserved_paths: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    dry_run: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode.value,
            "work_path": self.work_path,
            "cleared_paths": self.cleared_paths,
            "preserved_paths": self.preserved_paths,
            "warnings": self.warnings,
            "dry_run": self.dry_run,
        }


class CacheManager:
    """Apply explicit cold/warm/compiled cache policy.

    Destructive clearing is disabled unless ``allow_clear=True`` and is limited
    to configured cache directories under ``work_path``. Callers can provide
    adapter-specific cache names or relative paths; all paths are resolved and
    checked against ``work_path`` before clearing.
    """

    DEFAULT_CACHE_NAMES = (".cache", "knowledge", "history", "compile", "rga")
    DEFAULT_NESTED_CACHE_PATHS = (
        ".cache/rga",
        ".cache/knowledge",
        ".cache/compile",
    )

    def __init__(
        self,
        work_path: str | Path,
        *,
        cache_names: Optional[Iterable[str]] = None,
        cache_paths: Optional[Iterable[str | Path]] = None,
        compiled_markers: Optional[Iterable[str]] = None,
    ) -> None:
        self.work_path = Path(work_path).expanduser().resolve()
        self.cache_names = tuple(cache_names or self.DEFAULT_CACHE_NAMES)
        self.cache_paths = tuple(cache_paths or self.DEFAULT_NESTED_CACHE_PATHS)
        self.compiled_markers = tuple(compiled_markers or ("compile", "compiled"))

    def prepare(self, mode: str | CacheMode, *, allow_clear: bool = False, dry_run: bool = False) -> CacheActionReport:
        try:
            cache_mode = CacheMode(str(mode).lower())
        except ValueError:
            cache_mode = CacheMode.NONE
        report = CacheActionReport(mode=cache_mode, work_path=str(self.work_path), dry_run=dry_run)
        if str(mode).lower() not in {m.value for m in CacheMode}:
            report.warnings.append(f"unknown cache mode '{mode}', falling back to 'none'")

        if cache_mode == CacheMode.NONE:
            report.warnings.append("cache mode 'none': no cache policy applied")
            return report

        self.work_path.mkdir(parents=True, exist_ok=True)
        cache_targets = self._known_cache_targets()

        if cache_mode == CacheMode.WARM:
            report.preserved_paths = [str(p) for p in cache_targets if p.exists()]
            return report

        if cache_mode == CacheMode.COMPILED:
            for path in cache_targets:
                if self._is_compiled_cache(path) and path.exists():
                    report.preserved_paths.append(str(path))
                elif path.exists() and allow_clear:
                    self._clear(path, report, dry_run)
                elif path.exists():
                    report.preserved_paths.append(str(path))
            if not report.preserved_paths:
                report.warnings.append("compiled cache mode requested, but no compile artifacts were found")
            return report

        if cache_mode == CacheMode.COLD:
            if not allow_clear:
                report.warnings.append("cold cache requested but allow_clear=False; no cache cleared")
                report.preserved_paths = [str(p) for p in cache_targets if p.exists()]
                return report
            for path in cache_targets:
                if path.exists():
                    self._clear(path, report, dry_run)
            return report

        return report

    def _known_cache_targets(self) -> List[Path]:
        targets: List[Path] = []
        for name in self.cache_names:
            if name:
                targets.append(self.work_path / str(name))
        for raw_path in self.cache_paths:
            path = Path(raw_path)
            targets.append(path if path.is_absolute() else self.work_path / path)
        # Deduplicate while preserving order and refusing out-of-work targets.
        seen = set()
        unique = []
        for path in targets:
            resolved = path.expanduser().resolve()
            if not _is_relative_to(resolved, self.work_path):
                continue
            key = str(resolved)
            if key not in seen:
                seen.add(key)
                unique.append(resolved)
        return unique

    def _is_compiled_cache(self, path: Path) -> bool:
        lowered = "/".join(part.lower() for part in path.parts)
        return any(marker.lower() in lowered for marker in self.compiled_markers)

    def _clear(self, path: Path, report: CacheActionReport, dry_run: bool) -> None:
        try:
            resolved = path.resolve()
            if not _is_relative_to(resolved, self.work_path):
                report.warnings.append(f"refused to clear path outside work_path: {resolved}")
                return
            if dry_run:
                report.cleared_paths.append(str(resolved))
                return
            if resolved.is_dir():
                shutil.rmtree(resolved)
            elif resolved.exists():
                resolved.unlink()
            report.cleared_paths.append(str(resolved))
        except Exception as exc:
            report.warnings.append(f"failed to clear {path}: {exc}")


def _is_relative_to(path: Path, base: Path) -> bool:
    try:
        path.relative_to(base)
        return True
    except ValueError:
        return False
