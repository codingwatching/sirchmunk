"""Timezone helpers for benchmark ResearchOps artifacts.

Human-visible run IDs and filenames use the local timezone so they match the
operator's wall clock.  UTC timestamps are still recorded alongside local fields
where reproducibility and cross-timezone comparison matter.
"""
from __future__ import annotations

from datetime import datetime, timezone


def now_local() -> datetime:
    """Return timezone-aware local time."""
    return datetime.now().astimezone()


def now_local_iso() -> str:
    """Return current local time as ISO-8601 with timezone offset."""
    return now_local().isoformat()


def now_utc_iso() -> str:
    """Return current UTC time as ISO-8601."""
    return datetime.now(timezone.utc).isoformat()


def local_timestamp() -> str:
    """Return local wall-clock timestamp for filenames: YYYYMMDD_HHMMSS."""
    return now_local().strftime("%Y%m%d_%H%M%S")


def local_timezone_name() -> str:
    """Return a readable local timezone name or offset."""
    tzname = now_local().tzname()
    return tzname or now_local().strftime("%z")
