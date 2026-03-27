from __future__ import annotations

from datetime import datetime, timezone
from zoneinfo import ZoneInfo


def ms_to_dt_utc(ms: int | float | None) -> datetime | None:
    if ms is None:
        return None
    return datetime.fromtimestamp(float(ms) / 1000.0, tz=timezone.utc)


def to_local_hour(ms: int | float | None, tz_name: str) -> int | None:
    dt = ms_to_dt_utc(ms)
    if dt is None:
        return None
    return dt.astimezone(ZoneInfo(tz_name)).hour


def assign_time_of_day(hour: int | None) -> str:
    if hour is None:
        return "unknown"
    if 5 <= hour < 9:
        return "morning"
    if 9 <= hour < 18:
        return "midday"
    if 18 <= hour < 23:
        return "evening"
    return "night"
