"""10-minute Beijing-time windows for current traffic vs yesterday."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from flashsale_traffic.metrics import BEIJING_TZ, DEFAULT_WINDOW_MINUTES

BEIJING = ZoneInfo(BEIJING_TZ)
RAPTOR_TS = "%Y%m%d%H%M%S"


@dataclass(frozen=True)
class ReportWindow:
    """Inclusive start / exclusive end, already aligned to the broadcast cadence."""

    start: datetime
    end: datetime

    @property
    def yesterday(self) -> "ReportWindow":
        delta = timedelta(days=1)
        return ReportWindow(self.start - delta, self.end - delta)

    def raptor_start(self) -> str:
        return self.start.strftime(RAPTOR_TS)

    def raptor_end(self) -> str:
        return self.end.strftime(RAPTOR_TS)

    def label(self) -> str:
        return self.end.strftime("%Y-%m-%d %H:%M")


def align_window(
    now: datetime | None = None,
    *,
    window_minutes: int = DEFAULT_WINDOW_MINUTES,
) -> ReportWindow:
    """Last completed N-minute bucket in Asia/Shanghai (e.g. 10:10–10:20 at 10:23)."""
    if window_minutes <= 0:
        raise ValueError("window_minutes must be positive")
    instant = now.astimezone(BEIJING) if now else datetime.now(BEIJING)
    minute = (instant.minute // window_minutes) * window_minutes
    end = instant.replace(minute=minute, second=0, microsecond=0)
    if end >= instant:
        end -= timedelta(minutes=window_minutes)
    start = end - timedelta(minutes=window_minutes)
    return ReportWindow(start, end)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)
