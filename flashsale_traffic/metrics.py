"""Metric names, day-over-day math, and chart matching for the flash-sale dashboard."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

DASHBOARD_ID = 146640
DASHBOARD_URL = (
    "https://raptor.mws.sankuai.com/dashboard/list"
    f"?dashboard={DASHBOARD_ID}&isCore=false&tabName=metric"
)
DEFAULT_WINDOW_MINUTES = 10
BEIJING_TZ = "Asia/Shanghai"


@dataclass(frozen=True)
class MetricSpec:
    """One series we must include in the 10-minute broadcast."""

    key: str
    display_name: str
    required_all: tuple[str, ...]
    forbidden: tuple[str, ...] = ()
    exact_aliases: tuple[str, ...] = ()


METRIC_SPECS: tuple[MetricSpec, ...] = (
    MetricSpec(
        key="hub",
        display_name="HUB闪购场景总流量",
        required_all=("HUB", "闪购"),
        forbidden=("化蝶",),
        exact_aliases=("HUB闪购场景总流量", "HUB闪购总流量", "HUB闪购"),
    ),
    MetricSpec(
        key="huadie",
        display_name="化蝶闪购场景总流量",
        required_all=("化蝶",),
        forbidden=("HUB",),
        exact_aliases=("化蝶闪购场景总流量", "化蝶闪购总流量", "化蝶闪购"),
    ),
    MetricSpec(
        key="overall",
        display_name="整体流量",
        required_all=("整体",),
        forbidden=("HUB", "化蝶", "场景"),
        exact_aliases=("整体流量", "总流量", "整体总流量"),
    ),
)


@dataclass(frozen=True)
class MetricSnapshot:
    """Current window value plus the same window a day earlier."""

    spec: MetricSpec
    current: float | None
    yesterday: float | None
    source_name: str = ""

    @property
    def dod(self) -> float | None:
        return day_over_day(self.current, self.yesterday)


def day_over_day(current: float | None, yesterday: float | None) -> float | None:
    """(today - yesterday) / yesterday. None when yesterday is missing or zero."""
    if current is None or yesterday is None:
        return None
    if yesterday == 0:
        return None
    return (current - yesterday) / yesterday


def format_traffic(value: float | None) -> str:
    """Human-readable QPS / count: 万 when large, otherwise up to 2 decimals."""
    if value is None:
        return "--"
    sign = "-" if value < 0 else ""
    magnitude = abs(value)
    if magnitude >= 10_000:
        return f"{sign}{magnitude / 10_000:.2f}万"
    if magnitude >= 100 or magnitude == int(magnitude):
        return f"{sign}{magnitude:.0f}"
    if magnitude >= 10:
        return f"{sign}{magnitude:.1f}"
    return f"{sign}{magnitude:.2f}"


def format_dod(ratio: float | None) -> str:
    if ratio is None:
        return "日环比 --"
    pct = ratio * 100
    sign = "+" if pct > 0 else ""
    return f"日环比 {sign}{pct:.1f}%"


def _fold(name: str) -> str:
    return "".join(name.strip().upper().split())


def score_chart_name(spec: MetricSpec, name: str) -> int:
    """Higher is a better match. 0 means this chart is not this metric."""
    raw = name.strip()
    if not raw:
        return 0
    folded = _fold(raw)
    for alias in spec.exact_aliases:
        if folded == _fold(alias):
            return 100
    for token in spec.forbidden:
        if _fold(token) in folded:
            return 0
    if not all(_fold(token) in folded for token in spec.required_all):
        return 0
    score = 50 + 5 * len(spec.required_all)
    if spec.display_name in raw:
        score += 20
    return score


def pick_chart_name(spec: MetricSpec, names: Iterable[str]) -> str | None:
    ranked = sorted(
        ((score_chart_name(spec, name), name) for name in names),
        key=lambda item: item[0],
        reverse=True,
    )
    if not ranked or ranked[0][0] <= 0:
        return None
    return ranked[0][1]


def assign_charts(
    named_values: Sequence[tuple[str, float]],
    specs: Sequence[MetricSpec] = METRIC_SPECS,
) -> dict[str, tuple[str, float]]:
    """Map each spec to at most one (chart_name, value), preferring exact titles."""
    unused = list(named_values)
    assigned: dict[str, tuple[str, float]] = {}
    for spec in specs:
        best_i = -1
        best_score = 0
        for i, (name, _value) in enumerate(unused):
            score = score_chart_name(spec, name)
            if score > best_score:
                best_score = score
                best_i = i
        if best_i >= 0:
            assigned[spec.key] = unused.pop(best_i)
    return assigned
