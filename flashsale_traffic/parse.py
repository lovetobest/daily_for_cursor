"""Pull named numeric series out of Raptor dashboard / chart JSON."""

from __future__ import annotations

from typing import Any, Iterator

NAME_KEYS = (
    "name",
    "title",
    "chartName",
    "metricName",
    "alias",
    "label",
    "displayName",
    "chartTitle",
)
VALUE_KEYS = (
    "current",
    "currentValue",
    "latest",
    "value",
    "avg",
    "average",
    "sum",
    "total",
    "count",
    "qps",
    "y",
)
RATIO_KEYS = (
    "dod",
    "dayOverDay",
    "dayOnDay",
    "ringRatio",
    "ring",
    "wow",
    "compare",
    "日环比",
    "环比",
)
SERIES_KEYS = ("datapoints", "dataPoints", "points", "series", "values", "dps")


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _as_float(value: Any) -> float | None:
    if _is_number(value):
        return float(value)
    if isinstance(value, str):
        text = value.strip().replace(",", "").replace("%", "")
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None
    return None


def named_values(payload: Any) -> list[tuple[str, float]]:
    """Collect (chart_or_metric_name, numeric_value) pairs from nested JSON."""
    found: list[tuple[str, float]] = []
    for name, value, _ratio in _walk(payload, inherited_name=""):
        if name and value is not None:
            found.append((name, value))
    return found


def named_values_with_ratio(payload: Any) -> list[tuple[str, float, float | None]]:
    """Like named_values, but keep a day-over-day ratio when Raptor already computed it."""
    found: list[tuple[str, float, float | None]] = []
    for name, value, ratio in _walk(payload, inherited_name=""):
        if name and value is not None:
            found.append((name, value, ratio))
    return found


def series_average(points: Any) -> float | None:
    """Average the numeric samples in a Raptor/CAT-style time series."""
    numbers: list[float] = []
    if isinstance(points, dict):
        for value in points.values():
            parsed = _point_value(value)
            if parsed is not None:
                numbers.append(parsed)
    elif isinstance(points, list):
        for item in points:
            parsed = _point_value(item)
            if parsed is not None:
                numbers.append(parsed)
    if not numbers:
        return None
    return sum(numbers) / len(numbers)


def _point_value(item: Any) -> float | None:
    if _is_number(item):
        return float(item)
    if isinstance(item, (list, tuple)) and len(item) >= 2:
        return _as_float(item[1]) if _as_float(item[1]) is not None else _as_float(item[0])
    if isinstance(item, dict):
        for key in ("value", "y", "v", "avg", "sum"):
            parsed = _as_float(item.get(key))
            if parsed is not None:
                return parsed
    return None


def _first_name(obj: dict[str, Any]) -> str:
    for key in NAME_KEYS:
        value = obj.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _first_ratio(obj: dict[str, Any]) -> float | None:
    for key in RATIO_KEYS:
        parsed = _as_float(obj.get(key))
        if parsed is None:
            continue
        # Raptor sometimes stores 5.2 meaning +5.2%, sometimes 0.052.
        if abs(parsed) > 1.5:
            return parsed / 100.0
        return parsed
    compare = obj.get("compare")
    if isinstance(compare, dict):
        for key in ("dod", "dayOverDay", "ratio", "value"):
            parsed = _as_float(compare.get(key))
            if parsed is not None:
                return parsed / 100.0 if abs(parsed) > 1.5 else parsed
    return None


def _first_scalar(obj: dict[str, Any]) -> float | None:
    for key in VALUE_KEYS:
        parsed = _as_float(obj.get(key))
        if parsed is not None:
            return parsed
    for key in SERIES_KEYS:
        if key in obj:
            avg = series_average(obj[key])
            if avg is not None:
                return avg
    data = obj.get("data")
    if isinstance(data, list) or isinstance(data, dict):
        avg = series_average(data)
        if avg is not None:
            return avg
    return None


def _walk(node: Any, inherited_name: str) -> Iterator[tuple[str, float | None, float | None]]:
    if isinstance(node, list):
        for item in node:
            yield from _walk(item, inherited_name)
        return
    if not isinstance(node, dict):
        return

    name = _first_name(node) or inherited_name
    value = _first_scalar(node)
    ratio = _first_ratio(node)
    if name and value is not None:
        yield name, value, ratio

    skip = set(VALUE_KEYS) | set(RATIO_KEYS) | set(SERIES_KEYS) | set(NAME_KEYS)
    for key, child in node.items():
        if key in skip:
            continue
        yield from _walk(child, name)
