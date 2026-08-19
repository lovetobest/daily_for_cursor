"""Build the 10-minute flash-sale traffic broadcast."""

from __future__ import annotations

from typing import Any, Sequence

from flashsale_traffic.metrics import (
    METRIC_SPECS,
    MetricSnapshot,
    MetricSpec,
    assign_charts,
    format_dod,
    format_traffic,
)
from flashsale_traffic.parse import named_values, named_values_with_ratio
from flashsale_traffic.raptor import dashboard_link
from flashsale_traffic.windows import ReportWindow


def snapshots_from_payloads(
    current_payload: Any,
    yesterday_payload: Any | None = None,
    *,
    specs: Sequence[MetricSpec] = METRIC_SPECS,
) -> list[MetricSnapshot]:
    current_map = assign_charts(named_values(current_payload), specs)
    ratio_by_name = {
        name: ratio for name, _value, ratio in named_values_with_ratio(current_payload) if ratio is not None
    }
    yesterday_map = assign_charts(named_values(yesterday_payload or {}), specs)

    snapshots: list[MetricSnapshot] = []
    for spec in specs:
        current_hit = current_map.get(spec.key)
        yesterday_hit = yesterday_map.get(spec.key)
        current_value = current_hit[1] if current_hit else None
        source_name = current_hit[0] if current_hit else spec.display_name
        yesterday_value = yesterday_hit[1] if yesterday_hit else None
        if yesterday_value is None and current_hit is not None:
            ratio = ratio_by_name.get(current_hit[0])
            if ratio is not None and current_value is not None and ratio != -1:
                # Reconstruct yesterday from 日环比 when Raptor only returns today + ratio.
                yesterday_value = current_value / (1.0 + ratio) if (1.0 + ratio) != 0 else None
        snapshots.append(
            MetricSnapshot(
                spec=spec,
                current=current_value,
                yesterday=yesterday_value,
                source_name=source_name,
            )
        )
    return snapshots


def format_report(
    snapshots: Sequence[MetricSnapshot],
    window: ReportWindow,
    *,
    link: str | None = None,
    missing_note: str | None = None,
) -> str:
    lines = [f"【闪购流量播报】{window.label()}（近{(window.end - window.start).seconds // 60}分钟）"]
    for snapshot in snapshots:
        current = format_traffic(snapshot.current)
        dod = format_dod(snapshot.dod)
        lines.append(f"{snapshot.spec.display_name}：{current}（{dod}）")
    lines.append(f"大盘：{link or dashboard_link(window)}")
    if missing_note:
        lines.append(missing_note)
    missing = [item.spec.display_name for item in snapshots if item.current is None]
    if missing:
        lines.append("未匹配到：" + "、".join(missing))
    return "\n".join(lines)


def build_report(
    current_payload: Any,
    yesterday_payload: Any | None,
    window: ReportWindow,
    *,
    link: str | None = None,
    missing_note: str | None = None,
) -> str:
    return format_report(
        snapshots_from_payloads(current_payload, yesterday_payload),
        window,
        link=link,
        missing_note=missing_note,
    )
