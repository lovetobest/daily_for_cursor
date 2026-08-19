"""Tests for the 10-minute flash-sale traffic broadcast."""

from __future__ import annotations

import io
import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest import mock
from zoneinfo import ZoneInfo

from flashsale_traffic.metrics import (
    METRIC_SPECS,
    assign_charts,
    day_over_day,
    format_dod,
    format_traffic,
    score_chart_name,
)
from flashsale_traffic.parse import named_values, named_values_with_ratio
from flashsale_traffic.report import build_report, snapshots_from_payloads
from flashsale_traffic.windows import BEIJING, align_window
from flashsale_traffic.__main__ import main


BEIJING_TZ = ZoneInfo("Asia/Shanghai")

SAMPLE_DASHBOARD = {
    "charts": [
        {
            "name": "HUB闪购场景总流量",
            "current": 123456,
            "datapoints": [[1, 120000], [2, 126912]],
        },
        {
            "title": "化蝶闪购场景总流量",
            "value": 81000,
        },
        {
            "chartName": "整体流量",
            "avg": 204000,
            "dod": 0.024,
        },
        {
            "name": "HUB 转化率",
            "value": 0.12,
        },
    ]
}

YESTERDAY_DASHBOARD = {
    "charts": [
        {"name": "HUB闪购场景总流量", "current": 111111},
        {"name": "化蝶闪购场景总流量", "value": 90000},
        {"name": "整体流量", "avg": 199219},
    ]
}


class DayOverDayTests(unittest.TestCase):
    def test_positive_ratio(self) -> None:
        self.assertAlmostEqual(day_over_day(110, 100), 0.10)

    def test_zero_yesterday_is_none(self) -> None:
        self.assertIsNone(day_over_day(10, 0))

    def test_missing_is_none(self) -> None:
        self.assertIsNone(day_over_day(None, 10))
        self.assertIsNone(day_over_day(10, None))


class FormatTests(unittest.TestCase):
    def test_wan_and_integer(self) -> None:
        self.assertEqual(format_traffic(123456), "12.35万")
        self.assertEqual(format_traffic(2040), "2040")
        self.assertEqual(format_traffic(None), "--")

    def test_dod_signs(self) -> None:
        self.assertEqual(format_dod(0.052), "日环比 +5.2%")
        self.assertEqual(format_dod(-0.031), "日环比 -3.1%")
        self.assertEqual(format_dod(None), "日环比 --")


class MatchingTests(unittest.TestCase):
    def test_exact_titles_win(self) -> None:
        names = [
            "HUB闪购场景总流量",
            "化蝶闪购场景总流量",
            "整体流量",
            "HUB 转化率",
        ]
        hub = next(spec for spec in METRIC_SPECS if spec.key == "hub")
        overall = next(spec for spec in METRIC_SPECS if spec.key == "overall")
        self.assertGreater(score_chart_name(hub, "HUB闪购场景总流量"), 0)
        self.assertEqual(score_chart_name(hub, "化蝶闪购场景总流量"), 0)
        self.assertEqual(score_chart_name(overall, "HUB闪购场景总流量"), 0)
        self.assertGreater(score_chart_name(overall, "整体流量"), 0)
        self.assertIn("HUB闪购", names[0])

    def test_assign_does_not_reuse_hub_as_overall(self) -> None:
        assigned = assign_charts(
            [
                ("HUB闪购场景总流量", 1.0),
                ("化蝶闪购场景总流量", 2.0),
                ("整体流量", 3.0),
            ]
        )
        self.assertEqual(assigned["hub"][1], 1.0)
        self.assertEqual(assigned["huadie"][1], 2.0)
        self.assertEqual(assigned["overall"][1], 3.0)


class ParseTests(unittest.TestCase):
    def test_named_values_from_dashboard(self) -> None:
        pairs = dict(named_values(SAMPLE_DASHBOARD))
        self.assertEqual(pairs["HUB闪购场景总流量"], 123456)
        self.assertEqual(pairs["化蝶闪购场景总流量"], 81000)
        self.assertEqual(pairs["整体流量"], 204000)

    def test_ratio_percent_normalized(self) -> None:
        payload = {"charts": [{"name": "整体流量", "value": 100, "日环比": 5.2}]}
        rows = named_values_with_ratio(payload)
        self.assertAlmostEqual(rows[0][2], 0.052)


class WindowTests(unittest.TestCase):
    def test_aligns_to_completed_ten_minutes(self) -> None:
        now = datetime(2026, 8, 19, 10, 23, tzinfo=BEIJING_TZ)
        window = align_window(now, window_minutes=10)
        self.assertEqual(window.start, datetime(2026, 8, 19, 10, 10, tzinfo=BEIJING))
        self.assertEqual(window.end, datetime(2026, 8, 19, 10, 20, tzinfo=BEIJING))
        self.assertEqual(window.raptor_start(), "20260819101000")
        self.assertEqual(window.yesterday.raptor_start(), "20260818101000")


class ReportTests(unittest.TestCase):
    def test_message_contains_three_metrics_and_dod(self) -> None:
        window = align_window(datetime(2026, 8, 19, 10, 23, tzinfo=BEIJING_TZ))
        text = build_report(SAMPLE_DASHBOARD, YESTERDAY_DASHBOARD, window)
        self.assertIn("【闪购流量播报】2026-08-19 10:20（近10分钟）", text)
        self.assertIn("HUB闪购场景总流量：12.35万（日环比 +11.1%）", text)
        self.assertIn("化蝶闪购场景总流量：8.10万（日环比 -10.0%）", text)
        self.assertIn("整体流量：20.40万", text)
        self.assertIn("dashboard=146640", text)

    def test_reconstruct_yesterday_from_embedded_ratio(self) -> None:
        window = align_window(datetime(2026, 8, 19, 10, 23, tzinfo=BEIJING_TZ))
        snapshots = snapshots_from_payloads(SAMPLE_DASHBOARD, None)
        overall = next(item for item in snapshots if item.spec.key == "overall")
        self.assertIsNotNone(overall.dod)
        text = build_report(SAMPLE_DASHBOARD, None, window)
        self.assertIn("整体流量：20.40万（日环比 +2.4%）", text)


class CliTests(unittest.TestCase):
    def test_from_json_prints_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            today = Path(tmp) / "today.json"
            yesterday = Path(tmp) / "yesterday.json"
            today.write_text(json.dumps(SAMPLE_DASHBOARD), encoding="utf-8")
            yesterday.write_text(json.dumps(YESTERDAY_DASHBOARD), encoding="utf-8")
            with mock.patch("flashsale_traffic.__main__.align_window") as mocked:
                mocked.return_value = align_window(
                    datetime(2026, 8, 19, 10, 23, tzinfo=BEIJING_TZ)
                )
                buf = io.StringIO()
                with mock.patch("sys.stdout", buf):
                    code = main(
                        [
                            "--from-json",
                            str(today),
                            "--from-json-yesterday",
                            str(yesterday),
                            "--no-webhook",
                        ]
                    )
        self.assertEqual(code, 0)
        written = buf.getvalue()
        self.assertIn("HUB闪购场景总流量", written)
        self.assertIn("化蝶闪购场景总流量", written)
        self.assertIn("整体流量", written)


if __name__ == "__main__":
    unittest.main()
