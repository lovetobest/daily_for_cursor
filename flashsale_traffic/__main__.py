"""Broadcast HUB / 化蝶 / overall flash-sale traffic every 10 minutes.

Usage (Meituan intranet + SSO cookie):

    export RAPTOR_COOKIE='ssoid=...; ...'
    python3 -m flashsale_traffic

Dry-run with saved dashboard JSON (no network):

    python3 -m flashsale_traffic --from-json today.json --from-json-yesterday yesterday.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from flashsale_traffic.metrics import DEFAULT_WINDOW_MINUTES
from flashsale_traffic.raptor import RaptorError, fetch_window, probe
from flashsale_traffic.report import build_report
from flashsale_traffic.webhook import post_text, webhook_url
from flashsale_traffic.windows import align_window


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="闪购场景流量播报（HUB / 化蝶 / 整体 + 日环比）")
    parser.add_argument("--window-minutes", type=int, default=DEFAULT_WINDOW_MINUTES)
    parser.add_argument("--from-json", type=Path, help="当前窗口大盘/图表 JSON")
    parser.add_argument("--from-json-yesterday", type=Path, help="昨日同时段 JSON；省略则只用今日接口里的日环比")
    parser.add_argument("--probe", action="store_true", help="探测 Raptor API 候选路径后退出")
    parser.add_argument("--no-webhook", action="store_true", help="只打印，不 POST 到 DX_ROBOT_URL/WEBHOOK_URL")
    args = parser.parse_args(argv)

    window = align_window(window_minutes=args.window_minutes)

    if args.probe:
        for row in probe():
            status = row.get("status")
            extra = row.get("error") or row.get("content_type") or ""
            print(f"{status}\t{row['url']}\t{extra}")
        return 0

    note = None
    try:
        current_payload, yesterday_payload = _load_payloads(args, window)
    except RaptorError as exc:
        print(f"Raptor 拉取失败：{exc}", file=sys.stderr)
        return 2
    except (OSError, json.JSONDecodeError) as exc:
        print(f"读取 JSON 失败：{exc}", file=sys.stderr)
        return 2

    if yesterday_payload is None and args.from_json_yesterday is None:
        note = "昨日同时段未单独查询：日环比仅在大盘字段自带时展示。"

    text = build_report(current_payload, yesterday_payload, window, missing_note=note)
    print(text)

    if not args.no_webhook and webhook_url():
        try:
            post_text(text)
        except RuntimeError as exc:
            print(str(exc), file=sys.stderr)
            return 3
    return 0


def _load_payloads(args: argparse.Namespace, window):
    if args.from_json:
        current = json.loads(args.from_json.read_text(encoding="utf-8"))
        yesterday = None
        if args.from_json_yesterday:
            yesterday = json.loads(args.from_json_yesterday.read_text(encoding="utf-8"))
        return current, yesterday

    current = fetch_window(window)
    yesterday = fetch_window(window.yesterday)
    return current, yesterday


if __name__ == "__main__":
    raise SystemExit(main())
