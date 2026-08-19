"""Optional POST of the broadcast text (大象机器人 / generic webhook)."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request


def webhook_url() -> str:
    return (os.environ.get("DX_ROBOT_URL") or os.environ.get("WEBHOOK_URL") or "").strip()


def post_text(text: str, url: str | None = None) -> None:
    target = url if url is not None else webhook_url()
    if not target:
        return
    payload = json.dumps({"msgtype": "text", "text": {"content": text}}).encode("utf-8")
    request = urllib.request.Request(
        target,
        data=payload,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            response.read()
    except urllib.error.URLError as exc:
        raise RuntimeError(f"webhook POST failed: {exc.reason}") from exc
