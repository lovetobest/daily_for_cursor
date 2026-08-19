"""Fetch Raptor dashboard 146640 chart data (Meituan intranet + SSO cookie)."""

from __future__ import annotations

import json
import os
import ssl
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from flashsale_traffic.metrics import DASHBOARD_ID
from flashsale_traffic.windows import ReportWindow

DEFAULT_BASE_URL = "https://raptor.mws.sankuai.com"

# Paths observed on MWS Raptor SPAs. Probe tries each until one returns JSON.
DASHBOARD_INFO_PATHS = (
    "/api/dashboard/getDashboard",
    "/raptor/api/dashboard/getDashboard",
    "/api/board/info",
    "/raptor/dashboard/api/v2/dashboard/info",
    f"/api/v1/dashboard/{DASHBOARD_ID}",
    f"/raptor/api/dashboard/{DASHBOARD_ID}",
)

CHART_DATA_PATHS = (
    "/api/dashboard/getChartData",
    "/raptor/api/dashboard/getChartData",
    "/api/board/chart/query",
    "/raptor/dashboard/api/v2/chart/data",
    "/api/metric/batchQuery",
    "/raptor/api/metric/batchQuery",
)


class RaptorError(RuntimeError):
    """Raptor is unreachable, unauthenticated, or returned a non-JSON body."""


def load_cookie() -> str:
    raw = os.environ.get("RAPTOR_COOKIE", "").strip()
    if raw:
        return raw
    cookie_file = os.environ.get("RAPTOR_COOKIE_FILE", "").strip()
    if cookie_file:
        with open(cookie_file, encoding="utf-8") as handle:
            return handle.read().strip()
    return ""


def base_url() -> str:
    return os.environ.get("RAPTOR_BASE_URL", DEFAULT_BASE_URL).rstrip("/")


def dashboard_link(window: ReportWindow) -> str:
    query = urllib.parse.urlencode(
        {
            "dashboard": DASHBOARD_ID,
            "isCore": "false",
            "tabName": "metric",
            "startDate": window.raptor_start(),
            "endDate": window.raptor_end(),
            "timeType": "custom",
        }
    )
    return f"{base_url()}/dashboard/list?{query}"


def fetch_window(window: ReportWindow, *, cookie: str | None = None) -> dict[str, Any]:
    """Load dashboard JSON for one time window. Raises RaptorError on failure."""
    token = cookie if cookie is not None else load_cookie()
    opener = _opener(token)
    last_error = "no candidate endpoint succeeded"
    payload = _try_info(opener, window, last_error)
    data = _try_chart_data(opener, window, payload)
    return {"window": {"start": window.raptor_start(), "end": window.raptor_end()}, "data": data}


def probe(cookie: str | None = None) -> list[dict[str, Any]]:
    """Hit candidate endpoints and return status/content-type for debugging on-intranet."""
    token = cookie if cookie is not None else load_cookie()
    opener = _opener(token)
    results: list[dict[str, Any]] = []
    for path in DASHBOARD_INFO_PATHS + CHART_DATA_PATHS:
        url = f"{base_url()}{path}"
        results.append(_probe_one(opener, url))
    return results


def _opener(cookie: str) -> urllib.request.OpenerDirector:
    handlers = [urllib.request.HTTPSHandler(context=ssl.create_default_context())]
    opener = urllib.request.build_opener(*handlers)
    headers = [
        ("Accept", "application/json, text/plain, */*"),
        ("User-Agent", "daily_for_cursor-flashsale-traffic/1.0"),
        ("Referer", f"{base_url()}/dashboard/list?dashboard={DASHBOARD_ID}&isCore=false"),
    ]
    if cookie:
        headers.append(("Cookie", cookie))
    opener.addheaders = headers
    return opener


def _try_info(
    opener: urllib.request.OpenerDirector,
    window: ReportWindow,
    last_error: str,
) -> dict[str, Any]:
    query = {
        "dashboard": str(DASHBOARD_ID),
        "dashboardId": str(DASHBOARD_ID),
        "id": str(DASHBOARD_ID),
        "isCore": "false",
        "startDate": window.raptor_start(),
        "endDate": window.raptor_end(),
        "timeType": "custom",
    }
    for path in DASHBOARD_INFO_PATHS:
        url = f"{base_url()}{path}"
        if "{}" not in path and not path.endswith(str(DASHBOARD_ID)):
            url = f"{url}?{urllib.parse.urlencode(query)}"
        try:
            body = _json_get(opener, url)
            if isinstance(body, dict) or isinstance(body, list):
                return body if isinstance(body, dict) else {"items": body}
        except RaptorError as exc:
            last_error = str(exc)
    raise RaptorError(
        "无法读取 Raptor 大盘 JSON。"
        f" 请在能解析 {base_url()} 的美团内网机器上运行，并设置 RAPTOR_COOKIE。"
        f" 最近错误：{last_error}"
    )


def _try_chart_data(
    opener: urllib.request.OpenerDirector,
    window: ReportWindow,
    info: dict[str, Any],
) -> dict[str, Any]:
    payload = {
        "dashboard": DASHBOARD_ID,
        "dashboardId": DASHBOARD_ID,
        "id": DASHBOARD_ID,
        "isCore": False,
        "startDate": window.raptor_start(),
        "endDate": window.raptor_end(),
        "startTime": window.raptor_start(),
        "endTime": window.raptor_end(),
        "timeType": "custom",
    }
    for path in CHART_DATA_PATHS:
        url = f"{base_url()}{path}"
        try:
            body = _json_post(opener, url, payload)
            if isinstance(body, (dict, list)):
                return {"info": info, "charts": body}
        except RaptorError:
            continue
    return {"info": info, "charts": info}


def _json_get(opener: urllib.request.OpenerDirector, url: str) -> Any:
    try:
        with opener.open(url, timeout=20) as response:
            raw = response.read()
            ctype = response.headers.get("Content-Type", "")
    except urllib.error.HTTPError as exc:
        raise RaptorError(f"GET {url} -> HTTP {exc.code}") from exc
    except urllib.error.URLError as exc:
        raise RaptorError(f"GET {url} failed: {exc.reason}") from exc
    return _decode_json(url, raw, ctype)


def _json_post(opener: urllib.request.OpenerDirector, url: str, payload: dict[str, Any]) -> Any:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with opener.open(request, timeout=20) as response:
            raw = response.read()
            ctype = response.headers.get("Content-Type", "")
    except urllib.error.HTTPError as exc:
        raise RaptorError(f"POST {url} -> HTTP {exc.code}") from exc
    except urllib.error.URLError as exc:
        raise RaptorError(f"POST {url} failed: {exc.reason}") from exc
    return _decode_json(url, raw, ctype)


def _decode_json(url: str, raw: bytes, content_type: str) -> Any:
    text = raw.decode("utf-8", errors="replace").strip()
    if not text:
        raise RaptorError(f"{url} returned an empty body")
    if "html" in content_type.lower() or text[:15].lower().startswith("<!doctype") or text[:6].lower().startswith("<html"):
        raise RaptorError(f"{url} returned HTML (usually SSO login). Set RAPTOR_COOKIE.")
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise RaptorError(f"{url} was not JSON: {exc}") from exc


def _probe_one(opener: urllib.request.OpenerDirector, url: str) -> dict[str, Any]:
    result: dict[str, Any] = {"url": url}
    try:
        with opener.open(url, timeout=10) as response:
            result["status"] = response.status
            result["content_type"] = response.headers.get("Content-Type", "")
            preview = response.read(180).decode("utf-8", errors="replace")
            result["preview"] = preview.replace("\n", " ")[:180]
    except urllib.error.HTTPError as exc:
        result["status"] = exc.code
        result["error"] = exc.reason
    except urllib.error.URLError as exc:
        result["status"] = 0
        result["error"] = str(exc.reason)
    return result
