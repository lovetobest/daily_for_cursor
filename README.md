# daily_for_cursor

Personal Cursor workspace. Includes a stdlib-only **beam search** demo and a **闪购流量播报** CLI that reads [Raptor dashboard 146640](https://raptor.mws.sankuai.com/dashboard/list?dashboard=146640&isCore=false&tabName=metric).

## 闪购流量播报

Every completed 10-minute bucket in `Asia/Shanghai`, print **当前流量 + 日环比** for:

- HUB闪购场景总流量
- 化蝶闪购场景总流量
- 整体流量

```bash
# On a Meituan-intranet host that can resolve raptor.mws.sankuai.com
export RAPTOR_COOKIE='ssoid=...; ...'   # copy Cookie from a logged-in Raptor tab
python3 -m flashsale_traffic
```

Crontab (every 10 minutes, on an intranet jumphost):

```cron
*/10 * * * * cd /path/to/daily_for_cursor && RAPTOR_COOKIE_FILE=$HOME/.raptor_cookie python3 -m flashsale_traffic
```

Optional: set `DX_ROBOT_URL` or `WEBHOOK_URL` to POST the same text to a 大象 robot. Use `--probe` to see which Raptor API paths respond. Offline:

```bash
python3 -m flashsale_traffic --from-json today.json --from-json-yesterday yesterday.json
```

## Beam search demo

```bash
python3 -m beamsearch
```

Greedy decoding locks onto `the cat sat on the mat` because `cat` is the locally better choice after `the`. A wider beam also keeps `dog`, and finishes with the globally better sequence `the dog ran away`.

```bash
python3 -m beamsearch --beam-width 3
```

## Tests

```bash
python3 -m unittest discover -s tests -v
```

## Layout

| Path | Role |
|------|------|
| `flashsale_traffic/` | 10-minute HUB / 化蝶 / 整体流量播报 |
| `beamsearch/search.py` | `beam_search` and `greedy_search` |
| `beamsearch/toy_lm.py` | Tiny hand-written next-token model |
| `beamsearch/__main__.py` | CLI demo |
| `tests/` | Unit tests |

No third-party packages are required (Python 3.9+).
