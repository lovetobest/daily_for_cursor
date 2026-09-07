# daily_for_cursor

Small, dependency-free Python demos (Python 3.9+).

```bash
python3 -m unittest discover -s tests -v
```

## Beam search

Keep the top *k* partial sequences while decoding, instead of always taking the single best next token (greedy search).

Greedy decoding locks onto `the cat sat on the mat` because `cat` is the locally better choice after `the`. A wider beam also keeps `dog`, and finishes with the globally better sequence `the dog ran away`.

```bash
python3 -m beamsearch
python3 -m beamsearch --beam-width 3
```

| Path | Role |
|------|------|
| `beamsearch/search.py` | `beam_search` and `greedy_search` |
| `beamsearch/toy_lm.py` | Tiny hand-written next-token model |
| `beamsearch/__main__.py` | CLI demo |
| `tests/test_search.py` | Unit tests |

## Recommendation freshness

A toy food-delivery ranker that shows **feature freshness** — the usual meaning of 推荐系统实时性 once serving latency is already small.

User `u1` liked coffee for hours. After the last snapshot job they click three hotpot shops. Ranking is `quality + recency-weighted category interest`.

- **Snapshot** (batch / nearline): the profile is frozen at flush time, so the next request still returns coffee.
- **Realtime**: every event is in the profile before the next request, so the list switches to hotpot.

```bash
python3 -m realtime_recsys
```

| Path | Role |
|------|------|
| `realtime_recsys/engine.py` | `FeatureStore`, `recommend` |
| `realtime_recsys/catalog.py` | Shops and the preference-shift log |
| `realtime_recsys/__main__.py` | CLI demo |
| `tests/test_realtime_recsys.py` | Unit tests |
