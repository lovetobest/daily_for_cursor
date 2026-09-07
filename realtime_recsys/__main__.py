"""Show why snapshot features miss a mid-session intent shift.

Usage:
    python3 -m realtime_recsys
"""

from __future__ import annotations

import argparse
import unicodedata

from realtime_recsys.catalog import (
    FLUSH_TS,
    HALF_LIFE,
    QUERY_TS,
    USER_ID,
    preference_shift_events,
    toy_catalog,
)
from realtime_recsys.engine import FeatureStore, ScoredItem, recommend, top_category


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Demo recommendation freshness: snapshot vs realtime features.",
    )
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--half-life", type=float, default=HALF_LIFE)
    args = parser.parse_args()

    catalog = toy_catalog()
    store = FeatureStore(catalog, half_life=args.half_life)
    store.ingest_many(preference_shift_events())
    store.flush(FLUSH_TS)

    print("Recommendation freshness demo")
    print(
        "User u1 liked coffee for hours, then clicked three hotpot shops "
        f"at t=10, 20, 30. Query at t={QUERY_TS:.0f}. "
        f"Half-life = {args.half_life:g}s."
    )

    _print_block(
        "Snapshot (batch / nearline, last flush t=0)",
        store,
        mode="snapshot",
        k=args.k,
        note="stale — still coffee; the hotpot clicks have not been flushed",
    )
    _print_block(
        "Realtime (profile rebuilt from every event)",
        store,
        mode="realtime",
        k=args.k,
        note="follows the latest clicks",
    )

    store.flush(QUERY_TS)
    _print_block(
        "Snapshot after a nearline flush at t=40",
        store,
        mode="snapshot",
        k=args.k,
        note="now matches realtime; freshness cost is the flush interval",
    )


def _print_block(
    title: str,
    store: FeatureStore,
    *,
    mode: str,
    k: int,
    note: str,
) -> None:
    profile = store.profile(USER_ID, QUERY_TS, mode=mode)
    ranking = recommend(store.catalog, profile, k=k)
    lag = profile.freshness_lag(QUERY_TS)
    lag_text = "inf" if lag == float("inf") else f"{lag:.1f}s"
    as_of = "none" if profile.as_of is None else f"{profile.as_of:.1f}"
    print(f"\n== {title} ==")
    print(f"  feature as_of: {as_of}    freshness lag: {lag_text}")
    _print_ranking(ranking)
    print(f"  top category: {top_category(ranking)}   ({note})")


def _print_ranking(ranking: list[ScoredItem]) -> None:
    name_width = max(_display_width(row.item.name) for row in ranking)
    for i, row in enumerate(ranking, start=1):
        name = _pad(row.item.name, name_width)
        print(f"  {i}. {name}  {row.item.category:<8} {row.score:.4f}")


def _display_width(text: str) -> int:
    width = 0
    for char in text:
        width += 2 if unicodedata.east_asian_width(char) in {"W", "F"} else 1
    return width


def _pad(text: str, width: int) -> str:
    return text + " " * max(width - _display_width(text), 0)


if __name__ == "__main__":
    main()
