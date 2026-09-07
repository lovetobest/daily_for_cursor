"""Snapshot vs realtime user features, and a quality + interest ranker.

Real-time recommendation is not only low serving latency. The more common
production gap is **feature freshness**: how soon a new click enters the
profile that ranking reads. Snapshot stores (batch / nearline) freeze
interest at flush time; realtime rebuilds it from every event before the
next request.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable

from realtime_recsys.catalog import Catalog, Event, Item

KIND_WEIGHT = {"click": 1.0, "order": 3.0}
MODES = frozenset({"realtime", "snapshot"})


@dataclass(frozen=True)
class UserProfile:
    """Category interest as of a feature-compute timestamp."""

    user_id: str
    as_of: float | None
    interest: dict[str, float] = field(default_factory=dict)
    event_count: int = 0
    latest_event_ts: float | None = None

    def freshness_lag(self, now: float) -> float:
        """Seconds between feature compute time and the recommend request.

        Realtime profiles are computed at ``now`` so the lag is 0. Snapshot
        profiles keep the last flush time, so lag grows until the next job.
        """
        if self.as_of is None:
            return math.inf
        return now - self.as_of


@dataclass(frozen=True)
class ScoredItem:
    item: Item
    score: float


class FeatureStore:
    """In-memory event log with an optional materialized snapshot."""

    def __init__(self, catalog: Catalog, *, half_life: float = 1800.0) -> None:
        if half_life <= 0:
            raise ValueError("half_life must be > 0")
        self.catalog = catalog
        self.half_life = half_life
        self._events: list[Event] = []
        self._snapshots: dict[str, UserProfile] = {}

    def ingest(self, event: Event) -> None:
        if event.kind not in KIND_WEIGHT:
            raise ValueError(f"unknown event kind: {event.kind!r}")
        if event.item_id not in self.catalog:
            raise ValueError(f"unknown item_id: {event.item_id!r}")
        if not event.user_id:
            raise ValueError("user_id must be non-empty")
        self._events.append(event)

    def ingest_many(self, events: Iterable[Event]) -> None:
        for event in events:
            self.ingest(event)

    def flush(self, now: float) -> None:
        """Materialize snapshot profiles as of ``now`` (batch / nearline job)."""
        users = {event.user_id for event in self._events}
        self._snapshots = {user_id: self._compute(user_id, as_of=now) for user_id in users}

    def profile(self, user_id: str, now: float, *, mode: str) -> UserProfile:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {sorted(MODES)}")
        if mode == "realtime":
            return self._compute(user_id, as_of=now)
        return self._snapshots.get(
            user_id,
            UserProfile(user_id=user_id, as_of=None),
        )

    def _compute(self, user_id: str, *, as_of: float) -> UserProfile:
        interest: dict[str, float] = {}
        count = 0
        latest: float | None = None
        for event in self._events:
            if event.user_id != user_id or event.ts > as_of:
                continue
            item = self.catalog[event.item_id]
            age = as_of - event.ts
            weight = KIND_WEIGHT[event.kind] * (0.5 ** (age / self.half_life))
            interest[item.category] = interest.get(item.category, 0.0) + weight
            count += 1
            if latest is None or event.ts > latest:
                latest = event.ts
        return UserProfile(
            user_id=user_id,
            as_of=as_of,
            interest=interest,
            event_count=count,
            latest_event_ts=latest,
        )


def recommend(
    catalog: Catalog,
    profile: UserProfile,
    *,
    k: int = 3,
) -> list[ScoredItem]:
    """Rank ``score = quality + category_interest``. Stable by ``item_id``."""
    if k < 1:
        raise ValueError("k must be >= 1")
    ranked = [
        ScoredItem(
            item=item,
            score=item.quality + profile.interest.get(item.category, 0.0),
        )
        for item in catalog
    ]
    ranked.sort(key=lambda row: (-row.score, row.item.item_id))
    return ranked[:k]


def top_category(ranking: list[ScoredItem]) -> str | None:
    if not ranking:
        return None
    return ranking[0].item.category
