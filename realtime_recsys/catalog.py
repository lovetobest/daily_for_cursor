"""Toy catalog and a preference-shift event log for the freshness demo.

The catalog is a tiny food-delivery set: coffee, hotpot, and milk tea.
Historical clicks are coffee; a later burst of hotpot clicks is the intent shift.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator

# Last batch/nearline materialization in the demo scenario.
FLUSH_TS = 0.0
QUERY_TS = 40.0
USER_ID = "u1"
HALF_LIFE = 1800.0


@dataclass(frozen=True)
class Item:
    """A recommendable shop with a static quality score in ``[0, 1]``."""

    item_id: str
    name: str
    category: str
    quality: float


@dataclass(frozen=True)
class Event:
    """A user interaction used to build the interest profile."""

    ts: float
    user_id: str
    item_id: str
    kind: str = "click"


class Catalog:
    """Lookup table of items by id."""

    def __init__(self, items: Iterable[Item]) -> None:
        self._items = {item.item_id: item for item in items}
        if not self._items:
            raise ValueError("catalog must contain at least one item")

    def __getitem__(self, item_id: str) -> Item:
        return self._items[item_id]

    def __contains__(self, item_id: object) -> bool:
        return item_id in self._items

    def __iter__(self) -> Iterator[Item]:
        return iter(self._items.values())

    def __len__(self) -> int:
        return len(self._items)


def toy_catalog() -> Catalog:
    """Nine shops. Coffee has the highest static quality, then hotpot, then milk tea."""
    return Catalog(
        [
            Item("manner", "Manner Coffee", "coffee", 0.90),
            Item("starbucks", "Starbucks", "coffee", 0.82),
            Item("arabica", "%Arabica", "coffee", 0.74),
            Item("haidilao", "Haidilao / 海底捞", "hotpot", 0.88),
            Item("xiaolongkan", "Xiaolongkan / 小龙坎", "hotpot", 0.80),
            Item("shudaxia", "Shu Daxia / 蜀大侠", "hotpot", 0.72),
            Item("heytea", "Heytea / 喜茶", "milktea", 0.70),
            Item("nayuki", "Nayuki / 奈雪", "milktea", 0.62),
            Item("mixue", "Mixue / 蜜雪冰城", "milktea", 0.54),
        ]
    )


def preference_shift_events(user_id: str = USER_ID) -> list[Event]:
    """Coffee history before the last flush, then three hotpot clicks after it.

    Times are seconds relative to the last snapshot job (``FLUSH_TS = 0``).
    """
    return [
        Event(-7200.0, user_id, "manner"),
        Event(-5400.0, user_id, "starbucks"),
        Event(-3600.0, user_id, "arabica"),
        Event(10.0, user_id, "haidilao"),
        Event(20.0, user_id, "xiaolongkan"),
        Event(30.0, user_id, "shudaxia"),
    ]
