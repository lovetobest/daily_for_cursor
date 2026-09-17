"""Tiny food-delivery catalog with hand-written 3-d embeddings.

Axes:
  0  Japanese / hot-pot style food
  1  cafe / coffee
  2  western fast food

The demo user browses cafe twice, then clicks ramen. Mean-pooling
(two-tower) still looks like cafe; last-click sequential models switch
to Japanese food.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Item:
    item_id: str
    name: str
    category: str
    embedding: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.embedding) != 3:
            raise ValueError("embedding must be 3-d")


ITEMS: tuple[Item, ...] = (
    Item("ramen", "Tonkotsu ramen", "japanese", (1.00, 0.05, 0.05)),
    Item("sushi", "Salmon sushi", "japanese", (0.95, 0.10, 0.00)),
    Item("hotpot", "Spicy hotpot", "japanese", (0.90, 0.00, 0.10)),
    Item("coffee", "Drip coffee", "cafe", (0.05, 1.00, 0.05)),
    Item("cake", "Cheesecake", "cafe", (0.05, 0.90, 0.15)),
    Item("latte", "Vanilla latte", "cafe", (0.00, 0.95, 0.05)),
    Item("burger", "Cheeseburger", "western", (0.05, 0.10, 1.00)),
    Item("pizza", "Pepperoni pizza", "western", (0.00, 0.15, 0.95)),
)

CATALOG: dict[str, Item] = {item.item_id: item for item in ITEMS}

# Cafe, cafe, then a Japanese click — long-term vs last-click conflict.
DEFAULT_HISTORY: tuple[str, ...] = ("coffee", "cake", "ramen")

# Shared [MASK] vector for BERT-style cloze: overlaps Japanese and cafe.
MASK_ID = "[MASK]"
MASK_EMBEDDING: tuple[float, ...] = (0.40, 0.40, 0.00)


def item_embedding(item_id: str) -> tuple[float, ...]:
    if item_id == MASK_ID:
        return MASK_EMBEDDING
    try:
        return CATALOG[item_id].embedding
    except KeyError as exc:
        raise KeyError(f"unknown item_id: {item_id!r}") from exc


def require_history(history: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    tokens = tuple(history)
    if not tokens:
        raise ValueError("history must be non-empty")
    for item_id in tokens:
        if item_id != MASK_ID and item_id not in CATALOG:
            raise KeyError(f"unknown item_id: {item_id!r}")
    return tokens
