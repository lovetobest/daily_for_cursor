"""Tiny food-delivery catalog with hand-written 3-d embeddings.

Axes of ``Item.embedding`` (id embedding, used by sequential / DIN models):
  0  Japanese / hot-pot style food
  1  cafe / coffee
  2  western fast food

``Item.tokens`` are a bag of semantic word pieces for the *text* dual-encoder
(BERT-style semantic retrieval). A cold-start item can be retrieved from
tokens even if nobody has clicked it.

The demo user browses cafe twice, then clicks ramen. Mean-pooling two-tower
still looks like cafe; last-click SASRec switches to Japanese food; DIN
builds a *different* user vector for each candidate.
"""

from __future__ import annotations

from dataclasses import dataclass


EMBED_DIM = 3


@dataclass(frozen=True)
class Item:
    item_id: str
    name: str
    category: str
    embedding: tuple[float, ...]
    tokens: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if len(self.embedding) != EMBED_DIM:
            raise ValueError(f"embedding must be {EMBED_DIM}-d")
        if not self.tokens:
            raise ValueError("tokens must be non-empty")


# Token geometry matches the three axes so "jp + noodle" lands near ramen.
TOKEN_EMB: dict[str, tuple[float, ...]] = {
    "jp": (1.00, 0.00, 0.00),
    "noodle": (0.90, 0.00, 0.10),
    "fish": (0.85, 0.10, 0.00),
    "spicy": (0.75, 0.00, 0.20),
    "savory": (0.80, 0.05, 0.10),
    "cafe": (0.00, 1.00, 0.00),
    "drink": (0.00, 0.95, 0.05),
    "sweet": (0.00, 0.80, 0.20),
    "milk": (0.00, 0.90, 0.10),
    "west": (0.00, 0.10, 1.00),
    "meat": (0.05, 0.00, 0.95),
    "meal": (0.10, 0.05, 0.85),
}

ITEMS: tuple[Item, ...] = (
    Item("ramen", "Tonkotsu ramen", "japanese", (1.00, 0.05, 0.05), ("jp", "noodle", "savory")),
    Item("sushi", "Salmon sushi", "japanese", (0.95, 0.10, 0.00), ("jp", "fish")),
    Item("hotpot", "Spicy hotpot", "japanese", (0.90, 0.00, 0.10), ("jp", "spicy", "meal")),
    Item("udon", "Udon noodles", "japanese", (0.85, 0.08, 0.05), ("jp", "noodle")),
    Item("coffee", "Drip coffee", "cafe", (0.05, 1.00, 0.05), ("cafe", "drink")),
    Item("cake", "Cheesecake", "cafe", (0.05, 0.90, 0.15), ("cafe", "sweet")),
    Item("latte", "Vanilla latte", "cafe", (0.00, 0.95, 0.05), ("cafe", "drink", "milk")),
    Item("burger", "Cheeseburger", "western", (0.05, 0.10, 1.00), ("west", "meat", "meal")),
    Item("pizza", "Pepperoni pizza", "western", (0.00, 0.15, 0.95), ("west", "meal")),
)

CATALOG: dict[str, Item] = {item.item_id: item for item in ITEMS}

# Cafe, cafe, then a Japanese click — long-term vs last-click conflict.
DEFAULT_HISTORY: tuple[str, ...] = ("coffee", "cake", "ramen")

# Shared [MASK] vector for BERT-style cloze: overlaps Japanese and cafe.
MASK_ID = "[MASK]"
MASK_EMBEDDING: tuple[float, ...] = (0.40, 0.40, 0.00)

# udon has no clicks in the demo session — semantic recall only.
COLD_START_ID = "udon"


def item_embedding(item_id: str) -> tuple[float, ...]:
    if item_id == MASK_ID:
        return MASK_EMBEDDING
    try:
        return CATALOG[item_id].embedding
    except KeyError as exc:
        raise KeyError(f"unknown item_id: {item_id!r}") from exc


def token_embedding(token: str) -> tuple[float, ...]:
    try:
        return TOKEN_EMB[token]
    except KeyError as exc:
        raise KeyError(f"unknown token: {token!r}") from exc


def require_history(history: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    tokens = tuple(history)
    if not tokens:
        raise ValueError("history must be non-empty")
    for item_id in tokens:
        if item_id != MASK_ID and item_id not in CATALOG:
            raise KeyError(f"unknown item_id: {item_id!r}")
    return tokens


def history_tokens(history: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    """Flatten item word-pieces in click order (semantic user tower)."""
    pieces: list[str] = []
    for item_id in require_history(history):
        if item_id == MASK_ID:
            continue
        pieces.extend(CATALOG[item_id].tokens)
    if not pieces:
        raise ValueError("history has no text tokens")
    return tuple(pieces)
