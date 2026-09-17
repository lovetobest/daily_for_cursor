"""BERT as a *semantic* two-tower: encode item text, not just item ids.

In search and content rec this is often more important than BERT4Rec.
Each tower is a tiny bidirectional transformer over word-pieces; score is
cosine. New items (cold start) get a vector from their title tokens
without any click history.

Real systems swap this block for BERT / BERT-mini / CLIP-text and train
with InfoNCE on (query, clicked-item) pairs.
"""

from __future__ import annotations

from recsys_models.catalog import (
    CATALOG,
    history_tokens,
    require_history,
    token_embedding,
)
from recsys_models.models import ScoredItem, self_attention
from recsys_models.vectors import Vector, cosine, mean_pool


class BertDualEncoder:
    """Two-tower whose towers are bidirectional token encoders."""

    name = "BERT dual-encoder (semantic 双塔)"

    def encode_tokens(self, tokens: tuple[str, ...] | list[str]) -> Vector:
        pieces = tuple(tokens)
        if not pieces:
            raise ValueError("tokens must be non-empty")
        embeddings = [list(token_embedding(tok)) for tok in pieces]
        hidden, _ = self_attention(embeddings, causal=False)
        return mean_pool(hidden)

    def user_embedding(self, history: tuple[str, ...] | list[str]) -> Vector:
        return self.encode_tokens(history_tokens(history))

    def item_embedding(self, item_id: str) -> Vector:
        if item_id not in CATALOG:
            raise KeyError(f"unknown item_id: {item_id!r}")
        return self.encode_tokens(CATALOG[item_id].tokens)

    def rank(
        self,
        history: tuple[str, ...] | list[str],
        *,
        candidates: list[str] | tuple[str, ...] | None = None,
    ) -> list[ScoredItem]:
        tokens = require_history(history)
        query = self.user_embedding(tokens)
        pool = candidates if candidates is not None else CATALOG.keys()
        ranked: list[ScoredItem] = []
        for item_id in pool:
            if item_id in tokens:
                continue
            item = CATALOG[item_id]
            ranked.append(
                ScoredItem(item=item, score=cosine(query, self.item_embedding(item_id)))
            )
        ranked.sort(key=lambda row: row.score, reverse=True)
        return ranked
