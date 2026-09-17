"""Three industrial recsys families, as tiny attention / pooling models.

Two-tower (双塔 / DSSM)
    User tower and item tower encode independently. Score is cosine
    similarity. Item vectors can be precomputed and retrieved with ANN —
    this is still the default industrial *recall* architecture.

Transformer (SASRec-style)
    One-head causal self-attention over the click sequence. Recency bias
    on the attention logits makes last-click intent dominate *ranking*.

BERT (BERT4Rec-style)
    The same attention block, but bidirectional, with a [MASK] token.
    Next-item inference puts [MASK] at the end (session bag). The
    distinctive BERT trick is *cloze*: fill a hole using left AND right
    context, which a causal transformer cannot see.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

from recsys_models.catalog import (
    CATALOG,
    MASK_ID,
    Item,
    item_embedding,
    require_history,
)
from recsys_models.vectors import Vector, dot, l2_normalize, mean_pool, softmax


@dataclass(frozen=True)
class ScoredItem:
    item: Item
    score: float


def rank_against(
    query: list[float] | tuple[float, ...],
    *,
    exclude: frozenset[str] | set[str] = frozenset(),
    cosine: bool = False,
) -> list[ScoredItem]:
    """Score every catalog item against ``query``, highest first."""
    q = l2_normalize(query) if cosine else list(query)
    ranked: list[ScoredItem] = []
    for item in CATALOG.values():
        if item.item_id in exclude:
            continue
        key = l2_normalize(item.embedding) if cosine else item.embedding
        ranked.append(ScoredItem(item=item, score=dot(q, key)))
    ranked.sort(key=lambda row: row.score, reverse=True)
    return ranked


def _sequence_embeddings(item_ids: tuple[str, ...]) -> list[Vector]:
    return [list(item_embedding(item_id)) for item_id in item_ids]


def self_attention(
    embeddings: list[Vector],
    *,
    causal: bool,
    recency_slope: float = 0.0,
) -> tuple[list[Vector], list[list[float]]]:
    """Single-head scaled dot-product attention, identity Q/K/V.

    ``recency_slope`` adds ``slope * j / max(n-1, 1)`` to the logit for
    key position ``j``, a toy stand-in for positional embeddings.
    """
    if not embeddings:
        raise ValueError("embeddings must be non-empty")
    n = len(embeddings)
    dim = len(embeddings[0])
    scale = sqrt(dim)
    denom = max(n - 1, 1)
    hidden: list[Vector] = []
    weights: list[list[float]] = []
    for i, query in enumerate(embeddings):
        logits: list[float] = []
        for j, key in enumerate(embeddings):
            if causal and j > i:
                logits.append(float("-inf"))
                continue
            recency = recency_slope * (j / denom)
            logits.append(dot(query, key) / scale + recency)
        attn = softmax(logits)
        mixed = [0.0] * dim
        for weight, value in zip(attn, embeddings):
            for d, x in enumerate(value):
                mixed[d] += weight * x
        hidden.append(mixed)
        weights.append(attn)
    return hidden, weights


class TwoTower:
    """User embedding = mean of history; item embedding is a lookup.

    Towers never look at each other until the final cosine — the property
    that makes two-tower retrieval cacheable.
    """

    name = "two-tower (双塔)"

    def user_embedding(self, history: tuple[str, ...] | list[str]) -> Vector:
        tokens = require_history(history)
        return mean_pool(_sequence_embeddings(tokens))

    def rank(self, history: tuple[str, ...] | list[str]) -> list[ScoredItem]:
        tokens = require_history(history)
        query = self.user_embedding(tokens)
        return rank_against(query, exclude=set(tokens), cosine=True)


class CausalTransformer:
    """SASRec-style next-item ranker: causal attention + recency."""

    name = "transformer (SASRec)"

    def __init__(self, *, recency_slope: float = 2.0) -> None:
        self.recency_slope = recency_slope

    def encode(
        self,
        history: tuple[str, ...] | list[str],
        *,
        recency_slope: float | None = None,
    ) -> tuple[list[Vector], list[list[float]]]:
        tokens = require_history(history)
        slope = self.recency_slope if recency_slope is None else recency_slope
        return self_attention(
            _sequence_embeddings(tokens),
            causal=True,
            recency_slope=slope,
        )

    def rank(self, history: tuple[str, ...] | list[str]) -> list[ScoredItem]:
        tokens = require_history(history)
        hidden, _ = self.encode(tokens)
        return rank_against(hidden[-1], exclude=set(tokens), cosine=False)


class Bert4Rec:
    """BERT4Rec-style bidirectional encoder with a [MASK] token."""

    name = "BERT (BERT4Rec)"

    def encode(
        self,
        history: tuple[str, ...] | list[str],
        *,
        causal: bool = False,
        recency_slope: float = 0.0,
    ) -> tuple[list[Vector], list[list[float]]]:
        tokens = require_history(history)
        return self_attention(
            _sequence_embeddings(tokens),
            causal=causal,
            recency_slope=recency_slope,
        )

    def rank(self, history: tuple[str, ...] | list[str]) -> list[ScoredItem]:
        """Next-item: append [MASK], read the masked position (bidirectional)."""
        tokens = require_history(history)
        masked = tokens + (MASK_ID,)
        hidden, _ = self.encode(masked)
        return rank_against(hidden[-1], exclude=set(tokens), cosine=False)

    def cloze(
        self,
        left: tuple[str, ...] | list[str],
        right: tuple[str, ...] | list[str],
        *,
        causal: bool = False,
    ) -> list[ScoredItem]:
        """Fill [MASK] between ``left`` and ``right``.

        ``causal=True`` is the SASRec ablation: the mask position may only
        attend left, so it cannot use future clicks.
        """
        left_t = require_history(left)
        right_t = require_history(right)
        tokens = left_t + (MASK_ID,) + right_t
        mask_index = len(left_t)
        hidden, _ = self.encode(tokens, causal=causal)
        exclude = {item_id for item_id in tokens if item_id != MASK_ID}
        return rank_against(hidden[mask_index], exclude=exclude, cosine=False)
