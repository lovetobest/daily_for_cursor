"""DIN-style target attention: a *per-candidate* user vector.

Two-tower and SASRec produce one user vector per request, so they can
retrieve with ANN. DIN (and BST with the candidate in the sequence)
compute attention *toward the candidate*. That is high-order interaction
and cannot be an inner-product search over millions of items — it is a
ranker on a few hundred recalled candidates.

Toy activation: ``softmax( <history_i, candidate> / temperature )``.
The real DIN paper uses an MLP on ``[h, c, h-c, h*c]``; the geometry is
the same idea.
"""

from __future__ import annotations

from recsys_models.catalog import CATALOG, item_embedding, require_history
from recsys_models.models import ScoredItem
from recsys_models.vectors import Vector, cosine, softmax, weighted_sum


class TargetAttentionRanker:
    """Alibaba DIN toy: candidate-aware pooling of the click sequence."""

    name = "DIN (target-attention)"

    def __init__(self, *, temperature: float = 0.2) -> None:
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        self.temperature = temperature

    def attend(
        self,
        history: tuple[str, ...] | list[str],
        candidate_id: str,
    ) -> tuple[Vector, list[float]]:
        """Return (user vector for this candidate, attention over history)."""
        tokens = require_history(history)
        if candidate_id not in CATALOG:
            raise KeyError(f"unknown item_id: {candidate_id!r}")
        cand = item_embedding(candidate_id)
        keys = [item_embedding(item_id) for item_id in tokens]
        logits = [cosine(item, cand) for item in keys]
        weights = softmax(logits, temperature=self.temperature)
        return weighted_sum(weights, keys), weights

    def score(self, history: tuple[str, ...] | list[str], candidate_id: str) -> float:
        user, _ = self.attend(history, candidate_id)
        return cosine(user, item_embedding(candidate_id))

    def rank(
        self,
        history: tuple[str, ...] | list[str],
        *,
        candidates: list[str] | tuple[str, ...] | None = None,
    ) -> list[ScoredItem]:
        tokens = require_history(history)
        pool = candidates if candidates is not None else [
            item_id for item_id in CATALOG if item_id not in tokens
        ]
        ranked: list[ScoredItem] = []
        for item_id in pool:
            if item_id in tokens:
                continue
            ranked.append(ScoredItem(item=CATALOG[item_id], score=self.score(tokens, item_id)))
        ranked.sort(key=lambda row: row.score, reverse=True)
        return ranked
