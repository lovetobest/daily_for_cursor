"""Recall-then-rank funnel: two-tower retrieves, a ranker reorders."""

from __future__ import annotations

from dataclasses import dataclass

from recsys_models.models import ScoredItem, TwoTower


@dataclass(frozen=True)
class FunnelResult:
    recalled: list[ScoredItem]
    reranked: list[ScoredItem]


def recall_then_rank(
    history: tuple[str, ...] | list[str],
    *,
    retriever: TwoTower | None = None,
    ranker,
    recall_k: int = 4,
) -> FunnelResult:
    """Industrial serving sketch: ANN retrieve ``recall_k``, then score them.

    ``ranker`` must expose ``rank(history, candidates=...)``.
    """
    if recall_k < 1:
        raise ValueError("recall_k must be >= 1")
    retriever = TwoTower() if retriever is None else retriever
    recalled = retriever.rank(history)[:recall_k]
    candidate_ids = [row.item.item_id for row in recalled]
    reranked = ranker.rank(history, candidates=candidate_ids)
    return FunnelResult(recalled=recalled, reranked=reranked)
