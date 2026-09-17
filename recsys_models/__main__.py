"""Compare two-tower, Transformer, and BERT on a toy food-delivery session.

Usage:
    python3 -m recsys_models
    python3 -m recsys_models --history coffee,cake,ramen
"""

from __future__ import annotations

import argparse

from recsys_models.catalog import CATALOG, DEFAULT_HISTORY
from recsys_models.models import Bert4Rec, CausalTransformer, ScoredItem, TwoTower


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Demo two-tower / Transformer / BERT recommenders.",
    )
    parser.add_argument(
        "--history",
        default=",".join(DEFAULT_HISTORY),
        help="comma-separated item ids (default: coffee,cake,ramen)",
    )
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()
    history = _parse_history(args.history)
    k = args.top_k

    print("History:", " → ".join(history))
    print("  (cafe, cafe, then ramen — long-term interest vs last click)")
    print()

    two_tower = TwoTower()
    transformer = CausalTransformer()
    bert = Bert4Rec()

    _section(
        "Two-tower (双塔 / DSSM) — recall",
        "User = mean of history. Items scored with cosine. Order is ignored, "
        "so two cafe clicks outvote one ramen click. Item tower is independent, "
        "which is why this is the industrial ANN retrieval model.",
        two_tower.rank(history),
        k,
    )
    _section(
        "Transformer (SASRec) — sequential ranking",
        "Causal self-attention + recency bias. The last hidden state follows "
        "ramen, so Japanese food rises even though cafe is the majority.",
        transformer.rank(history),
        k,
    )
    _section(
        "BERT (BERT4Rec) — next item via [MASK] at the end",
        "Bidirectional attention over the whole session. Without strong "
        "recency, this behaves like a session bag — closer to two-tower "
        "than to last-click Transformer.",
        bert.rank(history),
        k,
    )

    if history[0] in CATALOG and history[-1] in CATALOG and history[0] != history[-1]:
        left, right = (history[0],), (history[-1],)
        print("=" * 72)
        print(f"Cloze: {left[0]} → [MASK] → {right[0]}")
        print("  BERT can read both sides. A causal Transformer cannot look right.")
        print()
        _section(
            "Causal cloze (left context only)",
            "Mask position attends only to the left item, so cafe wins.",
            bert.cloze(left, right, causal=True),
            k,
        )
        _section(
            "BERT cloze (left AND right)",
            "Right-side ramen pulls the hole toward Japanese food (sushi).",
            bert.cloze(left, right, causal=False),
            k,
        )


def _parse_history(raw: str) -> tuple[str, ...]:
    tokens = tuple(part.strip() for part in raw.split(",") if part.strip())
    if not tokens:
        raise SystemExit("history must contain at least one item id")
    unknown = [item_id for item_id in tokens if item_id not in CATALOG]
    if unknown:
        known = ", ".join(CATALOG)
        raise SystemExit(f"unknown item id(s) {unknown}. catalog: {known}")
    return tokens


def _section(title: str, blurb: str, ranked: list[ScoredItem], k: int) -> None:
    print(title)
    print(f"  {blurb}")
    for i, row in enumerate(ranked[:k], start=1):
        print(
            f"  {i}. {row.item.item_id:<8}  {row.score:7.3f}  "
            f"({row.item.category})"
        )
    print()


if __name__ == "__main__":
    main()
