"""Walk through two-tower, Transformer, BERT, DIN on a toy session.

Usage:
    python3 -m recsys_models
    python3 -m recsys_models --history coffee,cake,ramen
    python3 -m recsys_models --short
"""

from __future__ import annotations

import argparse

from recsys_models.catalog import CATALOG, DEFAULT_HISTORY
from recsys_models.lecture import run as run_lecture
from recsys_models.models import Bert4Rec, CausalTransformer, ScoredItem, TwoTower


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Demo two-tower / Transformer / BERT / DIN recommenders.",
    )
    parser.add_argument(
        "--history",
        default=",".join(DEFAULT_HISTORY),
        help="comma-separated item ids (default: coffee,cake,ramen)",
    )
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument(
        "--short",
        action="store_true",
        help="only print the three next-item lists (old demo)",
    )
    args = parser.parse_args()
    history = _parse_history(args.history)

    if args.short:
        _short(history, args.top_k)
        return
    run_lecture(history, top_k=args.top_k)


def _short(history: tuple[str, ...], k: int) -> None:
    print("History:", " → ".join(history))
    print()
    _section("Two-tower (双塔)", TwoTower().rank(history), k)
    _section("Transformer (SASRec user tower)", CausalTransformer().rank(history), k)
    _section("BERT4Rec ([MASK] at end)", Bert4Rec().rank(history), k)


def _parse_history(raw: str) -> tuple[str, ...]:
    tokens = tuple(part.strip() for part in raw.split(",") if part.strip())
    if not tokens:
        raise SystemExit("history must contain at least one item id")
    unknown = [item_id for item_id in tokens if item_id not in CATALOG]
    if unknown:
        known = ", ".join(CATALOG)
        raise SystemExit(f"unknown item id(s) {unknown}. catalog: {known}")
    return tokens


def _section(title: str, ranked: list[ScoredItem], k: int) -> None:
    print(title)
    for i, row in enumerate(ranked[:k], start=1):
        print(
            f"  {i}. {row.item.item_id:<8}  {row.score:7.3f}  "
            f"({row.item.category})"
        )
    print()


if __name__ == "__main__":
    main()
