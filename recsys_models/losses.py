"""InfoNCE / sampled-softmax as used to train two-tower retrievers."""

from __future__ import annotations

import math

from recsys_models.vectors import cosine, softmax


def infonce_loss(
    user: list[float] | tuple[float, ...],
    positive: list[float] | tuple[float, ...],
    negatives: list[list[float] | tuple[float, ...]],
    *,
    temperature: float = 0.07,
) -> tuple[float, list[float]]:
    """Return (loss, softmax probs over [positive, *negatives]).

    Industrial two-towers train with this (or sampled softmax, which is the
    same shape): one positive item, in-batch / random / hard negatives,
    temperature-scaled cosine logits.
    """
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    if not negatives:
        raise ValueError("need at least one negative")
    logits = [cosine(user, positive) / temperature]
    logits.extend(cosine(user, item) / temperature for item in negatives)
    probs = softmax(logits)
    loss = -math.log(max(probs[0], 1e-12))
    return loss, probs
