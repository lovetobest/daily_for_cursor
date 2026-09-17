"""Tiny stdlib vector helpers used by the toy recommenders."""

from __future__ import annotations

import math

Vector = list[float]


def dot(a: list[float] | tuple[float, ...], b: list[float] | tuple[float, ...]) -> float:
    if len(a) != len(b):
        raise ValueError("vectors must have the same dimension")
    return sum(x * y for x, y in zip(a, b))


def l2_norm(a: list[float] | tuple[float, ...]) -> float:
    return math.sqrt(sum(x * x for x in a))


def l2_normalize(a: list[float] | tuple[float, ...]) -> Vector:
    n = l2_norm(a)
    if n == 0.0:
        return [0.0] * len(a)
    return [x / n for x in a]


def cosine(a: list[float] | tuple[float, ...], b: list[float] | tuple[float, ...]) -> float:
    return dot(l2_normalize(a), l2_normalize(b))


def mean_pool(vectors: list[list[float] | tuple[float, ...]]) -> Vector:
    if not vectors:
        raise ValueError("cannot mean-pool an empty list")
    dim = len(vectors[0])
    acc = [0.0] * dim
    for vec in vectors:
        if len(vec) != dim:
            raise ValueError("all vectors must have the same dimension")
        for i, x in enumerate(vec):
            acc[i] += x
    n = float(len(vectors))
    return [x / n for x in acc]


def weighted_sum(
    weights: list[float],
    vectors: list[list[float] | tuple[float, ...]],
) -> Vector:
    if len(weights) != len(vectors):
        raise ValueError("weights and vectors must be the same length")
    if not vectors:
        raise ValueError("cannot pool an empty list")
    dim = len(vectors[0])
    acc = [0.0] * dim
    for weight, vec in zip(weights, vectors):
        for i, x in enumerate(vec):
            acc[i] += weight * x
    return acc


def softmax(logits: list[float], *, temperature: float = 1.0) -> list[float]:
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    scaled = [x / temperature for x in logits]
    finite = [x for x in scaled if x != float("-inf")]
    if not finite:
        raise ValueError("softmax of all -inf")
    peak = max(finite)
    exps = [0.0 if x == float("-inf") else math.exp(x - peak) for x in scaled]
    total = sum(exps)
    return [x / total for x in exps]


def format_vector(vec: list[float] | tuple[float, ...], *, digits: int = 2) -> str:
    body = ", ".join(f"{x:.{digits}f}" for x in vec)
    return f"[{body}]"
