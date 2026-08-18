"""Beam search and greedy decoding over a next-token log-probability function."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

NextLogProbs = Callable[[tuple[str, ...]], dict[str, float]]


@dataclass(frozen=True)
class Hypothesis:
    """A partial or finished token sequence and its cumulative log-probability."""

    tokens: tuple[str, ...]
    log_prob: float
    finished: bool = False

    def length_normalized_score(self) -> float:
        """Average log-probability per token, so shorter sequences are not favored."""
        n = max(len(self.tokens), 1)
        return self.log_prob / n


def greedy_search(
    next_log_probs: NextLogProbs,
    *,
    bos: str = "<s>",
    eos: str = "</s>",
    max_len: int = 16,
) -> Hypothesis:
    """Always extend the current sequence with the single highest-scoring next token."""
    results = beam_search(
        next_log_probs,
        beam_width=1,
        bos=bos,
        eos=eos,
        max_len=max_len,
        length_normalize=False,
    )
    return results[0]


def beam_search(
    next_log_probs: NextLogProbs,
    *,
    beam_width: int = 3,
    bos: str = "<s>",
    eos: str = "</s>",
    max_len: int = 16,
    length_normalize: bool = True,
) -> list[Hypothesis]:
    """Keep the top ``beam_width`` partial sequences at each decoding step.

    ``next_log_probs(prefix)`` should return a mapping of token -> log-probability
    for tokens that may follow ``prefix``. Missing tokens are treated as
    probability 0 (log-prob -inf) and never enter the beam.

    Returns hypotheses sorted best-first.
    """
    if beam_width < 1:
        raise ValueError("beam_width must be >= 1")
    if max_len < 1:
        raise ValueError("max_len must be >= 1")

    beam: list[Hypothesis] = [Hypothesis(tokens=(bos,), log_prob=0.0)]

    for _ in range(max_len):
        if all(h.finished for h in beam):
            break
        beam = _expand_beam(
            beam,
            next_log_probs=next_log_probs,
            beam_width=beam_width,
            eos=eos,
            length_normalize=length_normalize,
        )

    return _rank(beam, length_normalize=length_normalize)


def _expand_beam(
    beam: Iterable[Hypothesis],
    *,
    next_log_probs: NextLogProbs,
    beam_width: int,
    eos: str,
    length_normalize: bool,
) -> list[Hypothesis]:
    candidates: list[Hypothesis] = []
    for hyp in beam:
        if hyp.finished:
            candidates.append(hyp)
            continue
        for token, log_p in next_log_probs(hyp.tokens).items():
            candidates.append(
                Hypothesis(
                    tokens=hyp.tokens + (token,),
                    log_prob=hyp.log_prob + log_p,
                    finished=token == eos,
                )
            )
    if not candidates:
        return list(beam)
    return _rank(candidates, length_normalize=length_normalize)[:beam_width]


def _rank(hyps: Iterable[Hypothesis], *, length_normalize: bool) -> list[Hypothesis]:
    def key(h: Hypothesis) -> float:
        if length_normalize:
            return h.length_normalized_score()
        return h.log_prob

    return sorted(hyps, key=key, reverse=True)


def format_tokens(tokens: tuple[str, ...], *, bos: str = "<s>", eos: str = "</s>") -> str:
    """Render a hypothesis as a space-separated string without BOS/EOS markers."""
    visible = [t for t in tokens if t not in {bos, eos}]
    return " ".join(visible)
