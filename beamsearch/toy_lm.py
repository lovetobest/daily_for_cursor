"""A tiny hand-written language model used to demonstrate beam search.

Transition scores are log-probabilities of a next token given the previous token.
This is enough to show why beam width > 1 can recover from a locally greedy choice.
"""

from __future__ import annotations

from math import log

BOS = "<s>"
EOS = "</s>"

# P(next | previous). Rows sum to 1.0 before taking log.
_TRANSITIONS: dict[str, dict[str, float]] = {
    BOS: {"the": 1.0},
    "the": {"cat": 0.55, "dog": 0.45},
    # Greedy prefers "cat" here, but "dog ran ..." is a better full sequence.
    "cat": {"sat": 0.70, "slept": 0.30},
    "dog": {"ran": 0.90, "sat": 0.10},
    "sat": {"on": 0.80, EOS: 0.20},
    "slept": {EOS: 1.0},
    "ran": {"away": 0.95, EOS: 0.05},
    "on": {"the": 0.85, EOS: 0.15},
    "away": {EOS: 1.0},
}

# Second "the" (after "on") should go to a noun, not restart the sentence.
_AFTER_ON_THE: dict[str, float] = {"mat": 0.75, "floor": 0.25}
_NOUN_END: dict[str, float] = {EOS: 1.0}


def next_log_probs(prefix: tuple[str, ...]) -> dict[str, float]:
    """Return log P(token | last token), with a special case for "... on the ..."."""
    if not prefix:
        row = _TRANSITIONS[BOS]
    elif len(prefix) >= 2 and prefix[-2] == "on" and prefix[-1] == "the":
        row = _AFTER_ON_THE
    elif prefix[-1] in {"mat", "floor"}:
        row = _NOUN_END
    else:
        row = _TRANSITIONS.get(prefix[-1], {EOS: 1.0})
    return {token: log(prob) for token, prob in row.items()}
