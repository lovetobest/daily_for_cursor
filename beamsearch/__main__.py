"""Run the toy language model with greedy search and beam search.

Usage:
    python -m beamsearch
    python -m beamsearch --beam-width 3
"""

from __future__ import annotations

import argparse

from beamsearch.search import beam_search, format_tokens, greedy_search
from beamsearch.toy_lm import BOS, EOS, next_log_probs


def main() -> None:
    parser = argparse.ArgumentParser(description="Demo beam search on a tiny toy LM.")
    parser.add_argument("--beam-width", type=int, default=3)
    parser.add_argument("--max-len", type=int, default=12)
    args = parser.parse_args()

    greedy = greedy_search(next_log_probs, bos=BOS, eos=EOS, max_len=args.max_len)
    print("Greedy (beam width = 1)")
    _print_hyp(greedy)

    print(f"\nBeam search (beam width = {args.beam_width})")
    for i, hyp in enumerate(
        beam_search(
            next_log_probs,
            beam_width=args.beam_width,
            bos=BOS,
            eos=EOS,
            max_len=args.max_len,
        ),
        start=1,
    ):
        print(f"  #{i}")
        _print_hyp(hyp, indent="    ")


def _print_hyp(hyp, *, indent: str = "  ") -> None:
    text = format_tokens(hyp.tokens, bos=BOS, eos=EOS)
    print(f"{indent}text:  {text}")
    print(f"{indent}log p: {hyp.log_prob:.4f}")
    print(f"{indent}avg:   {hyp.length_normalized_score():.4f}")


if __name__ == "__main__":
    main()
