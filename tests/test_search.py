"""Tests for greedy vs beam search on the toy language model."""

import math
import unittest

from beamsearch.search import Hypothesis, beam_search, format_tokens, greedy_search
from beamsearch.toy_lm import BOS, EOS, next_log_probs


class BeamSearchTests(unittest.TestCase):
    def test_beam_width_one_matches_greedy(self) -> None:
        greedy = greedy_search(next_log_probs, bos=BOS, eos=EOS)
        beam = beam_search(
            next_log_probs,
            beam_width=1,
            bos=BOS,
            eos=EOS,
            length_normalize=False,
        )
        self.assertEqual(greedy.tokens, beam[0].tokens)
        self.assertAlmostEqual(greedy.log_prob, beam[0].log_prob)

    def test_greedy_locks_onto_cat_sat(self) -> None:
        greedy = greedy_search(next_log_probs, bos=BOS, eos=EOS)
        self.assertEqual(
            format_tokens(greedy.tokens),
            "the cat sat on the mat",
        )

    def test_wider_beam_recovers_higher_scoring_dog_ran(self) -> None:
        """Locally, 'cat' beats 'dog'. Globally, 'the dog ran away' scores higher."""
        greedy = greedy_search(next_log_probs, bos=BOS, eos=EOS)
        beams = beam_search(
            next_log_probs,
            beam_width=3,
            bos=BOS,
            eos=EOS,
            length_normalize=False,
        )
        best = beams[0]
        self.assertEqual(format_tokens(best.tokens), "the dog ran away")
        self.assertGreater(best.log_prob, greedy.log_prob)

    def test_all_hypotheses_end_at_eos_or_max_len(self) -> None:
        beams = beam_search(next_log_probs, beam_width=4, bos=BOS, eos=EOS, max_len=12)
        for hyp in beams:
            self.assertTrue(hyp.finished or len(hyp.tokens) == 1 + 12)

    def test_invalid_beam_width(self) -> None:
        with self.assertRaises(ValueError):
            beam_search(next_log_probs, beam_width=0)

    def test_length_normalized_score(self) -> None:
        hyp = Hypothesis(tokens=("a", "b", "c"), log_prob=-3.0)
        self.assertAlmostEqual(hyp.length_normalized_score(), -1.0)

    def test_empty_expansion_keeps_current_beam(self) -> None:
        def no_next(_prefix: tuple[str, ...]) -> dict[str, float]:
            return {}

        result = beam_search(no_next, beam_width=2, bos=BOS, eos=EOS, max_len=3)
        self.assertEqual(result[0].tokens, (BOS,))
        self.assertEqual(result[0].log_prob, 0.0)


class ToyLmTests(unittest.TestCase):
    def test_start_token_is_deterministic(self) -> None:
        scores = next_log_probs((BOS,))
        self.assertEqual(set(scores), {"the"})
        self.assertAlmostEqual(scores["the"], 0.0)

    def test_rows_are_valid_log_probs(self) -> None:
        for prefix in [(BOS,), ("the",), ("cat",), ("dog",), ("on", "the")]:
            scores = next_log_probs(prefix)
            total = sum(math.exp(p) for p in scores.values())
            self.assertAlmostEqual(total, 1.0, places=9)


if __name__ == "__main__":
    unittest.main()
