"""Tests for two-tower vs Transformer vs BERT on the toy catalog."""

import math
import unittest

from recsys_models.catalog import CATALOG, DEFAULT_HISTORY, MASK_ID, require_history
from recsys_models.models import Bert4Rec, CausalTransformer, TwoTower, self_attention
from recsys_models.vectors import dot, l2_normalize, mean_pool, softmax


def _ids(ranked, k: int = 3) -> list[str]:
    return [row.item.item_id for row in ranked[:k]]


class VectorTests(unittest.TestCase):
    def test_softmax_sums_to_one(self) -> None:
        weights = softmax([1.0, 2.0, 3.0])
        self.assertAlmostEqual(sum(weights), 1.0)
        self.assertGreater(weights[2], weights[0])

    def test_mean_pool(self) -> None:
        self.assertEqual(mean_pool([(1.0, 0.0), (0.0, 1.0)]), [0.5, 0.5])

    def test_l2_normalize_unit_length(self) -> None:
        vec = l2_normalize([3.0, 4.0])
        self.assertAlmostEqual(math.sqrt(vec[0] ** 2 + vec[1] ** 2), 1.0)


class CatalogTests(unittest.TestCase):
    def test_default_history_is_cafe_then_ramen(self) -> None:
        self.assertEqual(DEFAULT_HISTORY, ("coffee", "cake", "ramen"))
        self.assertEqual(CATALOG["coffee"].category, "cafe")
        self.assertEqual(CATALOG["ramen"].category, "japanese")

    def test_empty_history_rejected(self) -> None:
        with self.assertRaises(ValueError):
            require_history(())

    def test_unknown_item_rejected(self) -> None:
        with self.assertRaises(KeyError):
            require_history(("nope",))


class TwoTowerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model = TwoTower()

    def test_user_embedding_is_mean_of_history(self) -> None:
        got = self.model.user_embedding(DEFAULT_HISTORY)
        expected = mean_pool(
            [CATALOG[item_id].embedding for item_id in DEFAULT_HISTORY]
        )
        for a, b in zip(got, expected):
            self.assertAlmostEqual(a, b)

    def test_order_does_not_matter(self) -> None:
        a = self.model.user_embedding(("coffee", "ramen"))
        b = self.model.user_embedding(("ramen", "coffee"))
        self.assertEqual(a, b)

    def test_majority_cafe_history_ranks_latte_first(self) -> None:
        """Two cafe clicks outvote one ramen click when order is ignored."""
        ranked = self.model.rank(DEFAULT_HISTORY)
        self.assertEqual(ranked[0].item.item_id, "latte")
        self.assertEqual(ranked[0].item.category, "cafe")
        self.assertNotIn("coffee", _ids(ranked, k=len(ranked)))

    def test_scores_are_sorted_descending(self) -> None:
        scores = [row.score for row in self.model.rank(DEFAULT_HISTORY)]
        self.assertEqual(scores, sorted(scores, reverse=True))


class TransformerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model = CausalTransformer(recency_slope=2.0)

    def test_last_click_ramen_ranks_sushi_first(self) -> None:
        ranked = self.model.rank(DEFAULT_HISTORY)
        self.assertEqual(ranked[0].item.item_id, "sushi")
        self.assertEqual(ranked[0].item.category, "japanese")

    def test_sushi_beats_two_tower_cafe_winner(self) -> None:
        transformer_top = self.model.rank(DEFAULT_HISTORY)[0].item.item_id
        two_tower_top = TwoTower().rank(DEFAULT_HISTORY)[0].item.item_id
        self.assertEqual(transformer_top, "sushi")
        self.assertEqual(two_tower_top, "latte")

    def test_causal_mask_blocks_future_keys(self) -> None:
        hidden, weights = self.model.encode(("coffee", "cake", "ramen"))
        self.assertEqual(len(hidden), 3)
        self.assertEqual(weights[0][1], 0.0)
        self.assertEqual(weights[0][2], 0.0)
        self.assertEqual(weights[1][2], 0.0)
        self.assertAlmostEqual(sum(weights[0]), 1.0)
        self.assertAlmostEqual(sum(weights[-1]), 1.0)

    def test_recency_puts_more_mass_on_last_item(self) -> None:
        _, with_recency = self.model.encode(DEFAULT_HISTORY, recency_slope=2.0)
        _, without = self.model.encode(DEFAULT_HISTORY, recency_slope=0.0)
        self.assertGreater(with_recency[-1][-1], without[-1][-1])


class BertTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model = Bert4Rec()

    def test_next_item_mask_follows_session_bag_not_last_click(self) -> None:
        """End-[MASK] without recency stays cafe-like, unlike SASRec."""
        bert_top = self.model.rank(DEFAULT_HISTORY)[0]
        sasrec_top = CausalTransformer().rank(DEFAULT_HISTORY)[0]
        self.assertEqual(bert_top.item.category, "cafe")
        self.assertEqual(sasrec_top.item.item_id, "sushi")

    def test_cloze_causal_uses_left_cafe_context(self) -> None:
        ranked = self.model.cloze(("coffee",), ("ramen",), causal=True)
        self.assertEqual(ranked[0].item.item_id, "latte")

    def test_cloze_bert_uses_right_ramen_context(self) -> None:
        ranked = self.model.cloze(("coffee",), ("ramen",), causal=False)
        self.assertEqual(ranked[0].item.item_id, "sushi")

    def test_cloze_contrast_is_the_bert_point(self) -> None:
        left_only = self.model.cloze(("coffee",), ("ramen",), causal=True)[0]
        both_sides = self.model.cloze(("coffee",), ("ramen",), causal=False)[0]
        self.assertEqual(left_only.item.item_id, "latte")
        self.assertEqual(both_sides.item.item_id, "sushi")

    def test_bidirectional_mask_attends_right(self) -> None:
        tokens = ["coffee", MASK_ID, "ramen"]
        embeddings = [
            list(CATALOG["coffee"].embedding),
            [0.4, 0.4, 0.0],
            list(CATALOG["ramen"].embedding),
        ]
        self.assertEqual(tokens, ["coffee", MASK_ID, "ramen"])
        _, causal_w = self_attention(embeddings, causal=True)
        _, bert_w = self_attention(embeddings, causal=False)
        self.assertEqual(causal_w[1][2], 0.0)
        self.assertGreater(bert_w[1][2], 0.0)
        self.assertAlmostEqual(sum(bert_w[1]), 1.0)

    def test_next_item_excludes_history(self) -> None:
        ranked = self.model.rank(DEFAULT_HISTORY)
        for item_id in DEFAULT_HISTORY:
            self.assertNotIn(item_id, _ids(ranked, k=len(ranked)))


class AttentionTests(unittest.TestCase):
    def test_empty_sequence_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self_attention([], causal=True)

    def test_dot_rejects_mismatched_dims(self) -> None:
        with self.assertRaises(ValueError):
            dot([1.0, 2.0], [1.0])


if __name__ == "__main__":
    unittest.main()
