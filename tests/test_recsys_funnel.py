"""DIN, semantic BERT, InfoNCE, and the recall-then-rank funnel."""

import unittest

from recsys_models.catalog import CATALOG, COLD_START_ID, DEFAULT_HISTORY, item_embedding
from recsys_models.din import TargetAttentionRanker
from recsys_models.losses import infonce_loss
from recsys_models.models import CausalTransformer, TwoTower
from recsys_models.pipeline import recall_then_rank
from recsys_models.semantic import BertDualEncoder
from recsys_models.vectors import cosine, softmax


class TwoTowerExtraTests(unittest.TestCase):
    def test_long_term_mix_moves_toward_profile(self) -> None:
        model = TwoTower()
        session = model.user_embedding(DEFAULT_HISTORY)
        cafe = item_embedding("coffee")
        mixed = model.user_embedding(
            DEFAULT_HISTORY, long_term=cafe, long_term_weight=1.0
        )
        self.assertEqual(mixed, list(cafe))
        self.assertNotEqual(session, mixed)

    def test_rank_subset_stays_inside_candidates(self) -> None:
        ranked = TwoTower().rank(DEFAULT_HISTORY, candidates=("sushi", "burger"))
        self.assertEqual({row.item.item_id for row in ranked}, {"sushi", "burger"})


class DinTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model = TargetAttentionRanker(temperature=0.2)

    def test_user_vector_depends_on_candidate(self) -> None:
        u_sushi, w_sushi = self.model.attend(DEFAULT_HISTORY, "sushi")
        u_latte, w_latte = self.model.attend(DEFAULT_HISTORY, "latte")
        self.assertNotEqual(u_sushi, u_latte)
        ramen_idx = DEFAULT_HISTORY.index("ramen")
        coffee_idx = DEFAULT_HISTORY.index("coffee")
        self.assertGreater(w_sushi[ramen_idx], w_sushi[coffee_idx])
        self.assertGreater(w_latte[coffee_idx], w_latte[ramen_idx])

    def test_matched_candidate_beats_unrelated(self) -> None:
        self.assertGreater(
            self.model.score(DEFAULT_HISTORY, "sushi"),
            self.model.score(DEFAULT_HISTORY, "burger"),
        )
        self.assertGreater(
            self.model.score(DEFAULT_HISTORY, "latte"),
            self.model.score(DEFAULT_HISTORY, "burger"),
        )

    def test_two_tower_user_vector_is_candidate_independent(self) -> None:
        u = TwoTower().user_embedding(DEFAULT_HISTORY)
        self.assertEqual(u, TwoTower().user_embedding(DEFAULT_HISTORY))


class SemanticBertTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model = BertDualEncoder()

    def test_cold_start_udon_closer_than_burger(self) -> None:
        user = self.model.user_embedding(DEFAULT_HISTORY)
        udon = self.model.item_embedding(COLD_START_ID)
        burger = self.model.item_embedding("burger")
        self.assertGreater(cosine(user, udon), cosine(user, burger))

    def test_ramen_text_near_udon(self) -> None:
        ramen = self.model.item_embedding("ramen")
        udon = self.model.item_embedding("udon")
        pizza = self.model.item_embedding("pizza")
        self.assertGreater(cosine(ramen, udon), cosine(ramen, pizza))

    def test_rank_excludes_history(self) -> None:
        ids = {row.item.item_id for row in self.model.rank(DEFAULT_HISTORY)}
        for item_id in DEFAULT_HISTORY:
            self.assertNotIn(item_id, ids)


class InfonceTests(unittest.TestCase):
    def test_easy_negative_beats_hard_negative(self) -> None:
        user = TwoTower().user_embedding(DEFAULT_HISTORY)
        pos = item_embedding("latte")
        easy = item_embedding("burger")
        hard = item_embedding("sushi")
        loss_easy, p_easy = infonce_loss(user, pos, [easy])
        loss_hard, p_hard = infonce_loss(user, pos, [hard])
        self.assertLess(loss_easy, loss_hard)
        self.assertGreater(p_easy[0], p_hard[0])
        self.assertAlmostEqual(p_easy[0] + p_easy[1], 1.0)

    def test_rejects_empty_negatives(self) -> None:
        with self.assertRaises(ValueError):
            infonce_loss((1.0, 0.0, 0.0), (1.0, 0.0, 0.0), [])


class FunnelTests(unittest.TestCase):
    def test_sasrec_rerank_promotes_sushi(self) -> None:
        result = recall_then_rank(
            DEFAULT_HISTORY, ranker=CausalTransformer(), recall_k=4
        )
        self.assertEqual(result.recalled[0].item.item_id, "latte")
        self.assertEqual(result.reranked[0].item.item_id, "sushi")
        recalled_ids = {row.item.item_id for row in result.recalled}
        reranked_ids = {row.item.item_id for row in result.reranked}
        self.assertEqual(recalled_ids, reranked_ids)
        self.assertEqual(len(result.reranked), 4)

    def test_din_rerank_stays_in_recall_set(self) -> None:
        result = recall_then_rank(
            DEFAULT_HISTORY, ranker=TargetAttentionRanker(), recall_k=3
        )
        self.assertEqual(len(result.reranked), 3)
        self.assertEqual(
            {row.item.item_id for row in result.recalled},
            {row.item.item_id for row in result.reranked},
        )

    def test_invalid_recall_k(self) -> None:
        with self.assertRaises(ValueError):
            recall_then_rank(DEFAULT_HISTORY, ranker=CausalTransformer(), recall_k=0)


class SoftmaxTemperatureTests(unittest.TestCase):
    def test_lower_temperature_is_sharper(self) -> None:
        logits = [1.0, 0.0]
        sharp = softmax(logits, temperature=0.2)
        flat = softmax(logits, temperature=2.0)
        self.assertGreater(sharp[0], flat[0])

    def test_invalid_temperature(self) -> None:
        with self.assertRaises(ValueError):
            softmax([1.0], temperature=0.0)


class CatalogTokenTests(unittest.TestCase):
    def test_every_item_has_tokens(self) -> None:
        for item in CATALOG.values():
            self.assertTrue(item.tokens)
            self.assertEqual(len(item.embedding), 3)

    def test_udon_is_cold_start_not_in_default_history(self) -> None:
        self.assertNotIn(COLD_START_ID, DEFAULT_HISTORY)
        self.assertEqual(CATALOG[COLD_START_ID].category, "japanese")


if __name__ == "__main__":
    unittest.main()
