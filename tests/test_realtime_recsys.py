"""Tests for snapshot vs realtime recommendation freshness."""

import math
import unittest

from realtime_recsys.catalog import (
    FLUSH_TS,
    QUERY_TS,
    USER_ID,
    Catalog,
    Event,
    Item,
    preference_shift_events,
    toy_catalog,
)
from realtime_recsys.engine import FeatureStore, recommend, top_category


def _tiny_catalog() -> Catalog:
    return Catalog(
        [
            Item("c1", "Coffee One", "coffee", 0.9),
            Item("h1", "Hotpot One", "hotpot", 0.8),
        ]
    )


class FreshnessTests(unittest.TestCase):
    def test_realtime_follows_recent_hotpot_clicks(self) -> None:
        store = FeatureStore(toy_catalog())
        store.ingest_many(preference_shift_events())
        store.flush(FLUSH_TS)
        profile = store.profile(USER_ID, QUERY_TS, mode="realtime")
        ranking = recommend(store.catalog, profile, k=3)
        self.assertEqual(top_category(ranking), "hotpot")
        self.assertTrue(all(row.item.category == "hotpot" for row in ranking))

    def test_snapshot_without_flush_still_ranks_coffee(self) -> None:
        """Locally the user is in a hotpot session; the frozen profile misses it."""
        store = FeatureStore(toy_catalog())
        store.ingest_many(preference_shift_events())
        store.flush(FLUSH_TS)
        snapshot = store.profile(USER_ID, QUERY_TS, mode="snapshot")
        realtime = store.profile(USER_ID, QUERY_TS, mode="realtime")
        snap_rank = recommend(store.catalog, snapshot, k=3)
        real_rank = recommend(store.catalog, realtime, k=3)
        self.assertEqual(top_category(snap_rank), "coffee")
        self.assertTrue(all(row.item.category == "coffee" for row in snap_rank))
        self.assertEqual(top_category(real_rank), "hotpot")
        self.assertGreater(real_rank[0].score, snap_rank[0].score)

    def test_flush_after_shift_aligns_snapshot_with_realtime(self) -> None:
        store = FeatureStore(toy_catalog())
        store.ingest_many(preference_shift_events())
        store.flush(QUERY_TS)
        snapshot = store.profile(USER_ID, QUERY_TS, mode="snapshot")
        realtime = store.profile(USER_ID, QUERY_TS, mode="realtime")
        self.assertEqual(top_category(recommend(store.catalog, snapshot)), "hotpot")
        self.assertEqual(top_category(recommend(store.catalog, realtime)), "hotpot")
        self.assertAlmostEqual(snapshot.freshness_lag(QUERY_TS), 0.0)

    def test_realtime_freshness_lag_is_zero(self) -> None:
        store = FeatureStore(toy_catalog())
        store.ingest_many(preference_shift_events())
        profile = store.profile(USER_ID, QUERY_TS, mode="realtime")
        self.assertEqual(profile.as_of, QUERY_TS)
        self.assertEqual(profile.freshness_lag(QUERY_TS), 0.0)

    def test_snapshot_lag_grows_until_next_flush(self) -> None:
        store = FeatureStore(toy_catalog())
        store.ingest_many(preference_shift_events())
        store.flush(FLUSH_TS)
        profile = store.profile(USER_ID, QUERY_TS, mode="snapshot")
        self.assertEqual(profile.as_of, FLUSH_TS)
        self.assertAlmostEqual(profile.freshness_lag(QUERY_TS), QUERY_TS - FLUSH_TS)

    def test_unflushed_snapshot_has_infinite_lag(self) -> None:
        store = FeatureStore(_tiny_catalog())
        store.ingest(Event(0.0, "u", "c1"))
        profile = store.profile("u", 10.0, mode="snapshot")
        self.assertIsNone(profile.as_of)
        self.assertEqual(profile.freshness_lag(10.0), math.inf)
        ranking = recommend(store.catalog, profile, k=1)
        self.assertEqual(ranking[0].item.item_id, "c1")

    def test_half_life_halves_weight_after_one_half_life(self) -> None:
        store = FeatureStore(_tiny_catalog(), half_life=10.0)
        store.ingest(Event(0.0, "u", "h1", kind="click"))
        profile = store.profile("u", 10.0, mode="realtime")
        self.assertAlmostEqual(profile.interest["hotpot"], 0.5)

    def test_order_weighs_more_than_click(self) -> None:
        store = FeatureStore(_tiny_catalog(), half_life=10_000.0)
        store.ingest(Event(0.0, "u", "c1", kind="click"))
        store.ingest(Event(1.0, "u", "h1", kind="order"))
        profile = store.profile("u", 2.0, mode="realtime")
        ranking = recommend(store.catalog, profile, k=1)
        self.assertEqual(ranking[0].item.item_id, "h1")
        self.assertGreater(profile.interest["hotpot"], profile.interest["coffee"])

    def test_events_after_as_of_are_ignored(self) -> None:
        store = FeatureStore(_tiny_catalog(), half_life=10_000.0)
        store.ingest(Event(0.0, "u", "c1"))
        store.ingest(Event(50.0, "u", "h1"))
        store.flush(10.0)
        snapshot = store.profile("u", 60.0, mode="snapshot")
        self.assertNotIn("hotpot", snapshot.interest)
        self.assertAlmostEqual(snapshot.interest["coffee"], 0.5 ** (10.0 / 10_000.0))

    def test_empty_user_ranks_by_quality(self) -> None:
        store = FeatureStore(toy_catalog())
        profile = store.profile("nobody", QUERY_TS, mode="realtime")
        ranking = recommend(store.catalog, profile, k=1)
        self.assertEqual(ranking[0].item.item_id, "manner")
        self.assertEqual(profile.event_count, 0)

    def test_invalid_mode(self) -> None:
        store = FeatureStore(_tiny_catalog())
        with self.assertRaises(ValueError):
            store.profile("u", 0.0, mode="batch")

    def test_invalid_k(self) -> None:
        store = FeatureStore(_tiny_catalog())
        profile = store.profile("u", 0.0, mode="realtime")
        with self.assertRaises(ValueError):
            recommend(store.catalog, profile, k=0)

    def test_invalid_half_life(self) -> None:
        with self.assertRaises(ValueError):
            FeatureStore(_tiny_catalog(), half_life=0)

    def test_unknown_item_is_rejected(self) -> None:
        store = FeatureStore(_tiny_catalog())
        with self.assertRaises(ValueError):
            store.ingest(Event(0.0, "u", "missing"))


if __name__ == "__main__":
    unittest.main()
