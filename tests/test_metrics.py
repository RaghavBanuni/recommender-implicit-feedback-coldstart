"""Tests for the ranking metrics, every expected value computed by hand.

These functions decide which model wins, so none of them is checked against another
implementation of itself.  The NDCG case is worked out fully in the test body.
"""

from __future__ import annotations

import numpy as np
import pytest

from recsys.metrics import (
    average_precision_at_k,
    catalogue_coverage,
    cold_item_share,
    exposure_gini,
    hit_rate_at_k,
    intra_list_diversity,
    ndcg_at_k,
    novelty,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)

RANKED = ["a", "b", "c", "d"]
RELEVANT = {"b", "d", "z"}


def test_precision_counts_relevant_slots():
    assert precision_at_k(RANKED, RELEVANT, k=4) == pytest.approx(0.5)


def test_recall_divides_by_the_relevant_set():
    assert recall_at_k(RANKED, RELEVANT, k=4) == pytest.approx(2 / 3)


def test_recall_is_capped_by_basket_size():
    """A user with twenty relevant items cannot exceed 0.5 at k=10."""
    relevant = {f"item{index}" for index in range(20)}
    perfect = [f"item{index}" for index in range(10)]
    assert recall_at_k(perfect, relevant, k=10) == pytest.approx(0.5)


def test_hit_rate_is_binary():
    assert hit_rate_at_k(RANKED, RELEVANT, k=4) == 1.0
    assert hit_rate_at_k(["x", "y"], RELEVANT, k=2) == 0.0


def test_reciprocal_rank_uses_the_first_hit():
    assert reciprocal_rank(RANKED, RELEVANT, k=4) == pytest.approx(0.5)
    assert reciprocal_rank(["b"], RELEVANT, k=1) == pytest.approx(1.0)


def test_average_precision_normalises_by_achievable_hits():
    # hits at positions 2 and 4: (1/2 + 2/4) / min(3, 4) = 1.0 / 3
    assert average_precision_at_k(RANKED, RELEVANT, k=4) == pytest.approx(1 / 3)


def test_ndcg_matches_the_hand_computation():
    # gains [0, 1, 0, 1]; discounts 1/log2(2..5)
    dcg = 1 / np.log2(3) + 1 / np.log2(5)
    ideal = 1 / np.log2(2) + 1 / np.log2(3) + 1 / np.log2(4)  # min(k, 3) hits
    assert ndcg_at_k(RANKED, RELEVANT, k=4) == pytest.approx(dcg / ideal, rel=1e-9)


def test_ndcg_is_one_for_a_perfect_ranking():
    assert ndcg_at_k(["b", "d"], {"b", "d"}, k=10) == pytest.approx(1.0)


def test_ndcg_rewards_putting_the_hit_higher():
    early = ndcg_at_k(["b", "x", "y"], {"b"}, k=3)
    late = ndcg_at_k(["x", "y", "b"], {"b"}, k=3)
    assert early > late
    assert early == pytest.approx(1.0)


def test_metrics_reject_a_duplicated_list():
    with pytest.raises(ValueError, match="repeat an item"):
        precision_at_k(["a", "a"], RELEVANT, k=2)


def test_metrics_reject_an_empty_relevant_set():
    for function in (recall_at_k, average_precision_at_k, ndcg_at_k):
        with pytest.raises(ValueError, match="undefined"):
            function(RANKED, set(), k=4)


def test_k_must_be_positive():
    with pytest.raises(ValueError, match="k must be positive"):
        precision_at_k(RANKED, RELEVANT, k=0)


def test_coverage_counts_distinct_items_shown():
    recommendations = {"u1": ["a", "b"], "u2": ["b", "c"]}
    assert catalogue_coverage(recommendations, catalogue_size=10) == pytest.approx(0.3)


def test_novelty_rewards_the_long_tail():
    counts = {"a": 99.0, "b": 0.0}
    popular = novelty({"u": ["a"]}, counts)
    obscure = novelty({"u": ["b"]}, counts)
    assert obscure > popular
    assert popular == pytest.approx(-np.log2(100 / 101), rel=1e-9)
    assert obscure == pytest.approx(-np.log2(1 / 101), rel=1e-9)


def test_exposure_gini_is_zero_when_every_item_gets_a_turn():
    recommendations = {"u1": ["a", "b"], "u2": ["c", "d"]}
    assert exposure_gini(recommendations, catalogue_size=4) == pytest.approx(0.0, abs=1e-9)


def test_exposure_gini_is_high_when_one_item_takes_everything():
    recommendations = {"u1": ["a"], "u2": ["a"], "u3": ["a"], "u4": ["a"]}
    assert exposure_gini(recommendations, catalogue_size=4) == pytest.approx(0.75, abs=1e-9)


def test_intra_list_diversity_is_the_share_of_distinct_categories():
    categories = {"a": "shoes", "b": "shoes", "c": "bikes"}
    assert intra_list_diversity({"u": ["a", "b", "c"]}, categories) == pytest.approx(2 / 3)


def test_cold_item_share_counts_slots_not_users():
    assert cold_item_share({"u": ["a", "b"]}, cold_items={"b"}) == pytest.approx(0.5)
    assert cold_item_share({"u": ["a"]}, cold_items=set()) == pytest.approx(0.0)
