"""Tests for the business layer.

A hand-built catalogue is used rather than the generator, because these are guarantees the
merchandising team would be told about: nothing out of stock, never more than N of one
category, and the margin nudge bounded so the page stays a recommendation.
"""

from __future__ import annotations

import pandas as pd
import pytest

from recsys.rerank import RerankConfig, list_profile, rerank

CATALOGUE = pd.DataFrame(
    {
        "item_id": [f"I{index}" for index in range(8)],
        "category": ["shoes", "shoes", "shoes", "shoes", "bikes", "bikes", "food", "food"],
        "brand": ["A", "A", "B", "B", "A", "C", "C", "B"],
        "price": [100.0, 90.0, 80.0, 70.0, 60.0, 50.0, 40.0, 30.0],
        "margin": [10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 30.0],
        "in_stock": [True, False, True, True, True, True, True, True],
    }
)
# descending scores, so the model's preferred order is I0, I1, I2, ...
CANDIDATES = [(f"I{index}", 1.0 - index * 0.1) for index in range(8)]


def test_out_of_stock_items_are_removed_and_counted():
    result = rerank(CANDIDATES, CATALOGUE, k=8)
    assert "I1" not in result.items
    assert result.dropped_out_of_stock == 1


def test_out_of_stock_can_be_allowed_explicitly():
    result = rerank(
        CANDIDATES, CATALOGUE, k=8, config=RerankConfig(require_in_stock=False, max_per_category=8)
    )
    assert "I1" in result.items
    assert result.dropped_out_of_stock == 0


def test_the_category_cap_is_never_exceeded():
    result = rerank(
        CANDIDATES,
        CATALOGUE,
        k=8,
        config=RerankConfig(max_per_category=2, diversity_lambda=0.0, margin_weight=0.0),
    )
    counts = CATALOGUE.set_index("item_id").loc[result.items, "category"].value_counts()
    assert counts.max() <= 2


def test_relevance_order_survives_when_nothing_is_traded_away():
    """With diversity and margin switched off, the model's ranking is preserved."""
    result = rerank(
        CANDIDATES,
        CATALOGUE,
        k=4,
        config=RerankConfig(max_per_category=8, diversity_lambda=0.0, margin_weight=0.0),
    )
    assert result.items == ["I0", "I2", "I3", "I4"]  # I1 is out of stock


def test_diversity_spreads_the_list_across_categories():
    plain = rerank(
        CANDIDATES,
        CATALOGUE,
        k=3,
        config=RerankConfig(max_per_category=8, diversity_lambda=0.0, margin_weight=0.0),
    )
    diverse = rerank(
        CANDIDATES,
        CATALOGUE,
        k=3,
        config=RerankConfig(max_per_category=8, diversity_lambda=0.9, margin_weight=0.0),
    )
    categories = CATALOGUE.set_index("item_id")["category"]
    assert categories.loc[diverse.items].nunique() > categories.loc[plain.items].nunique()


def test_the_margin_nudge_breaks_ties_towards_profit():
    """Equal relevance, unequal margin: the profitable item wins, and only then."""
    flat = [(f"I{index}", 1.0) for index in range(8)]
    result = rerank(
        flat,
        CATALOGUE,
        k=1,
        config=RerankConfig(max_per_category=8, diversity_lambda=0.0, margin_weight=0.5),
    )
    assert result.items == ["I7"]  # by far the highest margin


def test_margin_cannot_override_a_clear_relevance_gap():
    result = rerank(
        CANDIDATES,
        CATALOGUE,
        k=1,
        config=RerankConfig(max_per_category=8, diversity_lambda=0.0, margin_weight=0.10),
    )
    assert result.items == ["I0"]


def test_unknown_items_are_dropped_and_reported():
    result = rerank([("ghost", 5.0), *CANDIDATES], CATALOGUE, k=3)
    assert "ghost" not in result.items
    assert result.dropped_unknown == 1


def test_a_short_candidate_list_returns_what_exists():
    result = rerank(CANDIDATES[:2], CATALOGUE, k=10)
    assert len(result) == 1  # I1 is out of stock
    assert result.items == ["I0"]


def test_everything_unavailable_yields_an_empty_list():
    result = rerank([("I1", 1.0)], CATALOGUE, k=5)
    assert result.items == []
    assert result.dropped_out_of_stock == 1


def test_reranking_is_deterministic():
    first = rerank(CANDIDATES, CATALOGUE, k=5)
    second = rerank(CANDIDATES, CATALOGUE, k=5)
    assert first.items == second.items


def test_diagnostics_describe_the_final_list():
    result = rerank(CANDIDATES, CATALOGUE, k=4)
    assert result.diagnostics["distinct_categories"] >= 2
    assert 0.0 < result.diagnostics["top_category_share"] <= 1.0
    assert result.diagnostics["mean_margin"] > 0.0


def test_configuration_is_validated():
    with pytest.raises(ValueError, match="max_per_category"):
        RerankConfig(max_per_category=0).validate()
    with pytest.raises(ValueError, match="diversity_lambda"):
        RerankConfig(diversity_lambda=1.5).validate()
    with pytest.raises(ValueError, match="margin_weight"):
        RerankConfig(margin_weight=2.0).validate()


def test_a_catalogue_without_the_required_columns_is_rejected():
    with pytest.raises(ValueError, match="missing columns"):
        rerank(CANDIDATES, CATALOGUE.drop(columns=["in_stock"]), k=3)


def test_list_profile_summarises_a_list():
    profile = list_profile(["I0", "I4", "I6"], CATALOGUE)
    assert profile["items"] == 3
    assert profile["distinct_categories"] == 3.0
    assert profile["out_of_stock"] == 0
    assert profile["mean_margin"] == pytest.approx((10.0 + 6.0 + 4.0) / 3, rel=1e-6)


def test_list_profile_handles_nothing_to_describe():
    assert list_profile([], CATALOGUE)["items"] == 0
    assert list_profile(["ghost"], CATALOGUE)["items"] == 0
