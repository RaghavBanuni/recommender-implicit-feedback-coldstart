"""Tests for the models themselves.

The cold-start tests are the point of this file.  They pin down a fact that is easy to
talk around: matrix factorization assigns *no* factors to an item nobody has touched, so
it can never recommend new stock, and no amount of tuning changes that.  The content model
can, and the hybrid's reserved slot is what actually puts new items on the page.
"""

from __future__ import annotations

import numpy as np
import pytest

from recsys.models import (
    ContentBasedRecommender,
    HybridRecommender,
    ImplicitALS,
    ItemItemCF,
    PopularityRecommender,
    build_model,
    reciprocal_ranks,
)


@pytest.fixture(scope="module")
def cold_ids(split):
    return {
        item_id
        for item_id, is_cold in zip(split.train.items.ids, split.train.cold_items)
        if bool(is_cold)
    }


@pytest.fixture(scope="module")
def heavy_users(users_by_history):
    users = users_by_history["heavy"][:5]
    assert users, "the fixture needs users with substantial history"
    return users


# ----------------------------------------------------------------- popularity
def test_popularity_is_identical_for_every_user(fitted):
    model = fitted["popularity"]
    assert np.array_equal(model.scores(0), model.scores(7))


def test_popularity_ranks_the_recent_bestsellers_first(fitted, split):
    model = fitted["popularity"]
    counts = split.train.recent_popularity(model.window_days)
    best = int(np.argmax(counts))
    assert int(np.argmax(model.scores(0))) == best


# --------------------------------------------------------------- shared logic
def test_recommendations_are_ranked_unique_and_capped(fitted, heavy_users):
    for name, model in fitted.items():
        items = model.recommend(heavy_users[0], k=10)
        assert len(items) <= 10, name
        assert len({item for item, _score in items}) == len(items), name
        if name != "hybrid":  # the hybrid appends its discovery slot out of rank order
            scores = [score for _item, score in items]
            assert scores == sorted(scores, reverse=True), name


def test_recommendations_never_repeat_the_user_history(fitted, split, heavy_users):
    for name, model in fitted.items():
        seen = set(split.train.items.decode(split.train.seen(heavy_users[0])))
        items = {item for item, _score in model.recommend(heavy_users[0], k=10)}
        assert not (items & seen), name


def test_an_allowed_mask_restricts_the_output(fitted, split, heavy_users):
    mask = np.zeros(len(split.train.items), dtype=bool)
    mask[[3, 8, 21, 55]] = True
    permitted = set(split.train.items.decode(np.flatnonzero(mask)))
    items = {item for item, _score in fitted["itemitem"].recommend(heavy_users[0], k=10, allowed=mask)}
    assert items <= permitted
    assert items


def test_an_unfitted_model_refuses_to_recommend():
    with pytest.raises(RuntimeError, match="has not been fitted"):
        ItemItemCF().recommend("U0000")


def test_k_must_be_positive(fitted, heavy_users):
    with pytest.raises(ValueError, match="k must be positive"):
        fitted["als"].recommend(heavy_users[0], k=0)


# ------------------------------------------------------------------------ ALS
def test_als_leaves_cold_items_without_factors(fitted, split):
    """The structural cold-start fact, asserted rather than described."""
    factors = fitted["als"].item_factors
    assert np.allclose(factors[split.train.cold_items], 0.0)
    assert not np.allclose(factors[~split.train.cold_items], 0.0)


def test_als_leaves_users_without_history_at_zero(fitted, split, users_by_history):
    assert users_by_history["none"], "the fixture needs a user with no training history"
    index = split.train.users.index_of(users_by_history["none"][0])
    assert np.allclose(fitted["als"].user_factors[index], 0.0)
    assert np.allclose(fitted["als"].scores(index), 0.0)


def test_als_never_recommends_cold_stock_to_a_user_with_history(fitted, heavy_users, cold_ids):
    recommendations = fitted["als"].recommend_many(heavy_users, k=10)
    for items in recommendations.values():
        assert not (set(items) & cold_ids)


def test_als_converges(fitted):
    losses = fitted["als"].loss_
    assert len(losses) == fitted["als"].iterations
    assert losses[-1] < losses[0]


def test_als_parameters_are_validated():
    with pytest.raises(ValueError, match="regularization must be positive"):
        ImplicitALS(regularization=0.0)
    with pytest.raises(ValueError, match="alpha must be positive"):
        ImplicitALS(alpha=0.0)
    with pytest.raises(ValueError, match="must be positive"):
        ImplicitALS(factors=0)


# ------------------------------------------------------------------ item-item
@pytest.fixture(scope="module")
def untruncated(split):
    """No neighbour truncation, so symmetry survives and pair statistics are complete."""
    return ItemItemCF(top_k_neighbours=10_000).fit(split.train)


def test_similarity_is_symmetric_with_a_zero_diagonal(untruncated):
    similarity = untruncated.similarity_
    assert np.allclose(similarity, similarity.T)
    assert np.allclose(np.diag(similarity), 0.0)
    assert (similarity >= 0).all()


def test_shrinkage_keeps_similarities_below_one(untruncated):
    """With shrinkage in the denominator a thin co-occurrence cannot look like a twin."""
    assert untruncated.similarity_.max() < 1.0


def test_truncation_drops_the_weak_neighbours(split, untruncated):
    tight = ItemItemCF(top_k_neighbours=5).fit(split.train)
    assert np.count_nonzero(tight.similarity_) < np.count_nonzero(untruncated.similarity_)


def test_similar_items_share_the_seed_category_more_often(untruncated, split):
    """Collaborative signal should recover the category structure it was never told about."""
    catalogue = split.train.catalogue
    same_category = np.equal.outer(
        catalogue["category"].to_numpy(), catalogue["category"].to_numpy()
    )
    np.fill_diagonal(same_category, False)
    similarity = untruncated.similarity_
    assert similarity[same_category].mean() > similarity[~same_category].mean()


def test_similar_items_returns_scored_neighbours(untruncated, split):
    item_id = split.train.items.ids[1]
    neighbours = untruncated.similar_items(item_id, k=5)
    assert len(neighbours) == 5
    assert item_id not in {item for item, _score in neighbours}
    scores = [score for _item, score in neighbours]
    assert scores == sorted(scores, reverse=True)


def test_itemitem_parameters_are_validated():
    with pytest.raises(ValueError, match="shrinkage"):
        ItemItemCF(shrinkage=-1.0)
    with pytest.raises(ValueError, match="damping"):
        ItemItemCF(damping=1.5)


# -------------------------------------------------------------------- content
def test_content_can_score_items_with_no_history(fitted, split, heavy_users):
    """The capability ALS structurally lacks."""
    model = fitted["content"]
    index = split.train.users.index_of(heavy_users[0])
    scores = model.scores(index)
    cold = split.train.cold_items
    assert cold.any()
    assert (scores[cold] > 0).mean() > 0.5


def test_content_profile_is_a_unit_vector(fitted, split, heavy_users):
    profile = fitted["content"].profile(split.train.users.index_of(heavy_users[0]))
    assert np.linalg.norm(profile) == pytest.approx(1.0, rel=1e-9)


def test_content_has_nothing_to_say_without_history(fitted, split, users_by_history):
    index = split.train.users.index_of(users_by_history["none"][0])
    assert np.allclose(fitted["content"].scores(index), 0.0)
    assert np.allclose(fitted["itemitem"].scores(index), 0.0)


def test_content_features_cover_the_catalogue(fitted, split):
    model = fitted["content"]
    assert model.features_.shape[0] == len(split.train.items)
    assert len(model.feature_names_) == model.features_.shape[1]
    norms = np.linalg.norm(model.features_, axis=1)
    assert np.allclose(norms, 1.0)


def test_price_bands_must_be_meaningful():
    with pytest.raises(ValueError, match="price_bands"):
        ContentBasedRecommender(price_bands=1)


# --------------------------------------------------------------------- hybrid
def test_hybrid_routes_on_available_history(fitted, users_by_history):
    model = fitted["hybrid"]
    assert model.route(users_by_history["none"][0]) == "popularity-only"
    assert model.route(users_by_history["heavy"][0]) == "collaborative"
    if users_by_history["sparse"]:
        assert model.route(users_by_history["sparse"][0]) == "content-led"


def test_hybrid_reserves_a_slot_for_new_stock(fitted, heavy_users, cold_ids):
    """The exposure quota, verified on the users least likely to receive new items."""
    model = fitted["hybrid"]
    for user_id in heavy_users:
        assert model.cold_candidates(user_id, 1), user_id
        items = {item for item, _score in model.recommend(user_id, k=10)}
        assert items & cold_ids, user_id


def test_without_the_quota_new_stock_never_surfaces(split, heavy_users, cold_ids):
    """Ranking alone cannot fix cold start: that is why the slot is reserved."""
    lean = HybridRecommender(
        explore_slots=0, als=ImplicitALS(seed=13, iterations=5)
    ).fit(split.train)
    for user_id in heavy_users:
        items = {item for item, _score in lean.recommend(user_id, k=10)}
        assert not (items & cold_ids), user_id


def test_the_quota_never_lengthens_the_list(fitted, heavy_users):
    assert len(fitted["hybrid"].recommend(heavy_users[0], k=10)) == 10


def test_a_user_without_history_gets_the_bestsellers(fitted, users_by_history):
    hybrid = fitted["hybrid"]
    popularity = fitted["popularity"]
    user_id = users_by_history["none"][0]
    index = hybrid.dataset.users.index_of(user_id)
    assert np.array_equal(hybrid.scores(index), popularity.scores(index))


def test_hybrid_parameters_are_validated():
    with pytest.raises(ValueError, match="unknown component weights"):
        HybridRecommender(weights={"prophet": 1.0})
    with pytest.raises(ValueError, match="non-negative"):
        HybridRecommender(weights={"als": -1.0})
    with pytest.raises(ValueError, match="explore_slots"):
        HybridRecommender(explore_slots=-1)


# ----------------------------------------------------------------- rank fusion
def test_reciprocal_ranks_follow_the_ordering():
    weights = reciprocal_ranks(np.array([3.0, 1.0, 2.0]), candidates=3, constant=60.0)
    assert weights[0] == pytest.approx(1 / 61)
    assert weights[2] == pytest.approx(1 / 62)
    assert weights[1] == pytest.approx(1 / 63)


def test_reciprocal_ranks_ignore_non_positive_scores():
    weights = reciprocal_ranks(np.array([1.0, 0.0, -5.0]), candidates=3)
    assert weights[0] > 0
    assert weights[1] == 0.0
    assert weights[2] == 0.0


def test_reciprocal_ranks_respect_the_candidate_cut():
    weights = reciprocal_ranks(np.array([5.0, 4.0, 3.0]), candidates=2)
    assert np.count_nonzero(weights) == 2


def test_build_model_is_the_single_source_of_names():
    assert isinstance(build_model("popularity"), PopularityRecommender)
    with pytest.raises(ValueError, match="unknown model"):
        build_model("transformer")
