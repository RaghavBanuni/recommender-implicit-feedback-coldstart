"""Tests for the evaluation harness.

The protocol tests come first: if the models are not scored on identical users with
identical ground truth, the comparison table is decoration.  After that come the two
findings the harness exists to produce - collaborative filtering beats popularity where
history exists, and only the hybrid gives new stock any exposure at all.
"""

from __future__ import annotations

import pytest

from recsys.evaluate import (
    EvalConfig,
    coldstart_report,
    compare_models,
    evaluate_model,
    evaluation_set,
    fit_models,
    per_user_scores,
    segment_report,
)


@pytest.fixture(scope="module")
def comparison(fitted, split, eval_config):
    return compare_models(fitted, split, eval_config).set_index("model")


def test_every_evaluated_user_has_relevant_items(split, eval_config):
    users, truth = evaluation_set(split, eval_config)
    assert users == sorted(truth)
    assert all(truth[user] for user in users)
    assert len(users) <= eval_config.max_users


def test_the_evaluation_set_is_reproducible(split, eval_config):
    first, _ = evaluation_set(split, eval_config)
    second, _ = evaluation_set(split, eval_config)
    assert first == second


def test_a_different_sample_seed_changes_the_users(split, eval_config):
    other = EvalConfig(k=eval_config.k, max_users=eval_config.max_users, seed=eval_config.seed + 1)
    first, _ = evaluation_set(split, eval_config)
    second, _ = evaluation_set(split, other)
    assert first != second


def test_ground_truth_never_contains_a_training_interaction(split, eval_config):
    _users, truth = evaluation_set(split, eval_config)
    for user_id, items in truth.items():
        seen = set(split.train.items.decode(split.train.seen(user_id)))
        assert not (items & seen)


def test_all_models_are_scored_on_the_same_users(comparison):
    assert comparison["users"].nunique() == 1


def test_the_table_is_sorted_best_first(comparison, eval_config):
    column = f"ndcg@{eval_config.k}"
    assert comparison[column].is_monotonic_decreasing


def test_collaborative_filtering_beats_the_popularity_baseline(comparison, eval_config):
    """If this fails, either the signal or the model is broken - both are worth knowing."""
    column = f"ndcg@{eval_config.k}"
    baseline = float(comparison.loc["popularity", column])
    best = max(float(comparison.loc[name, column]) for name in ("als", "itemitem", "hybrid"))
    assert best > baseline


def test_every_model_gets_something_right(comparison, eval_config):
    for name in comparison.index:
        assert float(comparison.loc[name, f"recall@{eval_config.k}"]) > 0.0
        assert float(comparison.loc[name, f"hit_rate@{eval_config.k}"]) > 0.0


def test_metrics_stay_inside_their_ranges(comparison, eval_config):
    for name in comparison.index:
        row = comparison.loc[name]
        for column in (f"recall@{eval_config.k}", f"ndcg@{eval_config.k}", "coverage", "mrr"):
            assert 0.0 <= float(row[column]) <= 1.0, (name, column)
        assert int(row["empty_lists"]) == 0, name


def test_popularity_is_the_least_novel_model(comparison):
    """By construction it recommends the most-interacted items to everyone."""
    assert float(comparison.loc["popularity", "novelty"]) < float(
        comparison.loc["content", "novelty"]
    )


def test_evaluate_model_reports_the_model_name(fitted, split, eval_config):
    result = evaluate_model(fitted["itemitem"], split, eval_config)
    assert result["model"] == "itemitem"
    assert f"ndcg@{eval_config.k}" in result


def test_per_user_scores_are_bounded(fitted, split, eval_config):
    users, truth = evaluation_set(split, eval_config)
    recommendations = fitted["hybrid"].recommend_many(users, k=eval_config.k)
    scores = per_user_scores(recommendations, truth, eval_config.k)
    assert len(scores) == len(users)
    for column in ("recall", "precision", "ndcg", "average_precision", "hit", "reciprocal_rank"):
        assert scores[column].between(0.0, 1.0).all(), column
    assert (scores["relevant_items"] > 0).all()


def test_segment_report_splits_the_same_users(fitted, split, eval_config):
    table = segment_report(fitted, split, eval_config)
    assert set(table["model"]) == set(fitted)
    totals = table.groupby("model")["users"].sum()
    assert totals.nunique() == 1
    assert "no history" in set(table["segment"])


def test_history_helps_the_collaborative_model(fitted, split, eval_config):
    """The segment cut that a single pooled number would hide."""
    table = segment_report(fitted, split, eval_config)
    als = table[table["model"] == "als"].set_index("segment")
    if "heavy" in als.index and "no history" in als.index:
        assert float(als.loc["heavy", "ndcg"]) > float(als.loc["no history", "ndcg"])


def test_only_the_hybrid_exposes_cold_stock(fitted, split, eval_config):
    table = coldstart_report(fitted, split, eval_config).set_index("model")
    assert int(table.loc["als", "cold_items_in_catalogue"]) > 0
    assert float(table.loc["hybrid", "cold_item_share"]) > 0.05
    assert float(table.loc["hybrid", "cold_item_share"]) > float(
        table.loc["als", "cold_item_share"]
    )
    assert float(table.loc["content", "cold_item_share"]) > 0.0


def test_coldstart_report_separates_new_users(fitted, split, eval_config):
    table = coldstart_report(fitted, split, eval_config)
    assert (table["new_users_scored"] > 0).all()
    assert set(table.columns) >= {"recall_new_users", "recall_established", "coverage"}


def test_unknown_models_are_refused(split):
    with pytest.raises(ValueError, match="unknown models"):
        fit_models(split, names=("transformer",))


def test_evaluation_config_is_validated(split):
    with pytest.raises(ValueError, match="k must be positive"):
        evaluation_set(split, EvalConfig(k=0))
    with pytest.raises(ValueError, match="fewer than ten users"):
        evaluation_set(split, EvalConfig(max_users=3))
