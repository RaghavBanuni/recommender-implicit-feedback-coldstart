"""Tests for the synthetic event log.

The generator has to contain real difficulty or the evaluation proves nothing, so each
intended property is asserted: causality (no event before a launch or a signup), a
power-law demand curve, concentrated per-user taste, drift across the timeline and repeat
purchases confined to consumables.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from recsys.data import (
    CATEGORIES,
    LogConfig,
    activity_profile,
    generate_events,
    gini,
    launch_profile,
    popularity_profile,
)


def test_log_is_reproducible(config):
    first = generate_events(config)
    second = generate_events(config)
    pd.testing.assert_frame_equal(first.events, second.events)
    pd.testing.assert_frame_equal(first.items, second.items)


def test_a_different_seed_gives_a_different_log(config):
    other = generate_events(LogConfig(**{**config.__dict__, "seed": config.seed + 1}))
    baseline = generate_events(config)
    assert not other.events.equals(baseline.events)


def test_days_stay_inside_the_timeline(log, config):
    assert log.events["day"].min() >= 0
    assert log.events["day"].max() < config.days


def test_nobody_interacts_before_signing_up(log):
    joined = log.users.set_index("user_id")["join_day"]
    earliest = log.events.groupby("user_id")["day"].min()
    assert (earliest >= joined.reindex(earliest.index)).all()


def test_no_item_is_touched_before_it_launches(log):
    """The causality guarantee the cold-start analysis depends on."""
    launch = log.items.set_index("item_id")["launch_day"]
    earliest = log.events.groupby("item_id")["day"].min()
    assert (earliest >= launch.reindex(earliest.index)).all()


def test_demand_follows_a_power_law(log):
    profile = popularity_profile(log)
    assert profile["top_10pct_share"] > 0.25
    assert profile["gini"] > 0.25
    assert profile["events"] == len(log.events)


def test_taste_is_concentrated_not_uniform(log):
    """Without real per-user structure no recommender could beat popularity."""
    frame = log.events.merge(log.items[["item_id", "category"]], on="item_id")
    shares = []
    for _user_id, group in frame.groupby("user_id"):
        if len(group) < 10:
            continue
        shares.append(group["category"].value_counts(normalize=True).iloc[0])
    assert np.mean(shares) > 0.35  # uniform taste would sit near 1/8


def test_taste_drifts_across_the_timeline(log, config):
    """Drift is what makes a random split dishonest and a temporal split necessary."""
    frame = log.events.merge(log.items[["item_id", "category"]], on="item_id")
    early = frame[frame["day"] < config.days // 2]["category"].value_counts(normalize=True)
    late = frame[frame["day"] >= config.days // 2]["category"].value_counts(normalize=True)
    aligned = pd.DataFrame({"early": early, "late": late}).fillna(0.0)
    total_variation = 0.5 * (aligned["early"] - aligned["late"]).abs().sum()
    assert total_variation > 0.015
    assert set(aligned.index) <= set(CATEGORIES)


def test_repeat_interactions_exist_and_favour_consumables(log):
    repeats = log.events[log.events.duplicated(["user_id", "item_id"], keep=False)]
    assert not repeats.empty
    consumable = log.items.set_index("item_id")["is_consumable"]
    share = float(consumable.reindex(repeats["item_id"]).mean())
    assert share > 0.5  # consumables are only a quarter of the catalogue


def test_recent_launches_have_little_history(log):
    profile = launch_profile(log).set_index("cohort")
    assert profile.loc["established", "mean_interactions"] > profile.loc[
        "recent launch", "mean_interactions"
    ]


def test_some_users_are_almost_new(log):
    profile = activity_profile(log)
    assert profile["share_under_3_events"] > 0.0
    assert profile["users"] == len(log.users)


def test_some_stock_is_unavailable(log):
    share = float((~log.items["in_stock"].astype(bool)).mean())
    assert 0.0 < share < 0.25


def test_every_category_is_represented(log):
    assert set(log.items["category"]) == set(CATEGORIES)


def test_gini_of_a_flat_distribution_is_zero():
    assert gini(np.ones(10)) == pytest.approx(0.0, abs=1e-9)


def test_gini_of_a_winner_take_all_distribution_is_high():
    values = np.zeros(10)
    values[0] = 10.0
    assert gini(values) == pytest.approx(0.9, abs=1e-9)


def test_gini_rejects_bad_input():
    with pytest.raises(ValueError, match="non-negative"):
        gini(np.array([-1.0, 2.0]))
    with pytest.raises(ValueError, match="at least one value"):
        gini(np.array([]))


def test_configuration_is_validated():
    with pytest.raises(ValueError, match="at least 120"):
        generate_events(LogConfig(days=60))
    with pytest.raises(ValueError, match=r"\[0, 1\)"):
        generate_events(LogConfig(repeat_rate=1.5))
    with pytest.raises(ValueError, match="too small"):
        generate_events(LogConfig(n_users=10))
    with pytest.raises(ValueError, match="before the holdout window"):
        generate_events(LogConfig(late_launch_window=10, holdout_days=30))
