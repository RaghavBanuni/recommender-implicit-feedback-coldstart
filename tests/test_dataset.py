"""Tests for the split and the confidence matrix.

The first two tests are the ones that matter: if any training row carries information from
after the cut-off, every accuracy number in this repository becomes fiction.
"""

from __future__ import annotations

import numpy as np
import pytest

from recsys.dataset import (
    DEFAULT_HALF_LIFE,
    EVENT_CONFIDENCE,
    IndexMap,
    build_split,
    confidence_weights,
    user_segments,
)


def test_training_data_stops_at_the_cut_off(split):
    assert split.train.events["day"].max() <= split.split_day
    assert split.test_events["day"].min() > split.split_day


def test_split_covers_every_event(split, log):
    assert len(split.train.events) + len(split.test_events) == len(log.events)


def test_matrix_keeps_a_row_for_everyone(split, log):
    """Dropping users and items without history would hide the cold-start cases."""
    assert split.train.matrix.shape == (len(log.users), len(log.items))
    assert len(split.train.users) == len(log.users)
    assert len(split.train.items) == len(log.items)


def test_confidence_matches_the_decayed_event_weights(split):
    """Hand-recompute one user's row from the raw events."""
    events = split.train.events
    busiest = events["user_id"].value_counts().index[0]
    mine = events[events["user_id"] == busiest].copy()
    mine["weight"] = mine["event_type"].map(EVENT_CONFIDENCE) * np.power(
        0.5, (split.split_day - mine["day"]) / DEFAULT_HALF_LIFE
    )
    expected = mine.groupby("item_id")["weight"].sum()

    row = split.train.matrix[split.train.users.index_of(busiest)]
    actual = {
        split.train.items.ids[index]: float(value) for index, value in zip(row.indices, row.data)
    }
    assert set(actual) == set(expected.index)
    for item_id, value in expected.items():
        assert actual[item_id] == pytest.approx(float(value), rel=1e-9)


def test_repeat_events_accumulate_confidence(log):
    """Two interactions with the same item are stronger evidence than one."""
    undecayed = build_split(log, half_life=None)
    repeats = undecayed.train.events[
        undecayed.train.events.duplicated(["user_id", "item_id"], keep=False)
    ]
    assert not repeats.empty
    user_id, item_id = repeats.iloc[0][["user_id", "item_id"]]
    same = repeats[(repeats["user_id"] == user_id) & (repeats["item_id"] == item_id)]
    expected = float(same["event_type"].map(EVENT_CONFIDENCE).sum())
    value = undecayed.train.matrix[
        undecayed.train.users.index_of(user_id), undecayed.train.items.index_of(item_id)
    ]
    assert float(value) == pytest.approx(expected)
    assert len(same) > 1


def test_decay_halves_confidence_after_one_half_life():
    import pandas as pd

    events = pd.DataFrame(
        {"event_type": ["purchase", "purchase"], "day": [100, 100 - int(DEFAULT_HALF_LIFE)]}
    )
    weights = confidence_weights(events, split_day=100, half_life=DEFAULT_HALF_LIFE)
    assert weights[0] == pytest.approx(EVENT_CONFIDENCE["purchase"])
    assert weights[1] == pytest.approx(EVENT_CONFIDENCE["purchase"] / 2)


def test_items_launched_after_the_cut_off_are_cold(split, log):
    launch = log.items.set_index("item_id")["launch_day"]
    cold = {
        item_id
        for item_id, is_cold in zip(split.train.items.ids, split.train.cold_items)
        if bool(is_cold)
    }
    late = set(launch.index[launch > split.split_day])
    assert late <= cold
    assert cold, "the fixture needs at least one item with no training history"


def test_ground_truth_excludes_what_the_user_already_saw(split):
    truth = split.ground_truth(exclude_seen=True)
    assert truth
    for user_id, items in truth.items():
        seen = set(split.train.items.decode(split.train.seen(user_id)))
        assert not (items & seen)


def test_keeping_seen_items_can_only_add_relevance(split):
    strict = split.ground_truth(exclude_seen=True)
    loose = split.ground_truth(exclude_seen=False)
    for user_id, items in strict.items():
        assert items <= loose[user_id]


def test_ground_truth_can_be_restricted_to_purchases(split):
    purchases = split.ground_truth(event_types=("purchase",))
    everything = split.ground_truth()
    assert purchases
    assert sum(len(items) for items in purchases.values()) < sum(
        len(items) for items in everything.values()
    )


def test_unknown_event_types_are_rejected(split):
    with pytest.raises(ValueError, match="unknown event types"):
        split.ground_truth(event_types=("rating",))


def test_evaluation_users_match_the_ground_truth(split):
    assert split.evaluation_users() == sorted(split.ground_truth())


def test_recent_popularity_only_counts_the_recent_window(split):
    window = 20
    counts = split.train.recent_popularity(window)
    assert counts.shape == (len(split.train.items),)
    assert (counts >= 0).all()
    recent = split.train.events[split.train.events["day"] > split.split_day - window]
    assert counts.sum() == pytest.approx(float(len(recent)))


def test_interaction_counts_are_distinct_pairs(split):
    pairs = split.train.events.drop_duplicates(["user_id", "item_id"])
    assert split.train.item_interactions.sum() == pytest.approx(float(len(pairs)))
    assert split.train.user_interactions.sum() == pytest.approx(float(len(pairs)))


def test_segments_label_every_user(split, log):
    frame = user_segments(split)
    assert len(frame) == len(log.users)
    assert set(frame["segment"]) <= {"no history", "sparse", "regular", "heavy"}
    no_history = frame[frame["segment"] == "no history"]
    assert (no_history["history"] == 0).all()
    assert (frame.loc[frame["history"] == 0, "segment"] == "no history").all()


def test_users_who_joined_after_the_cut_off_have_no_history(split):
    frame = user_segments(split)
    late = frame[frame["join_day"] > split.split_day]
    assert not late.empty
    assert (late["history"] == 0).all()


def test_split_day_must_leave_data_on_both_sides(log):
    with pytest.raises(ValueError, match="both sides"):
        build_split(log, split_day=0)
    with pytest.raises(ValueError, match="both sides"):
        build_split(log, split_day=int(log.events["day"].max()))


def test_half_life_must_be_positive(log):
    with pytest.raises(ValueError, match="half_life must be positive"):
        build_split(log, half_life=0.0)


def test_index_map_round_trips():
    mapping = IndexMap.from_ids(["a", "b", "c"])
    assert mapping.encode(["c", "a"]).tolist() == [2, 0]
    assert mapping.decode([1, 2]) == ["b", "c"]
    assert "b" in mapping and "z" not in mapping
    with pytest.raises(KeyError, match="unknown id"):
        mapping.index_of("z")


def test_index_map_rejects_duplicates():
    with pytest.raises(ValueError, match="unique"):
        IndexMap.from_ids(["a", "a"])
