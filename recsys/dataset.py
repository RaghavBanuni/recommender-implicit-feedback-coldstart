"""Turning an event log into a training matrix without leaking the future.

Three decisions live here, and each one is a place where recommender evaluations
usually go wrong.

**The split is a timestamp, not a shuffle.**  Training sees days ``0..split_day``;
ground truth is what happened afterwards.  A random split lets the model learn from
behaviour that had not happened yet and inflates every metric, which is why offline
numbers so often fail to reproduce online.

**Feedback is graded, not binary.**  A purchase is stronger evidence than a view, so
events carry a confidence weight.  The matrix stores summed confidence; the ALS solver
adds the ``1 +`` itself, because in the Hu-Koren-Volinsky formulation an unobserved cell
is a *low-confidence zero*, not a missing value.

**Old evidence counts for less.**  Confidence decays with an exponential half-life, so a
click from eight months ago does not carry the same weight as one from last week.  With
taste drift in the data this is worth real accuracy.

Every user and every catalogue item gets a row or column, including those with no
interactions at all.  Dropping them would hide exactly the cold-start cases the harness
exists to measure.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import sparse

from .data import EventLog

EVENT_CONFIDENCE: dict[str, float] = {"view": 1.0, "cart": 2.0, "purchase": 5.0}
DEFAULT_HALF_LIFE = 90.0


@dataclass(frozen=True)
class IndexMap:
    """Stable two-way mapping between ids and matrix positions."""

    ids: tuple[str, ...]
    position: dict[str, int] = field(repr=False)

    @classmethod
    def from_ids(cls, ids) -> "IndexMap":
        ordered = tuple(str(value) for value in ids)
        if len(set(ordered)) != len(ordered):
            raise ValueError("ids must be unique")
        return cls(ids=ordered, position={value: index for index, value in enumerate(ordered)})

    def __len__(self) -> int:
        return len(self.ids)

    def __contains__(self, value: object) -> bool:
        return str(value) in self.position

    def encode(self, values) -> np.ndarray:
        return np.array([self.position[str(value)] for value in values], dtype=int)

    def decode(self, indices) -> list[str]:
        return [self.ids[int(index)] for index in indices]

    def index_of(self, value: str) -> int:
        try:
            return self.position[str(value)]
        except KeyError as error:
            raise KeyError(f"unknown id: {value!r}") from error


@dataclass(frozen=True)
class Dataset:
    """The training view: a confidence matrix plus the side information models may use."""

    matrix: sparse.csr_matrix
    users: IndexMap
    items: IndexMap
    catalogue: pd.DataFrame
    events: pd.DataFrame
    split_day: int

    def __post_init__(self) -> None:
        if self.matrix.shape != (len(self.users), len(self.items)):
            raise ValueError("matrix shape does not match the index maps")
        if (self.matrix.data < 0).any():
            raise ValueError("confidence values cannot be negative")

    # --- counts ------------------------------------------------------------
    @property
    def item_interactions(self) -> np.ndarray:
        """Distinct users per item in the training window."""
        binary = self.matrix.copy()
        binary.data = np.ones_like(binary.data)
        return np.asarray(binary.sum(axis=0)).ravel()

    @property
    def user_interactions(self) -> np.ndarray:
        """Distinct items per user in the training window."""
        binary = self.matrix.copy()
        binary.data = np.ones_like(binary.data)
        return np.asarray(binary.sum(axis=1)).ravel()

    @property
    def cold_items(self) -> np.ndarray:
        """Boolean mask of items with no training interactions.

        Matrix factorization cannot place these anywhere: their factors never receive a
        gradient, so they can never be recommended.  Any claim about handling new stock
        has to be checked against this mask.
        """
        return self.item_interactions == 0

    @property
    def popularity(self) -> np.ndarray:
        return self.item_interactions

    def seen(self, user_id: str) -> np.ndarray:
        """Item indices this user already interacted with during training."""
        row = self.matrix[self.users.index_of(user_id)]
        return row.indices.copy()

    def history(self, user_id: str) -> pd.DataFrame:
        """The user's training events, most recent first."""
        return (
            self.events[self.events["user_id"] == str(user_id)]
            .sort_values("day", ascending=False)
            .reset_index(drop=True)
        )

    def recent_popularity(self, window_days: int = 30) -> np.ndarray:
        """Interaction counts restricted to the tail of the training window.

        Recent popularity beats all-time popularity as a baseline whenever taste drifts,
        and it is the baseline a real team would actually deploy.
        """
        if window_days < 1:
            raise ValueError("window_days must be positive")
        recent = self.events[self.events["day"] > self.split_day - window_days]
        counts = recent["item_id"].value_counts()
        return counts.reindex(list(self.items.ids), fill_value=0).to_numpy(dtype=float)


@dataclass(frozen=True)
class Split:
    """Training data, the held-out events, and the cut-off between them."""

    train: Dataset
    test_events: pd.DataFrame
    split_day: int
    log: EventLog

    def ground_truth(
        self, exclude_seen: bool = True, event_types: tuple[str, ...] | None = None
    ) -> dict[str, set[str]]:
        """Relevant items per user in the holdout window.

        ``exclude_seen`` mirrors what the serving pipeline does - it will not show an item
        the user already interacted with, so rewarding the model for re-predicting it
        measures nothing.  Repeat purchases of consumables are a separate problem with a
        separate protocol; set the flag to ``False`` to score that instead.
        """
        frame = self.test_events
        if event_types is not None:
            unknown = set(event_types) - set(EVENT_CONFIDENCE)
            if unknown:
                raise ValueError(f"unknown event types: {sorted(unknown)}")
            frame = frame[frame["event_type"].isin(event_types)]

        truth: dict[str, set[str]] = {}
        for user_id, group in frame.groupby("user_id", sort=True):
            items = set(group["item_id"])
            if exclude_seen and str(user_id) in self.train.users:
                already = set(self.train.items.decode(self.train.seen(str(user_id))))
                items -= already
            if items:
                truth[str(user_id)] = items
        return truth

    def evaluation_users(
        self, exclude_seen: bool = True, event_types: tuple[str, ...] | None = None
    ) -> list[str]:
        return sorted(self.ground_truth(exclude_seen, event_types))


def confidence_weights(
    events: pd.DataFrame, split_day: int, half_life: float | None = DEFAULT_HALF_LIFE
) -> np.ndarray:
    """Event confidence, optionally decayed by age at the split."""
    unknown = set(events["event_type"]) - set(EVENT_CONFIDENCE)
    if unknown:
        raise ValueError(f"unknown event types: {sorted(unknown)}")
    weights = events["event_type"].map(EVENT_CONFIDENCE).to_numpy(dtype=float)
    if half_life is None:
        return weights
    if half_life <= 0:
        raise ValueError("half_life must be positive, or None to disable decay")
    age = (split_day - events["day"].to_numpy(dtype=float)).clip(min=0.0)
    return weights * np.power(0.5, age / half_life)


def build_split(
    log: EventLog,
    split_day: int | None = None,
    half_life: float | None = DEFAULT_HALF_LIFE,
) -> Split:
    """Cut the log at ``split_day`` and build the training matrix from the past only."""
    cutoff = log.split_day if split_day is None else int(split_day)
    if cutoff < 1 or cutoff >= int(log.events["day"].max()):
        raise ValueError("split_day must leave events on both sides of the cut")

    train_events = log.events[log.events["day"] <= cutoff].reset_index(drop=True)
    test_events = log.events[log.events["day"] > cutoff].reset_index(drop=True)
    if train_events.empty or test_events.empty:
        raise ValueError("the split left one side empty")

    users = IndexMap.from_ids(log.users["user_id"])
    items = IndexMap.from_ids(log.items["item_id"])

    weights = confidence_weights(train_events, cutoff, half_life)
    rows = users.encode(train_events["user_id"])
    columns = items.encode(train_events["item_id"])
    matrix = sparse.coo_matrix(
        (weights, (rows, columns)), shape=(len(users), len(items)), dtype=float
    ).tocsr()
    matrix.sum_duplicates()  # repeat events on the same pair accumulate confidence

    train = Dataset(
        matrix=matrix,
        users=users,
        items=items,
        catalogue=log.items.set_index("item_id").loc[list(items.ids)].reset_index(),
        events=train_events,
        split_day=cutoff,
    )
    return Split(train=train, test_events=test_events, split_day=cutoff, log=log)


def user_segments(split: Split, cold_threshold: int = 3, heavy_quantile: float = 0.8):
    """Label every user by how much history the model actually has for them.

    Reporting one pooled accuracy number hides the fact that recommenders are excellent
    for heavy users and near-useless for newcomers - and newcomers are the ones a growing
    business keeps acquiring.
    """
    if cold_threshold < 1:
        raise ValueError("cold_threshold must be positive")
    counts = pd.Series(
        split.train.user_interactions, index=list(split.train.users.ids), name="history"
    )
    warm = counts[counts >= cold_threshold]
    heavy_cut = float(warm.quantile(heavy_quantile)) if not warm.empty else float("inf")

    def label(value: int) -> str:
        if value == 0:
            return "no history"
        if value < cold_threshold:
            return "sparse"
        if value >= heavy_cut:
            return "heavy"
        return "regular"

    frame = counts.to_frame()
    frame["segment"] = [label(int(value)) for value in counts]
    join_day = split.log.users.set_index("user_id")["join_day"]
    frame["join_day"] = join_day.reindex(frame.index).to_numpy()
    frame["new_user"] = frame["join_day"] > split.split_day - split.log.config.new_user_window
    return frame.reset_index(names="user_id")
