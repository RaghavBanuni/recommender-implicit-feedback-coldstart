"""Synthetic interaction log with the pathologies that break naive recommenders.

A generator is only useful here if the difficulties are real, so each is deliberate:

* **latent taste** - every user draws a sparse Dirichlet preference over categories, so
  collaborative signal genuinely exists and a model that finds it should beat
  popularity.  Without this, no recommender can win and the evaluation says nothing.
* **power-law popularity** - a handful of items absorb most of the interactions.  This is
  what makes a popularity baseline embarrassingly strong and catalogue coverage worth
  measuring.
* **staggered launches** - about a sixth of the catalogue appears only in the last weeks.
  These items cannot have collaborative factors, which is the cold-start problem in its
  honest form rather than a hypothetical.
* **late-joining users** - some users arrive days before the evaluation window with one or
  two clicks to their name.
* **taste drift** - preferences rotate over the timeline, so a model fitted on old data
  decays and a time-based split is the only fair one.
* **repeat purchases** - consumables get re-bought, so "filter everything the user has
  already seen" is not universally correct.
* **out-of-stock items** - a fraction cannot be sold at all, which the re-ranking layer has
  to respect no matter what the model scored them.

Events are views, cart-adds and purchases: graded evidence, never a rating.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

CATEGORIES: tuple[str, ...] = (
    "running",
    "yoga",
    "cycling",
    "hiking",
    "swimming",
    "strength",
    "recovery",
    "nutrition",
)
BRANDS: tuple[str, ...] = ("Northwind", "Kestrel", "Alto", "Vireo", "Basalt")
CONSUMABLE_CATEGORIES: frozenset[str] = frozenset({"nutrition", "recovery"})
EVENT_TYPES: tuple[str, ...] = ("view", "cart", "purchase")


@dataclass(frozen=True)
class LogConfig:
    """Knobs for the generated log.  Defaults give ~30k events over eight months."""

    days: int = 240
    n_users: int = 1_200
    n_items: int = 400
    holdout_days: int = 30
    late_launch_share: float = 0.15
    late_launch_window: int = 60
    new_user_share: float = 0.12
    new_user_window: int = 30
    taste_concentration: float = 0.25
    popularity_sigma: float = 1.1
    repeat_rate: float = 0.25
    drift_strength: float = 0.35
    mean_events_per_user: float = 26.0
    activity_shape: float = 1.4
    out_of_stock_share: float = 0.08
    seed: int = 23

    def validate(self) -> None:
        if self.days < 120:
            raise ValueError("days must be at least 120 so a temporal split has room")
        if self.n_users < 50 or self.n_items < len(CATEGORIES) * 4:
            raise ValueError("the catalogue and audience are too small to be interesting")
        if not 1 <= self.holdout_days < self.days // 2:
            raise ValueError("holdout_days must be positive and well inside the timeline")
        for name, value in (
            ("late_launch_share", self.late_launch_share),
            ("new_user_share", self.new_user_share),
            ("repeat_rate", self.repeat_rate),
            ("drift_strength", self.drift_strength),
            ("out_of_stock_share", self.out_of_stock_share),
        ):
            if not 0.0 <= value < 1.0:
                raise ValueError(f"{name} must lie in [0, 1)")
        if self.late_launch_window <= self.holdout_days:
            raise ValueError(
                "late launches must start before the holdout window, or cold items have "
                "no training history at all and the comparison is vacuous"
            )
        if self.taste_concentration <= 0 or self.popularity_sigma <= 0:
            raise ValueError("taste_concentration and popularity_sigma must be positive")
        if self.mean_events_per_user <= 1 or self.activity_shape <= 0:
            raise ValueError("activity parameters must be positive")


@dataclass(frozen=True)
class EventLog:
    """Everything downstream code needs: the events plus both side tables."""

    events: pd.DataFrame
    items: pd.DataFrame
    users: pd.DataFrame
    config: LogConfig

    @property
    def last_day(self) -> int:
        return int(self.events["day"].max())

    @property
    def split_day(self) -> int:
        """Last day of the training window under the default holdout."""
        return self.config.days - self.config.holdout_days - 1


def _item_ids(count: int) -> list[str]:
    return [f"I{index:04d}" for index in range(count)]


def _user_ids(count: int) -> list[str]:
    return [f"U{index:04d}" for index in range(count)]


def build_catalogue(config: LogConfig, rng: np.random.Generator) -> pd.DataFrame:
    """Item side table: attributes known at launch, plus a launch day and stock flag.

    Categories are assigned by cycling before shuffling, which guarantees every category
    is represented; sampling them independently occasionally leaves one empty and then a
    content model has a dead feature column.
    """
    count = config.n_items
    categories = np.array([CATEGORIES[index % len(CATEGORIES)] for index in range(count)])
    rng.shuffle(categories)
    brands = rng.choice(BRANDS, size=count)
    price = np.round(np.exp(rng.normal(3.2, 0.6, size=count)), 2)
    margin_rate = rng.uniform(0.15, 0.45, size=count)

    # heavy-tailed appeal: a few items will absorb most of the demand
    base_popularity = np.exp(rng.normal(0.0, config.popularity_sigma, size=count))
    base_popularity /= base_popularity.mean()

    launch_day = np.zeros(count, dtype=int)
    order = rng.permutation(count)
    late_count = int(round(count * config.late_launch_share))
    late = order[:late_count]
    launch_day[late] = rng.integers(
        config.days - config.late_launch_window, config.days - config.holdout_days, size=late_count
    )
    # a slower trickle of mid-timeline launches, so "new" is a gradient not a flag
    mid = order[late_count : late_count + int(round(count * 0.10))]
    launch_day[mid] = rng.integers(1, max(2, config.days // 2), size=len(mid))

    return pd.DataFrame(
        {
            "item_id": _item_ids(count),
            "category": categories,
            "brand": brands,
            "price": price,
            "margin": np.round(price * margin_rate, 2),
            "launch_day": launch_day,
            "base_popularity": np.round(base_popularity, 5),
            "in_stock": rng.random(count) > config.out_of_stock_share,
            "is_consumable": np.isin(categories, list(CONSUMABLE_CATEGORIES)),
        }
    )


def build_users(
    config: LogConfig, rng: np.random.Generator
) -> tuple[pd.DataFrame, np.ndarray]:
    """User side table and the latent taste matrix that drives their choices.

    The taste matrix is the ground truth no model is allowed to see; it exists so that
    "did the model find real structure" is a question with an answer.
    """
    count = config.n_users
    taste = rng.dirichlet(np.full(len(CATEGORIES), config.taste_concentration), size=count)
    activity = rng.gamma(
        config.activity_shape, config.mean_events_per_user / config.activity_shape, size=count
    )

    join_day = np.zeros(count, dtype=int)
    order = rng.permutation(count)
    new_count = int(round(count * config.new_user_share))
    newcomers = order[:new_count]
    join_day[newcomers] = rng.integers(
        config.days - config.new_user_window, config.days - 1, size=new_count
    )
    ramping = order[new_count : new_count + int(round(count * 0.15))]
    join_day[ramping] = rng.integers(1, max(2, config.days // 2), size=len(ramping))

    users = pd.DataFrame(
        {
            "user_id": _user_ids(count),
            "join_day": join_day,
            "activity": np.round(activity, 4),
        }
    )
    return users, taste


def _affinity(taste: np.ndarray, items: pd.DataFrame) -> np.ndarray:
    """Users x items appetite: category taste times intrinsic popularity."""
    one_hot = np.zeros((len(items), len(CATEGORIES)))
    lookup = {category: index for index, category in enumerate(CATEGORIES)}
    one_hot[np.arange(len(items)), [lookup[value] for value in items["category"]]] = 1.0
    # the floor keeps every item reachable: real logs are never perfectly targeted
    return (taste @ one_hot.T + 0.02) * items["base_popularity"].to_numpy()


def _drifted(taste: np.ndarray, strength: float) -> np.ndarray:
    """Rotate taste towards the neighbouring category to age the training data."""
    rotated = (1.0 - strength) * taste + strength * np.roll(taste, 1, axis=1)
    return rotated / rotated.sum(axis=1, keepdims=True)


def generate_events(config: LogConfig | None = None) -> EventLog:
    """Generate the full log.  Identical seeds give byte-identical output."""
    settings = config or LogConfig()
    settings.validate()
    rng = np.random.default_rng(settings.seed)

    items = build_catalogue(settings, rng)
    users, taste = build_users(settings, rng)

    early_weights = _affinity(taste, items)
    late_weights = _affinity(_drifted(taste, settings.drift_strength), items)
    launch_day = items["launch_day"].to_numpy()
    consumable = items["is_consumable"].to_numpy()
    midpoint = settings.days // 2

    user_ids = users["user_id"].to_numpy()
    item_ids = items["item_id"].to_numpy()
    join_days = users["join_day"].to_numpy()
    activity = users["activity"].to_numpy()

    records: list[tuple[str, str, int, str]] = []
    for user_index in range(settings.n_users):
        start = int(join_days[user_index])
        span = settings.days - start
        expected = activity[user_index] * span / settings.days
        count = int(rng.poisson(max(expected, 0.4)))
        if count == 0:
            continue  # users with no history at all: the hardest cold-start case
        event_days = np.sort(rng.integers(start, settings.days, size=count))

        history: list[int] = []
        for day in event_days:
            weights = (late_weights if day >= midpoint else early_weights)[user_index].copy()
            weights[launch_day > day] = 0.0  # an unlaunched item cannot be clicked
            total = weights.sum()
            if total <= 0:
                continue

            repeatable = [index for index in history if consumable[index] and launch_day[index] <= day]
            if repeatable and rng.random() < settings.repeat_rate:
                item_index = int(repeatable[int(rng.integers(len(repeatable)))])
            else:
                item_index = int(rng.choice(settings.n_items, p=weights / total))

            purchase_probability = 0.20 if consumable[item_index] else 0.09
            draw = rng.random()
            if draw < purchase_probability:
                event_type = "purchase"
            elif draw < purchase_probability + 0.18:
                event_type = "cart"
            else:
                event_type = "view"

            history.append(item_index)
            records.append((user_ids[user_index], item_ids[item_index], int(day), event_type))

    events = pd.DataFrame(records, columns=["user_id", "item_id", "day", "event_type"])
    if events.empty:
        raise ValueError("the generator produced no events; check the activity parameters")
    events = events.sort_values(["day", "user_id", "item_id"]).reset_index(drop=True)
    return EventLog(events=events, items=items, users=users, config=settings)


# ------------------------------------------------------------------- profiling
def gini(values: np.ndarray) -> float:
    """Gini coefficient of a non-negative distribution; 0 is uniform, 1 is winner-take-all."""
    array = np.sort(np.asarray(values, dtype=float).ravel())
    if array.size == 0:
        raise ValueError("gini needs at least one value")
    if (array < 0).any():
        raise ValueError("gini is only defined for non-negative values")
    total = array.sum()
    if total == 0:
        return 0.0
    index = np.arange(1, array.size + 1)
    return float((2.0 * (index * array).sum()) / (array.size * total) - (array.size + 1) / array.size)


def popularity_profile(log: EventLog) -> dict[str, float]:
    """How concentrated demand is - the reason a popularity baseline is hard to beat."""
    counts = log.events["item_id"].value_counts()
    full = counts.reindex(log.items["item_id"], fill_value=0).to_numpy(dtype=float)
    ordered = np.sort(full)[::-1]
    total = ordered.sum()
    top_one_percent = max(1, int(round(len(ordered) * 0.01)))
    top_decile = max(1, int(round(len(ordered) * 0.10)))
    return {
        "events": int(total),
        "items": int(len(ordered)),
        "items_never_touched": int((full == 0).sum()),
        "top_1pct_share": round(float(ordered[:top_one_percent].sum() / total), 4),
        "top_10pct_share": round(float(ordered[:top_decile].sum() / total), 4),
        "gini": round(gini(full), 4),
    }


def activity_profile(log: EventLog) -> dict[str, float]:
    """Distribution of history length, which is what cold-start routing keys on."""
    counts = log.events.groupby("user_id").size()
    full = counts.reindex(log.users["user_id"], fill_value=0)
    return {
        "users": int(len(full)),
        "users_with_no_events": int((full == 0).sum()),
        "median_events": float(full.median()),
        "p90_events": float(full.quantile(0.90)),
        "share_under_3_events": round(float((full < 3).mean()), 4),
        "gini": round(gini(full.to_numpy(dtype=float)), 4),
    }


def launch_profile(log: EventLog) -> pd.DataFrame:
    """Interactions per item bucketed by launch timing.

    The point of the table: recently launched items have almost no training signal, so
    any model that learns only from co-occurrence will ignore them.
    """
    counts = log.events["item_id"].value_counts()
    frame = log.items.copy()
    frame["interactions"] = frame["item_id"].map(counts).fillna(0).astype(int)
    cutoff = log.config.days - log.config.late_launch_window
    frame["cohort"] = np.where(
        frame["launch_day"] == 0,
        "established",
        np.where(frame["launch_day"] >= cutoff, "recent launch", "mid-timeline launch"),
    )
    grouped = (
        frame.groupby("cohort")
        .agg(
            items=("item_id", "size"),
            mean_interactions=("interactions", "mean"),
            median_interactions=("interactions", "median"),
            never_touched=("interactions", lambda column: int((column == 0).sum())),
        )
        .round(3)
        .reset_index()
    )
    return grouped.sort_values("mean_interactions", ascending=False).reset_index(drop=True)
