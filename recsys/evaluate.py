"""Evaluation harness: identical users, identical ground truth, identical k.

Every model is scored on the same evaluation set, and the popularity baseline is always
in the table.  That constraint is the whole value of the harness - a comparison where each
model picks its own user set or its own k can be made to say anything.

Three views are produced, and they routinely disagree:

* **pooled** - one row per model, accuracy plus catalogue health.
* **by segment** - the same metrics split by how much history the user has.  A model can win
  overall and be worse than popularity for newcomers.
* **cold start** - who can actually surface items with no interaction history.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .dataset import Split, user_segments
from .metrics import (
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
from .models import (
    ContentBasedRecommender,
    HybridRecommender,
    ImplicitALS,
    ItemItemCF,
    PopularityRecommender,
    Recommender,
)

DEFAULT_MODELS: tuple[str, ...] = ("popularity", "content", "itemitem", "als", "hybrid")


@dataclass(frozen=True)
class EvalConfig:
    """Evaluation protocol.  Sampling users is for speed and is seeded."""

    k: int = 10
    exclude_seen: bool = True
    event_types: tuple[str, ...] | None = None
    max_users: int | None = 500
    seed: int = 5

    def validate(self) -> None:
        if self.k < 1:
            raise ValueError("k must be positive")
        if self.max_users is not None and self.max_users < 10:
            raise ValueError("evaluating fewer than ten users tells you nothing")


def evaluation_set(split: Split, config: EvalConfig) -> tuple[list[str], dict[str, set[str]]]:
    """Users with at least one relevant held-out item, and their ground truth."""
    config.validate()
    truth = split.ground_truth(config.exclude_seen, config.event_types)
    users = sorted(truth)
    if not users:
        raise ValueError("no user has any held-out relevant item")
    if config.max_users is not None and len(users) > config.max_users:
        rng = np.random.default_rng(config.seed)
        chosen = rng.choice(len(users), size=config.max_users, replace=False)
        users = sorted(users[int(index)] for index in chosen)
        truth = {user: truth[user] for user in users}
    return users, truth


def per_user_scores(
    recommendations: Mapping[str, list[str]], truth: Mapping[str, set[str]], k: int
) -> pd.DataFrame:
    """One row per evaluated user, so any segment cut can be taken afterwards."""
    rows = []
    for user_id, items in recommendations.items():
        relevant = truth[user_id]
        rows.append(
            {
                "user_id": user_id,
                "recall": recall_at_k(items, relevant, k),
                "precision": precision_at_k(items, relevant, k),
                "ndcg": ndcg_at_k(items, relevant, k),
                "average_precision": average_precision_at_k(items, relevant, k),
                "hit": hit_rate_at_k(items, relevant, k),
                "reciprocal_rank": reciprocal_rank(items, relevant, k),
                "relevant_items": len(relevant),
            }
        )
    return pd.DataFrame(rows)


def score_recommendations(
    recommendations: Mapping[str, list[str]],
    truth: Mapping[str, set[str]],
    split: Split,
    k: int,
) -> dict[str, float]:
    """Accuracy and catalogue health for one model's output."""
    scores = per_user_scores(recommendations, truth, k)
    dataset = split.train
    counts = dict(zip(dataset.items.ids, dataset.item_interactions.astype(float)))
    categories = dict(zip(dataset.catalogue["item_id"], dataset.catalogue["category"]))
    cold = [
        item_id
        for item_id, is_cold in zip(dataset.items.ids, dataset.cold_items)
        if bool(is_cold)
    ]
    empty_lists = sum(1 for items in recommendations.values() if not items)
    return {
        "users": int(len(scores)),
        f"recall@{k}": round(float(scores["recall"].mean()), 5),
        f"ndcg@{k}": round(float(scores["ndcg"].mean()), 5),
        f"map@{k}": round(float(scores["average_precision"].mean()), 5),
        f"hit_rate@{k}": round(float(scores["hit"].mean()), 5),
        f"precision@{k}": round(float(scores["precision"].mean()), 5),
        "mrr": round(float(scores["reciprocal_rank"].mean()), 5),
        "coverage": round(catalogue_coverage(recommendations, len(dataset.items)), 5),
        "novelty": round(novelty(recommendations, counts), 4),
        "exposure_gini": round(exposure_gini(recommendations, len(dataset.items)), 4),
        "list_diversity": round(intra_list_diversity(recommendations, categories), 4),
        "cold_item_share": round(cold_item_share(recommendations, cold), 5),
        "empty_lists": int(empty_lists),
    }


def recommend_for_users(
    model: Recommender, split: Split, users: list[str], config: EvalConfig
) -> dict[str, list[str]]:
    return model.recommend_many(users, k=config.k, exclude_seen=config.exclude_seen)


def evaluate_model(
    model: Recommender, split: Split, config: EvalConfig | None = None
) -> dict[str, float]:
    settings = config or EvalConfig()
    users, truth = evaluation_set(split, settings)
    recommendations = recommend_for_users(model, split, users, settings)
    return {"model": model.name, **score_recommendations(recommendations, truth, split, settings.k)}


def fit_models(
    split: Split, names: tuple[str, ...] = DEFAULT_MODELS, seed: int = 11
) -> dict[str, Recommender]:
    """Fit the requested models on the training window.

    The hybrid gets its own component instances rather than borrowing the standalone ones,
    so a change to one cannot silently alter the other's reported score.
    """
    catalogue: dict[str, Recommender] = {
        "popularity": PopularityRecommender(),
        "content": ContentBasedRecommender(),
        "itemitem": ItemItemCF(),
        "als": ImplicitALS(seed=seed),
        "hybrid": HybridRecommender(als=ImplicitALS(seed=seed)),
    }
    unknown = set(names) - set(catalogue)
    if unknown:
        raise ValueError(f"unknown models: {sorted(unknown)}")
    return {name: catalogue[name].fit(split.train) for name in names}


def compare_models(
    models: Mapping[str, Recommender], split: Split, config: EvalConfig | None = None
) -> pd.DataFrame:
    """One row per model on identical users and ground truth."""
    settings = config or EvalConfig()
    users, truth = evaluation_set(split, settings)
    rows = []
    for name, model in models.items():
        recommendations = recommend_for_users(model, split, users, settings)
        rows.append({"model": name, **score_recommendations(recommendations, truth, split, settings.k)})
    frame = pd.DataFrame(rows)
    return frame.sort_values(f"ndcg@{settings.k}", ascending=False).reset_index(drop=True)


def segment_report(
    models: Mapping[str, Recommender],
    split: Split,
    config: EvalConfig | None = None,
    cold_threshold: int = 3,
) -> pd.DataFrame:
    """Accuracy by how much history the model had for the user.

    This is where a pooled win often falls apart: collaborative models depend on history,
    and the users a growing business keeps adding do not have any.
    """
    settings = config or EvalConfig()
    users, truth = evaluation_set(split, settings)
    segments = user_segments(split, cold_threshold=cold_threshold).set_index("user_id")

    rows = []
    for name, model in models.items():
        recommendations = recommend_for_users(model, split, users, settings)
        scores = per_user_scores(recommendations, truth, settings.k)
        scores["segment"] = segments["segment"].reindex(scores["user_id"]).to_numpy()
        grouped = scores.groupby("segment").agg(
            users=("user_id", "size"),
            recall=("recall", "mean"),
            ndcg=("ndcg", "mean"),
            hit_rate=("hit", "mean"),
        )
        for segment, row in grouped.iterrows():
            rows.append(
                {
                    "model": name,
                    "segment": segment,
                    "users": int(row["users"]),
                    "recall": round(float(row["recall"]), 5),
                    "ndcg": round(float(row["ndcg"]), 5),
                    "hit_rate": round(float(row["hit_rate"]), 5),
                }
            )
    order = {"no history": 0, "sparse": 1, "regular": 2, "heavy": 3}
    frame = pd.DataFrame(rows)
    frame["_order"] = frame["segment"].map(order).fillna(9)
    return frame.sort_values(["_order", "model"]).drop(columns="_order").reset_index(drop=True)


def coldstart_report(
    models: Mapping[str, Recommender], split: Split, config: EvalConfig | None = None
) -> pd.DataFrame:
    """Can each model surface stock it has never seen anybody interact with?

    Matrix factorization scores zero here by construction, which is the finding rather than
    a defect: the fix is routing, not more factors.
    """
    settings = config or EvalConfig()
    users, truth = evaluation_set(split, settings)
    dataset = split.train
    cold = [
        item_id for item_id, is_cold in zip(dataset.items.ids, dataset.cold_items) if bool(is_cold)
    ]
    segments = user_segments(split).set_index("user_id")
    new_users = set(segments.index[segments["new_user"].astype(bool)])

    rows = []
    for name, model in models.items():
        recommendations = recommend_for_users(model, split, users, settings)
        scores = per_user_scores(recommendations, truth, settings.k)
        is_new = scores["user_id"].isin(new_users)
        rows.append(
            {
                "model": name,
                "cold_items_in_catalogue": len(cold),
                "cold_item_share": round(cold_item_share(recommendations, cold), 5),
                "coverage": round(catalogue_coverage(recommendations, len(dataset.items)), 5),
                "recall_new_users": round(float(scores.loc[is_new, "recall"].mean()), 5)
                if is_new.any()
                else float("nan"),
                "recall_established": round(float(scores.loc[~is_new, "recall"].mean()), 5)
                if (~is_new).any()
                else float("nan"),
                "new_users_scored": int(is_new.sum()),
            }
        )
    return pd.DataFrame(rows).sort_values("cold_item_share", ascending=False).reset_index(drop=True)
