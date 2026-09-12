"""Ranking and catalogue metrics.

Accuracy metrics answer "did we put something relevant near the top".  They are
necessary and nowhere near sufficient: a model can win on Recall@10 by recommending the
same ten bestsellers to everyone, which sells nothing new and buries the catalogue.  So
the accuracy metrics here are deliberately accompanied by coverage, novelty, concentration
and cold-item exposure, and the harness always reports them together.

Every function takes an *ordered* list of recommended ids and a set of relevant ids, and
every one is small enough to verify by hand - which the tests do.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence

import numpy as np


def _prepare(recommended: Sequence[str], relevant: Iterable[str], k: int):
    if k < 1:
        raise ValueError("k must be positive")
    ranked = [str(item) for item in recommended[:k]]
    if len(set(ranked)) != len(ranked):
        raise ValueError("a recommendation list must not repeat an item")
    return ranked, {str(item) for item in relevant}


def precision_at_k(recommended: Sequence[str], relevant: Iterable[str], k: int = 10) -> float:
    """Share of the k slots that were relevant."""
    ranked, truth = _prepare(recommended, relevant, k)
    if not ranked:
        return 0.0
    return sum(item in truth for item in ranked) / k


def recall_at_k(recommended: Sequence[str], relevant: Iterable[str], k: int = 10) -> float:
    """Share of the relevant items that made it into the top k.

    Note the ceiling: a user with 40 relevant items cannot exceed 0.25 at k=10, so pooled
    recall is partly a statement about how much the users bought, not only about the model.
    That is why NDCG is reported next to it.
    """
    ranked, truth = _prepare(recommended, relevant, k)
    if not truth:
        raise ValueError("recall is undefined without relevant items")
    return sum(item in truth for item in ranked) / len(truth)


def hit_rate_at_k(recommended: Sequence[str], relevant: Iterable[str], k: int = 10) -> float:
    """Did we get anything at all right - the metric a product manager understands."""
    ranked, truth = _prepare(recommended, relevant, k)
    return float(any(item in truth for item in ranked))


def reciprocal_rank(recommended: Sequence[str], relevant: Iterable[str], k: int = 10) -> float:
    ranked, truth = _prepare(recommended, relevant, k)
    for position, item in enumerate(ranked, start=1):
        if item in truth:
            return 1.0 / position
    return 0.0


def average_precision_at_k(
    recommended: Sequence[str], relevant: Iterable[str], k: int = 10
) -> float:
    """Mean of the precisions at each hit, normalised by the achievable number of hits."""
    ranked, truth = _prepare(recommended, relevant, k)
    if not truth:
        raise ValueError("average precision is undefined without relevant items")
    hits = 0
    total = 0.0
    for position, item in enumerate(ranked, start=1):
        if item in truth:
            hits += 1
            total += hits / position
    return total / min(len(truth), k)


def ndcg_at_k(recommended: Sequence[str], relevant: Iterable[str], k: int = 10) -> float:
    """Binary-gain NDCG: position-discounted, normalised by the best possible ordering.

    The ideal DCG uses ``min(k, |relevant|)`` hits, so a user with two relevant items can
    still score 1.0.  Normalising by k instead silently caps such users and makes the
    metric depend on basket size.
    """
    ranked, truth = _prepare(recommended, relevant, k)
    if not truth:
        raise ValueError("NDCG is undefined without relevant items")
    gains = np.array([1.0 if item in truth else 0.0 for item in ranked])
    discounts = 1.0 / np.log2(np.arange(2, len(ranked) + 2))
    dcg = float((gains * discounts).sum())
    ideal_length = min(k, len(truth))
    ideal = float((1.0 / np.log2(np.arange(2, ideal_length + 2))).sum())
    return dcg / ideal if ideal > 0 else 0.0


# ------------------------------------------------------- catalogue-level views
def catalogue_coverage(recommendations: Mapping[str, Sequence[str]], catalogue_size: int) -> float:
    """Fraction of the catalogue that appears in anybody's list.

    Low coverage means most of the inventory is unsellable through the recommender, which
    is a merchandising problem even when accuracy looks fine.
    """
    if catalogue_size < 1:
        raise ValueError("catalogue_size must be positive")
    shown = {str(item) for items in recommendations.values() for item in items}
    return len(shown) / catalogue_size


def novelty(
    recommendations: Mapping[str, Sequence[str]], interaction_counts: Mapping[str, float]
) -> float:
    """Mean self-information ``-log2 p(item)`` of what was recommended.

    Higher means less obvious.  Counts are Laplace-smoothed so an item nobody has touched
    scores high but finite rather than infinite.
    """
    if not interaction_counts:
        raise ValueError("interaction counts are required")
    total = float(sum(interaction_counts.values()))
    catalogue_size = len(interaction_counts)
    scores: list[float] = []
    for items in recommendations.values():
        for item in items:
            count = float(interaction_counts.get(str(item), 0.0))
            probability = (count + 1.0) / (total + catalogue_size)
            scores.append(-np.log2(probability))
    if not scores:
        return 0.0
    return float(np.mean(scores))


def exposure_gini(recommendations: Mapping[str, Sequence[str]], catalogue_size: int) -> float:
    """Gini of how often each item is recommended; 0 is even, 1 is one item everywhere."""
    if catalogue_size < 1:
        raise ValueError("catalogue_size must be positive")
    counts = Counter(str(item) for items in recommendations.values() for item in items)
    exposure = np.zeros(catalogue_size, dtype=float)
    exposure[: len(counts)] = np.array(sorted(counts.values()), dtype=float)
    exposure = np.sort(exposure)
    total = exposure.sum()
    if total == 0:
        return 0.0
    index = np.arange(1, exposure.size + 1)
    return float(
        (2.0 * (index * exposure).sum()) / (exposure.size * total)
        - (exposure.size + 1) / exposure.size
    )


def intra_list_diversity(
    recommendations: Mapping[str, Sequence[str]], categories: Mapping[str, str]
) -> float:
    """Mean share of distinct categories within a list.

    1.0 means every slot is a different category; 0.1 at k=10 means ten of the same thing,
    which is what an accuracy-only objective tends to produce.
    """
    shares: list[float] = []
    for items in recommendations.values():
        if not items:
            continue
        labels = [categories.get(str(item), "unknown") for item in items]
        shares.append(len(set(labels)) / len(labels))
    return float(np.mean(shares)) if shares else 0.0


def cold_item_share(
    recommendations: Mapping[str, Sequence[str]], cold_items: Iterable[str]
) -> float:
    """Share of recommended slots given to items with no training history.

    The number that exposes what a pure matrix-factorization pipeline quietly does: zero.
    """
    cold = {str(item) for item in cold_items}
    total = 0
    hits = 0
    for items in recommendations.values():
        for item in items:
            total += 1
            hits += int(str(item) in cold)
    return hits / total if total else 0.0
