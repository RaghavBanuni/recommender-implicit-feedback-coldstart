"""The business layer between the model and the page.

Model output is a candidate list, not a merchandising decision.  Four constraints are
applied here, in the order a real system applies them:

1. **availability** - an out-of-stock item cannot be sold no matter how well it scored.
2. **already-seen** - handled upstream, except for consumables, which people legitimately
   re-buy; a blanket "never repeat" rule loses that revenue.
3. **per-category cap** - ten variants of the same thing is a bad page even when every one
   of them is individually well predicted.
4. **diversity and margin** - a greedy MMR pass trades a little relevance for spread, and a
   bounded margin nudge lets commercial reality in without letting it take over.

Each step reports what it removed, because "the recommender got worse after we added
business rules" needs an itemised answer, and the cost of each constraint is a decision
for the business rather than for the model.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RerankConfig:
    """Merchandising rules.  All of them are business inputs, not tuned parameters."""

    max_per_category: int = 3
    diversity_lambda: float = 0.3
    margin_weight: float = 0.10
    require_in_stock: bool = True

    def validate(self) -> None:
        if self.max_per_category < 1:
            raise ValueError("max_per_category must be positive")
        if not 0.0 <= self.diversity_lambda <= 1.0:
            raise ValueError("diversity_lambda must lie in [0, 1]")
        if not 0.0 <= self.margin_weight <= 1.0:
            raise ValueError(
                "margin_weight must lie in [0, 1]; above that the page is an ad, not a "
                "recommendation"
            )


@dataclass(frozen=True)
class RerankResult:
    """The final list plus an audit trail of what each rule removed."""

    items: list[str]
    scores: list[float]
    dropped_out_of_stock: int = 0
    dropped_category_cap: int = 0
    dropped_unknown: int = 0
    diagnostics: dict[str, float] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.items)


def _normalise(values: np.ndarray) -> np.ndarray:
    """Min-max to [0, 1] so diversity and margin trade against relevance comparably."""
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return array
    low, high = float(array.min()), float(array.max())
    if high - low < 1e-12:
        return np.ones_like(array)
    return (array - low) / (high - low)


def _similarity(left: pd.Series, right: pd.Series) -> float:
    """Cheap catalogue similarity: same category is the big penalty, same brand a smaller one."""
    score = 0.0
    if left["category"] == right["category"]:
        score += 0.8
    if left["brand"] == right["brand"]:
        score += 0.2
    return score


def rerank(
    candidates: list[tuple[str, float]],
    catalogue: pd.DataFrame,
    k: int = 10,
    config: RerankConfig | None = None,
) -> RerankResult:
    """Apply the merchandising rules to a scored candidate list.

    ``candidates`` should be longer than ``k`` - typically 3-5x - because every constraint
    removes items and a list exactly k long leaves the page short.
    """
    rules = config or RerankConfig()
    rules.validate()
    if k < 1:
        raise ValueError("k must be positive")

    required = {"item_id", "category", "brand", "margin", "in_stock"}
    missing = required - set(catalogue.columns)
    if missing:
        raise ValueError(f"catalogue is missing columns: {sorted(missing)}")
    indexed = catalogue.set_index("item_id")

    known = [(item, score) for item, score in candidates if item in indexed.index]
    dropped_unknown = len(candidates) - len(known)

    if rules.require_in_stock:
        available = [(item, score) for item, score in known if bool(indexed.at[item, "in_stock"])]
    else:
        available = list(known)
    dropped_out_of_stock = len(known) - len(available)

    if not available:
        return RerankResult(
            items=[],
            scores=[],
            dropped_out_of_stock=dropped_out_of_stock,
            dropped_unknown=dropped_unknown,
        )

    item_ids = [item for item, _score in available]
    relevance = _normalise(np.array([score for _item, score in available]))
    margin = _normalise(indexed.loc[item_ids, "margin"].to_numpy(dtype=float))
    utility = relevance * (1.0 + rules.margin_weight * margin)

    selected: list[int] = []
    category_counts: dict[str, int] = {}
    dropped_cap = 0
    remaining = set(range(len(item_ids)))

    while remaining and len(selected) < k:
        best_index: int | None = None
        best_value = -np.inf
        for position in sorted(remaining):
            category = str(indexed.at[item_ids[position], "category"])
            if category_counts.get(category, 0) >= rules.max_per_category:
                continue
            penalty = 0.0
            if selected:
                penalty = max(
                    _similarity(indexed.loc[item_ids[position]], indexed.loc[item_ids[chosen]])
                    for chosen in selected
                )
            value = (1.0 - rules.diversity_lambda) * utility[position] - (
                rules.diversity_lambda * penalty
            )
            if value > best_value:
                best_value = value
                best_index = position
        if best_index is None:
            dropped_cap += len(remaining)  # everything left is capped out
            break
        selected.append(best_index)
        remaining.discard(best_index)
        category = str(indexed.at[item_ids[best_index], "category"])
        category_counts[category] = category_counts.get(category, 0) + 1

    chosen_ids = [item_ids[position] for position in selected]
    return RerankResult(
        items=chosen_ids,
        scores=[float(available[position][1]) for position in selected],
        dropped_out_of_stock=dropped_out_of_stock,
        dropped_category_cap=dropped_cap,
        dropped_unknown=dropped_unknown,
        diagnostics={
            "distinct_categories": float(
                indexed.loc[chosen_ids, "category"].nunique() if chosen_ids else 0
            ),
            "mean_margin": float(
                indexed.loc[chosen_ids, "margin"].mean() if chosen_ids else 0.0
            ),
            "top_category_share": float(
                indexed.loc[chosen_ids, "category"].value_counts(normalize=True).iloc[0]
                if chosen_ids
                else 0.0
            ),
        },
    )


def list_profile(items: list[str], catalogue: pd.DataFrame) -> dict[str, float]:
    """Describe one list, so before/after re-ranking can be compared honestly."""
    indexed = catalogue.set_index("item_id")
    present = [item for item in items if item in indexed.index]
    if not present:
        return {
            "items": 0,
            "distinct_categories": 0.0,
            "top_category_share": 0.0,
            "mean_margin": 0.0,
            "out_of_stock": 0,
        }
    rows = indexed.loc[present]
    return {
        "items": len(present),
        "distinct_categories": float(rows["category"].nunique()),
        "top_category_share": round(
            float(rows["category"].value_counts(normalize=True).iloc[0]), 4
        ),
        "mean_margin": round(float(rows["margin"].mean()), 3),
        "out_of_stock": int((~rows["in_stock"].astype(bool)).sum()),
    }
