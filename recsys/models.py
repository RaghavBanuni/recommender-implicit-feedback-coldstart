"""Four recommenders and a router, all implemented directly on numpy/scipy.

They are written from scratch because the interesting parts of implicit-feedback
recommendation are exactly the parts a library hides:

* **PopularityRecommender** - the baseline that must be beaten.  It scores *recent*
  popularity rather than all-time, because with drifting taste that is both stronger and
  what an honest team would actually deploy.
* **ItemItemCF** - cosine similarity with shrinkage, plus per-user damping so a user with
  three hundred clicks does not define everybody's neighbourhoods.  Shrinkage is what
  stops a pair that co-occurred twice from looking like a perfect match.
* **ImplicitALS** - Hu, Koren & Volinsky alternating least squares.  Unobserved cells are
  low-confidence zeros rather than missing values, and the confidence of an observed cell
  grows with the strength of the evidence.
* **ContentBasedRecommender** - cosine in attribute space.  Weaker than collaborative
  filtering for established items, and the *only* one of the four that can score an item
  nobody has touched yet.
* **HybridRecommender** - routes by how much history a user has, fuses components by
  reciprocal rank, and reserves a slot for new stock.

A deliberate choice runs through the ALS implementation: entities with no interactions keep
zero factors.  It would be easy to hide that behind a fallback and report a flattering
number; instead the cold case stays visible, and the hybrid is what addresses it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from scipy import sparse

from .dataset import Dataset


class Recommender(ABC):
    """Common ranking machinery: subclasses only produce a score vector."""

    name: str = "recommender"

    def __init__(self) -> None:
        self.dataset: Dataset | None = None

    # --- to implement ------------------------------------------------------
    @abstractmethod
    def fit(self, dataset: Dataset) -> "Recommender":
        ...

    @abstractmethod
    def scores(self, user_index: int) -> np.ndarray:
        """Score for every item, in catalogue order."""

    # --- shared ------------------------------------------------------------
    def _require_fit(self) -> Dataset:
        if self.dataset is None:
            raise RuntimeError(f"{self.name} has not been fitted")
        return self.dataset

    def blocked_mask(
        self, user_index: int, exclude_seen: bool, allowed: np.ndarray | None
    ) -> np.ndarray:
        """Items this user must not be shown: already seen, or filtered out upstream."""
        dataset = self._require_fit()
        blocked = np.zeros(len(dataset.items), dtype=bool)
        if exclude_seen:
            blocked[dataset.matrix[user_index].indices] = True
        if allowed is not None:
            mask = np.asarray(allowed, dtype=bool).ravel()
            if mask.size != blocked.size:
                raise ValueError("allowed mask must cover the whole catalogue")
            blocked |= ~mask
        return blocked

    def recommend(
        self,
        user_id: str,
        k: int = 10,
        exclude_seen: bool = True,
        allowed: np.ndarray | None = None,
    ) -> list[tuple[str, float]]:
        """Top ``k`` items with scores, highest first.

        Ties break on item index so a run is reproducible; a random tie-break makes offline
        metrics wobble between runs for no reason.
        """
        dataset = self._require_fit()
        if k < 1:
            raise ValueError("k must be positive")
        user_index = dataset.users.index_of(user_id)
        values = np.array(self.scores(user_index), dtype=float)
        if values.shape != (len(dataset.items),):
            raise ValueError(f"{self.name} produced a score vector of the wrong shape")

        values[self.blocked_mask(user_index, exclude_seen, allowed)] = -np.inf
        order = np.lexsort((np.arange(values.size), -values))
        chosen = [index for index in order if np.isfinite(values[index])][:k]
        return [(dataset.items.ids[index], float(values[index])) for index in chosen]

    def recommend_many(
        self, user_ids, k: int = 10, exclude_seen: bool = True
    ) -> dict[str, list[str]]:
        return {
            str(user_id): [item for item, _score in self.recommend(str(user_id), k, exclude_seen)]
            for user_id in user_ids
        }


class PopularityRecommender(Recommender):
    """Recent bestsellers for everyone.  Not personalised, frequently competitive."""

    name = "popularity"

    def __init__(self, window_days: int | None = 45) -> None:
        super().__init__()
        if window_days is not None and window_days < 1:
            raise ValueError("window_days must be positive or None for all-time popularity")
        self.window_days = window_days
        self.scores_: np.ndarray | None = None

    def fit(self, dataset: Dataset) -> "PopularityRecommender":
        self.dataset = dataset
        counts = (
            dataset.item_interactions.astype(float)
            if self.window_days is None
            else dataset.recent_popularity(self.window_days)
        )
        # log damping: the top item is popular, not a thousand times more relevant
        self.scores_ = np.log1p(counts)
        return self

    def scores(self, user_index: int) -> np.ndarray:
        if self.scores_ is None:
            raise RuntimeError("popularity model has not been fitted")
        return self.scores_


class ItemItemCF(Recommender):
    """Item-item cosine similarity with shrinkage and per-user damping."""

    name = "itemitem"

    def __init__(
        self, top_k_neighbours: int = 100, shrinkage: float = 25.0, damping: float = 0.5
    ) -> None:
        super().__init__()
        if top_k_neighbours < 1:
            raise ValueError("top_k_neighbours must be positive")
        if shrinkage < 0:
            raise ValueError("shrinkage must be non-negative")
        if not 0.0 <= damping <= 1.0:
            raise ValueError("damping must lie in [0, 1]")
        self.top_k_neighbours = top_k_neighbours
        self.shrinkage = shrinkage
        self.damping = damping
        self.similarity_: np.ndarray | None = None

    def fit(self, dataset: Dataset) -> "ItemItemCF":
        self.dataset = dataset
        matrix = dataset.matrix.tocsr(copy=True)

        # damp prolific users before computing co-occurrence
        row_totals = np.asarray(matrix.sum(axis=1)).ravel()
        scale = np.divide(
            1.0,
            np.power(np.maximum(row_totals, 1e-9), self.damping),
            out=np.ones_like(row_totals),
            where=row_totals > 0,
        )
        damped = sparse.diags(scale) @ matrix

        gram = np.asarray((damped.T @ damped).todense())
        norms = np.sqrt(np.maximum(np.diag(gram), 0.0))
        denominator = np.outer(norms, norms) + self.shrinkage
        similarity = np.divide(gram, denominator, out=np.zeros_like(gram), where=denominator > 0)
        np.fill_diagonal(similarity, 0.0)

        # keep only the strongest neighbours: the long tail of tiny similarities is noise
        keep = min(self.top_k_neighbours, similarity.shape[1] - 1)
        if keep < similarity.shape[1] - 1:
            cut = np.partition(similarity, -keep, axis=1)[:, -keep][:, None]
            similarity = np.where(similarity >= cut, similarity, 0.0)
        self.similarity_ = similarity
        return self

    def scores(self, user_index: int) -> np.ndarray:
        dataset = self._require_fit()
        if self.similarity_ is None:
            raise RuntimeError("item-item model has not been fitted")
        row = dataset.matrix[user_index]
        if row.nnz == 0:
            return np.zeros(len(dataset.items))
        return np.asarray(row @ self.similarity_).ravel()

    def similar_items(self, item_id: str, k: int = 10) -> list[tuple[str, float]]:
        """Neighbours of one item - the cheapest sanity check on a similarity matrix."""
        dataset = self._require_fit()
        if self.similarity_ is None:
            raise RuntimeError("item-item model has not been fitted")
        index = dataset.items.index_of(item_id)
        row = self.similarity_[index]
        order = np.lexsort((np.arange(row.size), -row))[:k]
        return [(dataset.items.ids[position], float(row[position])) for position in order]


class ImplicitALS(Recommender):
    """Alternating least squares for implicit feedback (Hu, Koren & Volinsky 2008).

    For user ``u`` with observed confidences ``c_u`` on items ``I_u``:

        x_u = (Y'Y + Y_u'(C_u - I)Y_u + lambda*I)^-1 Y_u' c_u

    The ``Y'Y`` term is precomputed once per sweep, which is what makes the *entire* zero
    matrix affordable instead of only the observed cells.  That is the whole point of the
    method: with implicit data, the zeros carry information.

    Rows with no observations keep zero factors, so a brand-new item scores zero for
    everyone.  That is not a bug to paper over - it is the cold-start problem, and the
    hybrid router is where it gets addressed.
    """

    name = "als"

    def __init__(
        self,
        factors: int = 32,
        regularization: float = 0.05,
        alpha: float = 25.0,
        iterations: int = 15,
        seed: int = 11,
    ) -> None:
        super().__init__()
        if factors < 1 or iterations < 1:
            raise ValueError("factors and iterations must be positive")
        if regularization <= 0:
            raise ValueError("regularization must be positive or the solve can be singular")
        if alpha <= 0:
            raise ValueError("alpha must be positive; it is what makes confidence graded")
        self.factors = factors
        self.regularization = regularization
        self.alpha = alpha
        self.iterations = iterations
        self.seed = seed
        self.user_factors: np.ndarray | None = None
        self.item_factors: np.ndarray | None = None
        self.loss_: list[float] = []

    def fit(self, dataset: Dataset) -> "ImplicitALS":
        self.dataset = dataset
        rng = np.random.default_rng(self.seed)
        by_user = dataset.matrix.tocsr()
        by_item = dataset.matrix.T.tocsr()

        self.user_factors = rng.normal(0.0, 0.01, size=(by_user.shape[0], self.factors))
        self.item_factors = rng.normal(0.0, 0.01, size=(by_item.shape[0], self.factors))
        self.loss_ = []

        for _iteration in range(self.iterations):
            self.user_factors = self._sweep(by_user, self.item_factors)
            self.item_factors = self._sweep(by_item, self.user_factors)
            self.loss_.append(self._observed_loss(by_user))
        return self

    def _sweep(self, matrix: sparse.csr_matrix, other: np.ndarray) -> np.ndarray:
        gram = other.T @ other
        ridge = self.regularization * np.eye(self.factors)
        result = np.zeros((matrix.shape[0], self.factors))
        indptr, indices, data = matrix.indptr, matrix.indices, matrix.data

        for row in range(matrix.shape[0]):
            start, end = indptr[row], indptr[row + 1]
            if start == end:
                continue  # no evidence: factors stay at zero, honestly
            columns = indices[start:end]
            confidence = 1.0 + self.alpha * data[start:end]
            observed = other[columns]
            left = gram + (observed * (confidence - 1.0)[:, None]).T @ observed + ridge
            right = observed.T @ confidence
            try:
                result[row] = np.linalg.solve(left, right)
            except np.linalg.LinAlgError:  # pragma: no cover - numerical safety net
                result[row] = np.linalg.lstsq(left, right, rcond=None)[0]
        return result

    def _observed_loss(self, matrix: sparse.csr_matrix) -> float:
        """Confidence-weighted squared error on observed cells only.

        The full objective includes every zero cell; this partial version is enough to see
        whether the sweeps are converging, which is all it is used for.
        """
        assert self.user_factors is not None and self.item_factors is not None
        coo = matrix.tocoo()
        predicted = np.einsum("ij,ij->i", self.user_factors[coo.row], self.item_factors[coo.col])
        confidence = 1.0 + self.alpha * coo.data
        return round(float((confidence * (1.0 - predicted) ** 2).sum()), 4)

    def scores(self, user_index: int) -> np.ndarray:
        if self.user_factors is None or self.item_factors is None:
            raise RuntimeError("ALS has not been fitted")
        return self.item_factors @ self.user_factors[user_index]


class ContentBasedRecommender(Recommender):
    """Cosine similarity in item-attribute space.

    The only model here that can rank an item with no interaction history, because every
    feature it uses - category, brand, price band - exists the moment the item is created.
    That is what makes it the right fallback rather than a weaker duplicate of ALS.
    """

    name = "content"

    def __init__(self, price_bands: int = 5) -> None:
        super().__init__()
        if price_bands < 2:
            raise ValueError("price_bands must be at least 2")
        self.price_bands = price_bands
        self.features_: np.ndarray | None = None
        self.feature_names_: list[str] = []

    def fit(self, dataset: Dataset) -> "ContentBasedRecommender":
        self.dataset = dataset
        catalogue = dataset.catalogue
        blocks = [
            pd.get_dummies(catalogue["category"], prefix="cat", dtype=float),
            pd.get_dummies(catalogue["brand"], prefix="brand", dtype=float),
            pd.get_dummies(
                pd.qcut(catalogue["price"], self.price_bands, labels=False, duplicates="drop"),
                prefix="price",
                dtype=float,
            ),
            catalogue[["is_consumable"]].astype(float),
        ]
        features = pd.concat(blocks, axis=1)
        self.feature_names_ = list(features.columns)
        matrix = features.to_numpy(dtype=float)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        self.features_ = np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms > 0)
        return self

    def profile(self, user_index: int) -> np.ndarray:
        """Confidence-weighted average of the attributes a user has engaged with."""
        dataset = self._require_fit()
        if self.features_ is None:
            raise RuntimeError("content model has not been fitted")
        row = dataset.matrix[user_index]
        if row.nnz == 0:
            return np.zeros(self.features_.shape[1])
        weights = row.data / row.data.sum()
        vector = weights @ self.features_[row.indices]
        norm = np.linalg.norm(vector)
        return vector / norm if norm > 0 else vector

    def scores(self, user_index: int) -> np.ndarray:
        dataset = self._require_fit()
        if self.features_ is None:
            raise RuntimeError("content model has not been fitted")
        vector = self.profile(user_index)
        if not vector.any():
            return np.zeros(len(dataset.items))
        return self.features_ @ vector


def reciprocal_ranks(scores: np.ndarray, candidates: int = 200, constant: float = 60.0) -> np.ndarray:
    """Reciprocal-rank weights for one component's top candidates.

    Rank fusion instead of score addition, because an ALS dot product, a cosine and a log
    count are not on the same scale and normalising them is guesswork.  Items outside the
    candidate set contribute nothing.
    """
    values = np.asarray(scores, dtype=float).ravel()
    if candidates < 1:
        raise ValueError("candidates must be positive")
    if constant <= 0:
        raise ValueError("constant must be positive")
    weights = np.zeros(values.size)
    order = np.lexsort((np.arange(values.size), -values))[: min(candidates, values.size)]
    for rank, index in enumerate(order):
        if values[index] <= 0:
            continue  # a non-positive score is not a recommendation
        weights[index] = 1.0 / (constant + rank + 1)
    return weights


class HybridRecommender(Recommender):
    """Routes by available history, fuses by reciprocal rank, and funds discovery.

    Two mechanisms, and the first matters more than the fusion weights:

    **Routing.**  With one interaction, ALS and item-item are fitting noise, so they are
    excluded rather than down-weighted; with none, nothing personal is known and recent
    popularity is the only defensible answer.  Pretending a personalised model works for a
    user it has never seen is how cold-start bugs reach production.

    **An exposure quota.**  ``explore_slots`` of every ``k`` are reserved for the best
    content-matched item that has no interaction history.  A score boost cannot fix this:
    new items lose on every collaborative signal by construction, so with pure ranking they
    never appear, never accumulate feedback, and stay cold forever.  Reserving a slot breaks
    that loop, and the cost - one slot of a ranked list - is explicit and adjustable rather
    than buried in a weight.
    """

    name = "hybrid"

    def __init__(
        self,
        cold_threshold: int = 3,
        weights: dict[str, float] | None = None,
        candidates: int = 200,
        explore_slots: int = 1,
        als: ImplicitALS | None = None,
        item_item: ItemItemCF | None = None,
        content: ContentBasedRecommender | None = None,
        popularity: PopularityRecommender | None = None,
    ) -> None:
        super().__init__()
        if cold_threshold < 1:
            raise ValueError("cold_threshold must be positive")
        if explore_slots < 0:
            raise ValueError("explore_slots cannot be negative")
        self.cold_threshold = cold_threshold
        self.candidates = candidates
        self.explore_slots = explore_slots
        self.weights = dict(
            weights or {"als": 1.0, "itemitem": 0.7, "content": 0.35, "popularity": 0.15}
        )
        unknown = set(self.weights) - {"als", "itemitem", "content", "popularity"}
        if unknown:
            raise ValueError(f"unknown component weights: {sorted(unknown)}")
        if min(self.weights.values(), default=0.0) < 0:
            raise ValueError("component weights must be non-negative")
        self.als = als or ImplicitALS()
        self.item_item = item_item or ItemItemCF()
        self.content = content or ContentBasedRecommender()
        self.popularity = popularity or PopularityRecommender()
        self.cold_items_: np.ndarray | None = None

    def fit(self, dataset: Dataset) -> "HybridRecommender":
        self.dataset = dataset
        for component in (self.als, self.item_item, self.content, self.popularity):
            component.fit(dataset)
        self.cold_items_ = dataset.cold_items.copy()
        return self

    def route(self, user_id: str) -> str:
        """Which strategy this user gets: useful in logs and in tests."""
        dataset = self._require_fit()
        history = int(dataset.user_interactions[dataset.users.index_of(user_id)])
        if history == 0:
            return "popularity-only"
        if history < self.cold_threshold:
            return "content-led"
        return "collaborative"

    def scores(self, user_index: int) -> np.ndarray:
        dataset = self._require_fit()
        history = int(dataset.user_interactions[user_index])
        if history == 0:
            return self.popularity.scores(user_index).copy()

        if history < self.cold_threshold:
            active = {"content": self.content, "popularity": self.popularity}
            weights = {"content": 1.0, "popularity": 0.5}
        else:
            active = {
                "als": self.als,
                "itemitem": self.item_item,
                "content": self.content,
                "popularity": self.popularity,
            }
            weights = self.weights

        fused = np.zeros(len(dataset.items))
        for key, component in active.items():
            weight = float(weights.get(key, 0.0))
            if weight <= 0:
                continue
            fused += weight * reciprocal_ranks(component.scores(user_index), self.candidates)
        return fused

    def cold_candidates(
        self, user_id: str, limit: int, exclude_seen: bool = True, allowed=None
    ) -> list[str]:
        """Best content matches among items with no interaction history."""
        dataset = self._require_fit()
        if self.cold_items_ is None or limit < 1 or not self.cold_items_.any():
            return []
        user_index = dataset.users.index_of(user_id)
        content_scores = np.array(self.content.scores(user_index), dtype=float)
        if not content_scores.any():
            # no profile to match against: rank new stock by price band alone would be
            # arbitrary, so decline rather than fill the slot with noise
            return []
        eligible = self.cold_items_ & ~self.blocked_mask(user_index, exclude_seen, allowed)
        values = np.where(eligible, content_scores, -np.inf)
        order = np.lexsort((np.arange(values.size), -values))
        return [
            dataset.items.ids[index]
            for index in order[:limit]
            if np.isfinite(values[index]) and values[index] > 0
        ]

    def recommend(
        self,
        user_id: str,
        k: int = 10,
        exclude_seen: bool = True,
        allowed: np.ndarray | None = None,
    ) -> list[tuple[str, float]]:
        """Fused ranking with the discovery quota applied at the bottom of the list.

        New items take the last slots rather than the first: the quota buys them exposure
        without displacing the most relevant result on the page.
        """
        ranked = super().recommend(user_id, k, exclude_seen, allowed)
        quota = min(self.explore_slots, max(k - 1, 0))
        if quota < 1:
            return ranked

        already = {item for item, _score in ranked}
        fresh = [item for item in self.cold_candidates(user_id, quota, exclude_seen, allowed) if item not in already]
        if not fresh:
            return ranked

        kept = ranked[: max(k - len(fresh), 0)]
        # cold items carry no comparable score; report 0.0 rather than invent one
        return kept + [(item, 0.0) for item in fresh]


MODEL_FACTORIES: dict[str, callable] = {
    "popularity": PopularityRecommender,
    "itemitem": ItemItemCF,
    "als": ImplicitALS,
    "content": ContentBasedRecommender,
    "hybrid": HybridRecommender,
}


def build_model(name: str, **kwargs) -> Recommender:
    """Factory used by the CLI so model names are one source of truth."""
    try:
        factory = MODEL_FACTORIES[name]
    except KeyError as error:
        raise ValueError(
            f"unknown model {name!r}; choose from {sorted(MODEL_FACTORIES)}"
        ) from error
    return factory(**kwargs)
