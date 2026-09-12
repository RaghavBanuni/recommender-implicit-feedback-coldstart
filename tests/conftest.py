"""Shared fixtures.

A smaller catalogue and audience than the CLI defaults, chosen so the ALS sweeps and the
full model comparison run in seconds while keeping every structural property the tests
care about: staggered launches, late-joining users, cold items and sparse histories.
"""

from __future__ import annotations

import pytest

from recsys.data import LogConfig, generate_events
from recsys.dataset import build_split
from recsys.evaluate import EvalConfig, fit_models

CONFIG = LogConfig(
    days=180,
    n_users=300,
    n_items=120,
    holdout_days=21,
    late_launch_window=45,
    new_user_window=25,
    seed=7,
)


@pytest.fixture(scope="session")
def config() -> LogConfig:
    return CONFIG


@pytest.fixture(scope="session")
def log():
    return generate_events(CONFIG)


@pytest.fixture(scope="session")
def split(log):
    return build_split(log)


@pytest.fixture(scope="session")
def fitted(split):
    return fit_models(split, seed=13)


@pytest.fixture(scope="session")
def eval_config():
    return EvalConfig(k=10, max_users=120, seed=3)


@pytest.fixture(scope="session")
def users_by_history(split):
    """One user id per history regime, so cold-start behaviour can be asserted directly."""
    counts = split.train.user_interactions
    ids = split.train.users.ids
    buckets: dict[str, list[str]] = {"none": [], "sparse": [], "regular": [], "heavy": []}
    for user_id, count in zip(ids, counts):
        value = int(count)
        if value == 0:
            buckets["none"].append(user_id)
        elif value < 3:
            buckets["sparse"].append(user_id)
        elif value < 15:
            buckets["regular"].append(user_id)
        else:
            buckets["heavy"].append(user_id)
    return buckets
