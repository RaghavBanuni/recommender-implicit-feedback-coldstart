"""Command line interface: ``python -m recsys.cli <command>``."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .data import (
    EventLog,
    LogConfig,
    activity_profile,
    generate_events,
    launch_profile,
    popularity_profile,
)
from .dataset import build_split, user_segments
from .evaluate import (
    DEFAULT_MODELS,
    EvalConfig,
    coldstart_report,
    compare_models,
    fit_models,
    segment_report,
)
from .rerank import RerankConfig, list_profile, rerank


def _log(args: argparse.Namespace) -> EventLog:
    return generate_events(
        LogConfig(
            days=args.days,
            n_users=args.users,
            n_items=args.items,
            holdout_days=args.holdout,
            seed=args.seed,
        )
    )


def _write(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    print(f"wrote {path}")


def _models(args: argparse.Namespace) -> tuple[str, ...]:
    requested = tuple(name.strip() for name in args.models.split(",") if name.strip())
    unknown = set(requested) - set(DEFAULT_MODELS)
    if unknown:
        raise SystemExit(f"unknown models: {sorted(unknown)}; choose from {list(DEFAULT_MODELS)}")
    return requested


def cmd_data(args: argparse.Namespace) -> int:
    """Generate the log and show the structure that makes the problem hard."""
    log = _log(args)
    events = log.events
    print(
        f"{len(events)} events, {events['user_id'].nunique()} active users of "
        f"{len(log.users)}, {events['item_id'].nunique()} items touched of {len(log.items)}, "
        f"days {events['day'].min()}-{events['day'].max()}\n"
    )
    print("event mix")
    print(events["event_type"].value_counts(normalize=True).round(4).to_string(), "\n")

    print("demand concentration - why a popularity baseline is hard to beat")
    print(json.dumps(popularity_profile(log), indent=2), "\n")
    print("history length - who the cold-start router has to catch")
    print(json.dumps(activity_profile(log), indent=2), "\n")
    print("interactions by launch cohort")
    print(launch_profile(log).to_string(index=False), "\n")

    split = build_split(log)
    cold = int(split.train.cold_items.sum())
    print(
        f"training window: days 0-{split.split_day}, holdout: "
        f"{split.split_day + 1}-{log.last_day}"
    )
    print(
        f"{cold} of {len(log.items)} items have no training interactions at all - "
        "matrix factorization cannot rank any of them"
    )
    segments = user_segments(split)
    print("\nuser segments at the split")
    print(segments["segment"].value_counts().to_string())

    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    events.to_csv(destination, index=False)
    log.items.to_csv(destination.with_name("items.csv"), index=False)
    print(f"\nwrote {destination} and {destination.with_name('items.csv')}")
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    """All models on identical users, with catalogue health alongside accuracy."""
    log = _log(args)
    split = build_split(log)
    config = EvalConfig(k=args.k, max_users=args.max_users)
    models = fit_models(split, names=_models(args), seed=args.seed)

    table = compare_models(models, split, config)
    print(f"ranking quality at k={args.k} (identical users and ground truth)\n")
    accuracy = [
        "model",
        "users",
        f"recall@{args.k}",
        f"ndcg@{args.k}",
        f"map@{args.k}",
        f"hit_rate@{args.k}",
        "mrr",
    ]
    print(table[accuracy].to_string(index=False), "\n")
    print("catalogue health - accuracy alone would hide all of this")
    print(
        table[
            ["model", "coverage", "novelty", "exposure_gini", "list_diversity", "cold_item_share"]
        ].to_string(index=False),
        "\n",
    )

    best = table.iloc[0]
    baseline = table[table["model"] == "popularity"]
    if not baseline.empty:
        lift = best[f"ndcg@{args.k}"] / max(float(baseline.iloc[0][f"ndcg@{args.k}"]), 1e-9) - 1.0
        print(
            f"best model: {best['model']} (NDCG@{args.k} {best[f'ndcg@{args.k}']:.4f}, "
            f"{lift:+.1%} against the popularity baseline)"
        )
    _write(table, Path(args.out) / "model_comparison.csv")
    return 0


def cmd_segments(args: argparse.Namespace) -> int:
    """The same metrics split by how much history each user has."""
    log = _log(args)
    split = build_split(log)
    config = EvalConfig(k=args.k, max_users=args.max_users)
    models = fit_models(split, names=_models(args), seed=args.seed)

    table = segment_report(models, split, config)
    print(f"accuracy by user segment at k={args.k}\n")
    print(table.to_string(index=False), "\n")
    print(
        "read it this way: collaborative models earn their keep on users with history, \n"
        "and for users without any they are no better than showing the bestsellers."
    )
    _write(table, Path(args.out) / "segment_report.csv")
    return 0


def cmd_coldstart(args: argparse.Namespace) -> int:
    """Who can actually surface stock with no interaction history."""
    log = _log(args)
    split = build_split(log)
    config = EvalConfig(k=args.k, max_users=args.max_users)
    models = fit_models(split, names=_models(args), seed=args.seed)

    table = coldstart_report(models, split, config)
    print(f"cold-start exposure at k={args.k}\n")
    print(table.to_string(index=False), "\n")
    print(
        "ALS scores exactly zero cold items by construction: an item nobody has touched \n"
        "never receives a gradient, so its factors stay at the initialisation. The fix is \n"
        "routing to a content model, not more factors or more iterations."
    )
    _write(table, Path(args.out) / "coldstart_report.csv")
    return 0


def cmd_recommend(args: argparse.Namespace) -> int:
    """One user's list, before and after the business rules."""
    log = _log(args)
    split = build_split(log)
    models = fit_models(split, names=(args.model,), seed=args.seed)
    model = models[args.model]

    user_id = args.user_id
    if user_id not in split.train.users:
        raise SystemExit(f"unknown user {user_id!r}")

    history = split.train.history(user_id)
    print(f"user {user_id}: {len(history)} training events")
    if not history.empty:
        merged = history.merge(log.items[["item_id", "category", "brand"]], on="item_id")
        print("\nmost recent history")
        print(merged.head(8).to_string(index=False))
        print("\ncategories engaged with")
        print(merged["category"].value_counts().head(5).to_string())
    if hasattr(model, "route"):
        print(f"\nrouting decision: {model.route(user_id)}")

    candidates = model.recommend(user_id, k=args.k * 4)
    if not candidates:
        raise SystemExit("the model produced no candidates for this user")
    raw = [item for item, _score in candidates][: args.k]

    result = rerank(
        candidates,
        log.items,
        k=args.k,
        config=RerankConfig(
            max_per_category=args.max_per_category,
            diversity_lambda=args.diversity,
            margin_weight=args.margin_weight,
        ),
    )

    catalogue = log.items.set_index("item_id")
    print(f"\nmodel output (top {args.k} of {len(candidates)} candidates)")
    print(catalogue.loc[raw, ["category", "brand", "price", "margin", "in_stock"]].to_string())
    print(f"\nafter business rules ({len(result)} slots)")
    print(
        catalogue.loc[result.items, ["category", "brand", "price", "margin", "in_stock"]].to_string()
    )
    print(
        f"\nremoved: {result.dropped_out_of_stock} out of stock, "
        f"{result.dropped_category_cap} blocked by the per-category cap"
    )

    comparison = pd.DataFrame(
        [
            {"list": "model output", **list_profile(raw, log.items)},
            {"list": "after rules", **list_profile(result.items, log.items)},
        ]
    )
    print("\nwhat the rules changed")
    print(comparison.to_string(index=False))
    print(
        "\nthe trade is deliberate: a little relevance for spread, availability and margin. \n"
        "Whether it is worth it is a business question, and only an experiment settles it."
    )
    return 0


def cmd_similar(args: argparse.Namespace) -> int:
    """Item neighbourhoods - the cheapest sanity check on a similarity matrix."""
    log = _log(args)
    split = build_split(log)
    model = fit_models(split, names=("itemitem",), seed=args.seed)["itemitem"]

    if args.item_id not in split.train.items:
        raise SystemExit(f"unknown item {args.item_id!r}")
    catalogue = log.items.set_index("item_id")
    seed_row = catalogue.loc[args.item_id]
    print(
        f"{args.item_id}: {seed_row['category']} / {seed_row['brand']} at {seed_row['price']}\n"
    )
    neighbours = model.similar_items(args.item_id, k=args.k)
    frame = pd.DataFrame(neighbours, columns=["item_id", "similarity"])
    frame = frame.merge(
        log.items[["item_id", "category", "brand", "price"]], on="item_id", how="left"
    )
    print(frame.round(4).to_string(index=False))
    same = float((frame["category"] == seed_row["category"]).mean())
    print(f"\n{same:.0%} of the neighbours share the seed item's category")
    return 0


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--days", type=int, default=240)
    parser.add_argument("--users", type=int, default=1_200)
    parser.add_argument("--items", type=int, default=400)
    parser.add_argument("--holdout", type=int, default=30)
    parser.add_argument("--seed", type=int, default=23)


def _eval_args(parser: argparse.ArgumentParser) -> None:
    _common(parser)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--max-users", dest="max_users", type=int, default=500)
    parser.add_argument("--out", default="reports")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="recsys",
        description="Implicit-feedback recommenders with cold-start routing and business rules.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    data_parser = subparsers.add_parser("data", help="generate and describe the event log")
    _common(data_parser)
    data_parser.add_argument("--out", default="data/events.csv")
    data_parser.set_defaults(func=cmd_data)

    evaluate_parser = subparsers.add_parser("evaluate", help="compare all models")
    _eval_args(evaluate_parser)
    evaluate_parser.set_defaults(func=cmd_evaluate)

    segments_parser = subparsers.add_parser("segments", help="accuracy by user segment")
    _eval_args(segments_parser)
    segments_parser.set_defaults(func=cmd_segments)

    cold_parser = subparsers.add_parser("coldstart", help="exposure of items with no history")
    _eval_args(cold_parser)
    cold_parser.set_defaults(func=cmd_coldstart)

    recommend_parser = subparsers.add_parser("recommend", help="one user's list, before and after rules")
    _common(recommend_parser)
    recommend_parser.add_argument("--user-id", dest="user_id", default="U0007")
    recommend_parser.add_argument("--model", default="hybrid", choices=list(DEFAULT_MODELS))
    recommend_parser.add_argument("--k", type=int, default=10)
    recommend_parser.add_argument(
        "--max-per-category", dest="max_per_category", type=int, default=3
    )
    recommend_parser.add_argument("--diversity", type=float, default=0.3)
    recommend_parser.add_argument(
        "--margin-weight", dest="margin_weight", type=float, default=0.10
    )
    recommend_parser.set_defaults(func=cmd_recommend)

    similar_parser = subparsers.add_parser("similar", help="item-item neighbours for one item")
    _common(similar_parser)
    similar_parser.add_argument("--item-id", dest="item_id", default="I0001")
    similar_parser.add_argument("--k", type=int, default=10)
    similar_parser.set_defaults(func=cmd_similar)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
