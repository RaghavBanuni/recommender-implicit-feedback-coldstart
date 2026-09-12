# Implicit-Feedback Recommender with Cold-Start Routing

A ranking system built the way a real catalogue forces you to build one: nobody rates
anything, the popularity distribution is a power law, a sixth of the catalogue launched
too recently to have any interaction history, and the list that goes on the page is not
the raw model output.

Four recommenders are implemented from scratch on top of numpy and scipy, evaluated
against a **time-based** split, and then passed through the business layer that decides
what a customer actually sees.

```
events (user, item, type, day)
        |
        v
temporal split  ->  train window        holdout window (ground truth)
        |
        v
models: popularity | item-item CF | ALS (implicit) | content-based | hybrid
        |
        v
ranking metrics: Recall@K, NDCG@K, MAP@K  +  coverage, novelty, Gini
        |            (broken out by user segment and item age)
        v
re-ranking: drop seen/unavailable, cap per category, MMR diversity, margin boost
```

## Why this is not another MovieLens notebook

**Implicit feedback is not a rating.** A view is weak evidence of interest, a purchase is
strong evidence, and a *missing* interaction is not evidence of dislike - it usually means
the item was never shown. The ALS implementation follows Hu, Koren & Volinsky: every
user-item cell is a training example, with a *confidence* weight derived from the event
type, and unobserved cells are treated as low-confidence zeros rather than dropped. Squeezing
implicit events into a rating matrix and running plain SVD is the single most common way this
problem is done wrong.

**A random split leaks the future.** Splitting interactions at random lets a model learn from
next month's behaviour to predict last month's, and inflates every metric. The split here is a
timestamp cut: the model sees days `0..T`, and is scored on what happened after `T` - including
on users and items that barely existed at `T`.

**Cold start is not an edge case, it is a permanent segment.** Matrix factorization has no
factors for an item nobody has touched, so ALS structurally cannot recommend new stock; its
embedding stays near the initialization and the item never surfaces. The evaluation therefore
reports **cold-item exposure** alongside accuracy, and the hybrid routes cold users and cold
items to a content model that only needs attributes known at launch.

**Accuracy alone ships a bad page.** A model optimised for Recall@10 will happily return ten
variants of the same product, items that are out of stock, and things the customer bought
yesterday. The re-ranking layer applies the constraints a merchandiser would insist on, and the
report shows what each constraint costs in recall - because that trade is a business decision,
not a modelling one.

## What the harness reports

| Question | Metric |
| --- | --- |
| Did we put something relevant in the top K? | Recall@K, Hit-rate@K |
| Was it near the top? | NDCG@K, MAP@K, MRR |
| How much of the catalogue can we even sell? | Catalogue coverage |
| Are we just re-selling the bestsellers? | Novelty, Gini of recommended popularity |
| Can new stock get exposure? | Cold-item share of recommendations |
| Does it work for people with no history? | Metrics split by user activity segment |

Every model is scored on identical users, identical ground truth and identical K, and the
popularity baseline is always in the table. A collaborative model that cannot beat "show
everyone the bestsellers" is not worth its serving cost, and that comparison is easy to
quietly omit.

## Layout

```
recsys/
  data.py       synthetic event log: latent taste, power-law popularity, launches, drift
  dataset.py    id encoding, sparse confidence matrix, temporal split, ground truth
  models.py     popularity, item-item CF, implicit ALS, content-based, hybrid router
  metrics.py    ranking and catalogue metrics, each hand-verifiable
  evaluate.py   evaluation harness, segment and cold-start breakdowns
  rerank.py     business rules: availability, per-category caps, MMR diversity, margin
  cli.py        data / evaluate / segments / coldstart / recommend
tests/          leakage, model behaviour, metric arithmetic, re-ranking guarantees
```

## Running it

```bash
pip install -r requirements.txt

python -m recsys.cli data                # generate the log and describe its structure
python -m recsys.cli evaluate            # all models against the temporal split
python -m recsys.cli segments            # accuracy by user activity and tenure
python -m recsys.cli coldstart           # who can actually surface new stock
python -m recsys.cli recommend --user-id U0007   # a real list, before and after re-ranking

pytest -q
```

## Honest limitations

- The event log is synthetic. It contains popularity bias, repeat purchases, taste drift and
  staggered launches on purpose, but no real catalogue is this well behaved.
- Offline ranking metrics are a proxy. They can only reward re-discovering what the logging
  policy already showed the user; the true test is an online experiment, and no offline number
  settles it.
- ALS here is exact-solve per user, which is the clear implementation rather than the fastest
  one. At tens of millions of interactions you would want conjugate-gradient solves and
  approximate nearest neighbours at serving time.
- Confidence weights per event type are a business assumption, not a fitted parameter. They are
  in one place so they can be argued about.

MIT licensed.
