"""
clinician/common/aggregate.py -- combine a rater's item scores into one disorder score.

The weights are learned on patient transcripts, against the human per-image
TLI score, and then FROZEN for scoring client outputs. So they must be judged
out of sample: every number reported here comes from participants held out of
the fit (grouped K-fold, repeated), never from the rows the weights were fit on.

Weights are non-negative least squares on z-scored items, fit to the rank of
the target (the target is 60% zeros; we care about ordering). Non-negative
because every item is a symptom rating: a negative weight would mean "more of
this symptom = less disorder", which is overfitting noise or, worse, a length
proxy (see non_sequitur).
"""

import json

import numpy as np
import pandas as pd
from scipy.optimize import nnls
from scipy.stats import rankdata, spearmanr


def zfit(X):
    mu, sd = X.mean(axis=0), X.std(axis=0)
    return mu, np.where(sd > 0, sd, 1.0)


def fit_nnls(X, y):
    mu, sd = zfit(X)
    Z = (X - mu) / sd
    t = (rankdata(y) - 0.5) / len(y)                 # target ranks in (0, 1)
    A = np.column_stack([Z, np.ones(len(Z)), -np.ones(len(Z))])  # free-sign intercept
    w, _ = nnls(A, t)
    return {"mean": mu, "sd": sd, "weights": w[:-2], "intercept": w[-2] - w[-1]}


def predict(model, X):
    return ((X - model["mean"]) / model["sd"]) @ model["weights"] + model["intercept"]


def group_folds(groups, k, seed):
    rng = np.random.default_rng(seed)
    uniq = np.array(sorted(set(groups)))
    rng.shuffle(uniq)
    fold_of = {g: i % k for i, g in enumerate(uniq)}
    return np.array([fold_of[g] for g in groups])


def oof(X, y, groups, fitter, k=10, seed=0):
    """Out-of-fold predictions: each participant scored by weights fit without them."""
    pred = np.full(len(y), np.nan)
    folds = group_folds(groups, k, seed)
    for f in range(k):
        tr, te = folds != f, folds == f
        pred[te] = predict(fitter(X[tr], y[tr]), X[te])
    return pred


def rank_partial(x, y, zs):
    Z = np.column_stack([np.ones(len(x))] + [rankdata(z) for z in zs])
    res = lambda v: rankdata(v) - Z @ np.linalg.lstsq(Z, rankdata(v), rcond=None)[0]
    return float(np.corrcoef(res(x), res(y))[0, 1])


def save(model, features, path, extra):
    with open(path, "w") as f:
        json.dump({"features": features,
                   "mean": list(map(float, model["mean"])),
                   "sd": list(map(float, model["sd"])),
                   "weights": list(map(float, model["weights"])),
                   "intercept": float(model["intercept"]), **extra}, f, indent=2)


def load(path):
    with open(path) as f:
        m = json.load(f)
    for k in ("mean", "sd", "weights"):
        m[k] = np.array(m[k])
    return m


def apply(ratings, model):
    """Score a ratings table (same rater columns) with frozen weights. Higher =
    more disordered; the scale is the fitted rank scale, not TLI units."""
    missing = [f for f in model["features"] if f not in ratings]
    if missing:
        raise SystemExit(f"ratings lack features the weights were fit on: {missing}")
    return pd.Series(predict(model, ratings[model["features"]].to_numpy(float)),
                     index=ratings.index, name="disorder_score")
