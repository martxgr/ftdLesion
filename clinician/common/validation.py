"""
clinician/common/validation.py -- the numbers every rater is judged by.

Crude, single_shot and fine_tune all report through here, so their validation
against human TLI is on the same footing: same correlation, same CI, same
aggregation from picture rows to sessions.
"""

import numpy as np
import pandas as pd
from scipy import stats


def ccc(y, p):
    """Lin's concordance correlation: agreement, not just association."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    my, mp = y.mean(), p.mean()
    return 2 * np.cov(y, p, bias=True)[0, 1] / (y.var() + p.var() + (my - mp) ** 2)


def cluster_bootstrap(x, y, groups, fn, n_boot=2000, seed=0):
    """95% CI for fn(x, y), resampling whole clusters (participants), because
    rows from one person are not independent."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"x": x, "y": y, "g": groups})
    by = {g: d for g, d in df.groupby("g")}
    keys = np.array(list(by))
    vals = []
    for _ in range(n_boot):
        sample = pd.concat([by[k] for k in rng.choice(keys, len(keys))])
        if sample.x.nunique() > 1 and sample.y.nunique() > 1:
            vals.append(fn(sample.x.values, sample.y.values))
    return np.percentile(vals, [2.5, 97.5]) if vals else (np.nan, np.nan)


def correlate(df, rater_col, human_col, cluster_col, n_boot=2000):
    """One row of the validation table."""
    d = df[[rater_col, human_col, cluster_col]].dropna()
    out = {"rater": rater_col, "human": human_col, "n": len(d),
           "human_nonzero": int((d[human_col] > 0).sum())}
    if len(d) < 5 or d[rater_col].nunique() < 2 or d[human_col].nunique() < 2:
        return {**out, "spearman": np.nan, "lo": np.nan, "hi": np.nan,
                "pearson": np.nan, "ccc": np.nan, "p": np.nan}
    rho, p = stats.spearmanr(d[rater_col], d[human_col])
    lo, hi = cluster_bootstrap(d[rater_col], d[human_col], d[cluster_col],
                               lambda a, b: stats.spearmanr(a, b)[0], n_boot)
    return {**out, "spearman": rho, "lo": lo, "hi": hi,
            "pearson": stats.pearsonr(d[rater_col], d[human_col])[0],
            "ccc": ccc(d[human_col], d[rater_col]), "p": p}


def to_sessions(df, session_col, value_cols, keep_cols=()):
    """Mean over a session's picture rows: per-image TLI is noisy, and the
    session mean is the scale the TLI was designed for."""
    agg = {c: "mean" for c in value_cols}
    agg.update({c: "first" for c in keep_cols})
    return df.groupby(session_col, as_index=False).agg(agg)
