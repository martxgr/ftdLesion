#!/usr/bin/env python
"""
clinician/crude/validate/aggregate.py -- which combination of the rater's items
best tracks human per-image disorganisation? Freeze it for scoring the client.

Candidates, all scored on held-out participants (10-fold grouped CV, 5 repeats):
  tli_disorg_sum   Llama's 5 TLI disorganisation items, summed (no fitting)
  tli_total_sum    Llama's TLI impoverishment + disorganisation, summed
  crude_mean       mean of the crude items in the pool
  pool_z_mean      every pooled item z-scored, equal weights
  pool_nnls        every pooled item, non-negative weights fit to the target
  tli_nnls         the 8 TLI items only, non-negative weights

For each: out-of-sample Spearman rho with the target (95% CI by participant
bootstrap), rho at session level, rho with word count, and the partial rho
after removing impoverishment and word count -- a score that only tracks
length is no use once client outputs are length-matched.

The chosen model (aggregate.choose, default pool_nnls) is refit on all rows
and written to <output_root>/<name>/aggregate.json for the client.

  python clinician/crude/validate/aggregate.py [--config ...]
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import instruments as ins  # noqa: E402
from common import aggregate as agg  # noqa: E402
from common.config import load, resolve  # noqa: E402
from common.validation import cluster_bootstrap  # noqa: E402

DISORG = ["looseness", "peculiar_use_of_words", "peculiar_sentences", "peculiar_logic",
          "distractibility"]
IMPOV = ["poverty_of_speech", "weakening_of_goal"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ins.CRUDE, "configure.yaml"))
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--n-boot", type=int, default=500)
    args = ap.parse_args()
    cfg = load(args.config)
    a = {"target": "tli_disorganisation", "exclude": ["non_sequitur", "tangential"],
         "choose": "pool_nnls"}
    a.update(cfg.get("aggregate") or {})

    od = ins.out_dir(cfg)
    r = pd.read_csv(os.path.join(od, "ratings.csv"))
    pre = "rater_ev_" if any(c.startswith("rater_ev_") for c in r) else "rater_"
    v = cfg["validate"]
    h = pd.read_csv(resolve(v["data"]))
    h = h[h.row_type == v["row_type"]] if v.get("row_type") else h
    h = h.drop(columns=[c for c in ("prompt", "response", "exchange") if c in h])
    d = h.merge(r, left_on="row_id", right_on="key").dropna(subset=[a["target"]])

    items = [c[len(pre):] for c in r if c.startswith(pre)]
    pool = [f"{pre}{i}" for i in items if i not in a["exclude"]]
    tli = [f"{pre}tli_{k}" for k in DISORG + IMPOV + ["perseveration_of_ideas"] if f"{pre}tli_{k}" in d]
    crude = [c for c in pool if not c[len(pre):].startswith("tli_")]
    y, g = d[a["target"]].to_numpy(float), d.participant.to_numpy()
    words, impov = d.n_words_participant.to_numpy(float), d.tli_impoverishment.to_numpy(float)

    fixed = {
        "tli_disorg_sum": lambda D: D[[f"{pre}tli_{k}" for k in DISORG]].sum(axis=1).to_numpy(),
        "tli_total_sum": lambda D: D[[f"{pre}tli_{k}" for k in DISORG + IMPOV]].sum(axis=1).to_numpy(),
        "crude_mean": lambda D: D[crude].mean(axis=1).to_numpy(),
        "pool_z_mean": lambda D: ((D[pool] - D[pool].mean()) / D[pool].std()).mean(axis=1).to_numpy(),
    }
    fitted = {"pool_nnls": pool, "tli_nnls": tli}

    rows, preds = [], {}
    for name, fn in fixed.items():
        preds[name] = [fn(d)] * args.repeats
    for name, feats in fitted.items():
        X = d[feats].to_numpy(float)
        preds[name] = [agg.oof(X, y, g, agg.fit_nnls, k=10, seed=s) for s in range(args.repeats)]

    for name, ps in preds.items():
        rhos = [spearmanr(p, y)[0] for p in ps]
        p0 = ps[0]
        lo, hi = cluster_bootstrap(p0, y, g, lambda x, t: spearmanr(x, t)[0], args.n_boot)
        sess = pd.DataFrame({"s": d.session_key, "p": np.mean(ps, axis=0), "y": y}).groupby("s").mean()
        rows.append({"model": name, "rho_oos": np.mean(rhos), "lo": lo, "hi": hi,
                     "rho_session": spearmanr(sess.p, sess.y)[0],
                     "rho_words": spearmanr(p0, words)[0],
                     "partial_impov_words": agg.rank_partial(p0, y, [impov, words])})
    res = pd.DataFrame(rows).sort_values("rho_oos", ascending=False)
    res.to_csv(os.path.join(od, "aggregate_cv.csv"), index=False)

    pd.set_option("display.width", 200)
    print(f"target: human per-image {a['target']} | n={len(d)} images, "
          f"{d.participant.nunique()} participants | pool excludes {a['exclude']}")
    print("all rho below are out of sample (held-out participants)\n")
    print(res.round(3).to_string(index=False))

    feats = fitted.get(a["choose"])
    if feats:
        final = agg.fit_nnls(d[feats].to_numpy(float), y)
        out = os.path.join(od, "aggregate.json")
        agg.save(final, feats, out, {
            "target": a["target"], "fit_on": f"{cfg['name']} validate, {len(d)} images",
            "model": a["choose"],
            "cv": res[res.model == a["choose"]].round(4).to_dict("records")[0]})
        w = pd.Series(final["weights"], index=[f[len(pre):] for f in feats]).sort_values(ascending=False)
        print(f"\n{a['choose']} weights (on z-scored items), refit on all rows -> {out}")
        print(w.round(3).to_string())


if __name__ == "__main__":
    main()
