#!/usr/bin/env python
"""
clinician/crude/validate/report.py -- how well does the rater track human TLI?

Joins <output_root>/<name>/ratings.csv to the human TLI scores and writes
validation.csv, with Spearman rho (95% CI by participant-cluster bootstrap),
Pearson and CCC for:
  - each rater TLI item vs the same human item
  - rater TLI composites vs human composites (published definitions:
    impoverishment = poverty + weakening of goal; disorganisation = looseness,
    peculiar words, peculiar sentences, peculiar logic, distractibility;
    total = the two; perseveration belongs to neither)
  - each crude item, and their mean, vs human disorganisation and total
at two levels: picture rows, and sessions (mean over a session's pictures),
overall and split by dx_group. With the local backend the expected score (ev)
is used unless report.use_ev is false. Human items with almost no non-zero
ratings are reported but can't validate anything -- see human_nonzero.

  python clinician/crude/validate/report.py [--config ...]
"""

import argparse
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import instruments as ins  # noqa: E402
from common.config import load, resolve  # noqa: E402
from common.validation import correlate, to_sessions  # noqa: E402

IMPOVERISHMENT = ["poverty_of_speech", "weakening_of_goal"]
DISORGANISATION = ["looseness", "peculiar_use_of_words", "peculiar_sentences",
                   "peculiar_logic", "distractibility"]


def build(cfg):
    od = ins.out_dir(cfg)
    r = pd.read_csv(os.path.join(od, "ratings.csv"))
    ev = [c for c in r if c.startswith("rater_ev_")]
    if ev and (cfg.get("report") or {}).get("use_ev", True):
        r = r.drop(columns=[c.replace("rater_ev_", "rater_") for c in ev])
        r = r.rename(columns={c: c.replace("rater_ev_", "rater_") for c in ev})
    else:
        r = r.drop(columns=ev)
    v = cfg["validate"]
    h = pd.read_csv(resolve(v["data"]))
    h = h[h.row_type == v["row_type"]] if v.get("row_type") else h
    h = h.drop(columns=[c for c in ("prompt", "response", "exchange") if c in h])
    d = h.merge(r, left_on="row_id", right_on="key", how="inner")

    tli = [f"rater_tli_{k}" for k in ins.TLI_ITEMS if f"rater_tli_{k}" in d]
    if len(tli) == len(ins.TLI_ITEMS):
        d["rater_tli_impoverishment"] = d[[f"rater_tli_{k}" for k in IMPOVERISHMENT]].sum(axis=1, min_count=2)
        d["rater_tli_disorganisation"] = d[[f"rater_tli_{k}" for k in DISORGANISATION]].sum(axis=1, min_count=5)
        d["rater_tli_total"] = d.rater_tli_impoverishment + d.rater_tli_disorganisation
    crude = [c for c in r if c.startswith("rater_") and not c.startswith("rater_tli")]
    if crude:
        d["rater_crude_mean"] = d[crude].mean(axis=1)
    return d, tli, crude, od


def pairs(d, tli, crude):
    out = [(c, "item_" + c[len("rater_tli_"):]) for c in tli]
    out += [("rater_tli_total", "tli_total"),
            ("rater_tli_disorganisation", "tli_disorganisation"),
            ("rater_tli_impoverishment", "tli_impoverishment")]
    for c in crude + (["rater_crude_mean"] if crude else []):
        out += [(c, "tli_disorganisation"), (c, "tli_total")]
    return [(a, b) for a, b in out if a in d and b in d]


def run(cfg, n_boot=2000):
    if cfg["mode"] != "validate":
        raise SystemExit("report.py validates against human TLI: set mode: validate")
    d, tli, crude, od = build(cfg)
    prs = pairs(d, tli, crude)
    value_cols = sorted({c for p in prs for c in p})
    sess = to_sessions(d, "session_key", value_cols, keep_cols=["participant", "dx_group"])
    rows = []
    for level, frame in (("picture", d), ("session", sess)):
        for group in ("all", "patient", "control"):
            f = frame if group == "all" else frame[frame.dx_group == group]
            for a, b in prs:
                rows.append({"level": level, "group": group,
                             **correlate(f, a, b, "participant", n_boot)})
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(od, "validation.csv"), index=False)

    show = res[res.group == "all"].copy()
    show["rho [95% CI]"] = show.apply(lambda r: f"{r.spearman:+.2f} [{r.lo:+.2f}, {r.hi:+.2f}]", axis=1)
    pd.set_option("display.width", 200)
    for level in ("picture", "session"):
        s = show[show.level == level]
        print(f"\n== {level} level, n={s.n.max()} ==")
        print(s[["rater", "human", "rho [95% CI]", "pearson", "ccc", "human_nonzero"]]
              .round(2).to_string(index=False))
    print(f"\nfull table (incl. patient / control splits): {od}/validation.csv")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ins.CRUDE, "configure.yaml"))
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()
    run(load(args.config), args.n_boot)


if __name__ == "__main__":
    main()
