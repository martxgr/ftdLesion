#!/usr/bin/env python
"""
clinician/crude/validate/report.py -- how well does the API rater track human TLI?

Joins <output_root>/<name>/ratings.csv to the human TLI scores and writes
validation.csv, with Spearman rho (95% CI by participant-cluster bootstrap),
Pearson and CCC for:
  - each Claude TLI item vs the same human item
  - Claude TLI composites vs human composites (published definitions:
    impoverishment = poverty + weakening of goal; disorganisation = looseness,
    peculiar words, peculiar sentences, peculiar logic, distractibility;
    total = the two; perseveration belongs to neither)
  - each crude item, and their mean, vs human disorganisation and total
at two levels: picture rows, and sessions (mean over a session's pictures).
Also split by dx_group. Human items with almost no non-zero ratings are
reported but can't validate anything -- see human_nonzero.

  python clinician/crude/validate/report.py [--config ...]
"""

import argparse
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", ".."))
import rate_api  # noqa: E402
from common.validation import correlate, to_sessions  # noqa: E402

IMPOVERISHMENT = ["poverty_of_speech", "weakening_of_goal"]
DISORGANISATION = ["looseness", "peculiar_use_of_words", "peculiar_sentences",
                   "peculiar_logic", "distractibility"]


def build(cfg):
    od = rate_api.out_dir(cfg)
    r = pd.read_csv(os.path.join(od, "ratings.csv"))
    v = cfg["validate"]
    h = pd.read_csv(rate_api.resolve(v["data"]))
    h = h[h.row_type == v["row_type"]] if v.get("row_type") else h
    h = h.drop(columns=[c for c in ("prompt", "response", "exchange") if c in h])
    d = h.merge(r, left_on="row_id", right_on="key", how="inner")

    tli = [f"claude_tli_{k}" for k in rate_api.TLI_ITEMS if f"claude_tli_{k}" in d]
    if tli:
        d["claude_tli_impoverishment"] = d[[f"claude_tli_{k}" for k in IMPOVERISHMENT]].sum(axis=1, min_count=2)
        d["claude_tli_disorganisation"] = d[[f"claude_tli_{k}" for k in DISORGANISATION]].sum(axis=1, min_count=5)
        d["claude_tli_total"] = d.claude_tli_impoverishment + d.claude_tli_disorganisation
    crude = [c for c in d if c.startswith("claude_") and not c.startswith("claude_tli")]
    if crude:
        d["claude_crude_mean"] = d[crude].mean(axis=1)
    return d, tli, crude, od


def pairs(d, tli, crude):
    out = [(c, "item_" + c[len("claude_tli_"):]) for c in tli]
    if tli:
        out += [("claude_tli_total", "tli_total"),
                ("claude_tli_disorganisation", "tli_disorganisation"),
                ("claude_tli_impoverishment", "tli_impoverishment")]
    for c in crude + (["claude_crude_mean"] if crude else []):
        out += [(c, "tli_disorganisation"), (c, "tli_total")]
    return [(a, b) for a, b in out if a in d and b in d]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(rate_api.CRUDE, "configure.yaml"))
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()
    cfg = rate_api.load_cfg(args.config)
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
                             **correlate(f, a, b, "participant", args.n_boot)})
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


if __name__ == "__main__":
    main()
