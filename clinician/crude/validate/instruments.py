"""
clinician/crude/validate/instruments.py -- what the crude rater asks, for every backend.

Both backends (rate_api.py, rate_local.py) build their questions here, so the
wording can't drift between them. The rater sees the response only.
"""

import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CRUDE = os.path.dirname(HERE)
sys.path.insert(0, os.path.dirname(CRUDE))
from common.config import resolve  # noqa: E402

TLI_ITEMS = ["poverty_of_speech", "weakening_of_goal", "perseveration_of_ideas",
             "looseness", "peculiar_use_of_words", "peculiar_sentences",
             "peculiar_logic", "distractibility"]
TLI_LEVELS = [0.0, 0.25, 0.5, 1.0, 2.0]       # answer code i -> TLI_LEVELS[i]
CRUDE_LEVELS = [float(k) for k in range(11)]  # answer code k -> k


def out_dir(cfg):
    return os.path.join(resolve(cfg[cfg["mode"]]["output_root"]), cfg["name"])


def load_rows(cfg):
    """DataFrame [idx, key, text] of what will be rated."""
    if cfg["mode"] == "validate":
        v = cfg["validate"]
        d = pd.read_csv(resolve(v["data"]), keep_default_na=False)
        d = d[d.row_type == v["row_type"]] if v.get("row_type") else d
        rows = pd.DataFrame({"key": d.row_id, "text": d[v["text_col"]].astype(str)})
        limit = v.get("limit")
    elif cfg["mode"] == "rate":
        r = cfg["rate"]
        d = pd.read_csv(resolve(r["run_csv"]), keep_default_na=False)
        rows = pd.DataFrame({"key": d.row_id, "text": d[r["text_col"]].astype(str)})
        limit = None
    else:
        raise SystemExit(f"mode must be validate or rate, not {cfg['mode']!r}")
    rows = rows[rows.text.str.strip() != ""].reset_index(drop=True)
    if limit:
        rows = rows.head(int(limit))
    rows.insert(0, "idx", range(len(rows)))
    return rows


def crude_items(cfg):
    items = pd.read_csv(os.path.join(CRUDE, "items.csv"))
    want = (cfg["instruments"].get("crude") or {}).get("items") or []
    missing = set(want) - set(items.variable)
    if missing:
        raise SystemExit(f"items not in items.csv: {sorted(missing)}")
    return items.set_index("variable").loc[want, "question"].to_dict()


def crude_user(question, text):
    # llm_coding sent: question + response + "how the response was prompted".
    # Response-only keeps the question verbatim and drops the prompt.
    return f"{question} Here is the response: '{text}'"


def tli_system(one_item=False):
    with open(os.path.join(CRUDE, "tli_prompt.md"), encoding="utf-8") as f:
        s = f.read().rstrip()
    if one_item:
        s = s.replace("Return the eight scores.", "You will be asked for one item at a time.")
    return s


def tli_item_user(text, item):
    codes = ", ".join(f"{i} = {lv:g}" for i, lv in enumerate(TLI_LEVELS))
    return (f"Transcript:\n\n{text}\n\nScore only this item: {item}. "
            f"Answer with a single digit code ({codes}).")


def questions(cfg, rows, tli_per_item):
    """One dict per question: cid, idx, key, item, levels, system, user.
    tli_per_item=False gives one TLI question per response (API, JSON answer)."""
    qs = []
    crude = crude_items(cfg)
    use_tli = bool(cfg["instruments"].get("tli"))
    sys_all, sys_one = (tli_system(False), tli_system(True)) if use_tli else (None, None)
    for r in rows.itertuples():
        base = {"idx": r.idx, "key": r.key}
        for var, q in crude.items():
            qs.append({**base, "cid": f"r{r.idx:05d}-{var}", "item": var,
                       "levels": CRUDE_LEVELS, "system": None, "user": crude_user(q, r.text)})
        if not use_tli:
            continue
        if tli_per_item:
            for k in TLI_ITEMS:
                qs.append({**base, "cid": f"r{r.idx:05d}-tli_{k}", "item": f"tli_{k}",
                           "levels": TLI_LEVELS, "system": sys_one, "user": tli_item_user(r.text, k)})
        else:
            qs.append({**base, "cid": f"r{r.idx:05d}-tli", "item": "tli", "levels": None,
                       "system": sys_all, "user": f"Transcript:\n\n{r.text}"})
    return qs


def write_wide(long, od):
    """ratings.csv: one row per response, rater_<item> (+ rater_ev_<item>)."""
    wide = long.pivot_table(index="key", columns="item", values="score", aggfunc="first")
    wide.columns = [f"rater_{c}" for c in wide.columns]
    if "ev" in long and long.ev.notna().any():
        ev = long.pivot_table(index="key", columns="item", values="ev", aggfunc="first")
        ev.columns = [f"rater_ev_{c}" for c in ev.columns]
        wide = wide.join(ev)
    wide.reset_index().to_csv(os.path.join(od, "ratings.csv"), index=False)
    return wide
