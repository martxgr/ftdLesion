#!/usr/bin/env python
"""
clinician/crude/validate/rate_api.py -- rate a vector of responses with Claude.

Sends every (response, instrument) pair as one request in a Message Batch
(asynchronous, half price, usually done within an hour). The rater sees the
response only, never the prompt that produced it.

  python clinician/crude/validate/rate_api.py plan      # size + cost; sends nothing
  python clinician/crude/validate/rate_api.py submit    # send the batch
  python clinician/crude/validate/rate_api.py status    # poll
  python clinician/crude/validate/rate_api.py collect   # write ratings once ended

Instruments (clinician/crude/configure.yaml):
  crude  one call per item; question text verbatim from items.csv, 0-10
  tli    one call per response scoring all 8 TLI items (tli_prompt.md)

Outputs, in <output_root>/<name>/:
  ids.csv            request index -> row key (no text)
  state.json         batch id, config snapshot, timestamps
  ratings_long.csv   one row per (response, instrument item), with usage
  ratings.csv        one row per response: claude_<item> columns
Credentials come from the environment (ANTHROPIC_API_KEY, or an `ant auth
login` profile); nothing here stores a key.
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone

import pandas as pd
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
CRUDE = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(CRUDE))

TLI_ITEMS = ["poverty_of_speech", "weakening_of_goal", "perseveration_of_ideas",
             "looseness", "peculiar_use_of_words", "peculiar_sentences",
             "peculiar_logic", "distractibility"]
TLI_LEVELS = ["0", "0.25", "0.5", "1", "2"]


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def resolve(path):
    path = os.path.expanduser(os.path.expandvars(path))
    return path if os.path.isabs(path) else os.path.join(REPO, path)


def load_cfg(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def out_dir(cfg):
    root = cfg[cfg["mode"]]["output_root"]
    return os.path.join(resolve(root), cfg["name"])


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
    want = cfg["instruments"].get("crude", {}).get("items") or []
    missing = set(want) - set(items.variable)
    if missing:
        raise SystemExit(f"items not in items.csv: {sorted(missing)}")
    return items.set_index("variable").loc[want, "question"].to_dict()


def crude_schema():
    return {"type": "object",
            "properties": {"score": {"type": "integer", "enum": list(range(11))}},
            "required": ["score"], "additionalProperties": False}


def tli_schema():
    return {"type": "object",
            "properties": {k: {"type": "string", "enum": TLI_LEVELS} for k in TLI_ITEMS},
            "required": TLI_ITEMS, "additionalProperties": False}


def build_requests(cfg, rows):
    """[(custom_id, instrument, params)]. custom_ids carry only the request
    index, never a participant or row identifier."""
    base = {"model": cfg["model"], "max_tokens": int(cfg["max_tokens"])}
    reqs = []
    questions = crude_items(cfg)
    tli_system = None
    if cfg["instruments"].get("tli"):
        with open(os.path.join(CRUDE, "tli_prompt.md"), encoding="utf-8") as f:
            tli_system = f.read()
    for r in rows.itertuples():
        for var, q in questions.items():
            # The llm_coding call was: question + response + "how the response was
            # prompted". Response-only keeps the question verbatim and drops the prompt.
            reqs.append((f"r{r.idx:05d}-{var}", var, {
                **base,
                "messages": [{"role": "user", "content": f"{q} Here is the response: '{r.text}'"}],
                "output_config": {"effort": cfg["effort"],
                                  "format": {"type": "json_schema", "schema": crude_schema()}},
            }))
        if tli_system:
            reqs.append((f"r{r.idx:05d}-tli", "tli", {
                **base,
                "system": tli_system,
                "messages": [{"role": "user", "content": f"Transcript:\n\n{r.text}"}],
                "output_config": {"effort": cfg["effort"],
                                  "format": {"type": "json_schema", "schema": tli_schema()}},
            }))
    return reqs


def estimate(cfg, reqs):
    price = cfg["prices"][cfg["model"]]
    est_out = cfg["estimate_output_tokens"]
    tin = tout = 0
    for _, inst, p in reqs:
        chars = len(p.get("system", "")) + sum(len(m["content"]) for m in p["messages"]) + 200
        tin += chars / 3.5
        tout += est_out["tli" if inst == "tli" else "crude"]
    cost = (tin * price["input"] + tout * price["output"]) / 1e6 / 2  # batch = half price
    return tin, tout, cost


def client():
    import anthropic
    return anthropic.Anthropic()


def cmd_plan(cfg, args):
    rows = load_rows(cfg)
    reqs = build_requests(cfg, rows)
    tin, tout, cost = estimate(cfg, reqs)
    n_inst = len(crude_items(cfg)) + bool(cfg["instruments"].get("tli"))
    print(f"{cfg['name']}: {len(rows):,} responses x {n_inst} instruments = {len(reqs):,} requests "
          f"-> {cfg['model']} (effort {cfg['effort']}), Batches API")
    print(f"  estimate: {tin/1e6:.2f}M input + {tout/1e6:.2f}M output tokens "
          f"~ ${cost:,.0f} at batch prices (output guess from estimate_output_tokens)")
    print(f"  output dir: {out_dir(cfg)}")
    # show the SHAPE of a request without printing anyone's speech
    cid, inst, p = reqs[0]
    shown = json.loads(json.dumps(p))
    for m in shown["messages"]:
        m["content"] = m["content"].replace(rows.text[0], "<response text>")
    print(f"  example request {cid}:\n" + json.dumps(shown, indent=2)[:1500])


def cmd_submit(cfg, args):
    od = out_dir(cfg)
    state_path = os.path.join(od, "state.json")
    if os.path.exists(state_path) and not args.force:
        with open(state_path) as f:
            st = json.load(f)
        raise SystemExit(f"{od} already has batch {st['batch_id']} -- use status/collect, "
                         "or a new `name` to rate again (--force resubmits)")
    rows = load_rows(cfg)
    reqs = build_requests(cfg, rows)
    os.makedirs(od, exist_ok=True)
    rows[["idx", "key"]].to_csv(os.path.join(od, "ids.csv"), index=False)
    batch = client().messages.batches.create(
        requests=[{"custom_id": cid, "params": p} for cid, _, p in reqs])
    st = {"batch_id": batch.id, "submitted": now(), "n_requests": len(reqs),
          "n_responses": len(rows), "config": cfg}
    with open(state_path, "w") as f:
        json.dump(st, f, indent=2)
    print(f"submitted batch {batch.id}: {len(reqs):,} requests. "
          f"`status` to poll, `collect` once it has ended.")


def read_state(cfg):
    path = os.path.join(out_dir(cfg), "state.json")
    if not os.path.exists(path):
        raise SystemExit(f"nothing submitted yet for {cfg['name']} ({path})")
    with open(path) as f:
        return json.load(f)


def cmd_status(cfg, args):
    st = read_state(cfg)
    b = client().messages.batches.retrieve(st["batch_id"])
    c = b.request_counts
    print(f"{st['batch_id']}: {b.processing_status} | processing {c.processing}, "
          f"succeeded {c.succeeded}, errored {c.errored}, canceled {c.canceled}, expired {c.expired}")


def cmd_collect(cfg, args):
    st = read_state(cfg)
    od = out_dir(cfg)
    cl = client()
    b = cl.messages.batches.retrieve(st["batch_id"])
    if b.processing_status != "ended":
        raise SystemExit(f"batch {st['batch_id']} is still {b.processing_status}")
    ids = pd.read_csv(os.path.join(od, "ids.csv"), keep_default_na=False)
    key_of = dict(zip(ids.idx, ids.key))
    long, tin, tout = [], 0, 0
    for res in cl.messages.batches.results(st["batch_id"]):
        idx_s, inst = res.custom_id.split("-", 1)
        base = {"key": key_of[int(idx_s[1:])], "instrument": "tli" if inst == "tli" else "crude"}
        if res.result.type != "succeeded":
            long.append({**base, "item": inst, "score": None, "status": res.result.type})
            continue
        msg = res.result.message
        tin += msg.usage.input_tokens
        tout += msg.usage.output_tokens
        status = msg.stop_reason
        text = next((blk.text for blk in msg.content if blk.type == "text"), None)
        try:
            data = json.loads(text) if text and status != "refusal" else None
        except json.JSONDecodeError:
            data, status = None, "bad_json"
        if inst == "tli":
            for k in TLI_ITEMS:
                long.append({**base, "item": f"tli_{k}",
                             "score": float(data[k]) if data else None, "status": status})
        else:
            long.append({**base, "item": inst,
                         "score": float(data["score"]) if data else None, "status": status})
    long = pd.DataFrame(long)
    long.to_csv(os.path.join(od, "ratings_long.csv"), index=False)
    wide = long.pivot_table(index="key", columns="item", values="score", aggfunc="first")
    wide.columns = [f"claude_{c}" for c in wide.columns]
    wide.reset_index().to_csv(os.path.join(od, "ratings.csv"), index=False)

    price = cfg["prices"][cfg["model"]]
    cost = (tin * price["input"] + tout * price["output"]) / 1e6 / 2
    st.update({"collected": now(), "input_tokens": tin, "output_tokens": tout,
               "cost_usd_batch": round(cost, 2),
               "status_counts": long.status.value_counts().to_dict()})
    with open(os.path.join(od, "state.json"), "w") as f:
        json.dump(st, f, indent=2)
    print(f"collected {len(wide):,} responses -> {od}/ratings.csv")
    print(f"  status: {st['status_counts']}")
    print(f"  usage: {tin:,} in / {tout:,} out tokens ~ ${cost:,.2f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", choices=["plan", "submit", "status", "collect"])
    ap.add_argument("--config", default=os.path.join(CRUDE, "configure.yaml"))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    cfg = load_cfg(args.config)
    {"plan": cmd_plan, "submit": cmd_submit, "status": cmd_status,
     "collect": cmd_collect}[args.cmd](cfg, args)


if __name__ == "__main__":
    main()
