#!/usr/bin/env python
"""
clinician/crude/validate/rate_api.py -- rate a vector of responses with Claude.

The alternative to rate_local.py for when API access is available. Sends
every question as one request in a Message Batch (asynchronous, half price,
usually done within an hour). Same questions as the local backend
(instruments.py), except TLI is one request per response returning all eight
items as JSON.

  python clinician/crude/validate/rate_api.py plan      # size + cost; sends nothing
  python clinician/crude/validate/rate_api.py submit    # send the batch
  python clinician/crude/validate/rate_api.py status    # poll
  python clinician/crude/validate/rate_api.py collect   # write ratings once ended

Credentials come from the environment (ANTHROPIC_API_KEY, or an `ant auth
login` profile); nothing here stores a key. custom_ids carry only a request
index, never a participant or row identifier.
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import instruments as ins  # noqa: E402
from common.config import load  # noqa: E402


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def schema(q):
    if q["item"] == "tli":
        return {"type": "object",
                "properties": {k: {"type": "string", "enum": [f"{v:g}" for v in ins.TLI_LEVELS]}
                               for k in ins.TLI_ITEMS},
                "required": ins.TLI_ITEMS, "additionalProperties": False}
    return {"type": "object",
            "properties": {"score": {"type": "integer", "enum": list(range(11))}},
            "required": ["score"], "additionalProperties": False}


def requests(cfg, rows):
    a = cfg["api"]
    out = []
    for q in ins.questions(cfg, rows, tli_per_item=False):
        p = {"model": a["model"], "max_tokens": int(a["max_tokens"]),
             "messages": [{"role": "user", "content": q["user"]}],
             "output_config": {"effort": a["effort"],
                               "format": {"type": "json_schema", "schema": schema(q)}}}
        if q["system"]:
            p["system"] = q["system"]
        out.append((q, p))
    return out


def estimate(cfg, reqs):
    a = cfg["api"]
    price, est = a["prices"][a["model"]], a["estimate_output_tokens"]
    tin = sum((len(p.get("system", "")) + len(p["messages"][0]["content"]) + 200) / 3.5
              for _, p in reqs)
    tout = sum(est["tli" if q["item"] == "tli" else "crude"] for q, _ in reqs)
    return tin, tout, (tin * price["input"] + tout * price["output"]) / 1e6 / 2


def client():
    import anthropic
    return anthropic.Anthropic()


def cmd_plan(cfg, args):
    rows = ins.load_rows(cfg)
    reqs = requests(cfg, rows)
    tin, tout, cost = estimate(cfg, reqs)
    print(f"{cfg['name']}: {len(rows):,} responses -> {len(reqs):,} requests to "
          f"{cfg['api']['model']} (effort {cfg['api']['effort']}), Batches API")
    print(f"  estimate: {tin/1e6:.2f}M input + {tout/1e6:.2f}M output tokens ~ ${cost:,.0f} "
          f"at batch prices")
    print(f"  output dir: {ins.out_dir(cfg)}")
    q, p = reqs[0]
    shown = json.loads(json.dumps(p))
    shown["messages"][0]["content"] = shown["messages"][0]["content"].replace(rows.text[0], "<response text>")
    print(f"  example request {q['cid']}:\n" + json.dumps(shown, indent=2)[:1200])


def state_path(cfg):
    return os.path.join(ins.out_dir(cfg), "state.json")


def cmd_submit(cfg, args):
    od = ins.out_dir(cfg)
    if os.path.exists(state_path(cfg)) and not args.force:
        raise SystemExit(f"{od} already has a batch -- use status/collect, or a new `name`")
    rows = ins.load_rows(cfg)
    reqs = requests(cfg, rows)
    os.makedirs(od, exist_ok=True)
    rows[["idx", "key"]].to_csv(os.path.join(od, "ids.csv"), index=False)
    batch = client().messages.batches.create(
        requests=[{"custom_id": q["cid"], "params": p} for q, p in reqs])
    with open(state_path(cfg), "w") as f:
        json.dump({"batch_id": batch.id, "backend": "api", "submitted": now(),
                   "n_requests": len(reqs), "config": cfg}, f, indent=2)
    print(f"submitted batch {batch.id}: {len(reqs):,} requests")


def read_state(cfg):
    if not os.path.exists(state_path(cfg)):
        raise SystemExit(f"nothing submitted yet for {cfg['name']}")
    with open(state_path(cfg)) as f:
        return json.load(f)


def cmd_status(cfg, args):
    st = read_state(cfg)
    b = client().messages.batches.retrieve(st["batch_id"])
    c = b.request_counts
    print(f"{st['batch_id']}: {b.processing_status} | processing {c.processing}, "
          f"succeeded {c.succeeded}, errored {c.errored}, canceled {c.canceled}, expired {c.expired}")


def cmd_collect(cfg, args):
    st = read_state(cfg)
    od = ins.out_dir(cfg)
    cl = client()
    if cl.messages.batches.retrieve(st["batch_id"]).processing_status != "ended":
        raise SystemExit("batch has not ended yet")
    ids = pd.read_csv(os.path.join(od, "ids.csv"), keep_default_na=False)
    key_of = dict(zip(ids.idx, ids.key))
    long, tin, tout = [], 0, 0
    for res in cl.messages.batches.results(st["batch_id"]):
        idx_s, item = res.custom_id.split("-", 1)
        key = key_of[int(idx_s[1:])]
        if res.result.type != "succeeded":
            long.append({"key": key, "item": item, "score": None, "status": res.result.type})
            continue
        msg = res.result.message
        tin, tout = tin + msg.usage.input_tokens, tout + msg.usage.output_tokens
        status = msg.stop_reason
        text = next((blk.text for blk in msg.content if blk.type == "text"), None)
        try:
            data = json.loads(text) if text and status != "refusal" else None
        except json.JSONDecodeError:
            data, status = None, "bad_json"
        if item == "tli":
            for k in ins.TLI_ITEMS:
                long.append({"key": key, "item": f"tli_{k}", "status": status,
                             "score": float(data[k]) if data else None})
        else:
            long.append({"key": key, "item": item, "status": status,
                         "score": float(data["score"]) if data else None})
    long = pd.DataFrame(long)
    long.to_csv(os.path.join(od, "ratings_long.csv"), index=False)
    ins.write_wide(long, od)
    a = cfg["api"]
    price = a["prices"][a["model"]]
    cost = (tin * price["input"] + tout * price["output"]) / 1e6 / 2
    st.update({"collected": now(), "input_tokens": tin, "output_tokens": tout,
               "cost_usd_batch": round(cost, 2), "status_counts": long.status.value_counts().to_dict()})
    with open(state_path(cfg), "w") as f:
        json.dump(st, f, indent=2)
    print(f"collected -> {od}/ratings.csv | {st['status_counts']} | ~${cost:,.2f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", choices=["plan", "submit", "status", "collect"])
    ap.add_argument("--config", default=os.path.join(ins.CRUDE, "configure.yaml"))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    cfg = load(args.config)
    {"plan": cmd_plan, "submit": cmd_submit, "status": cmd_status,
     "collect": cmd_collect}[args.cmd](cfg, args)


if __name__ == "__main__":
    main()
