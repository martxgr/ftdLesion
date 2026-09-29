#!/usr/bin/env python
"""
clinician/crude/validate/rate_local.py -- rate a vector of responses with an
open model on our own GPUs (default Llama-3.3-70B-Instruct, already cached on
Bouchet).

No text is generated. Each question ends at the start of the model's answer,
and one forward pass gives the next-token distribution over the allowed answer
codes (0..10 for crude items, 0..4 for TLI). From it:
  score   the most likely answer
  ev      the probability-weighted mean answer -- continuous, better for correlation
  p_mass  probability on valid answers at all; low means the model wanted to
          say something else, so treat that score with suspicion

Resumable: rows already in ratings_long.csv are skipped.

  python clinician/crude/validate/rate_local.py [--config ...] [--report]
"""

import argparse
import json
import os
import platform
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import instruments as ins  # noqa: E402
from common.config import load  # noqa: E402


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class LocalRater:
    def __init__(self, lc):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        print(f"Loading {lc['model']} ...", flush=True)
        kw = {"dtype": getattr(torch, lc.get("dtype", "bfloat16"))}
        if lc.get("device_map"):
            kw["device_map"] = lc["device_map"]
        self.model = AutoModelForCausalLM.from_pretrained(lc["model"], **kw).eval()
        self.tok = AutoTokenizer.from_pretrained(lc["model"])
        self.tok.pad_token = self.tok.pad_token or self.tok.eos_token
        self.tok.padding_side = "left"   # last position = next token for every row
        self.max_length = int(lc.get("max_length", 4096))
        self.device = self.model.get_input_embeddings().weight.device
        self._code_ids = {}

    def code_ids(self, n):
        """Token ids for answer codes 0..n-1 (with and without a leading space).
        Every code must be a single token, or reading it off one position is wrong."""
        if n not in self._code_ids:
            ids = []
            for k in range(n):
                variants = [self.tok.encode(v, add_special_tokens=False) for v in (str(k), f" {k}")]
                single = sorted({v[0] for v in variants if len(v) == 1})
                if not single:
                    raise SystemExit(f"answer code {k} is not a single token for this tokenizer")
                ids.append(single)
            self._code_ids[n] = ids
        return self._code_ids[n]

    def encode(self, q):
        msgs = ([{"role": "system", "content": q["system"]}] if q["system"] else []) + \
               [{"role": "user", "content": q["user"]}]
        return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    def score(self, qs):
        torch = self.torch
        texts = [self.encode(q) for q in qs]
        enc = self.tok(texts, return_tensors="pt", padding=True, add_special_tokens=False)
        if enc["input_ids"].shape[1] > self.max_length:
            raise SystemExit(f"a prompt is {enc['input_ids'].shape[1]} tokens > max_length="
                             f"{self.max_length}; raise it rather than truncate")
        with torch.no_grad():
            logits = self.model(**enc.to(self.device)).logits[:, -1, :].float()
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        out = []
        for q, p in zip(qs, probs):
            ids = self.code_ids(len(q["levels"]))
            pk = np.array([p[i].sum() for i in ids])
            mass = float(pk.sum())
            pk = pk / mass if mass > 0 else pk
            levels = np.array(q["levels"])
            out.append({"score": float(levels[int(pk.argmax())]),
                        "ev": float((pk * levels).sum()), "p_mass": mass,
                        "n_tokens": int(enc["attention_mask"][len(out)].sum())})
        return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", default=os.path.join(ins.CRUDE, "configure.yaml"))
    ap.add_argument("--report", action="store_true", help="run report.py afterwards (validate mode)")
    args = ap.parse_args()
    cfg = load(args.config)
    rows = ins.load_rows(cfg)
    qs = ins.questions(cfg, rows, tli_per_item=True)
    od = ins.out_dir(cfg)
    os.makedirs(od, exist_ok=True)
    long_path = os.path.join(od, "ratings_long.csv")
    done = set()
    if os.path.exists(long_path):
        prev = pd.read_csv(long_path, keep_default_na=False)
        done = set(zip(prev.key.astype(str), prev.item))
    todo = [q for q in qs if (str(q["key"]), q["item"]) not in done]
    print(f"{cfg['name']}: {len(rows):,} responses, {len(qs):,} questions, "
          f"{len(qs) - len(todo):,} already done -> {od}", flush=True)

    state = {"name": cfg["name"], "backend": "local", "model": cfg["local"]["model"],
             "started": now(), "host": platform.node(), "config": cfg,
             "slurm_job": os.environ.get("SLURM_JOB_ID")}
    if todo:
        rater = LocalRater(cfg["local"])
        B = int(cfg["local"].get("batch_size", 8))
        # similar lengths together -> less padding
        todo.sort(key=lambda q: len(q["user"]) + len(q["system"] or ""))
        for b in range(0, len(todo), B):
            chunk = todo[b:b + B]
            res = rater.score(chunk)
            pd.DataFrame([{"key": q["key"], "item": q["item"],
                           "instrument": "tli" if q["item"].startswith("tli_") else "crude",
                           **r} for q, r in zip(chunk, res)]).to_csv(
                long_path, mode="a", header=not os.path.exists(long_path), index=False)
            if (b // B) % 50 == 0:
                print(f"  {b + len(chunk):,}/{len(todo):,}", flush=True)

    long = pd.read_csv(long_path, keep_default_na=False)
    ins.write_wide(long, od)
    state.update({"finished": now(), "n_questions": len(long),
                  "p_mass_median": float(long.p_mass.median()),
                  "p_mass_below_0.5": int((long.p_mass < 0.5).sum())})
    with open(os.path.join(od, "state.json"), "w") as f:
        json.dump(state, f, indent=2, default=str)
    print(f"done: {len(long):,} ratings -> {od}/ratings.csv "
          f"(median p_mass {state['p_mass_median']:.2f}; "
          f"{state['p_mass_below_0.5']} answers below 0.5)", flush=True)

    if args.report and cfg["mode"] == "validate":
        import report
        report.run(cfg)


if __name__ == "__main__":
    main()
