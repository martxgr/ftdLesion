#!/usr/bin/env python
"""
clinician/fine_tune/validate/score.py -- score client outputs with the fine-tuned
TLI model. Ported from llmSchizophrenia/0_finetunerun/score_ratings.py: the
model loading, seed averaging and `scale` handling are unchanged.

  plan     login node, no GPU. Stacks the client runs, length-matches every
           output to human picture descriptions, writes to_score.csv (exactly
           the text the model will see), checks each adapter was trained on
           the same text column, and reports token lengths.
  score    one chunk of rows (default: this SLURM array task), every seed.
           Resumable per chunk. The last chunk to finish runs `merge`.
  merge    stack the chunks into ratings.csv + ratings.log.json.

Head outputs (train_ftd.py): eight TLI items, then [8] the per-image GLOBAL
disorganisation rating -- the target the model was trained on. Everything is
returned on the TLI scale (logits x each adapter's own `scale`), averaged over
seeds, with the across-seed SD.

  python clinician/fine_tune/validate/score.py plan|score|merge [--config ...]
"""

import argparse
import glob
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
FT = os.path.dirname(HERE)
sys.path.insert(0, os.path.dirname(FT))
from common import length_match  # noqa: E402
from common.config import REPO, load, resolve  # noqa: E402

# head output order, from llmSchizophrenia/0_finetunerun/ftd_common.py ITEMS
ITEMS = ["poverty_of_speech", "weakening_of_goal", "looseness", "peculiar_use_of_words",
         "peculiar_sentences", "peculiar_logic", "perseveration_of_ideas", "distractibility"]
GLOBAL_IDX = 8
META = ["row_id", "run", "model", "phase", "prompt_id", "rep", "manipulation", "side",
        "param_name", "param_value", "layer_strategy", "layer_indices", "n_words", "stop_reason"]


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def out_dir(cfg):
    return os.path.join(resolve(cfg["output_root"]), cfg["name"])


# --------------------------------------------------------------------------- #
# Adapters
# --------------------------------------------------------------------------- #
def adapters(cfg):
    pat = resolve(cfg["adapters"])
    dirs = sorted(d for d in glob.glob(pat) if os.path.isdir(d))
    if not dirs:
        raise SystemExit(f"no adapter directories match {pat}")
    out = []
    for d in dirs:
        rj = next((p for p in (os.path.join(d, "..", "results.json"), os.path.join(d, "results.json"))
                   if os.path.exists(p)), None)
        if rj is None:
            raise SystemExit(f"no results.json next to {d}: can't recover scale or training text")
        with open(rj) as f:
            res = json.load(f)
        targs = res.get("args", {})
        tcol = targs.get("text_col", "response")
        if tcol != cfg["expect_text_col"]:
            raise SystemExit(f"{d} was trained on text_col={tcol!r}, config expects "
                             f"{cfg['expect_text_col']!r}: scoring other text would be off-format")
        name = os.path.basename(os.path.dirname(os.path.normpath(d)))
        out.append({"dir": d, "name": name, "scale": float(res["scale"]),
                    "scalar_head": bool(targs.get("scalar_head", False)),
                    "target_col": targs.get("target_col"), "train": targs.get("train"),
                    "max_length": targs.get("max_length")})
    if len({a["scalar_head"] for a in out}) > 1:
        raise SystemExit("adapters disagree on scalar_head; score them separately")
    return out


# --------------------------------------------------------------------------- #
# Input
# --------------------------------------------------------------------------- #
def human_words(cfg):
    lm = cfg["length_match"]
    h = pd.read_csv(resolve(lm["reference"]), usecols=["row_type", "dx_group", "n_words_participant"])
    h = h[h.row_type == lm["row_type"]]
    if lm.get("group", "all") != "all":
        h = h[h.dx_group == lm["group"]]
    return h.n_words_participant.dropna().astype(int).to_numpy()


def build_input(cfg):
    frames = []
    for run in cfg["runs"]:
        p = os.path.join(resolve(cfg["results_root"]), run, f"{run}.csv")
        frames.append(pd.read_csv(p, keep_default_na=False, dtype={"output": str}))
    d = pd.concat(frames, ignore_index=True)
    if not d.row_id.is_unique:
        raise SystemExit("row_id collides across runs -- refusing to stack")
    hw = human_words(cfg)
    lm = cfg["length_match"]
    d = length_match.match(d, hw, text_col=cfg["text_col"], seed=lm.get("seed", 0),
                           min_frac=lm.get("min_frac", 0.5))
    keep = [c for c in META if c in d] + ["text_rated", "n_words_rated", "trim_target", "trim"]
    return d[keep], hw


def input_path(cfg):
    return os.path.join(out_dir(cfg), "to_score.csv")


def ensure_input(cfg):
    """to_score.csv is deterministic; build it once (atomically -- array tasks
    may race to it) and always read it back, so every chunk scores the same text."""
    p = input_path(cfg)
    if not os.path.exists(p):
        os.makedirs(out_dir(cfg), exist_ok=True)
        d, _ = build_input(cfg)
        tmp = f"{p}.tmp-{os.getpid()}"
        d.to_csv(tmp, index=False, encoding="utf-8")
        os.replace(tmp, p)
    return pd.read_csv(p, keep_default_na=False, dtype={"text_rated": str})


# --------------------------------------------------------------------------- #
# plan
# --------------------------------------------------------------------------- #
def cmd_plan(cfg, args):
    ads = adapters(cfg)
    d, hw = build_input(cfg)
    os.makedirs(out_dir(cfg), exist_ok=True)
    d.to_csv(input_path(cfg), index=False, encoding="utf-8")
    q = lambda s: f"median {int(np.median(s))}, p75 {int(np.percentile(s, 75))}, max {int(np.max(s))}"
    print(f"{cfg['name']}: {len(d):,} rows from {cfg['runs']}")
    print(f"  words, generated : {q(d.n_words)}")
    print(f"  words, human     : {q(hw)}  ({len(hw)} picture descriptions, group={cfg['length_match'].get('group', 'all')})")
    print(f"  words, rated     : {q(d.n_words_rated)}")
    print(f"  cut              : {d.trim.value_counts().to_dict()}  | empty rows: {(d.n_words_rated == 0).sum()}")
    print(f"  adapters ({len(ads)}): " + ", ".join(f"{a['name']} (scale {a['scale']:.3f})" for a in ads))
    print(f"  trained on       : text_col={cfg['expect_text_col']}, target={ads[0]['target_col']}, "
          f"train={os.path.basename(str(ads[0]['train']))}")
    if not args.no_tokens:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(cfg["model_id"])
        lens = np.array([len(x) for x in tok(d.text_rated.tolist())["input_ids"]])
        print(f"  tokens           : median {int(np.median(lens))}, p95 {int(np.percentile(lens, 95))}, "
              f"max {lens.max()} | over max_length={cfg['max_length']}: {(lens > cfg['max_length']).mean():.2%}")
    print(f"  -> {input_path(cfg)}")


# --------------------------------------------------------------------------- #
# score
# --------------------------------------------------------------------------- #
def load_model(cfg, ads):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg["model_id"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"          # SeqCls pools the last non-pad token
    kw = {"num_labels": 1 if ads[0]["scalar_head"] else 9,
          "torch_dtype": getattr(torch, cfg.get("dtype", "bfloat16"))}
    if cfg.get("quant_4bit", True):
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    if cfg.get("device_map"):
        kw["device_map"] = cfg["device_map"]
    base = AutoModelForSequenceClassification.from_pretrained(cfg["model_id"], **kw)
    base.config.pad_token_id = tok.pad_token_id
    base.eval()
    model = None
    for a in ads:
        if model is None:
            model = PeftModel.from_pretrained(base, a["dir"], adapter_name=a["name"])
        else:
            model.load_adapter(a["dir"], adapter_name=a["name"])
    model.eval()
    return model, tok


def score_texts(model, tok, texts, max_length, batch_size):
    import torch
    order = np.argsort([len(t) for t in texts])          # similar lengths together
    out = np.zeros((len(texts), model.config.num_labels), dtype=np.float32)
    dev = model.get_input_embeddings().weight.device
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            idx = order[i:i + batch_size]
            enc = tok([texts[j] for j in idx], truncation=True, max_length=max_length,
                      padding=True, return_tensors="pt").to(dev)
            out[idx] = model(**enc).logits.float().cpu().numpy()
    return out


def chunk_paths(cfg, n):
    return [os.path.join(out_dir(cfg), "parts", f"chunk_{i:03d}_of_{n:03d}.csv") for i in range(n)]


def cmd_score(cfg, args):
    n = int(args.n_chunks or os.environ.get("SLURM_ARRAY_TASK_COUNT", 1))
    i = int(args.chunk if args.chunk is not None else os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    path = chunk_paths(cfg, n)[i]
    if os.path.exists(path):
        print(f"chunk {i}/{n} already scored: {path}")
    else:
        ads = adapters(cfg)
        d = ensure_input(cfg)
        rows = np.array_split(np.arange(len(d)), n)[i]
        part = d.iloc[rows].reset_index(drop=True)
        live = part.n_words_rated > 0                    # an empty output has nothing to rate
        texts = part.text_rated[live].tolist()
        print(f"chunk {i}/{n}: {len(part):,} rows ({(~live).sum()} empty), "
              f"{len(ads)} adapters, host {platform.node()}", flush=True)
        model, tok = load_model(cfg, ads)
        per = []
        for a in ads:
            model.set_adapter(a["name"])
            print(f"  {a['name']} (scale {a['scale']:.4f}) ...", flush=True)
            per.append(score_texts(model, tok, texts, cfg["max_length"], cfg["batch_size"]) * a["scale"])
        stack = np.stack(per, 0)                         # (seeds, rows, outputs)
        res = pd.DataFrame({"row_id": part.row_id})
        g = 0 if ads[0]["scalar_head"] else GLOBAL_IDX
        cols = {"disorg": stack[:, :, g].mean(0), "disorg_sd": stack[:, :, g].std(0)}
        for a, s in zip(ads, per):
            cols[f"disorg_{a['name']}"] = s[:, g]
        if not ads[0]["scalar_head"]:
            for j, it in enumerate(ITEMS):
                cols[f"pred_{it}"] = stack[:, :, j].mean(0)
        for c, v in cols.items():
            res[c] = np.nan
            res.loc[live.values, c] = v
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = f"{path}.tmp"
        res.to_csv(tmp, index=False)
        os.replace(tmp, path)
        print(f"chunk {i}/{n} done -> {path}", flush=True)
    if all(os.path.exists(p) for p in chunk_paths(cfg, n)):
        merge(cfg, n)


# --------------------------------------------------------------------------- #
# merge
# --------------------------------------------------------------------------- #
def git_commit():
    try:
        return subprocess.run(["git", "-C", REPO, "rev-parse", "HEAD"], capture_output=True,
                              text=True, timeout=20).stdout.strip() or None
    except Exception:
        return None


def merge(cfg, n):
    paths = chunk_paths(cfg, n)
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        raise SystemExit(f"{len(missing)} of {n} chunks not scored yet")
    scores = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    d = ensure_input(cfg)
    if len(scores) != len(d) or set(scores.row_id) != set(d.row_id):
        raise SystemExit("chunks don't cover to_score.csv exactly -- were chunks made with another n?")
    out = d.drop(columns=["text_rated"]).merge(scores, on="row_id")
    od = out_dir(cfg)
    tmp = os.path.join(od, "ratings.csv.tmp")
    out.to_csv(tmp, index=False)
    os.replace(tmp, os.path.join(od, "ratings.csv"))
    ads = adapters(cfg)
    log = {"name": cfg["name"], "merged_at": now(), "rows": len(out), "chunks": n,
           "runs": cfg["runs"], "config": cfg, "git_commit": git_commit(),
           "adapters": [{k: a[k] for k in ("name", "dir", "scale", "target_col", "train")} for a in ads],
           "trim": out.trim.value_counts().to_dict(),
           "n_words_rated": out.n_words_rated.describe().round(1).to_dict(),
           "disorg": out.disorg.describe().round(4).to_dict(),
           "disorg_sd_mean": float(out.disorg_sd.mean())}
    with open(os.path.join(od, "ratings.log.json"), "w") as f:
        json.dump(log, f, indent=2, default=str)
    print(f"merged {len(out):,} rows -> {od}/ratings.csv")
    base = out[out.manipulation == "none"].disorg
    print(f"  baseline disorg: mean {base.mean():.3f} (n={len(base)}) | "
          f"across-seed sd, mean: {out.disorg_sd.mean():.3f}")
    top = (out[out.manipulation != "none"]
           .groupby(["manipulation", "param_value"]).disorg.mean().sort_values(ascending=False).head(8))
    print("  highest mean disorg (manipulation, level):\n" + top.round(3).to_string())


def cmd_merge(cfg, args):
    n = args.n_chunks
    if n is None:
        found = sorted(glob.glob(os.path.join(out_dir(cfg), "parts", "chunk_*_of_*.csv")))
        if not found:
            raise SystemExit("no chunks to merge")
        n = int(found[0].rsplit("_of_", 1)[1].split(".")[0])
    merge(cfg, int(n))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", choices=["plan", "score", "merge"])
    ap.add_argument("--config", default=os.path.join(FT, "configure.yaml"))
    ap.add_argument("--chunk", type=int, default=None)
    ap.add_argument("--n-chunks", type=int, default=None)
    ap.add_argument("--no-tokens", action="store_true", help="plan: skip the tokenizer pass")
    args = ap.parse_args()
    cfg = load(args.config)
    {"plan": cmd_plan, "score": cmd_score, "merge": cmd_merge}[args.cmd](cfg, args)


if __name__ == "__main__":
    main()
