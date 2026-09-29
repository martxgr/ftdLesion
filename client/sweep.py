#!/usr/bin/env python
"""
client/sweep.py -- layer manipulations on Llama-family models.

Ported from llmSchizophrenia/8_sweep.py. The manipulation maths is unchanged;
what moved is everything around it: prompts come from prompts/registry.csv,
levels / bands / models / prompts come from configure.yaml, and a run is a
directory with a frozen config, one part-CSV per task, and a merged CSV + log.

Architecture reminder (Llama MLP):
    y = down_proj( SiLU(gate_proj(x)) * up_proj(x) )
  - gate_proj(x)  = g_raw   (PRE-SiLU gate activation)   <- "keys" gating side
  - up_proj(x)    = u       (magnitude side of the key)
  - down_proj.weight columns = value vectors v_i          <- "values" (content)

MANIPULATIONS
  Gate side (keys):
    gate_flatten   convex flatten g_raw toward per-token mean      (lambda)
    gate_shift     selective shift: lift only g_raw<0 by beta*sigma (beta)
    gate_noise     Gaussian noise on g_raw                          (sigma)
  Up side (robustness):
    up_flatten     convex flatten u toward per-token mean           (lambda)
  Values side (content):
    values_flatten convex flatten W_down columns toward mean column (lambda)
    values_promote shift each v_i toward its top-2 token embedding  (alpha)
    values_demote  shift each v_i away from its top-1 token embedding(alpha)
    values_noise   Gaussian noise on the MLP output (down_proj out)  (sigma)
  Control:
    input_noise    Gaussian noise on the whole MLP input vector     (sigma)

COMMANDS
  plan       validate configure.yaml, create/resume the run dir, freeze config
  calibrate  per-layer SDs + promote/demote tensors, cached per model
  task       one (model, manipulation, band) slice -> parts/<task>.csv, resumable
  merge      check completeness, concatenate parts, write <run>.csv + <run>.log.json
  list-tasks print the task table of a planned run

client/run.sh drives these; you rarely call them by hand.
"""

import argparse
import glob
import hashlib
import json
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
from contextlib import contextmanager
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
REGISTRY = os.path.join(REPO, "prompts", "registry.csv")

# promote/demote safety: cap each per-cell value-shift to this multiple of the
# value vector's own norm, so a near-parallel (e_top1,e_top2) pair (tiny gamma
# denominator -> huge gamma) cannot blow up a column.
MAX_SHIFT_RATIO = 5.0
PD_CHUNK = 2048  # cells per chunk in the promote/demote unembedding matmul

# --------------------------------------------------------------------------- #
# Manipulation registry (levels live in configure.yaml)
# --------------------------------------------------------------------------- #
MANIP_SPECS = {
    "gate_flatten":   {"mode": "act_hook",  "site": "gate_proj", "param_name": "lambda", "side": "gate"},
    "up_flatten":     {"mode": "act_hook",  "site": "up_proj",   "param_name": "lambda", "side": "up"},
    "values_flatten": {"mode": "param_mod", "site": "down_proj", "param_name": "lambda", "side": "values"},
    "gate_shift":     {"mode": "act_hook",  "site": "gate_proj", "param_name": "beta",   "side": "gate"},
    "gate_noise":     {"mode": "act_hook",  "site": "gate_proj", "param_name": "sigma",  "side": "gate"},
    "values_promote": {"mode": "param_mod", "site": "down_proj", "param_name": "alpha",  "side": "values"},
    "values_demote":  {"mode": "param_mod", "site": "down_proj", "param_name": "alpha",  "side": "values"},
    "input_noise":    {"mode": "pre_hook",  "site": "mlp",       "param_name": "sigma",  "side": "input"},
    "values_noise":   {"mode": "act_hook",  "site": "down_proj", "param_name": "sigma",  "side": "values"},
}

BAND_GROUPS = {
    "deciles":   [f"{i*10}-{(i+1)*10}%" for i in range(10)],
    "quartiles": ["0-25%", "25-50%", "50-75%", "75-100%"],
}

# Scaled by calibrated SDs or flip/drop gammas; the flattens need nothing.
NEEDS_CALIB = {"gate_shift", "gate_noise", "input_noise", "values_noise",
               "values_promote", "values_demote"}

# Bump when calibration gains a quantity, so older caches aren't reused.
CALIB_VERSION = 2   # 2: + sigma_values (SD of down_proj output) for values_noise

# Config keys that may change without it counting as a different run.
RESUMABLE_KEYS = ("slurm",)


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
_ENV_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def expand_path(s):
    """${VAR} and ${VAR:-default}; relative results are relative to the repo."""
    def sub(m):
        val = os.environ.get(m.group(1))
        if val:
            return val
        if m.group(2) is not None:
            return m.group(2)
        raise SystemExit(f"config: ${{{m.group(1)}}} is unset and has no default")
    out = os.path.expanduser(_ENV_RE.sub(sub, str(s)))
    return out if os.path.isabs(out) else os.path.normpath(os.path.join(REPO, out))


def parse_band(name):
    """'all' -> (0, 100); 'a-b%' -> (a, b). Integer percentiles only."""
    if name == "all":
        return 0, 100
    m = re.fullmatch(r"(\d+)-(\d+)%", name)
    if not m:
        raise SystemExit(f"config: bad band '{name}' -- use deciles, quartiles, all, or 'a-b%'")
    a, b = int(m.group(1)), int(m.group(2))
    if not 0 <= a < b <= 100:
        raise SystemExit(f"config: band '{name}' must satisfy 0 <= a < b <= 100")
    return a, b


def band_layers(name, n):
    """Layers of band `name` in an n-layer model. Integer arithmetic, so it
    reproduces 8_sweep.py exactly: int(n*i/10) for deciles, n//4 etc. for
    quartiles."""
    a, b = parse_band(name)
    layers = list(range(n * a // 100, n * b // 100))
    if not layers:
        raise SystemExit(f"band '{name}' is empty for a {n}-layer model")
    return layers


def expand_bands(spec):
    out = []
    for b in spec:
        for name in BAND_GROUPS.get(b, [b]):
            parse_band(name)
            if name not in out:
                out.append(name)
    return out


def load_registry():
    return pd.read_csv(REGISTRY, keep_default_na=False)


def load_config(path):
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return validate_config(raw)


def validate_config(raw):
    cfg = dict(raw)
    for k in ("run_name", "models", "prompts", "reps", "manipulations", "bands"):
        if k not in cfg:
            raise SystemExit(f"config: missing '{k}'")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", str(cfg["run_name"])):
        raise SystemExit("config: run_name may only contain letters, digits, _ . -")
    if isinstance(cfg["models"], str):
        cfg["models"] = [cfg["models"]]

    reg = load_registry()
    ids = list(reg.prompt_id)
    for key in ("prompts", "calibration_prompts"):
        v = cfg.get(key, "all")
        v = ids if v == "all" else list(v)
        bad = [p for p in v if p not in ids]
        if bad:
            raise SystemExit(f"config: {key} not in prompts/registry.csv: {bad}")
        cfg[key] = v

    manips = cfg["manipulations"] or {}
    bad = [m for m in manips if m not in MANIP_SPECS]
    if bad:
        raise SystemExit(f"config: unknown manipulations {bad}; choices: {list(MANIP_SPECS)}")
    for m, levels in manips.items():
        if not levels or not all(isinstance(x, (int, float)) for x in levels):
            raise SystemExit(f"config: {m} needs a non-empty list of numeric levels")
        if len(set(levels)) != len(levels):
            raise SystemExit(f"config: {m} has duplicate levels")
    cfg["manipulations"] = {m: [float(x) for x in lv] for m, lv in manips.items()}
    cfg["bands"] = expand_bands(cfg["bands"])

    cfg["reps"] = int(cfg["reps"])
    cfg["batch_reps"] = int(cfg.get("batch_reps") or cfg["reps"])
    if cfg["reps"] < 1 or cfg["batch_reps"] < 1:
        raise SystemExit("config: reps and batch_reps must be >= 1")
    cfg.setdefault("seed", 0)
    cfg.setdefault("baseline", True)
    cfg.setdefault("metrics", True)
    cfg.setdefault("dtype", "bfloat16")
    cfg.setdefault("device_map", "auto")
    gen = {"temperature": 1.0, "top_k": 50, "max_words": None, "max_new_tokens": 400}
    gen.update(cfg.get("generation") or {})
    cfg["generation"] = gen
    paths = {"output_root": "${FTD_BASE:-client/outputs}/results",
             "calib_root": "${FTD_BASE:-client/outputs}/calibration"}
    paths.update(cfg.get("paths") or {})
    cfg["paths"] = paths
    merge = {"max_mb": 500, "xlsx": True}
    merge.update(cfg.get("merge") or {})
    cfg["merge"] = merge
    cfg.setdefault("slurm", {})
    return cfg


def run_identity(cfg):
    return {k: v for k, v in cfg.items() if k not in RESUMABLE_KEYS}


def model_slug(name):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", os.path.basename(name.rstrip("/\\")) or name)


def band_slug(name):
    return name.replace("%", "pct")


def build_tasks(cfg):
    tasks = []
    for model in cfg["models"]:
        if cfg["baseline"]:
            tasks.append({"model": model, "manipulation": "none", "band": "none"})
        for m in cfg["manipulations"]:
            for b in cfg["bands"]:
                tasks.append({"model": model, "manipulation": m, "band": b})
    for i, t in enumerate(tasks):
        t["index"] = i
        t["part"] = f"{model_slug(t['model'])}__{t['manipulation']}__{band_slug(t['band'])}.csv"
    return tasks


def task_levels(cfg, task):
    return [None] if task["manipulation"] == "none" else cfg["manipulations"][task["manipulation"]]


def rows_per_task(cfg, task):
    return len(task_levels(cfg, task)) * len(cfg["prompts"]) * cfg["reps"]


def calib_dir_for(cfg, model):
    reg = load_registry().set_index("prompt_id")
    texts = f"v{CALIB_VERSION}\x1e" + "\x1e".join(
        reg.loc[p, "text"] for p in cfg["calibration_prompts"])
    h = hashlib.sha256(texts.encode("utf-8")).hexdigest()[:12]
    return os.path.join(expand_path(cfg["paths"]["calib_root"]), model_slug(model), h)


def run_dir_for(cfg):
    return os.path.join(expand_path(cfg["paths"]["output_root"]), cfg["run_name"])


def read_frozen(run_dir):
    with open(os.path.join(run_dir, "config.frozen.yaml"), encoding="utf-8") as f:
        return validate_config(yaml.safe_load(f))


# --------------------------------------------------------------------------- #
# Keys, seeds, CSV helpers
# --------------------------------------------------------------------------- #
def pkey(v):
    """Canonical string for a param level; 'none' for baseline."""
    if v is None or v == "" or (isinstance(v, float) and np.isnan(v)):
        return "none"
    return f"{float(v):g}"


def cell_seed(base, *parts):
    s = "|".join([str(base)] + [str(p) for p in parts])
    return int(hashlib.sha256(s.encode()).hexdigest()[:8], 16) % (2**31 - 1)


def load_completed(csv_path):
    if not os.path.exists(csv_path):
        return set()
    try:
        df = pd.read_csv(csv_path, keep_default_na=False,
                         dtype={"prompt_id": str}, usecols=["param_value", "prompt_id", "rep"])
    except Exception as ex:
        raise SystemExit(f"cannot read existing part {csv_path}: {ex}\n"
                         "Move it aside to regenerate the task from scratch.")
    return {(pkey(v if v != "" else None), p, int(r))
            for v, p, r in zip(df.param_value, df.prompt_id, df.rep)}


def append_rows(csv_path, rows):
    pd.DataFrame(rows).to_csv(csv_path, mode="a", header=not os.path.exists(csv_path),
                              index=False, encoding="utf-8")


def git_state():
    def git(*a):
        try:
            return subprocess.run(["git", "-C", REPO, *a], capture_output=True,
                                  text=True, timeout=20).stdout.strip()
        except Exception:
            return ""
    return {"commit": git("rev-parse", "HEAD") or None,
            "dirty": bool(git("status", "--porcelain", "--", "client", "prompts"))}


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# Model engine
# --------------------------------------------------------------------------- #
class Engine:
    """One loaded model plus everything needed to manipulate and sample it."""

    def __init__(self, name, dtype="bfloat16", device_map="auto"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.name = name
        print(f"Loading {name} ...", flush=True)
        kw = {"dtype": getattr(torch, dtype)}
        if device_map:
            kw["device_map"] = device_map
        self.model = AutoModelForCausalLM.from_pretrained(name, **kw)
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.eval()
        self.layers = self.model.model.layers
        self.n_layers = len(self.layers)
        for i, layer in enumerate(self.layers):
            mlp = getattr(layer, "mlp", None)
            if not all(hasattr(mlp, s) for s in ("gate_proj", "up_proj", "down_proj")):
                raise SystemExit(f"{name}: layer {i} has no gated (SwiGLU) MLP -- "
                                 "only Llama-family architectures are supported")
        self.input_device = self.model.get_input_embeddings().weight.device
        self.out_emb = self.model.get_output_embeddings().weight  # [V, d]; NOT tied in Llama 3
        eos = self.model.generation_config.eos_token_id
        self.eos_ids = set(eos if isinstance(eos, (list, tuple)) else [eos]) - {None}
        self.eos_ids.add(self.tokenizer.eos_token_id)
        self.calib = None
        self.calib_dir = None
        self._pd_cache = {}
        print(f"Model loaded: {self.n_layers} layers, d_model={self.model.config.hidden_size}, "
              f"d_mlp={self.model.config.intermediate_size}, vocab={self.out_emb.shape[0]}",
              flush=True)

    # ---------------- prompts ---------------- #
    def encode(self, prompt, n=1):
        formatted = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
        # identical rows -> no padding, so batching changes nothing per row
        return self.tokenizer([formatted] * n, return_tensors="pt",
                              add_special_tokens=False).to(self.input_device)

    # ---------------- calibration ---------------- #
    def calibrate(self, prompts, calib_dir, calib_prompt_ids):
        torch = self.torch
        os.makedirs(os.path.join(calib_dir, "promote_demote"), exist_ok=True)
        gate_stds = {i: [] for i in range(self.n_layers)}
        input_stds = {i: [] for i in range(self.n_layers)}
        value_stds = {i: [] for i in range(self.n_layers)}
        hooks = []

        def gate_rec(idx):
            def h(m, inp, out):
                gate_stds[idx].append(out.detach().float().std().item())
            return h

        def value_rec(idx):
            def h(m, inp, out):
                value_stds[idx].append(out.detach().float().std().item())
            return h

        def input_rec(idx):
            def h(m, a):
                input_stds[idx].append(a[0].detach().float().std().item())
            return h

        for i, layer in enumerate(self.layers):
            hooks.append(layer.mlp.gate_proj.register_forward_hook(gate_rec(i)))
            hooks.append(layer.mlp.register_forward_pre_hook(input_rec(i)))
            hooks.append(layer.mlp.down_proj.register_forward_hook(value_rec(i)))
        print(f"Calibrating per-layer sigmas over {len(prompts)} prompts ...", flush=True)
        try:
            for p in prompts:
                with torch.no_grad():
                    self.model(**self.encode(p))
        finally:
            for h in hooks:
                h.remove()

        scalars = {"model": self.name, "n_layers": self.n_layers,
                   "calibration_prompts": list(calib_prompt_ids),
                   "sigma_gate": {i: float(np.mean(gate_stds[i])) for i in range(self.n_layers)},
                   "sigma_input": {i: float(np.mean(input_stds[i])) for i in range(self.n_layers)},
                   "sigma_values": {i: float(np.mean(value_stds[i])) for i in range(self.n_layers)},
                   "calib_version": CALIB_VERSION,
                   "created": now()}

        print("Calibrating promote/demote (unembedding projection) per layer ...", flush=True)
        for li in range(self.n_layers):
            self._promote_demote_for_layer(li, calib_dir)
            if (li + 1) % 10 == 0 or li == self.n_layers - 1:
                print(f"  ... layers done: {li + 1}/{self.n_layers}", flush=True)
        # written last: its presence marks the calibration complete
        with open(os.path.join(calib_dir, "calib_scalars.json"), "w") as f:
            json.dump(scalars, f, indent=2)
        print(f"Calibration complete -> {calib_dir}", flush=True)

    def _promote_demote_for_layer(self, layer_idx, calib_dir):
        """For each MLP cell, project its value vector through the unembedding,
        take top-2 tokens, and store the flip/drop gammas.

          gamma_flip = (l_top1 - l_top2) / (||e_top2||^2 - e_top1.e_top2)
          gamma_drop = (l_top1 - l_top2) / (||e_top1||^2 - e_top1.e_top2)

        Cells with a non-positive denominator (e_top1 ~ parallel e_top2) get
        gamma=0 (no shift) -- they cannot be flipped along this direction.
        """
        torch = self.torch
        E = self.out_emb
        Vcols = self.layers[layer_idx].mlp.down_proj.weight.t()  # [d_m, d]; row i = v_i
        d_m = Vcols.shape[0]
        top1_idx = torch.empty(d_m, dtype=torch.long)
        top2_idx = torch.empty(d_m, dtype=torch.long)
        gamma_flip = torch.empty(d_m, dtype=torch.float32)
        gamma_drop = torch.empty(d_m, dtype=torch.float32)
        n_zero = 0
        with torch.no_grad():
            for s in range(0, d_m, PD_CHUNK):
                e = min(s + PD_CHUNK, d_m)
                vc = Vcols[s:e].to(E.device)
                tv, ti = (vc @ E.t()).topk(2, dim=-1)
                l1, l2 = tv[:, 0].float(), tv[:, 1].float()
                i1, i2 = ti[:, 0], ti[:, 1]
                e1, e2 = E[i1].float(), E[i2].float()
                dot = (e1 * e2).sum(-1)
                n1, n2 = (e1 * e1).sum(-1), (e2 * e2).sum(-1)
                gap = l1 - l2
                denom_flip, denom_drop = n2 - dot, n1 - dot
                gf = torch.where(denom_flip > 1e-6, gap / denom_flip, torch.zeros_like(gap))
                gd = torch.where(denom_drop > 1e-6, gap / denom_drop, torch.zeros_like(gap))
                n_zero += int((denom_flip <= 1e-6).sum().item())
                top1_idx[s:e], top2_idx[s:e] = i1.cpu(), i2.cpu()
                gamma_flip[s:e], gamma_drop[s:e] = gf.cpu(), gd.cpu()
        torch.save({"top1_idx": top1_idx, "top2_idx": top2_idx,
                    "gamma_flip": gamma_flip, "gamma_drop": gamma_drop,
                    "n_degenerate": n_zero},
                   os.path.join(calib_dir, "promote_demote", f"layer_{layer_idx}.pt"))

    def load_calibration(self, calib_dir):
        path = os.path.join(calib_dir, "calib_scalars.json")
        if not os.path.exists(path):
            raise SystemExit(f"no calibration at {calib_dir} -- run `sweep.py calibrate` first")
        with open(path) as f:
            c = json.load(f)
        if c["n_layers"] != self.n_layers:
            raise SystemExit(f"calibration at {calib_dir} is for {c['n_layers']} layers, "
                             f"model has {self.n_layers}")
        c["sigma_gate"] = {int(k): v for k, v in c["sigma_gate"].items()}
        c["sigma_input"] = {int(k): v for k, v in c["sigma_input"].items()}
        c["sigma_values"] = {int(k): v for k, v in c.get("sigma_values", {}).items()}
        self.calib, self.calib_dir, self._pd_cache = c, calib_dir, {}

    def _pd(self, layer_idx):
        if layer_idx not in self._pd_cache:
            self._pd_cache[layer_idx] = self.torch.load(
                os.path.join(self.calib_dir, "promote_demote", f"layer_{layer_idx}.pt"))
        return self._pd_cache[layer_idx]

    # ---------------- hooks ---------------- #
    def _act_hook(self, name, layer_idx, param):
        torch = self.torch
        if name in ("gate_flatten", "up_flatten"):
            lam = float(param)
            def hook(m, inp, out):
                return (1.0 - lam) * out + lam * out.mean(dim=-1, keepdim=True)
            return hook
        if name == "gate_shift":
            s = float(param) * self.calib["sigma_gate"][layer_idx]
            def hook(m, inp, out):
                # lift ONLY the sub-threshold (SiLU-suppressed) cells
                return out + (out < 0).to(out.dtype) * s
            return hook
        if name == "gate_noise":
            s = float(param) * self.calib["sigma_gate"][layer_idx]
            def hook(m, inp, out):
                return out + torch.randn_like(out) * s
            return hook
        if name == "values_noise":
            # noise on what the values write back into the residual stream
            s = float(param) * self.calib["sigma_values"][layer_idx]
            def hook(m, inp, out):
                return out + torch.randn_like(out) * s
            return hook
        raise ValueError(name)

    def _pre_hook(self, name, layer_idx, param):
        torch = self.torch
        if name == "input_noise":
            s = float(param) * self.calib["sigma_input"][layer_idx]
            def pre(m, a):
                x = a[0]
                return (x + torch.randn_like(x) * s,)
            return pre
        raise ValueError(name)

    def _param_mod(self, name, layer_idx, param, clean_cpu):
        """Modified down_proj.weight [d, d_m], built FRESH from the clean
        snapshot (no accumulated drift)."""
        torch = self.torch
        dp = self.layers[layer_idx].mlp.down_proj.weight
        clean = clean_cpu.to(dp.device).float()
        if name == "values_flatten":
            lam = float(param)
            modified = (1.0 - lam) * clean + lam * clean.mean(dim=1, keepdim=True)
        elif name in ("values_promote", "values_demote"):
            alpha = float(param)
            pd_ = self._pd(layer_idx)
            E = self.out_emb
            if name == "values_promote":
                idx, gamma, sign = pd_["top2_idx"], pd_["gamma_flip"], 1.0
            else:
                idx, gamma, sign = pd_["top1_idx"], pd_["gamma_drop"], -1.0
            idx, gamma = idx.to(E.device), gamma.to(E.device)
            shift = ((alpha * gamma).unsqueeze(1) * E[idx].float()).to(dp.device)  # [d_m, d]
            # cap each cell's shift norm to MAX_SHIFT_RATIO x its value-vector norm
            col_norm = clean.norm(dim=0)
            scale = torch.clamp(MAX_SHIFT_RATIO * col_norm / (shift.norm(dim=1) + 1e-6), max=1.0)
            modified = clean + sign * (shift * scale.unsqueeze(1)).t()
        else:
            raise ValueError(name)
        return modified.to(dp.dtype)

    @contextmanager
    def manipulated(self, name, layer_indices, param):
        """Apply `name` at `param` to `layer_indices` for the duration of the
        block; always restores the clean model on exit."""
        if name == "none":
            yield
            return
        spec = MANIP_SPECS[name]
        if name in NEEDS_CALIB and self.calib is None:
            raise RuntimeError(f"{name} needs calibration loaded")
        torch = self.torch
        hooks, snapshot = [], {}
        try:
            if spec["mode"] in ("act_hook", "pre_hook"):
                for li in layer_indices:
                    mlp = self.layers[li].mlp
                    module = mlp if spec["site"] == "mlp" else getattr(mlp, spec["site"])
                    if spec["mode"] == "act_hook":
                        hooks.append(module.register_forward_hook(self._act_hook(name, li, param)))
                    else:
                        hooks.append(module.register_forward_pre_hook(self._pre_hook(name, li, param)))
            else:
                with torch.no_grad():
                    for li in layer_indices:
                        dp = self.layers[li].mlp.down_proj.weight
                        # clean snapshot on CPU (safe for the 'all' band)
                        snapshot[li] = dp.detach().to("cpu", copy=True)
                        dp.copy_(self._param_mod(name, li, param, snapshot[li]))
            yield
        finally:
            for h in hooks:
                h.remove()
            if snapshot:
                with torch.no_grad():
                    for li, w in snapshot.items():
                        dp = self.layers[li].mlp.down_proj.weight
                        dp.copy_(w.to(dp.device))

    # ---------------- generation ---------------- #
    def generate(self, prompt, n, gen, seed):
        """n samples of one prompt in one batch. Returns [(text, n_words, stop_reason)]."""
        torch = self.torch
        from transformers import StoppingCriteria, StoppingCriteriaList
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        inputs = self.encode(prompt, n)
        input_len = inputs["input_ids"].shape[1]
        max_words = gen.get("max_words")
        tok = self.tokenizer

        class WordCap(StoppingCriteria):
            # A word is complete once the next one starts, so stop at max_words+1.
            def __call__(self, input_ids, scores, **kw):
                texts = tok.batch_decode(input_ids[:, input_len:], skip_special_tokens=True)
                return torch.tensor([len(t.split()) > max_words for t in texts],
                                    device=input_ids.device)

        kw = dict(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],
                  do_sample=True, top_k=gen["top_k"], temperature=gen["temperature"],
                  max_new_tokens=gen["max_new_tokens"], pad_token_id=tok.eos_token_id)
        if max_words:
            kw["stopping_criteria"] = StoppingCriteriaList([WordCap()])
        with torch.no_grad():
            out = self.model.generate(**kw)

        results = []
        for row in out[:, input_len:].tolist():
            cut = next((i for i, t in enumerate(row) if t in self.eos_ids), None)
            text = tok.decode(row if cut is None else row[:cut], skip_special_tokens=True).strip()
            n_words = len(text.split())
            if max_words and n_words > max_words:
                end = [m.end() for m in re.finditer(r"\S+", text)][max_words - 1]
                text, n_words, reason = text[:end], max_words, "word_cap"
            elif cut is None and len(row) >= gen["max_new_tokens"]:
                reason = "token_cap"
            else:
                reason = "eos"
            results.append((text, n_words, reason))
        return results


# --------------------------------------------------------------------------- #
# Inline metrics (a cross-check only -- NOT the FTD scoring, which is clinician's)
# --------------------------------------------------------------------------- #
_metric_model = None
METRIC_COLS = ["ttr", "repetition_rate", "coherence", "topic_similarity"]


def compute_metrics(prompt, output):
    global _metric_model
    m = {k: "" for k in METRIC_COLS}
    try:
        words = output.split()
        if words:
            m["ttr"] = len(set(w.lower() for w in words)) / len(words)
            bigrams = list(zip(words[:-1], words[1:]))
            if bigrams:
                m["repetition_rate"] = 1.0 - len(set(bigrams)) / len(bigrams)
        if _metric_model is None:
            from sentence_transformers import SentenceTransformer
            _metric_model = SentenceTransformer("all-MiniLM-L6-v2")
        sents = [s.strip() for s in output.replace("!", ".").replace("?", ".").split(".")
                 if len(s.strip().split()) >= 3]
        if len(sents) >= 2:
            emb = _metric_model.encode(sents, convert_to_numpy=True, normalize_embeddings=True)
            m["coherence"] = float(np.mean([np.dot(emb[i], emb[i + 1]) for i in range(len(emb) - 1)]))
        if output.strip():
            pe, oe = _metric_model.encode([prompt, output], convert_to_numpy=True,
                                          normalize_embeddings=True)
            m["topic_similarity"] = float(np.dot(pe, oe))
    except Exception as ex:  # never let metrics kill a generation
        print(f"  [metrics warning] {ex}", flush=True)
    return m


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #
def cmd_plan(args):
    cfg = load_config(args.config)
    tasks = build_tasks(cfg)
    run_dir = run_dir_for(cfg)
    n_rows = sum(rows_per_task(cfg, t) for t in tasks)
    gens = max(rows_per_task(cfg, t) for t in tasks)
    batches = -(-cfg["reps"] // cfg["batch_reps"])
    summary = (f"run {cfg['run_name']}: {len(tasks)} tasks, {n_rows:,} rows "
               f"({len(cfg['models'])} model(s) x {len(cfg['prompts'])} prompts x {cfg['reps']} reps; "
               f"{len(cfg['manipulations'])} manipulations x {len(cfg['bands'])} bands)\n"
               f"  per task: up to {gens:,} generations in "
               f"{gens // cfg['reps'] * batches:,} batches of <= {cfg['batch_reps']}\n"
               f"  run dir : {run_dir}")
    if args.dry_run:
        print(summary)
        return

    frozen = os.path.join(run_dir, "config.frozen.yaml")
    if os.path.exists(frozen):
        old = read_frozen(run_dir)
        if run_identity(old) != run_identity(cfg):
            diff = sorted(k for k in set(old) | set(cfg)
                          if k not in RESUMABLE_KEYS and old.get(k) != cfg.get(k))
            raise SystemExit(f"{run_dir} already exists with a different config "
                             f"(differs in: {', '.join(diff)}).\n"
                             "Change run_name for a new run, or restore the old settings to resume.")
        print(f"resuming existing run -- {summary}", file=sys.stderr)
    else:
        print(f"new run -- {summary}", file=sys.stderr)
    os.makedirs(os.path.join(run_dir, "parts"), exist_ok=True)
    with open(args.config, encoding="utf-8") as f:
        raw_text = f.read()
    with open(frozen, "w", encoding="utf-8") as f:
        f.write(raw_text)
    reg = load_registry()
    reg[reg.prompt_id.isin(set(cfg["prompts"]) | set(cfg["calibration_prompts"]))].to_csv(
        os.path.join(run_dir, "prompts.frozen.csv"), index=False, encoding="utf-8")
    pd.DataFrame(tasks)[["index", "model", "manipulation", "band", "part"]].to_csv(
        os.path.join(run_dir, "tasks.tsv"), sep="\t", index=False)
    manifest_path = os.path.join(run_dir, "manifest.json")
    manifest = {}
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f)
    manifest.setdefault("created", now())
    manifest.setdefault("submissions", []).append(
        {"at": now(), "host": platform.node(), "git": git_state(),
         "slurm": cfg["slurm"], "mode": "local" if args.local else "slurm"})
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    if args.shell:
        s = cfg["slurm"]
        out = {"RUN_DIR": run_dir, "N_TASKS": len(tasks),
               "CALIB_NEEDED": " ".join(str(i) for i, m in enumerate(cfg["models"])
                                        if not os.path.exists(os.path.join(calib_dir_for(cfg, m),
                                                                           "calib_scalars.json"))),
               "N_MODELS": len(cfg["models"]),
               "S_PARTITION": s.get("partition", ""), "S_GPUS": s.get("gpus", ""),
               "S_CPUS": s.get("cpus", ""), "S_MEM": s.get("mem", ""),
               "S_TIME": s.get("time", ""), "S_MAXC": s.get("max_concurrent") or "",
               "S_CAL_TIME": s.get("calibrate_time", s.get("time", "")),
               "S_MERGE_MEM": s.get("merge_mem", "32G"),
               "S_MERGE_PARTITION": s.get("merge_partition") or s.get("partition", "")}
        for k, v in out.items():
            print(f"{k}={shlex.quote(str(v))}")


def cmd_list_tasks(args):
    cfg = read_frozen(args.run_dir)
    for t in build_tasks(cfg):
        print(f"{t['index']}\t{t['model']}\t{t['manipulation']}\t{t['band']}")


def cmd_calibrate(args):
    cfg = read_frozen(args.run_dir)
    models = cfg["models"] if args.model is None else [cfg["models"][args.model]]
    reg = load_registry().set_index("prompt_id")
    for model in models:
        cdir = calib_dir_for(cfg, model)
        if os.path.exists(os.path.join(cdir, "calib_scalars.json")) and not args.force:
            print(f"calibration exists for {model}: {cdir}", flush=True)
            continue
        eng = Engine(model, cfg["dtype"], cfg["device_map"])
        eng.calibrate([reg.loc[p, "text"] for p in cfg["calibration_prompts"]], cdir,
                      cfg["calibration_prompts"])
        del eng


def run_task(cfg, run_dir, task, engine=None):
    """Generate every missing row of one task. Returns the engine for reuse."""
    reg = load_registry().set_index("prompt_id")
    if engine is None or engine.name != task["model"]:
        engine = Engine(task["model"], cfg["dtype"], cfg["device_map"])
    if task["manipulation"] in NEEDS_CALIB:
        engine.load_calibration(calib_dir_for(cfg, task["model"]))
    manip, band = task["manipulation"], task["band"]
    spec = MANIP_SPECS.get(manip, {"side": "none", "param_name": ""})
    layers = [] if manip == "none" else band_layers(band, engine.n_layers)
    li_str = ",".join(map(str, layers))
    csv_path = os.path.join(run_dir, "parts", task["part"])
    done = load_completed(csv_path)
    reps, B = cfg["reps"], cfg["batch_reps"]
    total = rows_per_task(cfg, task)
    print(f"task {task['index']}: {task['model']} | {manip} | {band} "
          f"layers=[{li_str}] | {len(done)}/{total} rows already done", flush=True)

    for level in task_levels(cfg, task):
        for pid in cfg["prompts"]:
            prompt = reg.loc[pid, "text"]
            for b0 in range(0, reps, B):
                rep_ids = list(range(b0, min(b0 + B, reps)))
                missing = [r for r in rep_ids if (pkey(level), pid, r) not in done]
                if not missing:
                    continue
                seed = cell_seed(cfg["seed"], task["model"], manip, band, pkey(level), pid, b0)
                print(f"  {spec['param_name'] or 'baseline'}={pkey(level)} prompt={pid} "
                      f"reps {rep_ids[0]}-{rep_ids[-1]} seed={seed}", flush=True)
                with engine.manipulated(manip, layers, level):
                    outs = engine.generate(prompt, len(rep_ids), cfg["generation"], seed)
                rows = []
                for r, (text, n_words, reason) in zip(rep_ids, outs):
                    if r not in missing:
                        continue
                    row = {"run": cfg["run_name"], "model": task["model"],
                           "phase": "baseline" if manip == "none" else "sweep",
                           "prompt_id": pid, "rep": r, "manipulation": manip,
                           "side": spec["side"], "param_name": spec["param_name"],
                           "param_value": "" if level is None else level,
                           "layer_strategy": band, "layer_indices": li_str,
                           "output": text, "n_words": n_words, "stop_reason": reason,
                           "seed": seed, "timestamp": now()}
                    if cfg["metrics"]:
                        row.update(compute_metrics(prompt, text))
                    rows.append(row)
                append_rows(csv_path, rows)
    print(f"task {task['index']} complete.", flush=True)
    return engine


def cmd_task(args):
    cfg = read_frozen(args.run_dir)
    tasks = build_tasks(cfg)
    idx = args.index if args.index is not None else os.environ.get("SLURM_ARRAY_TASK_ID")
    if idx is None:
        raise SystemExit("task: pass --index or run inside a SLURM array")
    idxs = range(len(tasks)) if idx == "all" else [int(idx)]
    engine = None
    for i in idxs:
        engine = run_task(cfg, args.run_dir, tasks[i], engine)


def cmd_merge(args):
    run_dir = args.run_dir
    cfg = read_frozen(run_dir)
    tasks = build_tasks(cfg)
    name = cfg["run_name"]
    problems, frames, counts = [], [], []
    for t in tasks:
        path = os.path.join(run_dir, "parts", t["part"])
        expected = {(pkey(l), p, r) for l in task_levels(cfg, t)
                    for p in cfg["prompts"] for r in range(cfg["reps"])}
        if not os.path.exists(path):
            problems.append(f"task {t['index']} ({t['part']}): missing")
            continue
        df = pd.read_csv(path, keep_default_na=False, dtype={"prompt_id": str, "output": str})
        got = [(pkey(v if v != "" else None), p, int(r))
               for v, p, r in zip(df.param_value, df.prompt_id, df.rep)]
        if len(got) != len(set(got)):
            problems.append(f"task {t['index']} ({t['part']}): duplicate rows")
        if set(got) != expected:
            problems.append(f"task {t['index']} ({t['part']}): {len(expected - set(got))} "
                            f"missing, {len(set(got) - expected)} unexpected rows")
        df.insert(0, "task", t["index"])
        frames.append(df)
        counts.append({"task": t["index"], "part": t["part"], "rows": len(df)})
    if problems:
        print("merge refused -- run is incomplete:\n  " + "\n  ".join(problems), file=sys.stderr)
        sys.exit(1)

    order = {p: i for i, p in enumerate(cfg["prompts"])}
    allrows = pd.concat(frames, ignore_index=True)
    allrows["_p"] = allrows.prompt_id.map(order)
    allrows["_v"] = pd.to_numeric(allrows.param_value, errors="coerce").fillna(-1)
    allrows = (allrows.sort_values(["task", "_v", "_p", "rep"])
               .drop(columns=["_p", "_v"]).reset_index(drop=True))
    allrows.insert(0, "row_id", [f"{name}:{i}" for i in range(len(allrows))])

    csv_path = os.path.join(run_dir, f"{name}.csv")
    tmp = csv_path + ".tmp"
    allrows.to_csv(tmp, index=False, encoding="utf-8")
    size_mb = os.path.getsize(tmp) / 2**20
    merged = size_mb <= cfg["merge"]["max_mb"]
    if merged:
        os.replace(tmp, csv_path)
    else:
        os.remove(tmp)
        print(f"merged CSV would be {size_mb:.0f} MB > merge.max_mb={cfg['merge']['max_mb']}; "
              "keeping parts/ unmerged", file=sys.stderr)

    xlsx_path = None
    if merged and cfg["merge"]["xlsx"]:
        if len(allrows) < 1_048_000 and allrows.output.str.len().max() < 32_000:
            xlsx_path = os.path.join(run_dir, f"{name}.xlsx")
            allrows.to_excel(xlsx_path, index=False)
        else:
            print("too large for Excel; skipped .xlsx", file=sys.stderr)

    with open(os.path.join(run_dir, "manifest.json")) as f:
        manifest = json.load(f)
    calib = {}
    for m in cfg["models"]:
        cdir = calib_dir_for(cfg, m)
        p = os.path.join(cdir, "calib_scalars.json")
        calib[m] = {"dir": cdir, "sha256": sha256_file(p) if os.path.exists(p) else None}
    prompts_frozen = pd.read_csv(os.path.join(run_dir, "prompts.frozen.csv"), keep_default_na=False)
    log = {
        "run_name": name,
        "merged_at": now(),
        "rows": len(allrows),
        "csv": os.path.basename(csv_path) if merged else None,
        "csv_sha256": sha256_file(csv_path) if merged else None,
        "csv_mb": round(size_mb, 1),
        "xlsx": os.path.basename(xlsx_path) if xlsx_path else None,
        "parts_kept": not merged or args.keep_parts,
        "config": cfg,
        "prompts": {r.prompt_id: hashlib.sha256(r.text.encode("utf-8")).hexdigest()[:12]
                    for r in prompts_frozen.itertuples()},
        "calibration": calib,
        "submissions": manifest.get("submissions", []),
        "tasks": counts,
        "stop_reason": allrows.stop_reason.value_counts().to_dict(),
        "n_words": allrows.n_words.describe().round(1).to_dict(),
        "generated_from": allrows.timestamp.min(),
        "generated_to": allrows.timestamp.max(),
        "merge_host": platform.node(),
        "slurm_job": os.environ.get("SLURM_JOB_ID"),
    }
    with open(os.path.join(run_dir, f"{name}.log.json"), "w") as f:
        json.dump(log, f, indent=2, default=str)
    if merged and not args.keep_parts:
        shutil.rmtree(os.path.join(run_dir, "parts"))
    print(f"merged {len(allrows):,} rows -> {csv_path if merged else '(parts kept)'}; "
          f"log -> {name}.log.json", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("plan")
    p.add_argument("--config", default=os.path.join(HERE, "configure.yaml"))
    p.add_argument("--dry-run", action="store_true", help="print the run size and exit")
    p.add_argument("--shell", action="store_true", help="emit VAR=value lines for run.sh")
    p.add_argument("--local", action="store_true")
    p.set_defaults(fn=cmd_plan)

    p = sub.add_parser("list-tasks"); p.add_argument("--run-dir", required=True)
    p.set_defaults(fn=cmd_list_tasks)

    p = sub.add_parser("calibrate"); p.add_argument("--run-dir", required=True)
    p.add_argument("--model", type=int, default=None, help="model index; default all")
    p.add_argument("--force", action="store_true", help="recalibrate even if cached")
    p.set_defaults(fn=cmd_calibrate)

    p = sub.add_parser("task"); p.add_argument("--run-dir", required=True)
    p.add_argument("--index", default=None, help="task index, or 'all'; default $SLURM_ARRAY_TASK_ID")
    p.set_defaults(fn=cmd_task)

    p = sub.add_parser("merge"); p.add_argument("--run-dir", required=True)
    p.add_argument("--keep-parts", action="store_true")
    p.set_defaults(fn=cmd_merge)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
