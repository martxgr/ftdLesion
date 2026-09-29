#!/usr/bin/env python
"""
client/tests/test_tiny.py -- exercise the whole client on CPU in about a minute.

Builds a randomly initialised 10-layer Llama (same gate_proj / up_proj /
down_proj MLP as the 70B, tiny widths) and checks:

  1. band_layers reproduces 8_sweep.py's get_layer_strategies exactly
  2. every manipulation changes the logits, and the model is bit-identical
     to clean after the context exits (hooks removed, weights restored)
  3. the word cap stops and trims generation
  4. end to end via run.sh --local: plan -> calibrate -> tasks -> merge,
     resume without duplicates, merge refuses an incomplete run, a changed
     config under the same run_name is refused

  python client/tests/test_tiny.py

Needs torch, transformers, sentence-transformers, and the gpt2-medium
tokenizer in the local HF cache (it borrows gpt2's vocab).
"""

import os
import shutil
import subprocess
import sys
import tempfile

import pandas as pd
import torch
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
CLIENT = os.path.dirname(HERE)
sys.path.insert(0, CLIENT)
import sweep  # noqa: E402


def old_strategies(n):
    """Verbatim from llmSchizophrenia/8_sweep.py."""
    s = {}
    for i in range(10):
        s[f"{i*10}-{(i+1)*10}%"] = list(range(int(n * i / 10), int(n * (i + 1) / 10)))
    s["0-25%"] = list(range(0, n // 4))
    s["25-50%"] = list(range(n // 4, n // 2))
    s["50-75%"] = list(range(n // 2, 3 * n // 4))
    s["75-100%"] = list(range(3 * n // 4, n))
    s["all"] = list(range(n))
    return s


def test_bands():
    names = sweep.expand_bands(["deciles", "quartiles", "all"])
    for n in (10, 32, 80, 81, 126):
        old = old_strategies(n)
        assert names == list(old), names
        for b in names:
            assert sweep.band_layers(b, n) == old[b], (n, b)
    assert sweep.band_layers("35-45%", 80) == list(range(28, 36))
    print("ok  bands match 8_sweep.py for n in 10, 32, 80, 81, 126")


def build_tiny(path):
    from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM
    tok = AutoTokenizer.from_pretrained("gpt2-medium")
    tok.chat_template = ("{% for m in messages %}{{ m['content'] }}\n{% endfor %}"
                         "{% if add_generation_prompt %}Answer:{% endif %}")
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=len(tok), hidden_size=64, intermediate_size=128,
                      num_hidden_layers=10, num_attention_heads=4, num_key_value_heads=2,
                      max_position_embeddings=2048, tie_word_embeddings=False,
                      bos_token_id=tok.eos_token_id, eos_token_id=tok.eos_token_id)
    LlamaForCausalLM(cfg).save_pretrained(path)
    tok.save_pretrained(path)


def test_manipulations(model_dir, calib_dir):
    eng = sweep.Engine(model_dir, dtype="float32", device_map=None)
    reg = sweep.load_registry().set_index("prompt_id")
    eng.calibrate([reg.loc[p, "text"] for p in ("bird", "farm")], calib_dir, ["bird", "farm"])
    eng.load_calibration(calib_dir)
    x = eng.encode(reg.loc["farm", "text"])
    weights0 = {k: v.clone() for k, v in eng.model.state_dict().items()}

    def logits():
        with torch.no_grad():
            return eng.model(**x).logits
    clean = logits()
    levels = {"lambda": 0.7, "beta": 2.0, "sigma": 2.0, "alpha": 4.0}
    for name, spec in sweep.MANIP_SPECS.items():
        for band in ("0-10%", "40-60%", "all"):
            layers = sweep.band_layers(band, eng.n_layers)
            torch.manual_seed(1)
            with eng.manipulated(name, layers, levels[spec["param_name"]]):
                moved = logits()
            assert not torch.equal(moved, clean), f"{name} {band} changed nothing"
            assert torch.equal(logits(), clean), f"{name} {band} did not restore"
        for k, v in eng.model.state_dict().items():
            assert torch.equal(v, weights0[k]), f"{name} left weight {k} modified"
    assert all(not m._forward_hooks and not m._forward_pre_hooks for m in eng.model.modules())
    layers = sweep.band_layers("all", eng.n_layers)
    draws = []
    for ns in (11, 11, 12):
        with eng.manipulated("values_weight_noise", layers, 2.0, noise_seed=ns):
            draws.append(logits())
    assert torch.equal(draws[0], draws[1]) and not torch.equal(draws[0], draws[2])
    print("ok  values_weight_noise: same seed -> same lesion, new seed -> new lesion")
    print(f"ok  all {len(sweep.MANIP_SPECS)} manipulations move the logits and restore the model exactly")

    gen = {"temperature": 1.0, "top_k": 50, "max_words": 12, "max_new_tokens": 60}
    outs = eng.generate(reg.loc["bird", "text"], 6, gen, seed=3)
    assert len(outs) == 6
    for text, n_words, reason in outs:
        assert n_words == len(text.split()) <= 12, (n_words, text)
        assert reason in ("eos", "word_cap", "token_cap")
        if reason == "word_cap":
            assert n_words == 12
    again = eng.generate(reg.loc["bird", "text"], 6, gen, seed=3)
    assert again == outs, "same seed should reproduce the batch"
    print(f"ok  word cap: stop reasons {sorted(r for _, _, r in outs)}; seeded batch reproduces")


def write_cfg(path, model_dir, root, **over):
    cfg = {
        "run_name": "tiny", "models": [model_dir], "dtype": "float32", "device_map": None,
        "prompts": ["bird", "farm"], "reps": 3, "batch_reps": 2, "seed": 7, "baseline": True,
        "generation": {"temperature": 1.0, "top_k": 50, "max_words": 15, "max_new_tokens": 30},
        "metrics": True, "bands": ["quartiles", "all", "35-45%"],
        "manipulations": {m: [0.5, 2.0] for m in sweep.MANIP_SPECS},
        "calibration_prompts": ["bird", "farm"],
        "paths": {"output_root": os.path.join(root, "results"),
                  "calib_root": os.path.join(root, "calibration")},
        "merge": {"max_mb": 50}, "slurm": {"partition": "x"},
    }
    cfg.update(over)
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f)
    return cfg


def sh(*args, ok=True):
    env = dict(os.environ, PYTHON=sys.executable, HF_HUB_OFFLINE="1")
    env.pop("FTD_BASE", None)
    r = subprocess.run(list(args), capture_output=True, text=True, env=env)
    if ok and r.returncode != 0:
        raise AssertionError(f"{args} failed:\n{r.stdout[-3000:]}\n{r.stderr[-3000:]}")
    return r


def test_end_to_end(model_dir, root):
    py, sw, run_sh = sys.executable, os.path.join(CLIENT, "sweep.py"), os.path.join(CLIENT, "run.sh")
    cfg_path = os.path.join(root, "tiny.yaml")
    write_cfg(cfg_path, model_dir, root)
    run_dir = os.path.join(root, "results", "tiny")

    # plan + one task, then the same task again: resume must add nothing
    sh(py, sw, "plan", "--config", cfg_path, "--local")
    sh(py, sw, "calibrate", "--run-dir", run_dir)
    sh(py, sw, "task", "--run-dir", run_dir, "--index", "1")
    part = os.path.join(run_dir, "parts", sweep.build_tasks(sweep.read_frozen(run_dir))[1]["part"])
    n1 = len(pd.read_csv(part))
    sh(py, sw, "task", "--run-dir", run_dir, "--index", "1")
    assert len(pd.read_csv(part)) == n1 == 2 * 2 * 3, n1
    print(f"ok  resume: task rerun left {n1} rows, no duplicates")

    r = sh(py, sw, "merge", "--run-dir", run_dir, ok=False)
    assert r.returncode == 1 and "incomplete" in r.stderr, r.stderr
    assert os.path.isdir(os.path.join(run_dir, "parts"))
    print("ok  merge refuses an incomplete run and leaves parts alone")

    write_cfg(cfg_path, model_dir, root, reps=4)
    r = sh(py, sw, "plan", "--config", cfg_path, "--local", ok=False)
    assert r.returncode != 0 and "different config" in r.stderr, r.stderr
    write_cfg(cfg_path, model_dir, root, slurm={"partition": "other", "time": "01:00:00"})
    sh(py, sw, "plan", "--config", cfg_path, "--local")
    print("ok  changed config refused; changed SLURM resources resume")

    sh("bash", run_sh, "--local", cfg_path)
    df = pd.read_csv(os.path.join(run_dir, "tiny.csv"), keep_default_na=False)
    cfg = sweep.read_frozen(run_dir)
    expected = sum(sweep.rows_per_task(cfg, t) for t in sweep.build_tasks(cfg))
    assert len(df) == expected == 2 * 3 * (1 + len(sweep.MANIP_SPECS) * 2 * 6), (len(df), expected)
    assert df.row_id.is_unique and not df.duplicated(["task", "param_value", "prompt_id", "rep"]).any()
    assert set(df.manipulation) == set(sweep.MANIP_SPECS) | {"none"}
    assert (df.n_words <= 15).all()
    assert not os.path.exists(os.path.join(run_dir, "parts")), "parts should be purged"
    for f in ("tiny.log.json", "config.frozen.yaml", "prompts.frozen.csv"):
        assert os.path.exists(os.path.join(run_dir, f)), f
    base = df[df.manipulation == "none"]
    swept = df[(df.manipulation == "gate_noise") & (df.layer_strategy == "all")]
    assert (base.layer_indices == "").all() and len(swept) == 2 * 2 * 3
    print(f"ok  run.sh --local: {len(df)} rows merged, parts purged, csv + log written")
    print(df.stop_reason.value_counts().to_dict())


def main():
    test_bands()
    root = tempfile.mkdtemp(prefix="ftdlesion_tiny_")
    try:
        model_dir = os.path.join(root, "tiny-llama")
        build_tiny(model_dir)
        test_manipulations(model_dir, os.path.join(root, "calib_unit"))
        test_end_to_end(model_dir, root)
    finally:
        shutil.rmtree(root, ignore_errors=True)
    print("ALL PASSED")


if __name__ == "__main__":
    main()
