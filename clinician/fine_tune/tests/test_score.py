#!/usr/bin/env python
"""
clinician/fine_tune/tests/test_score.py -- the fine-tune scorer end to end on CPU.

Tiny random Llama classifier (9 outputs, as train_ftd.py builds it) with two
LoRA "seeds", two synthetic client runs, synthetic human word counts -- no
patient data. Checks length matching, the training-format guard, chunked
scoring, resume and merge.

  python clinician/fine_tune/tests/test_score.py
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
FT = os.path.dirname(HERE)
sys.path.insert(0, os.path.dirname(FT))
from common import length_match  # noqa: E402

SCORE = os.path.join(FT, "validate", "score.py")


def test_trim():
    t = "The man plows. The woman waits by the barn! Then they talk about the harvest and"
    kept, n, how = length_match.trim_to(t, 10)
    assert (kept, n, how) == ("The man plows. The woman waits by the barn!", 9, "sentence"), (kept, n, how)
    assert length_match.trim_to(t, 100)[2] == "none"
    # the nearest sentence end may lie past the target: 11 words is nearer 9 than 3 is
    assert length_match.trim_to("a b c. d e f g h i j k. l m n", 9)[1:] == (11, "sentence")
    kept, n, how = length_match.trim_to("one two three four five six seven", 4)
    assert (kept, n, how) == ("one two three four", 4, "hard")
    # a sentence end that would keep < half the target is not used
    kept, n, how = length_match.trim_to("Hi. one two three four five six seven eight", 8)
    assert how == "hard" and n == 8, (kept, n, how)
    assert length_match.row_target("r1", np.array([50, 100, 150])) == \
        length_match.row_target("r1", np.array([50, 100, 150]))
    print("ok  trimming: sentence cut, hard cut, no cut; targets fixed by row_id")


def build(root):
    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoTokenizer, LlamaConfig, LlamaForSequenceClassification
    tok = AutoTokenizer.from_pretrained("gpt2-medium")
    tok.pad_token = tok.eos_token
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=len(tok), hidden_size=64, intermediate_size=128,
                      num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
                      max_position_embeddings=2048, num_labels=9,
                      pad_token_id=tok.eos_token_id)
    base_dir = os.path.join(root, "base")
    LlamaForSequenceClassification(cfg).save_pretrained(base_dir)
    tok.save_pretrained(base_dir)
    for k, (scale, tcol) in enumerate([(0.5, "response"), (0.8, "response"), (0.5, "exchange")], 1):
        torch.manual_seed(k)
        m = get_peft_model(LlamaForSequenceClassification.from_pretrained(base_dir),
                           LoraConfig(task_type=TaskType.SEQ_CLS, r=4, target_modules=["q_proj", "v_proj"],
                                      modules_to_save=["score"], init_lora_weights=False))
        run = os.path.join(root, "runs" if tcol == "response" else "bad", f"tight_none_seed{k}")
        m.save_pretrained(os.path.join(run, "best_adapter"))
        json.dump({"args": {"text_col": tcol, "target_col": "target", "train": "x/tight_none_train.csv",
                            "scalar_head": False}, "scale": scale},
                  open(os.path.join(run, "results.json"), "w"))
    return base_dir


def client_runs(root):
    rng = np.random.default_rng(0)
    words = "the man plows the field while two women stand apart near the barn .".split()
    for run, n in (("runA", 23), ("runB", 7)):
        rows = []
        for i in range(n):
            text = " ".join(rng.choice(words, rng.integers(0, 120))) + " end"
            if run == "runB" and i == 5:
                text = ""                                # a lesion that said nothing
            rows.append({"row_id": f"{run}:{i}", "run": run, "manipulation": "none" if i < 3 else "gate_shift",
                         "param_value": "" if i < 3 else 2.5, "prompt_id": "farm", "rep": i,
                         "n_words": len(text.split()), "stop_reason": "eos",
                         "output": text.replace(" .", ".\n\n", 1)})
        os.makedirs(os.path.join(root, "results", run))
        pd.DataFrame(rows).to_csv(os.path.join(root, "results", run, f"{run}.csv"), index=False)
    pd.DataFrame({"row_type": "image", "dx_group": "patient",
                  "n_words_participant": rng.integers(10, 60, 50)}).to_csv(os.path.join(root, "human.csv"), index=False)


def main():
    test_trim()
    root = tempfile.mkdtemp(prefix="ftdlesion_ft_")
    try:
        base = build(root)
        client_runs(root)
        cfg = yaml.safe_load(open(os.path.join(FT, "configure.yaml"), encoding="utf-8"))
        cfg.update(name="t", runs=["runA", "runB"], results_root=os.path.join(root, "results"),
                   adapters=os.path.join(root, "runs", "tight_none_seed*", "best_adapter"),
                   model_id=base, quant_4bit=False, dtype="float32", device_map=None,
                   batch_size=4, output_root=os.path.join(root, "ratings"))
        cfg["length_match"]["reference"] = os.path.join(root, "human.csv")
        cp = os.path.join(root, "cfg.yaml")
        yaml.safe_dump(cfg, open(cp, "w"))
        run = lambda *a: subprocess.run([sys.executable, SCORE, *a, "--config", cp],
                                        capture_output=True, text=True)

        r = run("plan", "--no-tokens")
        assert r.returncode == 0, r.stdout + r.stderr[-3000:]
        d = pd.read_csv(os.path.join(root, "ratings", "t", "to_score.csv"), keep_default_na=False)
        assert len(d) == 30 and (d.n_words_rated <= 1.5 * d.trim_target).all()
        assert not d.text_rated.str.contains("\n").any()
        print(f"ok  plan: 30 rows stacked from 2 runs, cut to human lengths {d.trim.value_counts().to_dict()}")

        bad = dict(cfg, adapters=os.path.join(root, "bad", "*", "best_adapter"))
        yaml.safe_dump(bad, open(os.path.join(root, "bad.yaml"), "w"))
        r = subprocess.run([sys.executable, SCORE, "plan", "--no-tokens", "--config",
                            os.path.join(root, "bad.yaml")], capture_output=True, text=True)
        assert r.returncode != 0 and "off-format" in r.stderr, r.stderr[-500:]
        print("ok  adapters trained on another text column are refused")

        r0 = run("score", "--chunk", "0", "--n-chunks", "2")
        assert r0.returncode == 0 and "merged" not in r0.stdout, r0.stdout + r0.stderr[-3000:]
        r1 = run("score", "--chunk", "1", "--n-chunks", "2")
        assert r1.returncode == 0 and "merged 30 rows" in r1.stdout, r1.stdout + r1.stderr[-3000:]
        out = pd.read_csv(os.path.join(root, "ratings", "t", "ratings.csv"))
        live = out.n_words_rated > 0
        assert len(out) == 30 and out.loc[live, "disorg"].notna().all() and out.loc[~live, "disorg"].isna().all()
        assert {"disorg_sd", "disorg_tight_none_seed1", "disorg_tight_none_seed2", "pred_looseness"} <= set(out)
        # the mean of the seeds, each on its own scale
        assert np.allclose(out.loc[live, "disorg"],
                           out.loc[live, ["disorg_tight_none_seed1", "disorg_tight_none_seed2"]].mean(axis=1), atol=1e-5)
        print(f"ok  scored in 2 chunks, last chunk merged: {live.sum()} rated, {(~live).sum()} empty left blank")

        again = run("score", "--chunk", "0", "--n-chunks", "2")
        assert "already scored" in again.stdout and "Loading" not in again.stdout
        print("ok  finished chunks are skipped on resubmit")
        print("ALL PASSED")
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    main()
