#!/usr/bin/env python
"""
clinician/crude/tests/test_local.py -- the local rater end to end on CPU.

Uses a tiny random Llama (gpt2 vocab) and SYNTHETIC transcripts -- no patient
data -- and checks that rate_local.py scores every question, resumes without
duplicates, and that report.py produces the validation table.

  python clinician/crude/tests/test_local.py
"""

import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
CRUDE = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(CRUDE))
sys.path.insert(0, os.path.join(REPO, "client", "tests"))
from test_tiny import build_tiny  # noqa: E402

ITEMS = ["poverty_of_speech", "weakening_of_goal", "perseveration_of_ideas", "looseness",
         "peculiar_use_of_words", "peculiar_sentences", "peculiar_logic", "distractibility"]


def synthetic(path, n_sessions=12):
    rng = np.random.default_rng(0)
    words = "the man is plowing a field and two women stand near the barn looking away".split()
    rows = []
    for s in range(n_sessions):
        for pic in ("farm", "embrace", "bridge"):
            r = {"row_id": f"s{s}{pic}", "row_type": "image", "session_key": f"s{s}",
                 "participant": f"p{s // 2}", "dx_group": "patient" if s % 2 else "control",
                 "response": " ".join(rng.choice(words, rng.integers(8, 30))),
                 "prompt": "", "exchange": ""}
            r["n_words_participant"] = len(r["response"].split())
            for k in ITEMS:
                r[f"item_{k}"] = rng.choice([0, 0, 0.25, 0.5, 1])
            r["tli_impoverishment"] = r["item_poverty_of_speech"] + r["item_weakening_of_goal"]
            r["tli_disorganisation"] = sum(r[f"item_{k}"] for k in ITEMS[3:])
            r["tli_total"] = r["tli_impoverishment"] + r["tli_disorganisation"]
            rows.append(r)
    pd.DataFrame(rows).to_csv(path, index=False)
    return len(rows)


def main():
    root = tempfile.mkdtemp(prefix="ftdlesion_rate_")
    try:
        model = os.path.join(root, "tiny-llama")
        build_tiny(model)
        n = synthetic(os.path.join(root, "master.csv"))
        cfg = yaml.safe_load(open(os.path.join(CRUDE, "configure.yaml"), encoding="utf-8"))
        cfg["name"] = "tiny"
        cfg["local"].update(model=model, dtype="float32", device_map=None, batch_size=5)
        cfg["validate"].update(data=os.path.join(root, "master.csv"), output_root=root)
        cfg_path = os.path.join(root, "cfg.yaml")
        yaml.safe_dump(cfg, open(cfg_path, "w"))
        script = os.path.join(CRUDE, "validate", "rate_local.py")
        run = lambda: subprocess.run([sys.executable, script, "--config", cfg_path, "--report"],
                                     capture_output=True, text=True)
        r = run()
        assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-3000:]
        od = os.path.join(root, "tiny")
        long = pd.read_csv(os.path.join(od, "ratings_long.csv"))
        n_q = n * (len(cfg["instruments"]["crude"]["items"]) + len(ITEMS))
        assert len(long) == n_q, (len(long), n_q)
        assert long.p_mass.between(0, 1).all() and long.ev.notna().all()
        tli = long[long.instrument == "tli"]
        assert set(tli.score) <= {0, 0.25, 0.5, 1, 2} and tli.ev.between(0, 2).all()
        crude = long[long.instrument == "crude"]
        assert crude.score.between(0, 10).all() and set(crude.score) <= set(range(11))
        wide = pd.read_csv(os.path.join(od, "ratings.csv"))
        assert len(wide) == n and "rater_ev_tli_looseness" in wide and "rater_tangential" in wide
        print(f"ok  {len(long)} ratings for {n} synthetic responses; median p_mass "
              f"{long.p_mass.median():.3f} (random model, so ~chance)")
        r2 = run()
        assert r2.returncode == 0 and "already done" in r2.stdout
        assert len(pd.read_csv(os.path.join(od, "ratings_long.csv"))) == n_q
        print("ok  rerun resumed: nothing re-scored, no duplicates")
        val = pd.read_csv(os.path.join(od, "validation.csv"))
        assert {"picture", "session"} <= set(val.level) and "rater_tli_total" in set(val.rater)
        print(f"ok  report: {len(val)} validation rows")
        a = subprocess.run([sys.executable, os.path.join(CRUDE, "validate", "aggregate.py"),
                            "--config", cfg_path, "--repeats", "2", "--n-boot", "50"],
                           capture_output=True, text=True)
        assert a.returncode == 0, a.stdout[-2000:] + a.stderr[-3000:]
        sys.path.insert(0, os.path.dirname(CRUDE))
        from common import aggregate as agg
        m = agg.load(os.path.join(od, "aggregate.json"))
        s = agg.apply(wide, m)
        assert len(s) == n and s.notna().all() and (m["weights"] >= 0).all()
        cv = pd.read_csv(os.path.join(od, "aggregate_cv.csv"))
        assert {"pool_nnls", "tli_nnls", "tli_disorg_sum"} <= set(cv.model)
        print(f"ok  aggregate: {len(cv)} candidates cross-validated, weights frozen and re-applied")
        print("ALL PASSED")
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    main()
