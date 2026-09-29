# ftdLesion

Lesion Llama's MLP layers, rate the output for formal thought disorder, and
analyse which lesions produce human-like FTD rather than noise.

```
ftdLesion/
├── env.sh                 cluster paths + conda activation; every job script sources it
├── prompts/registry.csv   every prompt, stable IDs (client + analysis read it; clinician never does)
├── client/                the lesioned models
│   ├── configure.yaml     models, prompts, manipulations x levels, layer bands, length cap
│   ├── run.sh             `sbatch client/run.sh`: calibrate -> task array -> merge
│   ├── sweep.py           the engine (ported from llmSchizophrenia/8_sweep.py)
│   ├── pull.sh            copy a merged run from Bouchet to client/outputs/
│   ├── tests/             CPU test on a tiny random Llama
│   ├── outputs/           <run>/<run>.csv + .log.json              (gitignored)
│   └── archive/           legacy run_1 ... run_7                    (gitignored)
├── clinician/             the raters -- see response only, never the prompt
│   ├── common/            shared validation metrics
│   ├── crude/             per-item API rater
│   ├── single_shot/       placeholder
│   └── fine_tune/         LoRA TLI scorer
└── analysis/
    └── ratings/           clinician ratings of client runs            (gitignored)
```

## Data rule

Git holds code only. Patient transcripts live in
`Analyses/0_securedata/0_finetunedata/` (laptop) and `$FTD_STORE` (Bouchet);
sweep outputs in `$FTD_BASE/results` (Bouchet) or `client/outputs/` (laptop).
`.gitignore` ignores every CSV except `prompts/registry.csv` as a backstop.

## Client

```bash
bash client/run.sh --dry-run     # size of the run in configure.yaml
sbatch client/run.sh             # on Bouchet: submit it (from the repo root)
bash client/run.sh --local cfg.yaml   # on the laptop: run every step here
python client/tests/test_tiny.py # check the engine end to end on CPU
```

A run is a directory `<output_root>/<run_name>/` holding the frozen config,
the prompts it used, and after merging `<run_name>.csv` with its
`<run_name>.log.json` (config, git commit, calibration hash, per-task counts,
stop reasons). Resubmitting resumes; a changed config under the same
`run_name` is refused.

## Clinician (crude)

```bash
sbatch clinician/crude/rate.sbatch               # on Bouchet: one GPU job, rate then validate
python clinician/crude/tests/test_local.py       # CPU check on synthetic transcripts
```

`backend: local` scores with Llama-3.3-70B on our GPUs, reading each answer off
the next-token distribution (`score` = most likely answer, `ev` = expected
score, `p_mass` = probability on valid answers). `backend: api` does the same
questions through the Anthropic Batches API (`validate/rate_api.py`). Ratings
of patient transcripts go to `$FTD_RATINGS`, never the repo.
