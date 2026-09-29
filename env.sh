# env.sh -- cluster environment shared by every job script. SOURCE it, don't run it.
#
# This is the only file that knows where things live on Bouchet. Job scripts
# source it instead of hard-coding paths, so moving the work base is a one-line
# edit here rather than a per-script edit.
#
# Laptop runs (client/run.sh --local) don't need it: with FTD_BASE unset, the
# client writes to client/outputs/ inside the repo, which is gitignored.

# --- storage ----------------------------------------------------------------
# Work base: sweep results, calibration, SLURM logs. Project storage, persistent.
export FTD_BASE=/nfs/roberts/project/pi_prc29/mam475/kv_ablation

# Fine-tune store: training data, adapters, runs (see clinician/fine_tune).
export FTD_STORE=${FTD_STORE:-$HOME/project_pi_prc29/mam475/ftd}

# Patient transcripts + human TLI, and the rater's scores of them. Both stay on
# the allocation, never in the repo (clinician/crude/configure.yaml reads these).
export FTD_PATIENT_DATA=${FTD_PATIENT_DATA:-$FTD_STORE/data/tli_master.csv}
export FTD_RATINGS=${FTD_RATINGS:-$FTD_STORE/ratings}

# --- Hugging Face -----------------------------------------------------------
# Weights live on SCRATCH, which is purged after ~60 idle days. If a job fails
# with what looks like an auth error, check the cache first:
#   du -sh $HF_HOME/hub/models--meta-llama--Llama-3.3-70B-Instruct   # expect 132G
export HF_HOME=/nfs/roberts/scratch/pi_prc29/mam475/hf_cache
# Token kept in $HOME deliberately, so a scratch purge can't take it.
export HF_TOKEN_PATH=$HOME/.cache/huggingface/token
# Compute nodes have no network: fail loudly if weights aren't pre-staged.
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

# --- software ---------------------------------------------------------------
FTD_CONDA_ENV=${FTD_CONDA_ENV:-llm_ablation}

ftd_activate() {
  module load miniconda 2>/dev/null || true
  # batch shells don't run conda's init; without the hook `activate` fails
  eval "$(conda shell.bash hook)"
  conda activate "$FTD_CONDA_ENV"
  echo "python: $(command -v python)"
}

# --- laptop <-> cluster -----------------------------------------------------
# Used by client/pull.sh. VERIFY the login host before first use.
FTD_SSH=${FTD_SSH:-mam475@bouchet.ycrc.yale.edu}
