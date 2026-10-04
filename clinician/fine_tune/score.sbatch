#!/bin/bash
#SBATCH --job-name=ftd-ftscore
#SBATCH --partition=gpu_b200
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --array=0-15
#SBATCH --output=logs/%x_%A_%a.out
# clinician/fine_tune/score.sbatch -- score client outputs with the fine-tuned
# TLI model, one chunk of rows per array task (4-bit 70B fits one GPU). The
# last chunk to finish merges them into ratings.csv.
#
# From the repo root, after `python clinician/fine_tune/validate/score.py plan`:
#   sbatch clinician/fine_tune/score.sbatch [config]
#
# Resubmitting skips finished chunks. Keep --array the same between
# submissions: the chunk boundaries depend on the number of tasks.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
[ -f clinician/fine_tune/validate/score.py ] || { echo "submit from the ftdLesion repo root" >&2; exit 2; }
export FTD_CONDA_ENV=${FTD_CONDA_ENV:-ftd}   # the fine-tune env: peft + bitsandbytes
source ./env.sh
ftd_activate
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python clinician/fine_tune/validate/score.py score --config "${1:-clinician/fine_tune/configure.yaml}"
