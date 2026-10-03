#!/bin/bash
#SBATCH --job-name=ftd-rate
#SBATCH --partition=gpu_b200
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out
# clinician/crude/rate.sbatch -- the local rater as configure.yaml says, then
# (validate mode) the correlation table against human TLI.
#
# From the repo root:
#   sbatch clinician/crude/rate.sbatch                      # clinician/crude/configure.yaml
#   sbatch clinician/crude/rate.sbatch path/to/other.yaml
#
# Resubmitting resumes: questions already in ratings_long.csv are skipped.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
[ -f clinician/crude/validate/rate_local.py ] || { echo "submit from the ftdLesion repo root" >&2; exit 2; }
source ./env.sh
ftd_activate
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python clinician/crude/validate/rate_local.py --config "${1:-clinician/crude/configure.yaml}" --report
