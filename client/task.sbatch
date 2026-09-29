#!/bin/bash
# One (model, manipulation, band) task of a planned run. Submitted by run.sh,
# which supplies --array, resources and log paths; don't sbatch this directly.
#   $1 = run dir
set -euo pipefail
source ./env.sh      # run.sh sets --chdir to the repo root
ftd_activate
echo "task ${SLURM_ARRAY_TASK_ID} on $(hostname)  gpus=${CUDA_VISIBLE_DEVICES:-none}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python client/sweep.py task --run-dir "$1"
