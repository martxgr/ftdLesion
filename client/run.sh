#!/bin/bash
# client/run.sh -- run whatever client/configure.yaml describes.
#
#   client/run.sh                        submit to SLURM: calibrate -> task array -> merge
#   client/run.sh path/to/other.yaml     same, with another config
#   client/run.sh --local [config]       every step in this shell, no SLURM
#   client/run.sh --dry-run [config]     print the run size and exit
#
# Resubmitting the same config resumes: finished rows are skipped, calibration
# is reused. A changed config with the same run_name is refused -- rename it.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$HERE")"
MODE=slurm
case "${1:-}" in
  --local)   MODE=local; shift ;;
  --dry-run) MODE=dry;   shift ;;
esac
CONFIG="${1:-$HERE/configure.yaml}"
PY=${PYTHON:-python}

if [ "$MODE" = dry ]; then
  exec "$PY" "$HERE/sweep.py" plan --config "$CONFIG" --dry-run
fi

if [ "$MODE" = slurm ]; then
  source "$REPO/env.sh"
  ftd_activate
fi

LOCAL_FLAG=""; [ "$MODE" = local ] && LOCAL_FLAG="--local"
eval "$("$PY" "$HERE/sweep.py" plan --config "$CONFIG" --shell $LOCAL_FLAG)"
SWEEP="$HERE/sweep.py"

if [ "$MODE" = local ]; then
  "$PY" "$SWEEP" calibrate --run-dir "$RUN_DIR"
  "$PY" "$SWEEP" task --run-dir "$RUN_DIR" --index all
  "$PY" "$SWEEP" merge --run-dir "$RUN_DIR"
  exit 0
fi

# ---- SLURM: three jobs chained by dependency --------------------------------
LOGS="$FTD_BASE/logs"
mkdir -p "$LOGS"
COMMON=(--chdir="$REPO" --partition="$S_PARTITION" --cpus-per-task="$S_CPUS")

DEP=""
if [ -n "$CALIB_NEEDED" ]; then
  CAL_IDS=()
  for m in $CALIB_NEEDED; do
    CAL_IDS+=("$(sbatch --parsable "${COMMON[@]}" --gpus="$S_GPUS" --mem="$S_MEM" \
      --time="$S_CAL_TIME" --job-name=ftd-calib \
      --output="$LOGS/%x_%j.out" --error="$LOGS/%x_%j.err" \
      "$HERE/calibrate.sbatch" "$RUN_DIR" "$m")")
  done
  DEP="--dependency=afterok:$(IFS=:; echo "${CAL_IDS[*]}")"
  echo "calibration job(s): ${CAL_IDS[*]}"
else
  echo "calibration cached for every model"
fi

ARRAY="0-$((N_TASKS - 1))"; [ -n "$S_MAXC" ] && ARRAY="$ARRAY%$S_MAXC"
ARR_ID=$(sbatch --parsable $DEP "${COMMON[@]}" --gpus="$S_GPUS" --mem="$S_MEM" \
  --time="$S_TIME" --array="$ARRAY" --job-name=ftd-sweep \
  --output="$LOGS/%x_%A_%a.out" --error="$LOGS/%x_%A_%a.err" \
  "$HERE/task.sbatch" "$RUN_DIR")
echo "task array: $ARR_ID ($N_TASKS tasks)"

MRG_ID=$(sbatch --parsable --dependency=afterok:"$ARR_ID" --chdir="$REPO" \
  --partition="$S_MERGE_PARTITION" --cpus-per-task=2 --mem="$S_MERGE_MEM" \
  --time=02:00:00 --job-name=ftd-merge \
  --output="$LOGS/%x_%j.out" --error="$LOGS/%x_%j.err" \
  "$HERE/merge.sbatch" "$RUN_DIR")
echo "merge: $MRG_ID"
echo "run dir: $RUN_DIR"
