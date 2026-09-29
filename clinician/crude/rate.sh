#!/bin/bash
# clinician/crude/rate.sh -- run the local rater as configure.yaml says.
#
#   clinician/crude/rate.sh [config]          submit one GPU job: rate, then validate
#   clinician/crude/rate.sh --here [config]   run in this shell (a GPU node, or CPU tests)
#
# Resubmitting resumes: questions already in ratings_long.csv are skipped.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$(dirname "$HERE")")"
MODE=slurm
[ "${1:-}" = "--here" ] && { MODE=here; shift; }
CONFIG="$(cd "$(dirname "${1:-$HERE/configure.yaml}")" && pwd)/$(basename "${1:-$HERE/configure.yaml}")"
PY=${PYTHON:-python}

if [ "$MODE" = here ]; then
  exec "$PY" "$HERE/validate/rate_local.py" --config "$CONFIG" --report
fi

source "$REPO/env.sh"
ftd_activate
read -r S_PARTITION S_GPUS S_CPUS S_MEM S_TIME NAME < <("$PY" - "$CONFIG" <<'EOF'
import sys, yaml
c = yaml.safe_load(open(sys.argv[1])); s = c["slurm"]
print(s["partition"], s["gpus"], s["cpus"], s["mem"], s["time"], c["name"])
EOF
)
mkdir -p "$FTD_BASE/logs"
sbatch --chdir="$REPO" --job-name="rate-$NAME" --partition="$S_PARTITION" \
  --gpus="$S_GPUS" --cpus-per-task="$S_CPUS" --mem="$S_MEM" --time="$S_TIME" \
  --output="$FTD_BASE/logs/%x_%j.out" --error="$FTD_BASE/logs/%x_%j.err" \
  --wrap="source ./env.sh && ftd_activate && python clinician/crude/validate/rate_local.py --config '$CONFIG' --report"
