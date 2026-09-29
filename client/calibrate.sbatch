#!/bin/bash
# Calibrate one model of a planned run. Submitted by run.sh.
#   $1 = run dir   $2 = model index
set -euo pipefail
source ./env.sh
ftd_activate
python client/sweep.py calibrate --run-dir "$1" --model "$2"
