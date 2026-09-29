#!/bin/bash
# Concatenate a finished run's parts into <run>.csv + <run>.log.json, then
# purge the parts. Refuses (exit 1, parts untouched) if any task is incomplete.
# Submitted by run.sh after the task array.
#   $1 = run dir
set -euo pipefail
source ./env.sh
ftd_activate
python client/sweep.py merge --run-dir "$1"
