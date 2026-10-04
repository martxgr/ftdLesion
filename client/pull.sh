#!/bin/bash
# client/pull.sh RUN_NAME -- copy a merged run from Bouchet to client/outputs/.
# Copies the CSV, log and frozen config only (not parts/). Run on the laptop.
# One ssh connection for all files, so one Duo approval.
set -euo pipefail
[ $# -eq 1 ] || { echo "usage: client/pull.sh RUN_NAME" >&2; exit 2; }
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$(dirname "$HERE")/env.sh"      # FTD_BASE, FTD_SSH
SRC="$FTD_BASE/results/$1"
DEST="$HERE/outputs/$1"
mkdir -p "$DEST"
ssh "$FTD_SSH" "cd '$SRC' && tar cf - \$(ls '$1.csv' '$1.log.json' config.frozen.yaml prompts.frozen.csv 2>/dev/null)" \
  | tar xvf - -C "$DEST"
echo "-> $DEST"
