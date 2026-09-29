#!/bin/bash
# client/pull.sh RUN_NAME -- copy a merged run from Bouchet to client/outputs/.
# Copies the CSV, xlsx, log and frozen config only (not parts/). Run on the laptop.
set -euo pipefail
[ $# -eq 1 ] || { echo "usage: client/pull.sh RUN_NAME" >&2; exit 2; }
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$(dirname "$HERE")/env.sh"      # FTD_BASE, FTD_SSH
SRC="$FTD_BASE/results/$1"
DEST="$HERE/outputs/$1"
mkdir -p "$DEST"
for f in "$1.csv" "$1.log.json" config.frozen.yaml prompts.frozen.csv "$1.xlsx"; do
  scp "$FTD_SSH:$SRC/$f" "$DEST/" || echo "  (no $f)"
done
echo "-> $DEST"
