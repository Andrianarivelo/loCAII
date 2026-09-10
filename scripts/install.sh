#!/usr/bin/env bash
# Recreate the isolated environment and official standalone inference runtime.
set -euo pipefail
APP_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_BIN="${LOCAII_CONDA:-$HOME/miniconda3/bin/conda}"
PYTHON_BIN="${LOCAII_PYTHON:-$HOME/miniconda3/envs/locaii/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  "$CONDA_BIN" create -n locaii -c conda-forge python=3.12 ffmpeg -y
elif [[ ! -x "$(dirname -- "$PYTHON_BIN")/ffprobe" ]]; then
  "$CONDA_BIN" install -p "$(dirname -- "$(dirname -- "$PYTHON_BIN")")" -c conda-forge ffmpeg -y
fi
"$PYTHON_BIN" -m pip install -e "$APP_ROOT"
"$PYTHON_BIN" "$APP_ROOT/scripts/install_runtime.py"
printf '%s\n' 'Installation complete. Open launch.sh or install the desktop shortcut.'
