#!/usr/bin/env bash
# Launch loCAII with its isolated Python environment from any working directory.
set -euo pipefail
APP_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${LOCAII_PYTHON:-/home/andry/miniconda3/envs/locaii/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo 'loCAII Python environment is missing. Run scripts/install.sh first.' >&2
  command -v zenity >/dev/null && zenity --error --text='loCAII needs setup. Run scripts/install.sh in the application directory.' || true
  exit 1
fi
unset QT_PLUGIN_PATH QT_QPA_PLATFORM_PLUGIN_PATH
export PYTHONPATH="$APP_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PATH="$HOME/.local/bin:$PATH"
cd "$APP_ROOT"
exec "$PYTHON_BIN" -m locaii.app "$@"
