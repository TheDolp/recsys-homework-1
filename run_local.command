#!/bin/bash
set -euo pipefail

project_dir="$(cd "$(dirname "$0")" && pwd)"
venv_dir="$project_dir/.venv"
state_dir="$project_dir/.cache/jupyter"

if ! command -v python3 >/dev/null 2>&1; then
  echo "Нужен Python 3.9 или новее."
  exit 1
fi

if [ ! -x "$venv_dir/bin/python" ]; then
  python3 -m venv "$venv_dir"
fi

"$venv_dir/bin/python" -m pip install -q -r "$project_dir/requirements.txt"
export JUPYTER_CONFIG_DIR="$state_dir/config"
export JUPYTER_DATA_DIR="$state_dir/data"
export JUPYTER_RUNTIME_DIR="$state_dir/runtime"
export MPLCONFIGDIR="$state_dir/matplotlib"
export IPYTHONDIR="$state_dir/ipython"
export PYTHONPYCACHEPREFIX="$project_dir/.cache/pycache"
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
mkdir -p "$JUPYTER_CONFIG_DIR" "$JUPYTER_DATA_DIR" "$JUPYTER_RUNTIME_DIR" "$MPLCONFIGDIR" "$IPYTHONDIR" "$PYTHONPYCACHEPREFIX"
cd "$project_dir"
"$venv_dir/bin/python" -m jupyter lab --ServerApp.root_dir="$project_dir" "$project_dir/experiments.ipynb"
