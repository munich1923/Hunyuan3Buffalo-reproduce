#!/usr/bin/env bash
set -euo pipefail
ROOT=/root/userdata/zc/hunyuan3d-buffalo-1.0-editing
cd "$ROOT"
TASK_CACHE="$ROOT/environment/cache/stage2"
TASK_TMP="$ROOT/environment/tmp/stage2"
mkdir -p "$TASK_CACHE" "$TASK_TMP"
export TMPDIR="$TASK_TMP" TMP="$TASK_TMP" TEMP="$TASK_TMP"
export XDG_CACHE_HOME="$TASK_CACHE/xdg" HF_HOME="$TASK_CACHE/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"
export HF_DATASETS_CACHE="$HF_HOME/datasets" TRANSFORMERS_CACHE="$HF_HOME/transformers"
export TORCH_HOME="$TASK_CACHE/torch" TORCH_EXTENSIONS_DIR="$TASK_CACHE/torch_extensions"
export TRITON_CACHE_DIR="$TASK_CACHE/triton" CUDA_CACHE_PATH="$TASK_CACHE/cuda"
export PIP_CACHE_DIR="$TASK_CACHE/pip" CONDA_PKGS_DIRS="$TASK_CACHE/conda"
export MPLCONFIGDIR="$TASK_CACHE/matplotlib" PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
TASK_ENV=shape
if [[ "${1:-}" == cache_stage2_text.py ]]; then TASK_ENV=vlm; fi
TASK_SCRIPT="${1:?Pass a Stage 2 script name}"; shift
if [[ "$TASK_SCRIPT" == */* || "$TASK_SCRIPT" != *stage2*.py ]]; then
  echo "Only project Stage 2 script basenames are accepted" >&2; exit 2
fi
exec "$ROOT/environment/venvs/$TASK_ENV/bin/python" -B -u "$ROOT/scripts/$TASK_SCRIPT" "$@"
