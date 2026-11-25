#!/usr/bin/env bash
# Environment for a single-GPU Linux machine.
#
#   bash setup_env.sh install    # Python 3.12 venv, torch for the host driver, pinned packages
#   bash setup_env.sh download   # SD3.5 Medium, metric models, PartiPrompts (needs `hf auth login`)
#   bash setup_env.sh check      # GPU, CUDA, bf16, attention kernels
#
# Then, in every new shell: source $ROOT/shortcut_env.sh
#
#   ROOT      working directory for the venv, model cache, data and runs (~70 GB, default ~/shortcut_work)
#   TORCH_CU  torch CUDA build: auto (from the driver) | cu132 | cu130 | cu126
set -euo pipefail
ROOT=${ROOT:-$HOME/shortcut_work}
VENV=${VENV:-$ROOT/venv}
ENV_FILE=$ROOT/shortcut_env.sh
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TORCH_CU=${TORCH_CU:-auto}
log() { echo -e "\033[1;32m[setup]\033[0m $*" >&2; }
die() { echo -e "\033[1;31m[setup] $*\033[0m" >&2; exit 1; }
ver_ge() { [ "$(printf '%s\n%s\n' "$2" "$1" | sort -V | head -n1)" = "$2" ]; }

pick_torch() {
  if [ "$TORCH_CU" = auto ]; then
    local drv
    drv=$({ nvidia-smi 2>/dev/null || true; } | grep -oE 'CUDA Version: [0-9.]+' | grep -oE '[0-9.]+$' | head -n1)
    if [ -z "$drv" ] || ver_ge "$drv" 13.2; then TORCH_CU=cu132
    elif ver_ge "$drv" 13.0; then TORCH_CU=cu130
    elif ver_ge "$drv" 12.0; then TORCH_CU=cu126
    else die "the driver supports only CUDA $drv, at least 12.0 is needed"
    fi
    log "driver CUDA ${drv:-unknown} -> torch build $TORCH_CU"
  fi
  case $TORCH_CU in
    cu132) TORCH_VER=2.15.0 ;;
    cu130|cu126) TORCH_VER=2.14.1 ;;
    *) die "unsupported TORCH_CU=$TORCH_CU" ;;
  esac
}

find_python() {
  local c
  for c in python3.12 python3.13 python3; do
    if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys, venv, ensurepip; sys.exit(sys.version_info < (3, 12))' 2>/dev/null; then
      command -v "$c"; return
    fi
  done
  if command -v conda >/dev/null 2>&1; then
    conda create -y -q -p "$ROOT/py312" -c conda-forge python=3.12 >&2
    echo "$ROOT/py312/bin/python"; return
  fi
  die "Python >= 3.12 with venv (or conda) is required"
}

cpu_limit() {  # CPUs of the container quota, not of the host
  local q p
  if [ -r /sys/fs/cgroup/cpu.max ]; then
    read -r q p < /sys/fs/cgroup/cpu.max
    if [ "$q" != max ]; then echo $(( q / p )); return; fi
  fi
  nproc
}

cmd_install() {
  mkdir -p "$ROOT"
  pick_torch
  local py; py=$(find_python)
  [ -x "$VENV/bin/python" ] || "$py" -m venv "$VENV"
  local cpus; cpus=$(cpu_limit)
  cat > "$ENV_FILE" <<EOF
source "$VENV/bin/activate"
export ROOT="$ROOT" HF_HOME="$ROOT/hf"
export OMP_NUM_THREADS=$cpus MKL_NUM_THREADS=$cpus TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
EOF
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  python -m pip install -q -U pip
  pip install --no-cache-dir "torch==$TORCH_VER" --index-url "https://download.pytorch.org/whl/$TORCH_CU"
  pip install --no-cache-dir -r "$REPO/requirements.txt"
  python -c "import torch, diffusers; print('torch', torch.__version__, '| diffusers', diffusers.__version__)"
  log "done. In new shells: source $ENV_FILE"
}

cmd_download() {
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  hf auth whoami >/dev/null 2>&1 || die "run 'hf auth login' with an account that accepted the SD3.5 Medium license"
  hf download stabilityai/stable-diffusion-3.5-medium --include model_index.json \
    --include "scheduler/*" --include "transformer/*" --include "vae/*" \
    --include "text_encoder*/*" --include "tokenizer*/*"
  for repo in openai/clip-vit-large-patch14 yuvalkirstain/PickScore_v1; do
    hf download "$repo" --include "*.json" --include "*.txt" --include model.safetensors
  done
  hf download laion/CLIP-ViT-H-14-laion2B-s32B-b79K --include "*.json" --include "*.txt"
  python -c "from datasets import load_dataset; print(len(load_dataset('nateraw/parti-prompts', split='train')), 'PartiPrompts')"
  du -sh "$HF_HOME" >&2
}

cmd_check() {
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  python "$REPO/scripts/check_env.py"
}

case "${1:-}" in
  install) cmd_install ;;
  download) cmd_download ;;
  check) cmd_check ;;
  *) sed -n '2,11p' "$0"; exit 1 ;;
esac
