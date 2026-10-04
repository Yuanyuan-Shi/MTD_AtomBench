#!/usr/bin/env bash
# One-time setup of a fresh clone (e.g. on the AWS cluster), from the repository root:
#   bash setup.sh
# 1. OpenPI third-party repos (not stored in git), at the commits OpenPI pinned.
# 2. Python environment openpi/.venv from openpi/uv.lock, plus CUDA 12.8 PyTorch.
# 3. atom_mtd_offlinedata/runtime: transformers with OpenPI's patched model files.
# Every step is skipped when already done, so rerunning is safe.
# Afterwards: python -m atom_mtd_offlinedata.prepare --config <config> (dataset + base model).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

clone_pinned() {  # url, directory, commit
  if [[ -n "$(ls -A "$2" 2>/dev/null)" ]]; then
    echo "[third_party] $2 present, skipped"
    return
  fi
  git clone --quiet "$1" "$2"
  git -C "$2" checkout --quiet "$3"
  echo "[third_party] $2 @ $3"
}
clone_pinned https://github.com/Physical-Intelligence/aloha.git openpi/third_party/aloha \
  d1dc83afd89ded4379851257fe5d85632d31d5ec
clone_pinned https://github.com/Lifelong-Robot-Learning/LIBERO.git openpi/third_party/libero \
  f78abd68ee283de9f9be3c8f7e2a9ad60246e95c

if [[ -x openpi/.venv/bin/python ]]; then
  echo "[venv] openpi/.venv present, skipped"
else
  (cd openpi && GIT_LFS_SKIP_SMUDGE=1 uv sync)
  # CUDA 12.8 builds: required by Blackwell GPUs, also used on H100 for identical numerics.
  uv pip install --python openpi/.venv/bin/python --reinstall \
    torch==2.7.1+cu128 torchvision==0.22.1+cu128 \
    --index-url https://download.pytorch.org/whl/cu128
  echo "[venv] openpi/.venv created"
fi

# OpenPI's PyTorch pi0.5 needs patched transformers files (adaRMS, precision, KV cache).
# Instead of overwriting the environment's copy (OpenPI's README), keep a patched copy
# that env.sh puts first on PYTHONPATH.
runtime=atom_mtd_offlinedata/runtime/python/transformers
installed=$(env -u PYTHONPATH openpi/.venv/bin/python -c \
  "import os, transformers; print(transformers.__version__, os.path.dirname(transformers.__file__))")
version=${installed%% *}
if [[ "$version" != "4.53.2" ]]; then
  echo "transformers $version installed; OpenPI's patch files are written for 4.53.2" >&2
  exit 1
fi
rm -rf "$runtime"
mkdir -p "$(dirname "$runtime")"
cp -r "${installed#* }" "$runtime"
cp -r openpi/src/openpi/models_pytorch/transformers_replace/* "$runtime"/
echo "[runtime] $runtime rebuilt (transformers $version + OpenPI patch)"

mkdir -p atom_mtd_offlinedata/reports  # SLURM writes job logs here and needs the directory
echo "Setup complete. Next: source atom_mtd_offlinedata/env.sh, then run prepare.py."
