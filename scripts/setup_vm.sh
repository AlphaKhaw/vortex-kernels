#!/usr/bin/env bash
# Pixi-based bootstrap for a fresh GPU VM (RunPod / Lambda / any Ubuntu box).
# Target: Ubuntu 22.04, NVIDIA driver supporting CUDA 12.9+, A100/H100 GPU.
#
# Usage:  bash scripts/setup_vm.sh
#
# For the conda/micromamba fallback, see scripts/setup_vm_conda.sh.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "== 1. Install Pixi if missing =="
if ! command -v pixi &>/dev/null && [ ! -x "$HOME/.pixi/bin/pixi" ]; then
    curl -fsSL https://pixi.sh/install.sh | sh
fi
export PATH="$HOME/.pixi/bin:$PATH"
pixi --version

echo "== 2. Clone the vortex fork as a sibling dir (../vortex) =="
# pixi.toml editable-installs vtx from ../vortex, so this must exist before
# `pixi install`. The kernel work lands on the triton-hc-kernels branch and
# becomes the upstream PR. Prerequisite: fork Zymrael/vortex on GitHub once
# (`gh repo fork Zymrael/vortex --fork-name vortex`, or the web UI).
VORTEX_FORK="${VORTEX_FORK:-git@github.com:AlphaKhaw/vortex.git}"
VORTEX_DIR="$(dirname "$REPO_ROOT")/vortex"
if [ ! -d "$VORTEX_DIR" ]; then
    git clone "$VORTEX_FORK" "$VORTEX_DIR" || {
        echo "ERROR: could not clone $VORTEX_FORK"
        echo "Fork Zymrael/vortex on GitHub first, or rerun with VORTEX_FORK=<url>."
        exit 1
    }
    git -C "$VORTEX_DIR" remote add upstream https://github.com/Zymrael/vortex.git
    git -C "$VORTEX_DIR" checkout -b triton-hc-kernels
fi
echo "   vortex: $VORTEX_DIR  (branch $(git -C "$VORTEX_DIR" branch --show-current))"

echo "== 3. Resolve + install everything via pixi =="
export CUDA_HOME=/usr/local/cuda
export MAX_JOBS="${MAX_JOBS:-8}"
pixi install

echo "== 4. Sanity check =="
pixi run verify

cat <<'EOF'

================================================================
Setup complete. Use pixi for everything:

    pixi run check             # lint + typecheck + fast tests
    pixi run test-gpu          # GPU-marked tests
    pixi run profile           # baseline Evo2 profiling
    pixi shell                 # drop into the env manually

The vortex fork is editable-installed: edits to ../vortex/vortex/*.py are
live in this env. Verify:  python -c "import vortex; print(vortex.__file__)"

Fallback (no pixi, conda/micromamba instead):
    bash scripts/setup_vm_conda.sh
================================================================
EOF
