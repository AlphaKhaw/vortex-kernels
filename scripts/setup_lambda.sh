#!/usr/bin/env bash
# Pixi-based bootstrap for a fresh Lambda Cloud H100/A100 instance.
# Target: Ubuntu 22.04 + Lambda Stack (CUDA 12.8, nvcc present).
#
# Usage:  bash scripts/setup_lambda.sh
#
# For the conda/micromamba fallback, see scripts/setup_lambda_conda.sh.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "== 1. Install Pixi if missing =="
if ! command -v pixi &>/dev/null && [ ! -x "$HOME/.pixi/bin/pixi" ]; then
    curl -fsSL https://pixi.sh/install.sh | sh
fi
export PATH="$HOME/.pixi/bin:$PATH"
pixi --version

echo "== 2. Clone vortex for source-level kernel work (outside pixi env) =="
if [ ! -d "$HOME/vortex" ]; then
    git clone https://github.com/Zymrael/vortex.git "$HOME/vortex"
    (cd "$HOME/vortex" && git submodule update --init --recursive)
fi

echo "== 3. Resolve + install everything via pixi =="
export CUDA_HOME=/usr/local/cuda
export MAX_JOBS="${MAX_JOBS:-8}"
pixi install

echo "== 4. Sanity check =="
pixi run verify

cat <<'EOF'

================================================================
Setup complete. Use pixi for everything:

    pixi run test              # pytest
    pixi run profile           # baseline Evo2 profiling
    pixi run lint              # ruff check
    pixi run format            # ruff format
    pixi run typecheck         # pyright
    pixi shell                 # drop into the env manually

Optional FlashFFTConv (Tier-2 benchmark comparison):
    pixi install -e full       # adds flash-fft-conv feature

Fallback (no pixi, conda/micromamba instead):
    bash scripts/setup_lambda_conda.sh
================================================================
EOF
