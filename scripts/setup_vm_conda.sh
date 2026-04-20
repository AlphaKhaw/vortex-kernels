#!/usr/bin/env bash
# Conda/micromamba fallback bootstrap for a fresh GPU VM.
# Target: Ubuntu 22.04, NVIDIA driver supporting CUDA 12.9+, A100/H100 GPU.
#
# Usage:  bash scripts/setup_vm_conda.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "== 1. Install micromamba if missing =="
if ! command -v micromamba &>/dev/null && [ ! -x "$HOME/.local/bin/micromamba" ]; then
    "${SHELL}" <(curl -L micro.mamba.pm/install.sh) <<< $'\nY\n'"$HOME/micromamba"$'\nY\nY\n'
fi
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$HOME/micromamba}"
eval "$("$HOME/.local/bin/micromamba" shell hook -s bash)"

echo "== 2. Create conda env from environment.yml =="
if ! micromamba env list | grep -q "^  evo2 "; then
    micromamba create -y -n evo2 -f environment.yml
fi
micromamba activate evo2

echo "== 3. Install uv inside the env =="
pip install --quiet uv

echo "== 4. Stage torch first (flash-attn + TE need it at build time) =="
export CUDA_HOME=/usr/local/cuda
export MAX_JOBS="${MAX_JOBS:-8}"
uv pip install "torch==2.7.1" --index-url https://download.pytorch.org/whl/cu128

echo "== 5. Flash-attn prebuilt wheel (do NOT compile) =="
uv pip install "flash-attn==2.8.0.post2" --no-build-isolation

echo "== 6. Evo2 ecosystem =="
uv pip install "huggingface_hub>=0.24,<0.27"
uv pip install "evo2>=0.5.3"

echo "== 7. Clone vortex for kernel development (editable for PR work) =="
if [ ! -d "$HOME/vortex" ]; then
    git clone https://github.com/Zymrael/vortex.git "$HOME/vortex"
    (cd "$HOME/vortex" && git submodule update --init --recursive)
fi

echo "== 8. Optional: FlashFFTConv for Tier-2 benchmark =="
uv pip install --no-build-isolation \
    "git+https://github.com/HazyResearch/flash-fft-conv.git#subdirectory=csrc/flashfftconv" \
    || echo "    (FlashFFTConv install failed — Tier-2 bench will skip)"

echo "== 9. Install this repo in editable mode =="
uv pip install -e ".[dev,bench]"

echo "== 10. Sanity check =="
python scripts/verify_install.py

cat <<'EOF'

================================================================
Setup complete. Activate the env in future sessions with:
    eval "$($HOME/.local/bin/micromamba shell hook -s bash)"
    micromamba activate evo2

Next steps:
    python benchmarks/profile_evo2.py --model evo2_7b_base --seq-lens 8192 32768
================================================================
EOF
