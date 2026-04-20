# vortex-kernels

Optimized Triton inference kernels for [Vortex](https://github.com/Zymrael/vortex) /
[Evo2](https://github.com/ArcInstitute/evo2). Fills the empty scaffolding slots
`vortex/ops/hcl_interface.py`, `hcm_interface.py`, `hcs_interface.py` with fused
kernels that reduce FFT-conv kernel launches on the HCL/HCM hot paths.

**Third-party.** Not affiliated with Arc Institute or the Vortex core team. The
goal is an upstream PR against [Zymrael/vortex#16](https://github.com/Zymrael/vortex/issues/16);
until that merges, this package monkey-patches `vortex` at import time as a
drop-in accelerator.

## Scope

Evo2 7B spends ~40% of forward-pass CUDA time in FFT convolutions across HCL/HCM
layers. `use_flashfft: False` is the default in every shipped config; flipping it
recovers ~1.5x on HCL alone. But HCM's `parallel_fir` branch never dispatches to
FlashFFT — it always calls the unfused `fftconv_func`, even when `use_flashfft=True`.
That's the primary optimization target here.

Three planned deliverables:

| Interface | Target | Status |
|---|---|---|
| `hcm_interface.py` | Fused fftconv_func replacing 6-launch path in `parallel_fir` (`fir_length >= 128`) | stub |
| `hcs_interface.py` | Wire existing `vortex/ops/hyena_x/triton_indirect_fwd.py` into `parallel_fir` default branch | stub |
| `hcl_interface.py` | Fused scale/multiply around cuFFT in `parallel_iir` (marginal if FlashFFTConv is installed) | stub |

## Setup — GPU VM (RunPod / Lambda / any CUDA 12.9+ host)

### Primary: Pixi (fast, lockfile, task runner)

```bash
git clone <this repo> && cd vortex-kernels
bash scripts/setup_vm.sh
```

That installs [Pixi](https://pixi.sh) (Rust-based conda-forge resolver, 2-5× faster
than micromamba), resolves everything from `pixi.toml` (CUDA 12.9 + conda-forge
pytorch 2.7.x with CUDA build + TE 2.3.0 binary + flash-attn wheel + evo2),
clones vortex for source work, and runs the sanity check.

All subsequent commands are Pixi tasks:

```bash
pixi run verify        # sanity check imports
pixi run test          # pytest
pixi run profile       # baseline Evo2 profiling
pixi run lint          # ruff check
pixi run format        # ruff format
pixi run typecheck     # basedpyright
pixi shell             # drop into the activated env manually
```

For the Tier-2 FlashFFTConv benchmark comparison:

```bash
pixi install -e full   # pulls flash-fft-conv from git
```

### Fallback: conda/micromamba + uv (traditional)

Matches the install docs in the Vortex and Evo2 READMEs verbatim:

```bash
bash scripts/setup_vm_conda.sh
```

Uses `environment.yml` for the conda-forge base (CUDA + TE) and `uv` for the pip
layer. Works without Pixi.

### Local macOS development

CUDA bits won't work on macOS, but you can edit / type-check / write unit tests
that skip if `torch.cuda.is_available() is False`:

```bash
pip install -e ".[dev]"
pytest tests/  # most tests will skip without a GPU
```

## Workflow

1. **Profile baseline.** `pixi run profile` — confirms FFT dominance before
   touching any kernel code.
2. **Tier 2 comparison.** Enable `use_flashfft: True` in the Evo2 config and rerun.
3. **Write Triton kernel.** Start with `vortex_kernels/interfaces/hcm.py` (biggest
   unclaimed opportunity — HCM's FFT path is never accelerated by FlashFFTConv).
4. **Correctness tests.** `pixi run test`.
5. **Three-tier end-to-end.** Scripts under `benchmarks/bench_tier{1,2,3}_*.py`.
6. **Open the vortex issue** with data in hand — template in `docs/issue_draft.md`.

## License

Apache 2.0 (matches Vortex and Evo2).
