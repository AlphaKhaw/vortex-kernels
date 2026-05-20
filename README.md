# vortex-kernels

Profiling and benchmark harness for a third-party Triton kernel contribution to
[Vortex](https://github.com/Zymrael/vortex) /
[Evo2](https://github.com/ArcInstitute/evo2) — fused inference kernels for the
three Hyena conv layer kinds (HCS / HCM / HCL).

**Third-party.** Not affiliated with Arc Institute or the Vortex core team.
Tracks [Zymrael/vortex#16](https://github.com/Zymrael/vortex/issues/16) and
[#76](https://github.com/Zymrael/vortex/issues/76).

## How this is organized

The kernels are developed on a **vortex fork** (branch `triton-hc-kernels`),
each opt-in behind a `use_{hcs,hcm,hcl}_kernel` config flag — that branch is
the upstream PR. This repo is the **harness**: profiler, microbenches,
correctness check, results.

| Repo | Holds |
|---|---|
| vortex fork (`../vortex`, branch `triton-hc-kernels`) | the three kernels + the `use_{hcs,hcm,hcl}_kernel` dispatch — the PR |
| this repo | `benchmarks/`, `results/`, `tests/` |

## The kernels

The HCL/HCM/HCS Hyena conv layers are ~21% of Evo2 7B forward CUDA time at long
context (L=65k), and `use_flashfft: False` is the default in every shipped
config — so all three run unfused.

| Layer | Filter | Kernel | Dispatches in |
|---|---|---|---|
| HCS | length 7 | from-scratch depthwise-conv Triton kernel | `parallel_fir`, gated short conv |
| HCM | length 128 | fused FFT-conv epilogues around cuFFT | `parallel_fir`, `fir_length >= 128` |
| HCL | length L | tiled `compute_filter` + FFT-conv — avoids the `(D, state_size, L)` fp32 tensor that OOMs Evo2 at L=131k | `parallel_iir` |

Each is gated by a flag defaulting to off — zero behavioral change when off.

## Setup — Linux + CUDA 12.9 host

```bash
git clone https://github.com/AlphaKhaw/vortex-kernels.git ~/vortex-kernels
cd ~/vortex-kernels
bash scripts/setup_vm.sh
```

`setup_vm.sh` installs [Pixi](https://pixi.sh), clones the vortex fork as a
sibling (`../vortex`) on the `triton-hc-kernels` branch, then `pixi install`
resolves the full stack (CUDA 12.9, PyTorch 2.7 cuda build, Transformer
Engine 2.3, flash-attn, evo2) and editable-installs the fork.

```bash
pixi run verify          # sanity-check imports
pixi run check           # lint + typecheck + fast tests (pre-push gate)
```

## Benchmarks

### End-to-end profile sweep — Evo2 forward across `(seq_len, kernel-config)`

Output dir is chosen automatically: `results/<gpu>/baseline_profile/` when
`--triton` is empty, `results/<gpu>/progression/<flagset>/` when it is set.

```bash
# Stock baseline (flags all off).
pixi run profile --seq-lens 8192 32768 65536

# Progression — add one kernel at a time.
pixi run profile --seq-lens 8192 32768 65536 --triton hcs
pixi run profile --seq-lens 8192 32768 65536 --triton hcs,hcm
pixi run profile --seq-lens 8192 32768 65536 131072 --triton hcs,hcm,hcl
```

Each run emits per-`seq_len` JSON summaries, per-layer breakdowns, op-category
breakdowns, plots, and a `report.md`. Chrome traces are written but
`.gitignore`'d (they are multi-MB) — load locally at `chrome://tracing`.

### Per-kernel microbench — kernel vs stock reference

```bash
pixi run python -m benchmarks.bench_hcl    # vs unfused PyTorch FFT-IIR
pixi run python -m benchmarks.bench_hcm    # vs FFTConv
pixi run python -m benchmarks.bench_hcs    # vs cuDNN depthwise
```

Each emits a single JSON under `results/<gpu>/microbench/` with stock-vs-kernel
timing, peak memory, max numerical diff, and the OOM-crossover seq_len if any.

### End-to-end numerical correctness

Stock baseline vs all-kernels-on on the same loaded checkpoint, fixed seed.
Runs in the model's native inference dtype (bf16).

```bash
pixi run python -m benchmarks.correctness_evo2 \
    --models evo2_7b --seq-lens 8192 32768
```

Reports `max_abs_diff`, `mean_abs_diff`, `cosine_sim_last_token`,
`cosine_sim_sequence_mean`, and `argmax_match_rate` per `(model, seq_len)` to
`results/<gpu>/correctness/correctness.json`.

### Aggregate progression rows across GPUs

```bash
pixi run python -m benchmarks.compare_progression
```

Walks every `results/<gpu>/progression/*/combined_summary.json`, pulls the
headline forward-pass numbers, and writes the cross-GPU comparison table to
`results/comparison.md`.

## Results layout

All measurements land under `results/<gpu_slug>/`. The slug is auto-detected
from `torch.cuda.get_device_name()` (`rtx-4090`, `h100`, ...) so artifacts
from different GPUs stay in separate trees. Every emitted JSON wraps its
measurements in a `run_meta` envelope (timestamp, gpu, cuda/driver/torch/
triton versions, repo SHAs, config) so re-runs are traceable.

```
results/
├── <gpu>/
│   ├── baseline_profile/      flags all off
│   ├── microbench/            bench_{hcl,hcm,hcs}.json
│   ├── progression/{hcs,hcs_hcm,final}/
│   └── correctness/           stock-vs-kernels logit comparison
└── comparison.md              cross-GPU progression table
```

## License

Apache 2.0 (matches Vortex and Evo2).
