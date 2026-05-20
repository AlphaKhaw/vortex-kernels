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
each opt-in behind a `use_{hcs,hcm,hcl}_kernel` config flag — that branch is the upstream
PR. This repo is the **harness**: the profiler, the benchmarks, the measured
results, and the writeup.

| Repo | Holds |
|---|---|
| vortex fork (`../vortex`) | the three kernels + the `use_{hcs,hcm,hcl}_kernel` dispatch — the PR |
| this repo | `benchmarks/` profiler, `results/` artifacts, `docs/` |

## The kernels

The HCL/HCM/HCS Hyena conv layers are ~21% of Evo2 7B forward CUDA time at long
context (L=65k), and `use_flashfft: False` is the default in every shipped
config — so all three run unfused. The three target layer kinds:

| Layer | Filter | Kernel | Dispatches in |
|---|---|---|---|
| HCS | length 7 | from-scratch depthwise-conv Triton kernel | `parallel_fir`, gated short conv |
| HCM | length 128 | fused FFT-conv epilogues around cuFFT | `parallel_fir`, `fir_length >= 128` |
| HCL | length L | tiled `compute_filter` + FFT-conv — avoids the `(D, state_size, L)` fp32 tensor that OOMs Evo2 at L=131k | `parallel_iir` |

Each is gated by a flag defaulting to off — zero behavioral change when off.

## Setup — Linux + CUDA 12.9 host

```bash
git clone <this repo> ~/vortex-kernels && cd ~/vortex-kernels
bash scripts/setup_vm.sh
```

`setup_vm.sh` installs [Pixi](https://pixi.sh), clones vortex as a sibling
(`../vortex`) on the `triton-hc-kernels` branch, then `pixi install` resolves
the full stack (CUDA 12.9, PyTorch 2.7 cuda build, Transformer Engine,
flash-attn, evo2) and editable-installs the fork.

```bash
pixi run verify        # sanity-check imports
pixi run test          # pytest (GPU tests excluded)
pixi run test-gpu      # GPU-marked tests
pixi run check         # lint + typecheck + fast tests
pixi run profile       # Evo2 profiling
```

## Results layout

All measurements land under `results/<gpu_slug>/`. The slug is auto-detected
from `torch.cuda.get_device_name()` (`rtx-4090`, `h100`, ...) so 4090 and
H100 artefacts stay in separate trees. Each phase folder carries JSON
measurements, `plots/`, and a `report.md`. Every emitted JSON wraps its
measurements in a `run_meta` envelope (timestamp, gpu, cuda/driver/torch/
triton versions, repo SHAs, config) so re-runs are traceable.

```
results/
├── rtx-4090/                  dev box
│   ├── baseline_profile/      no kernels — flag-off baseline
│   ├── microbench/            bench_{hcs,hcm,hcl}.json
│   └── progression/{base,hcs,hcs_hcm,final}/
└── h100/                      sign-off pod (produced on demand)
```

## Workflow

1. **Profile the baseline** — `pixi run profile`; artefacts in
   `results/<gpu>/baseline_profile/`.
2. **Implement a kernel** in the fork, behind its `use_*_kernel` flag.
3. **Measure the progression** — toggle flags and re-profile; rows land in
   `results/<gpu>/progression/{base,hcs,hcs_hcm,final}/`.
4. **Upstream PR** — the fork branch; scope tracked in
   [Zymrael/vortex#76](https://github.com/Zymrael/vortex/issues/76).

## License

Apache 2.0 (matches Vortex and Evo2).
