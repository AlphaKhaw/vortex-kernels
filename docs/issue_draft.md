# Vortex issue draft

Posting target: https://github.com/Zymrael/vortex/issues/new

Placeholders filled:
- vortex main SHA: [`cb229ae`](https://github.com/Zymrael/vortex/commit/cb229ae50581aa072972c33a81e23eb43a1330ab) (as of 2026-04-21)
- prototype repo: https://github.com/AlphaKhaw/vortex-kernels

Before clicking Submit: confirm the repo is Public (Settings → General →
Change visibility) and the results/ artifacts are pushed to origin/main.

---

**Title:** Proposal: fill the three empty `vortex/ops/hc*_interface.py` files with fused Triton FFT-conv kernels (refs #16)

**Body:**

Hi @garykbrixi @Zymrael — proposing to fill the three empty `vortex/ops/hc{l,m,s}_interface.py` scaffolds referenced in #16 with fused Triton kernels, following @garykbrixi's note in #69 that inference-performance contributions are welcome. Numbers below from profiling `evo2_7b` on one H100 80GB SXM.

Measurements ran against `vtx==1.0.8` on PyPI; current vortex main is [`cb229ae`](https://github.com/Zymrael/vortex/commit/cb229ae50581aa072972c33a81e23eb43a1330ab) and the three `hc*_interface.py` files are still empty there. I haven't re-profiled against post-#75 main yet — flagging that explicitly — but the kernel opportunity is orthogonal to PR #75's TE-optional fallback path and applies in either mode.

### Motivation

**1. Single-GPU inference OOMs at L=131k, well below `max_seqlen: 1048576`.** Root cause is `vortex/model/model.py::compute_filter` materializing a `(D=4096, state_size=16, L)` fp32 tensor:

| seq_len | filter tensor | fits on 80 GB? |
|---|---:|---|
| 65,536 | 17.2 GiB | yes (barely) |
| 131,072 | 34.4 GiB | no |
| 1,048,576 | 262 GiB | no |

Understood that 1M is pipeline-parallel deployment; this is context for why a tiled HCL kernel that avoids the `(D, L)` allocation would unlock longer single-GPU seq_len — common for variant scoring / small-lab workflows. With `BLOCK_L=4096`, filter-tile peak drops to ~1 GiB.

**2. All three hyena variants run unfused in the default config.** `use_flashfft: False` in `shc-evo2-7b-8k-2T-v2` means HCL skips FlashFFTConv too, not just HCM. Per forward at L=65k: **~80 kernel launches across the conv system** (6+ per layer × 27 conv layers) that can collapse to ~20 with fusion.

**3. The conv system is 21.5% of CUDA time** at L=65k:

| Layer kind | CUDA ms (5 runs) | % of total | Layers |
|---|---:|---:|---:|
| **hcl** | 16,392.7 | **12.7%** | 9 |
| **hcm** | 6,868.4 | **5.3%** | 9 |
| **hcs** | 4,522.4 | **3.5%** | 9 |
| attn | 5,592.9 | 4.3% | 5 |

Remaining 49% is TE's FP8 quantize/dequantize stack around Linear projections — out of scope. Full data: [report.md](https://github.com/AlphaKhaw/vortex-kernels/blob/main/results/baseline_profile/report.md), [stacked plot](https://github.com/AlphaKhaw/vortex-kernels/blob/main/results/baseline_profile/plots/stacked_by_layer_kind.png).

### Prior art

Read the `kernels` branch (last real commit `2cd0338`, Jan 2025): its scope is training-path CGCG + vendored Mamba `causal_conv1d`. The `HyenaMR(pass)` / `HyenaLI(pass)` stubs there confirm HCM and HCL were never started. Planning to work fresh on `main` but open to reconciling if you'd prefer a rebase.

### Proposed scope

**One PR** covering all three kernels (or three on request, HCS → HCM → HCL). Each is opt-in behind a per-interface config flag defaulting to `False`:

- **`hcl_interface.py`** — Triton kernel tiling over L, computing `h = residues * (log_poles * t).exp()` per tile, fused with the FFT-conv. Avoids `(D, L)` materialization. Gated by `use_triton_hcl`.
- **`hcm_interface.py`** — `hcm_fft_conv(u, k, D, …)` matching `fftconv_func`'s signature; fuses scale/multiply around cuFFT, collapsing 6 launches to 4. Dispatched in `engine.py::parallel_fir`'s `fir_length >= 128` branch. Gated by `use_triton_hcm`.
- **`hcs_interface.py`** — direct Triton depthwise conv for `hcs_filter_length: 7` (FFT round-trip is a net loss at this size). Gated by `use_triton_hcs`.

**Zero behavioral change when flags are off, zero API changes.** Acceptance: `max_diff < 5e-2`, `mean_diff < 5e-3` vs reference bf16 on `evo2_7b` at L ∈ {8192, 32768, 65536}; bit-exact fallback when flags are off.

### Prototype repo

All profiling, kernel code, correctness tests, results at https://github.com/AlphaKhaw/vortex-kernels. Monkey-patches vortex at import time so I could iterate without forking. The profiler itself (`benchmarks/profile_evo2.py`) is reusable.

### Questions

1. Fresh modules on `main`, or rebase on the `kernels` branch?
2. Per-interface flags (`use_triton_{hcl,hcm,hcs}`) or one unified `use_triton_kernels`?
3. One PR covering all three kernels, or three smaller PRs (HCS → HCM → HCL) for staged review?

Tests would follow the existing `test/test_evo2_forward.py` convention from PR #41 unless you'd prefer otherwise. No rush — holding PRs until you weigh in. Thanks for the scaffolding.

---

### Appendix — reproducibility

RunPod H100 80GB SXM. Python 3.12, torch 2.7.1 (cuda12_9 build), TE 2.3.0, flash-attn 2.8.0.post2, evo2 0.5.3, **vtx 1.0.8** (PyPI release — pre-dates PR #75). Profiler: 5 timed forwards + 3 warmup, `record_shapes=True`, `profile_memory=True`. Forward-pass run-to-run std <2% across all completed seq_lens.
