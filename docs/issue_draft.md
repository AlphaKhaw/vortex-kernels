# Vortex issue draft — post AFTER results/ is committed + repo flipped public

Posting target: https://github.com/Zymrael/vortex/issues/new

> Fill `<COMMIT_SHA>` (current Zymrael/vortex main SHA) and `<REPO_URL>` (your
> public vortex-kernels repo) before posting. Every other number is already
> from real measurement — do not edit them.

---

**Title:** Proposal: fill the three empty `vortex/ops/hc*_interface.py` files with fused Triton FFT-conv kernels (refs #16)

**Body:**

Hi @garykbrixi @Zymrael — I've been profiling Evo2-7B inference on an H100 80GB SXM and I'd like to propose filling the three empty `vortex/ops/hc{l,m,s}_interface.py` scaffolding files that are referenced in #16. The combined FFT-conv system is **21.5% of CUDA time at seq_len=65k** and — more importantly — the default HCL path has a memory footprint that caps single-GPU inference well below the model's advertised `max_seqlen`.

I'm not a maintainer and don't want to step on #16 or the `kernels` branch work. Opening this as a discussion before writing PRs.

### Motivation

#### 1. Single-GPU inference OOMs far below `max_seqlen: 1048576`

Running stock `evo2_7b` on one H100 80GB SXM with `torch 2.7.1 / TE 2.3.0 / flash-attn 2.8.0.post2` (against vortex main @ `<COMMIT_SHA>`):

- L=65,536: **fits**, forward 6681 ± 12 ms
- L=131,072: **OOM** during warmup, even with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`

Traceback locates the allocation in `vortex/model/model.py::compute_filter`:

```python
h = (residues[..., None] * (log_poles * self.t).exp()).sum(1)[None]
```

That materializes a `(D=4096, state_size=16, L)` fp32 tensor. At L=131k that's 32 GiB for a single intermediate, before activation memory. At L=1M it would be 256 GiB — physically impossible on current single-GPU hardware.

I understand the 1M claim is pipeline-parallel; this isn't a bug report. It's context for why a memory-efficient HCL kernel that avoids materializing the full filter would directly unlock longer seq_len on single-GPU deployments, which is the common case for many downstream users (variant scoring pipelines, small labs, etc.).

Size math for the filter-realization tensor at `D=4096, state_size=16`:

| seq_len | Filter tensor size (fp32) |
|---|---:|
| 65,536 | 17.2 GiB |
| 131,072 | 34.4 GiB |
| 262,144 | 68.7 GiB |
| 1,048,576 | 262 GiB |

Tiling this over L with a modest `BLOCK_L=4096` reduces peak filter-tile allocation to ~1 GiB. Whether the L=131k forward fits in 80 GB after tiling depends on the input-FFT intermediates (separate, not tiled by this proposal) plus activation memory — to be confirmed during PR review.

#### 2. All three hyena variants run unfused by default

From the L=65,536 profile (full data linked below):

| Category | CUDA ms (5 runs) | % of total |
|---|---:|---:|
| other (TE FP8 quantize/dequantize + misc) | 62,900.9 | 48.8% |
| gemm | 25,894.1 | 20.1% |
| elementwise | 10,347.7 | 8.0% |
| cast_copy | 9,486.9 | 7.4% |
| conv | 7,985.1 | 6.2% |
| **fft** | **7,967.4** | **6.2%** |
| attention | 3,852.7 | 3.0% |
| reshape | 455.1 | 0.4% |

Breaking down by layer kind at L=65,536:

| Layer kind | CUDA ms | % of total | Layers |
|---|---:|---:|---:|
| **hcl** | 16,392.7 | **12.7%** | 9 |
| **hcm** | 6,868.4 | **5.3%** | 9 |
| **hcs** | 4,522.4 | **3.5%** | 9 |
| attn | 5,592.9 | 4.3% | 5 |

Combined **hyena conv = 21.5%** of inference CUDA time at L=65k. The remaining 49% ("other") is TE's FP8 quantize/dequantize stack around Linear projections — out of scope for this proposal.

The `fft` category alone grows monotonically with seq_len (3.9% @ L=2k → 6.2% @ L=65k), consistent with the `O(L log L)` asymptote — the relative weight of the conv system increases as users go longer.

#### 3. Default config doesn't take advantage of existing FlashFFTConv wiring

`shc-evo2-7b-8k-2T-v2`'s default config has `use_flashfft: False`, so HCL runs through the unfused `fftconv_func` path even on H100s where FlashFFTConv would work. HCM's `parallel_fir` `fir_length >= 128` branch is unfused regardless of the flag. HCS's short-filter path is also unfused. So all three variants currently hit the same unfused pattern: separate cuFFT on input, cuFFT on filter, complex multiply, iFFT, then scale/bias/activation — 6+ launches per layer instead of 2 (cuFFT + iFFT) with fusion.

Per forward at L=65k: **~80 unfused conv-system kernel launches** that can collapse to ~20 with fused Triton.

### Prior art reviewed

I read the `kernels` branch (last real commit `2cd0338` "feat: kernel interface and cgcg", Jan 2025). Its scope is the training-path CGCG (Chunked Gated Conv Gated) + a vendored Mamba `causal_conv1d`. The `HyenaMR(nn.Module): pass` and `HyenaLI(nn.Module): pass` stubs in `vortex/ops/conv/hyena_ops/interface.py` confirm HCM and HCL were intended but not implemented on that branch, and the Triton kernels present aren't wired into `engine.py`. I'd plan to **start fresh on `main`** for the inference FFT-conv path rather than rebase — but happy to reconcile differently if you have a reason to prefer the branch.

### Proposed scope

**One PR** covering all three interfaces, with the three kernels cleanly separated internally. Happy to split into three smaller PRs if you'd prefer staged review — just say so and I'll sequence them HCS → HCM → HCL (smallest risk to largest).

Each interface is opt-in behind its own config flag defaulting to `False`:

- **`hcl_interface.py`** — Triton kernel that tiles over L, computes the filter `h = residues * (log_poles * t).exp()` inside the tile, and fuses the FFT-conv without materializing the full `(D, L)` fp32 intermediate. Gated by `use_triton_hcl`.
- **`hcm_interface.py`** — `hcm_fft_conv(u, k, D, …)` matching `fftconv_func`'s signature. Fused scale/multiply around cuFFT collapses the 6-launch unfused pattern to 2× cuFFT + 2× Triton. Dispatched in `engine.py::parallel_fir`'s `fir_length >= 128` branch. Gated by `use_triton_hcm`.
- **`hcs_interface.py`** — Direct Triton depthwise conv for `hcs_filter_length: 7`. At this filter size the FFT round-trip is a net loss; a direct conv wins. Dispatched in `parallel_fir`'s `fir_length < 128` branch. Gated by `use_triton_hcs`.

**Zero behavioral change when flags are off, zero API changes.** All three keep bit-exact fallback to the current code paths.

### Acceptance criteria (per PR)

- **Correctness**: `max_diff < 5e-2`, `mean_diff < 5e-3` vs reference bf16 output on `evo2_7b` at seq_len ∈ {8192, 32768, 65536}
- **Microbench**: kernel-level speedup measured in isolation at Evo2-7B shapes (`B=1, D=4096, filter_len={filter-specific}`)
- **End-to-end**: logit agreement within tolerance above on a 4k-token smoke test for patched vs unpatched model
- **No regression**: bit-exact output path unchanged when the `use_triton_*` flag is `False`
- **Memory** (HCL only): peak allocator bytes reduced enough to fit L=131k on H100 80GB

### Standalone prototype repo

All profiling, kernel code, correctness tests, and results artifacts live in `<REPO_URL>`. It monkey-patches vortex at import time so I could iterate without maintaining a fork of your main. The profiler itself (`benchmarks/profile_evo2.py`) is the tool that produced the numbers above and is reusable across evo2 variants. Happy to hand-port the three kernels and their tests into `vortex/ops/` and `vortex/ops/tests/` once we align on scope.

### Questions before I open PRs

1. **Branch strategy**: fresh modules on `main`, or re-export from `vortex.ops.conv.hyena_ops.*` (i.e., rebased on the `kernels` branch)? I'd prefer the former given the branch state, but you may have a reason.
2. **Config-key naming**: `use_triton_hcl` / `_hcm` / `_hcs` per-kernel flags, or one unified `use_triton_kernels` bool, or auto-dispatch when importable? Per-kernel flags give reviewers the safest rollback path.
3. **Test placement**: I don't see a top-level `tests/` directory on main. Under `vortex/ops/tests/` for the whole family, or one test file per interface alongside the implementation?
4. **Single PR or three**: default plan is one PR with all three kernels + their tests; I'll split into three (HCS → HCM → HCL) if you'd rather stage the review.
5. **Scope of #16**: does this proposal fit within #16, or would you prefer a new issue and have this one close?

I'll hold off on opening any PR until you've had a chance to respond — no rush.

Thanks for the model and the scaffolding work you've already put in.

---

### Appendix — reproducibility

- Hardware: RunPod H100 80GB SXM
- Software: python 3.12, torch 2.7.1 (cuda12_9 build), TE 2.3.0, flash-attn 2.8.0.post2, evo2 0.5.3, vortex main @ `<COMMIT_SHA>`
- Profiler: 5 timed forwards + 3 warmup, `record_shapes=True`, `profile_memory=True`
- Full data: `<REPO_URL>/blob/main/results/baseline_profile/report.md`
- Stacked layer-kind plot: `<REPO_URL>/blob/main/results/baseline_profile/plots/stacked_by_layer_kind.png`
- Forward-pass run-to-run std was <2% across all completed seq_lens
