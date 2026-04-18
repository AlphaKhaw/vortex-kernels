# Vortex issue draft — post AFTER Phase 1 profiling

Posting target: https://github.com/Zymrael/vortex/issues/new

> Fill in `<PROFILE_PCT>`, `<MICRO_SPEEDUP>`, `<COMMIT_SHA>`, `<REPO_URL>` from
> real numbers on a Lambda H100 before submitting. Do not post this with
> placeholders.

---

**Title:** Proposal: fill `vortex/ops/hcm_interface.py` with fused FFT-conv Triton kernel (refs #16)

**Body:**

Hi @garykbrixi @Zymrael — I'd like to propose filling one of the empty
`vortex/ops/hc*_interface.py` slots that referenced in #16, starting with HCM.

### Motivation

Profiling Evo2-7B on an H100 at seq_len=32768 shows ~`<PROFILE_PCT>%` of
CUDA time is in FFT operations across HCL and HCM layers. Reviewing
`vortex/model/engine.py` at `<COMMIT_SHA>`:

- **HCL** (`parallel_iir`): already accelerated when `use_flashfft=True` via
  the `fftconv_fn` dispatch. Works fine once the YAML flag is flipped.
- **HCM** (`parallel_fir`, `fir_length >= 128` branch): calls pure-torch
  `fftconv_func(u.fp32, weight.fp32, ...)` under `torch.autocast("cuda")`,
  **regardless of `use_flashfft`**. FlashFFTConv is only wired into
  `parallel_iir`, never into `parallel_fir`. 9 HCM layers × 3 FFT launches
  each = 27 unfused cuFFT calls per forward pass that remain even after
  enabling every existing optimization flag.

So HCM is the biggest unclaimed optimization target, and it sits in one of the
empty interface files you already scaffolded.

### Prior art reviewed

I read the `kernels` branch (`8046e67`, last real commit `2cd0338` "feat:
kernel interface and cgcg", Jan 2025). Its scope is training-path CGCG
(Chunked Gated Conv Gated) + a vendored Mamba `causal_conv1d` — the
`HyenaMR(nn.Module): pass` and `HyenaLI(nn.Module): pass` stubs in
`vortex/ops/conv/hyena_ops/interface.py` confirm HCM and HCL were intended
but not started on that branch. The Triton kernels there also aren't wired
into `engine.py` on that branch. I'd plan to **start fresh on `main`** for
the inference FFT-conv path rather than rebase — happy to reconcile differently
if you prefer.

### Proposed scope (this issue + first PR)

Fill `vortex/ops/hcm_interface.py` with:

```python
def hcm_fft_conv(u, k, D, *, dropout_mask=None, gelu=True, k_rev=None,
                 bidirectional=False, **kwargs) -> torch.Tensor:
    """Matches fftconv_func signature. Fused scale/multiply around cuFFT
    replaces 6 kernel launches with 2 cuFFT + 2 Triton = 4 launches."""
```

Then a one-line change in `engine.py::parallel_fir`'s `fir_length >= 128`
branch to dispatch to `hcm_fft_conv` when available, falling back to
`fftconv_func` otherwise. Guarded by a new config key `use_triton_hcm` that
defaults to False — **zero behavioral change when the flag is off, zero API
changes**.

### Acceptance criteria

- Correctness: `max_diff < 5e-2`, `mean_diff < 5e-3` vs the existing
  `fftconv_func` reference in bf16 on evo2_7b_base at seq_len ∈ {8192, 32768}.
- Microbench: `≥ <MICRO_SPEEDUP>x` on the HCM layer in isolation at the
  shapes Evo2 7B actually uses (`B=1, D=4096, L=32768, filter_len=128`).
- End-to-end: logits from patched vs unpatched model match within tolerance
  above on a 4k-token smoke test.
- No regression when `use_triton_hcm=False` (bit-exact output path unchanged).

### Standalone test repo

All the above is prototyped in a standalone package that monkey-patches
vortex at import time: `<REPO_URL>`. Happy to hand-port the kernel and its
tests into `vortex/ops/hcm_interface.py` + `vortex/ops/tests/` for the PR.

### Questions for you before I open the PR

1. Do you want `hcm_interface.py` filled via re-export from
   `vortex.ops.conv.hyena_ops.*` (i.e., rebased on the `kernels` branch), or
   as a new fresh module on `main`? I'd prefer the latter given the branch
   state, but you may have a reason.
2. Config-key name preference — `use_triton_hcm`, `use_hcm_kernel`, something
   else? Or would you prefer auto-dispatch (use it if importable, fall back
   silently)?
3. Is there a preferred place for Triton kernel tests in vortex? I couldn't
   find a `tests/` top-level on main — should they live under
   `vortex/ops/hcm_interface/tests/` or add a new `vortex/ops/tests/`?
4. If HCM lands cleanly, would you want a follow-up PR for `hcs_interface.py`
   (wiring the existing `vortex/ops/hyena_x/triton_indirect_fwd.py` which is
   already written but never imported from engine.py)?

I'll hold off on opening a PR until you've had a chance to respond — no rush.

Thanks for the interesting code to work on.
