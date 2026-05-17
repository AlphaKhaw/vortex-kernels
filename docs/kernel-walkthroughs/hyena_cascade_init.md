# `HyenaCascade.__init__` — parameter declarations

> **Tour target 4 of 12** · [index](README.md) · prev: [parallel_iir](parallel_iir.md) · next: [HyenaCascade.parallel_forward](hyena_cascade_parallel_forward.md)
>
> **Depth: light** — data contract only. See the README for the two-tier
> scheme.

## Source

| | |
|---|---|
| Upstream | `vortex/model/model.py:126-213` |

## Overall purpose

**Declare every learnable parameter that flows into `parallel_fir` /
`parallel_iir`** and configure the per-layer dispatch flags. This
constructor runs once per Hyena layer when the model is instantiated;
its outputs are what `parallel_forward` reads at every inference step.

The constructor branches on `fir_inner_filter_length`:

- **`None`** → this is an **HCL layer**. Declares modal-decomposition
  params (`residues`, `log_poles`) for the IIR filter. `self.h` is
  `None` and gets built lazily by `compute_filter` at each forward.
- **`< 128`** → this is an **HCS cascade layer**. Declares
  `self.h` directly as a learnable per-group filter `(g, 1, hl)`.
  `self.D` is `None` (no skip-gain for short filters).
- **`>= 128`** → this is an **HCM cascade layer**. Same `self.h` as
  HCS but with `self.D` as a per-channel zeros-initialized
  parameter (gated bias enabled at this length).

## Inputs (constructor args)

| Arg | Role |
|---|---|
| `config` | The model config object (carries `hidden_size`, `state_size`, `short_filter_length`, etc.) |
| `layer_idx` | Layer index; threaded through for activation-logging support |
| `hyena_filter_groups` | Number of filter groups `g`. For evo2_7b: 256 (HCS), 128 (HCM), 4096 (HCL = D) |
| `fir_inner_filter_length` | `None` (HCL), `7` (HCS), or `128` (HCM). The dispatch axis. |

## Declared parameters by layer type

```
                              HCS              HCM            HCL
                          (fir_inner=7)   (fir_inner=128)   (None)
                          ─────────────   ───────────────   ─────────────

ALWAYS DECLARED (every Hyena layer)
─────────────────────────────────────
short_filter_weight       (12288, 1, 3)   (12288, 1, 3)    (12288, 1, 3)   = (3*D, 1, short_filter_length=3)
short_filter_bias         None            None             None             (config.short_filter_bias=False in evo2)
engine                    HIE             HIE              HIE              (HyenaInferenceEngine instance)
fir_fn                    F.conv1d        F.conv1d         F.conv1d
fir_inner_fn              F.conv1d        F.conv1d         F.conv1d
fftconv_fn                None            None             None             (set later if use_flashfft=True)
long_fir_threshold        None            None             None             (evo2 default)

BRANCH: fir_inner_filter_length IS set (HCS/HCM cascade)
────────────────────────────────────────────────────────
h                         (256, 1, 7)     (128, 1, 128)    None             (g, 1, fir_inner_filter_length)
D                         None            (4096,)          (4096,)          gated bias for >= 128, zeros init
log_poles                 (not declared)  (not declared)
residues                  (not declared)  (not declared)

BRANCH: fir_inner_filter_length is None (HCL)
──────────────────────────────────────────────
log_poles                 (not declared)  (not declared)   (4096, 16, 1)    fp32, randn init
residues                  (not declared)  (not declared)   (4096, 16)       fp32, randn init
D                         see above       see above        (4096,)          zeros init
h                         see above       see above        None             ← built on demand by compute_filter

ALWAYS DECLARED, value `None` until first forward
─────────────────────────────────────────────────
t                         None            None             None             ← built lazily by update_time
```

## Where these flow downstream

```
HyenaCascade.__init__
        │
        │ self.short_filter_weight, self.short_filter_bias
        ├──────────────────────────────────────────────► parallel_fir (FEATURIZER call, gate=False)
        │
        │ self.h (HCS/HCM), self.D (HCM only)
        ├──────────────────────────────────────────────► parallel_fir (CASCADE call, gate=True)
        │
        │ self.residues, self.log_poles, self.D
        ├──────────────────────────────────────────────► compute_filter ─► h ─► parallel_iir (HCL)
        │
        │ self.fir_fn, self.fir_inner_fn, self.fftconv_fn,
        │ self.long_fir_threshold, self.use_flashfft
        └──────────────────────────────────────────────► (config/dispatch knobs read at call-time)
```

## Key data-contract facts to remember

1. **`self.h` is `(g, 1, hl)` per-group**, NOT `(D, 1, hl)` per-channel.
   The expansion happens at runtime in `parallel_forward` via
   `h = h.repeat_interleave(D//g, 0)`. Your HCS adapter must invert
   that expansion (slice every `dg`-th channel) to feed CGCG's per-group
   contract. See [parallel_forward](hyena_cascade_parallel_forward.md)
   for where the expansion happens.

2. **`self.D` exists only for HCM and HCL** (not for HCS). HCS has
   `self.D = None`, so the bias-add in `parallel_fir`'s HCS branch
   would be a no-op IF `bias` were passed in — but the cascade call
   actually passes `D` as the `bias` argument anyway... wait, let me
   re-check. Actually the cascade call in `parallel_forward` passes
   `D` as the `bias` arg only for HCM+HCL. For HCS the call passes
   `self.D = None` so no bias-add happens.

3. **HCL params are fp32**, not bf16. The `log_poles` and `residues`
   are explicitly fp32. The skip param `D` defaults to model dtype
   (probably bf16 after `.to(bf16)` in StripedHyena.__init__) but the
   modal math always upcasts to fp32 internally.

4. **`use_flashfft` toggles the HCL fast path** — when True AND
   FlashFFTConv is installed, `self.fftconv_fn` gets the
   single-kernel callable assigned in `StripedHyena.__init__`
   (model.py:~584). Default `False`.

## What this means for the wire-up

Two consumers of this constructor's outputs:

- **HCS adapter** (`hcs_dispatch`) reads:
  - `self.h` (the per-group filter, `(g, 1, hl)`) — but it arrives
    via `parallel_fir`'s `weight` arg, already `repeat_interleave`d
    to `(D, 1, hl)`. Adapter de-expands via `weight[::dg]`.
  - `self.hyena_filter_groups` (= `g`) — arrives via `parallel_fir`'s
    `groups` arg. Used as the predicate signal AND the de-expand factor.

- **HCL kernel** (`hcl_fft_conv`) reads:
  - `self.residues`, `self.log_poles` (the modal params)
  - `self.D` (the skip-gain)
  - These arrive via `parallel_iir`'s `residues`/`poles`/`D` args after
    `compute_filter` builds `h`. **Our HCL kernel ignores the passed
    `h`** and re-derives chunks from `residues + log_poles` inline
    (the whole point of tiling).

## Cross-references

- [parallel_fir](parallel_fir.md) — Target 2 — consumes
  `short_filter_weight` (featurizer) and `h` (cascade)
- [parallel_iir](parallel_iir.md) — Target 3 — consumes
  `residues, log_poles, D` (via the `compute_filter` step)
- [HyenaCascade.parallel_forward](hyena_cascade_parallel_forward.md) —
  Target 5 — the orchestrator that reads these params and threads them
  into the engine methods
- [compute_filter_pure](compute_filter_pure.md) — Target 6 — builds
  `h` from `residues + log_poles + t`

---

*Last updated: 2026-05-12 (Phase 1 read-through, target 4/12, light)*
