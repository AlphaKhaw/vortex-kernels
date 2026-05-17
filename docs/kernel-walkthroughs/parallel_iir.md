# `HyenaInferenceEngine.parallel_iir` — the IIR dispatch method

> **Tour target 3 of 12** · [index](README.md) · prev: [parallel_fir](parallel_fir.md) · next: [HyenaCascade.\_\_init\_\_](hyena_cascade_init.md)

## Source

| | |
|---|---|
| Upstream | `vortex/model/engine.py:262-405` |
| Vendored copy | `vortex_kernels/reference/engine_ref.py:277-419` |
| Called from | `HyenaCascade.parallel_forward` (model.py:~324) — once per HCL layer |

## Overall purpose

**Apply the HCL long convolution and the gating sandwich in one shot,
with optional state caching for streaming inference.**

`parallel_iir` is the HCL sibling of `parallel_fir`. Two big differences:

1. **The filter is huge** — `h` arrives with shape `(1, D, L)` (the full
   sequence length), materialized by `compute_filter` from
   `(residues, log_poles)`. At L=131k this filter is 2 GiB per layer.
   Direct `conv1d` is hopeless; FFT-conv is mandatory.

2. **The FFT-conv is inlined** (not via `fftconv_func`). Three FFT ops
   (`rfft(h)`, `fft(x1v)`, `irfft(X*H)`) right inside the method body,
   plus a fork in the path: with FlashFFTConv installed, a single
   fused-kernel call replaces those three ops.

The method also handles:
- The CGCG gating sandwich (same as `parallel_fir`: split → `x1*v` →
  conv → `x2 * (...)`) but more tightly fused in the post-processing
- **Saving `X_s`** (the fft of `x1v`) for reuse by
  `prefill_via_modal_fft` during streaming inference
- A rarely-used `long_fir_threshold` fallback to truncated direct conv

## The HCL dispatch axis

Three possible code paths inside this method, decided by config:

```
                  parallel_iir is called
                          │
                          ▼
            ┌─────────────────────────────┐
            │ inference_params != None    │
            │ AND prefill_style ==        │
            │     "recurrence"?           │
            └──────┬────────────┬─────────┘
                   │ YES        │ NO
                   ▼            ▼
        recurrent prefill   ┌──────────────────────┐
        (streaming only)    │ use_flashfft         │
                            │ AND L % 2 == 0?      │
                            └──┬───────────┬───────┘
                               │ YES       │ NO
                               ▼           ▼
                         FlashFFTConv  ┌───────────────────────┐
                         single call   │ long_fir_threshold    │
                         (HCL fast     │ is None?              │
                          path)        └──┬────────────────┬───┘
                                          │ YES            │ NO
                                          ▼                ▼
                                   ★ FFT-conv inline    truncated
                                   ★ (3 ops)            direct conv
                                   ★ THE HCL TARGET     (rarely used)
```

**Our HCL kernel (Phase 4) replaces the marked path** — the inline
three-FFT branch (`elif long_fir_threshold is None`). That's the path
evo2_7b takes by default with `use_flashfft=False`.

The other paths stay untouched:
- FlashFFTConv path: already optimized; if the user installs
  FlashFFTConv they get it via the `fftconv_fn` slot
- Recurrent / `long_fir_threshold` paths: rarely-used edge cases

## Realistic input shapes (evo2_7b HCL at L=8192)

```
INPUTS to parallel_iir (the HCL FFT branch we target)
══════════════════════════════════════════════════════

z_pre    ( 1 , 12288 , 8192 )  bf16   ← pre-split cascade input (3*D channels)
         └─┬─┘└──┬──┘└──┬──┘             B × 3*D × L
           B   3*D    L

h        ( 1 , 4096 , 8192 )   fp32   ← MATERIALIZED filter from compute_filter
         └┬┘└──┬──┘└──┬──┘              the (D, S, L) → sum(S) → (1, D, L) result
          1   D=4096  L=8192             THIS IS THE OOM SOURCE we tile away

D        ( 4096 , )            fp32   ← skip-gain parameter
         └──┬──┘
          D=4096

t        ( 1 , 1 , 8192 )      fp32   ← arange(L)[None, None]; from update_time
         └─┬─┘└─┬┘└──┬──┘
           1   1   L

poles    ( 4096 , 16 , 1 )     fp32   ← log_poles (the "log" prefix is silent in the param name)
residues ( 4096 , 16 )         fp32

dims = (4096, 8 or similar, head_size, 16, 1)
fft_size = 2 * L = 16384
use_flashfft = False (evo2 default)
prefill_style = "fft"
long_fir_threshold = None (evo2 default)
column_split_hyena = True
inference_params = None (parallel prefill, our path)
```

## Inputs explained

### `z_pre` — pre-split cascade input

The 3*D-channel tensor coming out of `HyenaCascade.parallel_forward`
after the upstream featurizer + linear projections. Same shape role as
`u` in `parallel_fir`'s cascade call: gets split into `(x2, x1, v)`.

### `h` — the materialized filter

The HCL filter, already evaluated to shape `(1, D, L)`. Built by
`HyenaCascade.compute_filter` via the modal formula
`h[d, t] = Σ_s residue[d, s] * exp(log_poles[d, s] * t)`.

**This is the tensor whose intermediate `(D, S, L)` form is the OOM
source.** Once materialized to `(1, D, L)`, the per-layer footprint
is "only" `D * L * 4` bytes — 128 MiB at L=8k, 512 MiB at L=131k.
But the *intermediate* before `.sum(1)` is 16× larger because of the
state-size dim.

See [compute_filter_pure](compute_filter_pure.md) (target 6) for how `h`
is built.

### `D` — skip-gain parameter

Same `(D,)` fp32 tensor you've seen — the state-space direct-feedthrough
matrix. Applied as `x1v * D.unsqueeze(-1)` in this method (multiplicative
to `x1v`, not `u`, because gate-prep is structured differently here).

### `poles` (= `log_poles`), `residues`, `t`

Used **only** by the `prefill_via_modal_fft` call at the end (for
streaming inference state caching). The FFT-conv math doesn't use them
directly — `compute_filter` already converted them to `h`. The
parameters are passed in so the IIR state can be reconstructed via
modal FFT.

For parallel-prefill (our path with `inference_params=None`), these are
dead inputs.

### `dims`, `layer_idx`

Same shape-constants bundle and layer index pattern as `parallel_fir`.

### `inference_params` — streaming inference handle

Same role as in `parallel_fir`. For parallel prefill, this is `None`
and the entire `if inference_params is not None:` block at the end is
skipped.

### `prefill_style` — IIR prefill strategy

Determines how the state gets populated for streaming inference:

- `"fft"` (evo2 default) — use the cached `X_s` from FFT-conv to compute
  state via `prefill_via_modal_fft`
- `"recurrence"` — explicit step-by-step recurrence; takes a different
  branch at the top of the dispatch

For parallel prefill (`inference_params=None`), this argument is
ignored.

### `fftconv_fn` — FlashFFTConv slot

If FlashFFTConv is installed and configured, this slot holds the fused
single-kernel callable. Then `use_flashfft=True` triggers the fast path.

**Naming gotcha** (you've seen this before): `fftconv_fn` is **not** the
same as the `fftconv_func` function from
[Target 1](fftconv_func.md). `fftconv_func` is the unfused PyTorch
function used by HCM; `fftconv_fn` is the slot for FlashFFTConv used by
HCL. The names are similar by accident.

### `use_flashfft` — fast-path toggle

When `True` AND `L` is even AND `fftconv_fn` is set, the entire HCL
FFT-conv collapses to one call. Default `False` in every shipped evo2
config.

### `column_split_hyena` — split mode

Same role as in `parallel_fir`. evo2 uses `True`. The split logic
inside this method is more elaborate than `parallel_fir`'s because it
handles attention-head structure inline rather than via the
`column_split` helper.

### `long_fir_threshold` — direct-conv escape hatch

When set (default `None`), falls back to a truncated depthwise
`F.conv1d` with `h[..., :long_fir_threshold]`. Used for sanity tests
and not in evo2 defaults. Skip it mentally — our kernel doesn't need
to handle it (the predicate excludes it).

### `padding_mask` — sequence-mask tensor

Same role as in `parallel_fir`; rarely used in single-batch inference.

## Background concepts

### `fft` vs `rfft` — why x1v uses `fft`

`parallel_iir` calls `fft` on `x1v` (line ~333), not `rfft` like
`fftconv_func` does. Why?

```python
H   = torch.fft.rfft(h.to(fp32), n=fft_size) / fft_size   # rfft — N/2+1 bins
X_s = torch.fft.fft(x1v.to(fp32), n=fft_size)             # full fft — N bins
X   = X_s[..., : H.shape[-1]]                             # slice down to N/2+1
```

The reason: **`X_s` is saved for reuse**. The post-FFT-conv block uses
`X[..., :H.shape[-1]]` for the multiply, but `prefill_via_modal_fft`
later wants the full complex spectrum `X_s[..., None, :]` (all N bins)
to combine with `state_s` shaped `(D, S, N)`.

If `parallel_iir` had used `rfft`, the second half of `X_s` would be
missing and prefill caching would break. So it does the more expensive
full `fft` to keep `X_s` reusable downstream.

**This matters for our wire-up**: our HCL kernel must EITHER preserve
the same `X_s` reuse contract, OR guard on `inference_params is None`
so it only runs in the parallel-prefill case where `X_s` is discarded.
The simpler path is the guard; that's what the plan specifies.

### `x1v * D + y` then `* x2` — the post-processing

`parallel_iir`'s post-processing is *tighter* than `parallel_fir`'s.
One line does the skip-add and the post-gate together:

```python
y = (y + x1v * D.unsqueeze(-1)) * x2
```

vs `parallel_fir`'s two-step:

```python
z = z + bias[None, :, None]   # skip-add (bias here is the D param)
...
z = x2 * z                    # post-gate
```

Same math, different decomposition. The HCL form is fused into one
expression because everything's already in scope (no `padding_mask`
step in the middle).

## Step-by-step with shape tracking (HCL FFT branch)

Walking the path `inference_params=None, use_flashfft=False,
long_fir_threshold=None` — the HCL parallel-prefill case.

### Step 1: Compute fft_size + unpack dims

```python
fft_size = 2 * L
hidden_size, num_attention_heads, hidden_size_per_attention_head, _, _ = dims
```

```
SHAPES
  L = 8192 (arg from caller)
  fft_size = 16384
  hidden_size = 4096
  num_attention_heads = 8 (typically)
  hidden_size_per_attention_head = 512
```

**Purpose**: Bookkeeping. `fft_size = 2 * L` is the now-familiar
padding for linear conv (see [fftconv_func step 1](fftconv_func.md#step-1-set-the-fft-size)).
Dim unpacking is for the column-split branch in step 2.

### Step 2: Split `z_pre` into `(x2, x1, v)`

```python
if column_split_hyena:
    z = z_pre.reshape(
        z_pre.shape[0],
        num_attention_heads,
        3 * hidden_size_per_attention_head,
        z_pre.shape[2],
    )
    x2, x1, v = (
        z[:, :, :hidden_size_per_attention_head],
        z[:, :, hidden_size_per_attention_head : 2 * hidden_size_per_attention_head],
        z[:, :, 2 * hidden_size_per_attention_head :],
    )
    x2, x1, v = (
        x2.reshape(x2.shape[0], -1, x2.shape[-1]),
        x1.reshape(x1.shape[0], -1, x1.shape[-1]),
        v.reshape(v.shape[0], -1, v.shape[-1]),
    )
else:
    x2, x1, v = z_pre.split([hidden_size, hidden_size, hidden_size], dim=1)

if self.hyena_flip_x1x2:
    x1, x2 = x2, x1

x1v = x1 * v
```

```
SHAPES (column_split_hyena=True, the evo2 path)
  z_pre                       ( 1 , 12288 , 8192 )   bf16
  z (reshape)                 ( 1 ,    8 , 3*512 , 8192 )  bf16   (split heads)
  Each slice along dim=2:
    x2 raw                    ( 1 , 8 , 512 , 8192 )       bf16
    x1 raw                    ( 1 , 8 , 512 , 8192 )       bf16
    v raw                     ( 1 , 8 , 512 , 8192 )       bf16
  Reshape each to (B, -1, L):
    x2, x1, v                 ( 1 , 4096 , 8192 )          bf16    (heads × head_size = D)
  After x1v = x1 * v:
    x1v                       ( 1 , 4096 , 8192 )          bf16
```

**Purpose**: Same gate-prep idea as `parallel_fir` — separate the three
streams. The column-split-hyena branch is more elaborate (3D reshape
to respect attention-head structure, then re-flatten) but produces
the same logical `(x2, x1, v)`. The non-column-split branch is a plain
`u.split` like `parallel_fir`.

**Subtle differences from `parallel_fir`**:
1. The variable holding `x1*v` is named `x1v`, not rebound to `u`.
   That's a small style choice but makes it easier to track which one
   is the gating input later.
2. `x2` is held in scope for step 5 (same as `parallel_fir`).

### Step 3: HCL FFT-conv (the path we replace)

```python
elif long_fir_threshold is None:
    H = torch.fft.rfft(h.to(dtype=torch.float32), n=fft_size) / fft_size
    X_s = torch.fft.fft(x1v.to(dtype=torch.float32), n=fft_size)
    X = X_s[..., : H.shape[-1]]
    if len(z_pre.shape) > 3:
        H = H.unsqueeze(1)
    y = torch.fft.irfft(X * H, n=fft_size, norm="forward")[..., :L]
```

```
SHAPES (the four lines, with intermediate types)

  h                                ( 1 , 4096 , 8192 )      fp32
  h.float() (no-op if already fp32)( 1 , 4096 , 8192 )      fp32
  H = rfft(h, n=16384) / 16384     ( 1 , 4096 , 8193 )      complex64

  x1v                              ( 1 , 4096 , 8192 )      bf16
  x1v.float()                      ( 1 , 4096 , 8192 )      fp32
  X_s = fft(x1v, n=16384)          ( 1 , 4096 , 16384 )     complex64   ← FULL fft, not rfft
  X = X_s[..., :H.shape[-1]]       ( 1 , 4096 , 8193 )      complex64   ← slice to match H

  len(z_pre.shape) == 3 → skip H.unsqueeze(1)

  X * H                            ( 1 , 4096 , 8193 )      complex64
  irfft(X*H, n=16384, "forward")   ( 1 , 4096 , 16384 )     fp32
  [..., :L]                        ( 1 , 4096 , 8192 )      fp32

  y                                ( 1 , 4096 , 8192 )      fp32
```

**Purpose**: FFT-based depthwise long convolution. Three FFT ops:

1. `rfft(h)` to get filter spectrum (real-input symmetry → 8193 bins)
2. `fft(x1v)` to get input spectrum **using the full complex FFT** —
   not the half-bins rfft. Why: the full spectrum is reusable by
   `prefill_via_modal_fft` later; rfft would lose information needed
   for that.
3. `irfft(X * H)` for the inverse, with the same `norm="forward"` trick
   from `fftconv_func` (no /N because H was pre-divided).

The slice `X = X_s[..., :H.shape[-1]]` discards the redundant
high-frequency bins (since `H` is already in rfft-half form), keeping
only what's needed for the multiply.

**This is the path our HCL kernel replaces.** Three FFT launches plus
the elementwise mul plus the irfft trim → all fused, plus the tiled
`compute_filter` so `h` never materializes.

### Step 4: Cast back, skip-add, post-gate (one line)

```python
y = y.to(dtype=x1v.dtype)
y = (y + x1v * D.unsqueeze(-1)) * x2
```

```
SHAPES
  y                       ( 1 , 4096 , 8192 )  fp32
  y.to(bf16)              ( 1 , 4096 , 8192 )  bf16

  x1v                     ( 1 , 4096 , 8192 )  bf16  (held from step 2)
  D                       ( 4096 , )           fp32
  D.unsqueeze(-1)         ( 4096 , 1 )         fp32
  x1v * D[:, None]        ( 1 , 4096 , 8192 )  bf16  (autocast handles dtype)

  y + x1v * D[:, None]    ( 1 , 4096 , 8192 )  bf16
  x2                      ( 1 , 4096 , 8192 )  bf16  (held from step 2)
  (...) * x2              ( 1 , 4096 , 8192 )  bf16

  y (rebound)             ( 1 , 4096 , 8192 )  bf16
```

**Purpose**: Three operations fused into one expression:
1. Cast `y` back from fp32 (FFT output) to bf16 (inference dtype)
2. Add the skip-gain term `x1v * D[:, None]` — same as `D.unsqueeze(-1) * u`
   from [fftconv_func step 6](fftconv_func.md#step-6-skip-residual),
   just multiplied by `x1v` (the gated combination) instead of raw `u`
3. Apply the **second gate** `* x2` — closes the CGCG sandwich

Both gates in one expression. `parallel_fir` had this in two steps with
the padding-mask between them; `parallel_iir` doesn't run the padding
mask (it assumed handled elsewhere), so the two steps collapse.

### Step 5: Prefill caching (only if streaming inference)

```python
if inference_params is not None:
    if prefill_style == "fft":
        self.prefill_via_modal_fft(
            inference_params=inference_params,
            x1v=x1v,
            X_s=X_s,     # ← THE REUSE — full complex spectrum saved from step 3
            L=L,
            t=t,
            poles=poles,
            dims=dims,
            layer_idx=layer_idx,
            use_flashfft=use_flashfft,
            fftconv_fn=fftconv_fn,
        )
    elif prefill_style == "recurrence":
        pass
    else:
        raise NotImplementedError
    if self.low_mem_mode:
        del z_pre, x2, x1, v, x1v, h, poles, residues
        torch.cuda.empty_cache()
```

**Purpose**: When streaming inference is enabled, this populates the
IIR state from the current parallel-prefill result so that subsequent
token-by-token decoding can pick up where prefill left off. The state
construction uses:

- `x1v` (the gated input, already computed)
- `X_s` (the FFT of `x1v`, computed in step 3 with the full `fft` rather
  than `rfft` specifically to enable this reuse)
- `t, poles, dims` (to evaluate `state_s = (poles * t).exp()`)

This block is **skipped entirely** when `inference_params is None`
(parallel-prefill mode, our HCL kernel's target case). The `X_s`
variable is unreferenced after step 3 in that case, and Python's
garbage collector frees its memory.

### Step 6: Layout-swap return

```python
return y.permute(0, 2, 1)
```

```
SHAPES
  y                ( 1 , 4096 , 8192 )   bf16  (channel-second)
  y.permute(0,2,1) ( 1 , 8192 , 4096 )   bf16  (channel-last)
```

**Purpose**: Hand back to the caller in the layer's native layout
(channel-last). `parallel_fir` also produces channel-second output but
the caller permutes at line 316 of model.py; `parallel_iir` permutes
internally as part of its return contract. Different design choices,
no semantic difference.

## The other dispatch branches

### `prefill_style == "recurrence"` (streaming-only)

```python
if inference_params is not None and prefill_style == "recurrence":
    y = self.prefill_via_direct_recurrence(
        inference_params=inference_params,
        x1v=x1v,
        L=L,
        poles=poles,
        residues=residues,
    )
```

Explicit step-by-step recurrence over the IIR state. Used for streaming
inference when the user prefers recurrence over FFT-based prefill. Our
HCL kernel doesn't touch this branch (parallel-prefill assumes
`inference_params=None`).

### `use_flashfft and L % 2 == 0` (HCL fast path, if installed)

```python
if use_flashfft and (L % 2) == 0:
    y = fftconv_fn(
        x1v.to(dtype=torch.bfloat16).contiguous(),
        h.to(dtype=torch.float32),
    )
    X_s = None
```

Single fused-kernel call to FlashFFTConv. **Massively faster** than the
three-FFT-inline path when available. Default off in evo2 configs.

**Important**: `X_s = None` here, so the prefill_via_modal_fft path at
the bottom can't use the FFT-based prefill. The `use_flashfft + streaming
inference` combo requires `prefill_style="recurrence"` to avoid the
missing `X_s`.

### `long_fir_threshold is not None` (truncated direct conv)

```python
else:
    assert h.shape[0] == 1, "batch size must be 1 for long_fir_threshold"
    h = h[0][:, None]
    h = h[..., :long_fir_threshold]
    y = F.conv1d(
        x1v,
        h.to(dtype=x1v.dtype),
        stride=1,
        groups=x1v.shape[1],
        padding=h.shape[-1] - 1,
    )[..., :L]
```

When the user sets `long_fir_threshold=K` for some `K < L`, truncate
the filter to its first K taps and run direct depthwise conv1d. Sanity-
check option; not used in evo2 defaults.

## End-to-end shape pipeline (HCL FFT branch, parallel prefill)

```
z_pre  (1, 12288, 8192) bf16
       │ column_split + flip
       ▼
  x2 (1, 4096, 8192) bf16  ────────────────┐
  x1 (1, 4096, 8192) bf16                  │
  v  (1, 4096, 8192) bf16                  │
       │ x1 * v                            │
       ▼                                   │
 x1v  (1, 4096, 8192) bf16                 │
       │ cast to fp32                      │
       ▼                                   │
       (1, 4096, 8192) fp32                │
       │  fft, pad to 16384                │
       ▼                                   │
 X_s  (1, 4096, 16384) complex64           │
       │ [..., :8193] slice                │
       ▼                                   │
 X    (1, 4096, 8193) complex64 ──┐        │
                                  │        │
 h    (1, 4096, 8192) fp32        │        │
       │ rfft, pad to 16384       │        │
       ▼                          │        │
 H    (1, 4096, 8193) complex64   │        │
       │ /fft_size                │        │
       ▼                          │        │
       (1, 4096, 8193) complex64  │        │
                                  │        │
                          ┌───────┘        │
                          │ complex multiply
                          ▼                │
       (1, 4096, 8193) complex64           │
       │ irfft (no /N), back to 16384      │
       ▼                                   │
       (1, 4096, 16384) fp32               │
       │ [..., :8192] trim                 │
       ▼                                   │
 y    (1, 4096, 8192) fp32                 │
       │ cast to bf16                      │
       ▼                                   │
       (1, 4096, 8192) bf16                │
       │ + x1v * D[:, None]                │
       ▼                                   │
       (1, 4096, 8192) bf16                │
       │ × x2  ←───────────────────────────┘
       ▼
 y    (1, 4096, 8192) bf16
       │ permute(0, 2, 1)
       ▼
RETURN (1, 8192, 4096) bf16
```

## Per-step purpose, in one line each

| Step | Code | Purpose |
|---|---|---|
| 1 | `fft_size = 2 * L; unpack dims` | Bookkeeping + padding factor |
| 2 (gate) | `x2, x1, v = split(z_pre); x1v = x1 * v` | First gate of CGCG |
| 3 (HCL FFT) | `H = rfft(h)/N; X_s = fft(x1v); y = irfft(X_s[..., :H[-1]] * H)[..., :L]` | The 3-FFT depthwise long-conv (the path our kernel replaces) |
| 4 (combined) | `y = (y.to(bf16) + x1v * D[:, None]) * x2` | Cast + skip-add + post-gate, fused |
| 5 (caching) | `prefill_via_modal_fft(...) if inference_params is not None` | Streaming-inference state save (skipped for parallel prefill) |
| 6 | `return y.permute(0, 2, 1)` | Hand back as channel-last `(B, L, D)` |

## What changes for the wire-up

The HCL kernel (Phase 4) **replaces the entire `parallel_iir` method**
when used in parallel-prefill mode (because the dispatch logic is too
intertwined to hook one branch cleanly):

```python
def patched_parallel_iir(self, z_pre, h_unused, D, L, poles, residues, t,
                          dims, layer_idx, inference_params=None,
                          prefill_style="fft", fftconv_fn=None,
                          padding_mask=None, use_flashfft=False,
                          column_split_hyena=False, long_fir_threshold=None):
    # Predicate: target the HCL FFT branch (the path we replace)
    if (inference_params is None
            and not use_flashfft
            and long_fir_threshold is None):
        # ★ Our kernel runs the WHOLE thing — tiled compute_filter, FFT-conv,
        # bias-add, post-gate, layout swap — in one fused Triton+cuFFT path.
        # It also REQUIRES residues + log_poles inputs (not the pre-built h),
        # because tiling compute_filter is the whole point.
        return hcl_fused_kernel(z_pre, residues, poles, D, L, dims,
                                 column_split_hyena=column_split_hyena,
                                 hyena_flip_x1x2=self.hyena_flip_x1x2)

    # Otherwise fall through to upstream (recurrence / flashfft / long_fir_threshold)
    return _ORIGINALS["parallel_iir"](self, z_pre, h_unused, D, L, poles,
                                        residues, t, dims, layer_idx,
                                        inference_params=inference_params,
                                        prefill_style=prefill_style,
                                        fftconv_fn=fftconv_fn,
                                        padding_mask=padding_mask,
                                        use_flashfft=use_flashfft,
                                        column_split_hyena=column_split_hyena,
                                        long_fir_threshold=long_fir_threshold)
```

**Key difference from the HCS adapter**: HCS's adapter takes the
already-materialized weight and dispatches to CGCG. HCL's adapter
**bypasses `compute_filter` entirely** — the tiled kernel computes
filter chunks inline, in registers, never materializing the `(D, S, L)`
intermediate.

The caller (`HyenaCascade.parallel_forward`) still calls
`compute_filter` and passes `h` to `parallel_iir`. **That call becomes
wasted work** when our kernel fires (we ignore the passed `h`). One
optimization for Phase 4 polish: also patch `HyenaCascade.parallel_forward`
to skip the `compute_filter` call when the HCL kernel is active. But
that's not blocking — the wasted `h` allocation at L=131k is still
much smaller than what the OOM elision saves.

See `IMPLEMENTATION_PLAN.md` Phase 4 for the full kernel design
(overlap-save tiling, BLOCK_L=4096 sweet spot, the L=131k unlock test).

## Where `parallel_iir` is used

**Defined** in two places:

| Path | Line | Role |
|---|---|---|
| `vortex/model/engine.py` | 262 | Upstream method on `HyenaInferenceEngine` |
| `vortex_kernels/reference/engine_ref.py` | 277 | Vendored verbatim copy |

**Called** from `HyenaCascade.parallel_forward` once per HCL layer:

| Path | Line | Context |
|---|---|---|
| `vortex/model/model.py` | ~324 | After `compute_filter(L, device)` builds `h`. HCM/HCS layers DON'T reach this call — they take the `parallel_fir` path |

The model.py call site passes a lot of args (poles/residues/t for the
prefill caching, fftconv_fn for FlashFFTConv slot, etc.). Most are
unused in our parallel-prefill HCL path but the signature must match.

## Cross-references

- [fftconv_func](fftconv_func.md) — Target 1 — the HCM FFT-conv that
  parallel_iir's inline branch *mirrors* (same three-FFT pattern). Useful
  to read first because it's simpler.
- [parallel_fir](parallel_fir.md) — Target 2 — the FIR sibling
  method (HCS/HCM). Compare to see why HCL needed a separate method
  (filter shape and dispatch axis are different).
- [compute_filter_pure](compute_filter_pure.md) — Target 6 — how `h`
  is built. Critical for understanding what our HCL kernel must
  reimplement inline.
- [HyenaCascade.parallel_forward](hyena_cascade_parallel_forward.md) —
  Target 5 — the caller, which orchestrates `compute_filter` →
  `parallel_iir`.

## Open questions / to-revisit later

- **`column_split_hyena` reshape contiguity**: the nested reshape /
  slice / re-reshape in step 2 may produce non-contiguous tensors.
  Our HCL kernel asserts contiguous inputs; need to verify and add
  `.contiguous()` if needed.
- **Bypassing the wasted `compute_filter`**: when the HCL kernel
  fires, the upstream `compute_filter` call in `parallel_forward`
  still runs and builds `h`. Wasted ~512 MiB allocation per layer at
  L=131k. Patching `parallel_forward` too saves that. Worth it for
  the final wire-up but not blocking.
- **The `H.unsqueeze(1)` branch**: line ~351 has
  `if len(z_pre.shape) > 3: H = H.unsqueeze(1)`. When does `z_pre`
  have rank > 3? Some training configs with extra heads? Sanity-check
  during Phase 4 development.
- **`X_s` reuse for streaming inference**: our kernel guards on
  `inference_params is None`. Streaming inference users fall through
  to upstream and pay the unfused cost. Acceptable for the initial PR;
  could optimize later if there's demand.

---

*Last updated: 2026-05-12 (Phase 1 read-through, target 3/12)*
