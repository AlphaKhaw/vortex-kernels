# `HyenaCascade.parallel_forward` — the per-layer orchestrator

> **Tour target 5 of 12** · [index](README.md) · prev: [HyenaCascade.\_\_init\_\_](hyena_cascade_init.md) · next: [compute_filter_pure](compute_filter_pure.md)

## Source

| | |
|---|---|
| Upstream | `vortex/model/model.py:221-317` |
| Called by | `HyenaCascade.forward` (line 218) when streaming-inference state isn't active |
| Calls into | `self.engine.parallel_fir` (×2), `self.compute_filter`, `self.engine.parallel_iir` |

## Overall purpose

**This is THE per-layer orchestrator** — the bridge between the model's
parameter declarations (from `__init__`) and the kernel-level execution
(in the engine methods). Every HCS, HCM, and HCL forward pass during
parallel prefill (= our HCS adapter's path) goes through this method
once per layer.

The method does **the same three-step routine for every layer**:
1. **Featurizer conv** — apply the short input projection conv
   (`engine.parallel_fir` with `gate=False, fir_length=3`)
2. **Filter prep** — for HCS/HCM, read `self.h` (already declared);
   for HCL, build `h` via `compute_filter`. Then `repeat_interleave`
   to expand from `(g, 1, hl)` to `(D, 1, hl)`.
3. **Cascade conv** — branches on `fir_inner_filter_length`:
   - `is None` (HCL) → call `engine.parallel_iir` with the
     materialized `h`
   - `is not None` (HCS/HCM) → call `engine.parallel_fir` AGAIN with
     `gate=True` and the per-channel `h`

That's it. ~100 lines of orchestration; almost no math in this method
itself — it's all routing to the engine. **But the routing is exactly
what your HCS adapter must trace** to know which call to intercept,
what shape the args have, and what the post-conditions are.

## Why this is critical for the wire-up

Three specific lines in this method determine the predicate and
shape-handling for our HCS adapter:

1. **Line 262**: `h = h.repeat_interleave(D//g, 0)` — expands the
   per-group filter to per-channel. Your adapter must invert this
   (`weight[::dg]`) to get back to `(g, 1, hl)` for CGCG.
2. **Line 279**: `gated_bias=self.fir_inner_filter_length >= 128` —
   the distinguishing flag between HCS (`False`) and HCM (`True`)
   when both hit the cascade `parallel_fir` call.
3. **Line 285**: `groups=self.hyena_filter_groups` — this is the
   predicate signal that lets your adapter know "this is the HCS/HCM
   cascade case, not the featurizer".

The featurizer call (lines 233–246) doesn't pass `groups`, so the
adapter's predicate `groups is not None and groups > 1` fails there →
falls through to upstream. The cascade call (lines 271–286) does pass
`groups=hyena_filter_groups` → predicate matches → adapter fires.

## Realistic input shapes (single HCS layer in evo2_7b at L=8192)

```
INPUTS to parallel_forward
══════════════════════════

u                    ( 1 , 8192 , 12288 )  bf16   ← projected input
                     └─┬─┘└──┬──┘└──┬──┘            from ParallelGatedConvBlock.proj_norm
                       B    L     3*D                = (B, L, 3*hidden_size) after the up-projection
                                                       wait — actually u is just the input,
                                                       the 3*D expansion happens in the FEATURIZER.
                                                       Let me re-check.

Actually for the featurizer call:
u                    ( 1 , 8192 , 12288 )  bf16   ← already projected to 3*D channels
                                                     by ParallelGatedConvBlock.proj_norm at model.py:~538
                                                     (3*hidden_size linear projection)

inference_params     None  (parallel prefill mode — our path)
padding_mask         None  (typical for single-batch inference)
```

⚠️ Note: I was initially confused about where the `3*D` expansion
happens. It's NOT inside `HyenaCascade.parallel_forward` — it happens
upstream in `ParallelGatedConvBlock.proj_norm` (model.py around line
538) via a `TELinear(hidden_size, 3*hidden_size)`. So by the time `u`
arrives here, it's already `(B, L, 3*D)`.

## Inputs explained

| Arg | Role |
|---|---|
| `u` | The input activations from the previous block, shape `(B, L, 3*D)` bf16 (already projected by `proj_norm`) |
| `inference_params` | `None` for parallel prefill, an `InferenceParams` instance for streaming |
| `padding_mask` | Optional `(B, L)` bool mask; usually `None` in single-batch inference |

## Step-by-step with shape tracking (HCS layer path)

I'll walk the HCS layer case since that's the wire-up target. HCM and
HCL diverge at step 5; I'll call out the differences there.

### Step 1: Read L, bundle `dims`

```python
L = u.shape[1]
dims = (
    self.hidden_size,                       # 4096
    self.num_attention_heads,                # e.g., 8
    self.hidden_size_per_attention_head,     # 4096/8 = 512
    self.state_size,                         # 16
    self.hyena_filter_groups,                # 256 for HCS, 128 for HCM, 4096 for HCL
)
```

```
SHAPES
  u                       ( 1 , 8192 , 12288 )    bf16
  L = u.shape[1] = 8192
  dims = (4096, 8, 512, 16, 256) for HCS
```

**Purpose**: Bundle the shape constants that get passed through every
engine call. `L` is read from the actual tensor (not the config) so it
adapts to the runtime sequence length.

### Step 2: Featurizer call (the first `parallel_fir`)

```python
z_pre, fir_state = self.engine.parallel_fir(
    self.fir_fn,                 # F.conv1d
    u,                            # (B, L, 3*D)
    self.short_filter_weight,    # (3*D, 1, 3)
    self.short_filter_bias,      # None (config.short_filter_bias = False)
    L,
    dims=dims,
    gate=False,                  # ← FEATURIZER mode, no gating
    column_split_hyena=self.column_split_hyena,
    fir_length=self.short_filter_length,  # 3
    inference_params=inference_params,
    padding_mask=padding_mask,
    dim_last=True,                # ← channel-last input
)
```

```
SHAPES
  Input:
    u                       ( 1 , 8192 , 12288 )   bf16    (B, L, 3*D)
    short_filter_weight     ( 12288 , 1 , 3 )      bf16    (3*D, 1, 3)
    short_filter_bias       None
  Output:
    z_pre                   ( 1 , 12288 , 8192 )   bf16    (B, 3*D, L) — channel-second after permute
    fir_state               None (since inference_params is None)
```

**Purpose**: Apply the short input projection conv. This is the
"featurizer" — a length-3 depthwise conv on 3*D channels. Same call
path as [parallel_fir](parallel_fir.md) target 2 walked through.

Note `dim_last=True` here — `parallel_fir` will permute `(B, L, 3*D) →
(B, 3*D, L)` internally before the conv. The output comes back as
`(B, 3*D, L)`.

**The output `z_pre` is the pre-split tensor for the cascade conv.**
It still has 3*D channels; the next call will split it into x2/x1/v.

### Step 3: Streaming-inference state save (no-op for us)

```python
if inference_params:
    inference_params.fir_state_dict[self.layer_idx] = fir_state
```

Skipped when `inference_params is None`. Parallel prefill never enters
this branch.

### Step 4: Interleave (rarely used)

```python
if self.config.interleave:
    z_pre = interleave(z_pre)
```

`interleave` rearranges the x2/x1/v ordering inside `z_pre`. Used for
some training configs; evo2's default is `interleave=False`. Skipped.

### Step 5: Filter prep — `h` and repeat_interleave

```python
if self.h is None:
    h, _, _, _ = self.compute_filter(L, u.device)
else:
    h = self.h

D = self.D

if self.hyena_filter_groups > 1:
    h = h.repeat_interleave(self.hidden_size // self.hyena_filter_groups, 0)
```

```
SHAPES — depends on layer type

HCS:  self.h = (256, 1, 7)        ← stored param, per-group
      h = self.h
      D = self.D = None
      g=256, D//g = 4096/256 = 16
      After repeat_interleave:
        h = (4096, 1, 7)           ← per-channel, every 16 consecutive
                                      channels share the same filter

HCM:  self.h = (128, 1, 128)       ← stored param, per-group
      h = self.h
      D = self.D = (4096,)         ← gated bias
      g=128, D//g = 4096/128 = 32
      After repeat_interleave:
        h = (4096, 1, 128)         ← per-channel

HCL:  self.h is None → compute_filter(L, device) materializes:
        h = (1, 4096, 8192)         ← (1, D, L)
      D = self.D = (4096,)          ← skip-gain
      g=4096 (= D), D//g = 1
      After repeat_interleave with factor 1:
        h = (1, 4096, 8192)         ← unchanged (no-op for HCL)
```

**Purpose**: Build the filter tensor that gets passed to the engine.
Two design decisions to internalize:

1. **`self.h` is stored per-group; the engine wants per-channel.** The
   `repeat_interleave(D//g, 0)` is what reconciles them. Each block of
   `D//g` consecutive channels uses the same filter. This is the line
   your HCS adapter must mentally undo.

2. **For HCL, `self.h` is `None`**, so `compute_filter` builds it on
   the fly from `(residues, log_poles)`. This is **the OOM source**
   at large L — the intermediate `(D, S, L)` fp32 tensor inside
   `compute_filter`. Our HCL kernel skips this call entirely and
   computes filter chunks inline.

### Step 6: Cascade dispatch — HCS/HCM branch

```python
if self.fir_inner_filter_length is not None:
    y, fir_inner_state = self.engine.parallel_fir(
        self.fir_inner_fn,                                    # F.conv1d
        z_pre,                                                 # (B, 3*D, L)
        h,                                                     # (D, 1, hl) post-repeat
        D,                                                     # None (HCS) or (D,) (HCM)
        L,
        dims=dims,
        gate=True,                                             # ← CASCADE mode, gated
        gated_bias=self.fir_inner_filter_length >= 128,       # ← False for HCS, True for HCM
        dim_last=False,                                        # ← channel-second already
        column_split_hyena=self.column_split_hyena,
        fir_length=self.fir_inner_filter_length,              # 7 or 128
        inference_params=inference_params,
        padding_mask=padding_mask,
        groups=self.hyena_filter_groups,                      # 256 or 128 — THE PREDICATE SIGNAL
    )
    y = y.permute(0, 2, 1)                                    # (B, D, L) → (B, L, D)
    if inference_params:
        inference_params.fir_inner_state_dict[self.layer_idx] = fir_inner_state
```

```
SHAPES (HCS layer)
  Input to parallel_fir:
    z_pre         ( 1 , 12288 , 8192 )    bf16
    h             ( 4096 , 1 , 7 )        bf16    (after repeat_interleave)
    D = bias      None                              (HCS has self.D = None)
  Output of parallel_fir:
    y             ( 1 , 4096 , 8192 )     bf16    (B, D, L)
    fir_inner_state   None
  After y.permute(0, 2, 1):
    y             ( 1 , 8192 , 4096 )     bf16    (B, L, D) channel-last
```

**Purpose**: This is the **HCS/HCM cascade conv** — the second
`parallel_fir` call that does the long(ish) gated conv. The args here
are what defines the wire-up predicate:

- `gate=True` ✓
- `fir_length=7` (HCS) or `128` (HCM); for HCS `< 128` ✓
- `groups=256` (HCS) or `128` (HCM); both `> 1` ✓ ✓

So our HCS adapter's predicate (`gate=True AND fir_length<128 AND
groups>1`) fires on the HCS variant but not HCM (HCM's `fir_length`
fails the `<128` test). HCM goes through `fftconv_func` per
`parallel_fir`'s HCM branch.

The `gated_bias` arg is the HCS/HCM differentiator INSIDE `parallel_fir`.
For HCS it's `False` (simple bias-add); for HCM it's `True` (gated
bias-add `z + bias[None, :, None] * u`). Your HCS adapter should
assert `not gated_bias` since CGCG's contract is simple bias-add.

The final `y.permute(0, 2, 1)` converts back to channel-last for the
next layer.

### Step 7: HCL branch (parallel_iir)

```python
else:  # fir_inner_filter_length is None — HCL
    y = self.engine.parallel_iir(
        z_pre,                                          # (B, 3*D, L)
        h,                                              # (1, D, L) materialized by compute_filter
        D,                                              # (D,) skip-gain
        L,
        t=self.t,                                       # built lazily by update_time
        poles=self.log_poles,                            # (D, S, 1)
        residues=self.residues,                          # (D, S)
        dims=dims,
        inference_params=inference_params,
        layer_idx=self.layer_idx,
        prefill_style=self.config.get("prefill_style", "fft"),
        use_flashfft=self.use_flashfft,
        fftconv_fn=self.fftconv_fn,                      # FlashFFTConv slot (None unless installed)
        column_split_hyena=self.column_split_hyena,
        long_fir_threshold=self.long_fir_threshold,
        padding_mask=padding_mask,
    )
```

```
SHAPES (HCL layer)
  Input to parallel_iir:
    z_pre         ( 1 , 12288 , 8192 )    bf16
    h             ( 1 , 4096 , 8192 )     fp32   (materialized by compute_filter)
    D             ( 4096 , )              fp32   (zeros-initialized)
    poles, residues, t  ... (used for prefill caching only)
  Output of parallel_iir:
    y             ( 1 , 8192 , 4096 )     bf16   (B, L, D) — parallel_iir permutes internally
```

**Purpose**: The HCL FFT-conv branch. See [parallel_iir](parallel_iir.md)
target 3 for the inside.

Note that `parallel_iir` returns `y.permute(0, 2, 1)` internally, so
its output is already channel-last `(B, L, D)`. That's why the HCL
branch DOESN'T have an explicit `y = y.permute(0, 2, 1)` here, while
the HCS/HCM branch does (their output from `parallel_fir` is
channel-second).

### Step 8: Return

```python
return y, inference_params
```

```
SHAPES
  y                    ( 1 , 8192 , 4096 )    bf16   (B, L, D)
  inference_params     None or InferenceParams instance
```

**Purpose**: Hand back the layer output (channel-last) and the
inference state (untouched). The caller (`ParallelGatedConvBlock`)
applies a final output projection.

## End-to-end shape pipeline (HCS layer)

```
u                     (1, 8192, 12288) bf16    [B, L, 3*D]
  │ engine.parallel_fir #1 (FEATURIZER)
  │   gate=False, fir_length=3, dim_last=True
  │   F.conv1d on (1, 12288, 8192) post-permute
  ▼
z_pre                 (1, 12288, 8192) bf16    [B, 3*D, L]

  │ self.h is not None (HCS) → read directly
  ▼
h_raw                 (256, 1, 7) bf16         [g, 1, hl]   ← per-GROUP

  │ repeat_interleave(D//g=16, dim=0)
  ▼
h                     (4096, 1, 7) bf16        [D, 1, hl]   ← per-CHANNEL

  │ engine.parallel_fir #2 (CASCADE)
  │   gate=True, fir_length=7, dim_last=False, groups=256
  │   ★ THIS IS THE CALL YOUR HCS ADAPTER INTERCEPTS ★
  ▼
y_intermediate        (1, 4096, 8192) bf16     [B, D, L]
  │ permute(0, 2, 1)
  ▼
y                     (1, 8192, 4096) bf16     [B, L, D]
  │
  ▼
return (y, inference_params)
```

## Per-step purpose, in one line each

| Step | Code | Purpose |
|---|---|---|
| 1 | `L, dims = ...` | Read runtime sequence length + bundle shape constants |
| 2 | `engine.parallel_fir(gate=False, fir_length=3)` | Apply featurizer (input projection conv) |
| 3 | `inference_params.fir_state_dict[idx] = fir_state` | Save streaming state (no-op for prefill) |
| 4 | `if config.interleave: ...` | Optional channel reorder (default off in evo2) |
| 5a | `h = self.h or compute_filter(L)` | Read or build the cascade filter |
| 5b | `h = h.repeat_interleave(D//g, 0)` | Expand per-group filter to per-channel |
| 6 (HCS/HCM) | `engine.parallel_fir(gate=True, fir_length=hl, groups=g)` + permute | Apply cascade conv ★ |
| 7 (HCL) | `engine.parallel_iir(z_pre, h, D, ...)` | Apply IIR FFT-conv (returns already permuted) |
| 8 | `return y, inference_params` | Hand back layer output |

★ = the call your HCS adapter intercepts.

## The three layer-type paths through this method

```
                  parallel_forward called
                  ───────────────────────
                          │
                          ▼
              Step 1-2: ALWAYS run featurizer
                          │
                          ▼
              Step 5: Build/read h, repeat_interleave
                          │
                          ▼
                ┌─────────┴─────────┐
                │  fir_inner_       │
                │  filter_length    │
                │  is None?         │
                └─────┬───────┬─────┘
                      │YES    │NO
                      ▼       ▼
                ┌──────┐  ┌──────────────────────┐
                │ HCL  │  │ fir_inner_filter_    │
                │branch│  │ length >= 128?       │
                │      │  └──┬───────────────┬───┘
                │      │     │YES (HCM)      │NO (HCS)
                │      │     ▼               ▼
                │      │  ┌─────────────┐ ┌─────────────┐
                │      │  │ HCM branch  │ │ HCS branch  │
                │      │  │ gated_bias  │ │ gated_bias  │
                │      │  │  = True     │ │  = False    │
                │      │  │ fir_length= │ │ fir_length= │
                │      │  │  128        │ │  7          │
                │      │  └──────┬──────┘ └──────┬──────┘
                │      │         │               │
                │      │  goes through parallel_fir's
                │      │  HCM branch (calls fftconv_func)
                │      │                         │
                │      │                  goes through parallel_fir's
                │      │                  HCS branch (calls F.conv1d)
                │      │                         │
                ▼      ▼                         ▼
            parallel_iir's              parallel_fir's
            HCL FFT branch              cascade dispatch
            (the long FFT path)         (3-way: deprecated/HCM/HCS)
```

All three paths share steps 1, 2, 5 of this method. They diverge at
step 6/7.

## What changes for the wire-up

`HyenaCascade.parallel_forward` itself is **NOT patched** by Phase 2
(HCS) or Phase 3 (HCM) — the engine-level patches handle those. For
Phase 4 (HCL), this method is *also* a candidate for patching as an
optimization:

```python
# Phase 4 OPTIONAL polish: patch parallel_forward to skip compute_filter
# when our HCL kernel will fire. Saves the (1, D, L) fp32 allocation
# (~512 MiB at L=131k) since our kernel re-derives chunks inline.

def patched_parallel_forward(self, u, inference_params=None, padding_mask=None):
    if (self.fir_inner_filter_length is None
            and inference_params is None
            and not self.use_flashfft
            and self.long_fir_threshold is None):
        # HCL fast path: skip compute_filter; HCL kernel reads residues+log_poles directly
        return _hcl_fused_path(self, u, padding_mask)
    return _ORIGINALS["parallel_forward"](self, u, inference_params, padding_mask)
```

But this is a polish, not blocking. The base HCL kernel patches
`engine.parallel_iir` and accepts the wasted `compute_filter` work.

For Phase 2 (HCS) and Phase 3 (HCM), **no change to this method** — the
engine-level adapter handles it.

## Cross-references

- [parallel_fir](parallel_fir.md) — Target 2 — the engine method
  this calls twice (featurizer + cascade)
- [parallel_iir](parallel_iir.md) — Target 3 — the engine method
  this calls for HCL
- [HyenaCascade.\_\_init\_\_](hyena_cascade_init.md) — Target 4 — declares
  the parameters this method reads
- [compute_filter_pure](compute_filter_pure.md) — Target 6 — what
  `self.compute_filter(...)` calls (with the OOM-source intermediate)

## Open questions / to-revisit later

- **Where does `(B, L, 3*D)` actually come from?** Confirmed via line
  538 of model.py (`ParallelGatedConvBlock.proj_norm` does the
  `TELinear(hidden_size, 3*hidden_size)` projection). Worth a sanity
  re-check during e2e testing — the shape arriving here must already
  be `(B, L, 3*D)` for the featurizer to work.
- **The `column_split_hyena` toggle**: affects both the featurizer
  call and the cascade call. Need to verify the column-split path
  produces the same `(x2, x1, v)` decomposition as the simple split
  for our test inputs. Should be a quick numerical check.
- **`compute_filter` allocation when HCL kernel fires**: 512 MiB
  wasted per layer at L=131k. Phase 4 polish if motivated.
- **`interleave=True` configs**: would the HCS adapter still work if
  the user sets `interleave=True`? Probably yes (interleave just
  reorders channels in `z_pre`, doesn't change shape), but worth a
  test case.

---

*Last updated: 2026-05-12 (Phase 1 read-through, target 5/12)*
