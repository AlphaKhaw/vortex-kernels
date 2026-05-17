# `HyenaInferenceEngine.parallel_fir` — the FIR dispatch method

> **Tour target 2 of 12** · [index](README.md) · prev: [fftconv_func](fftconv_func.md) · next: [parallel_iir](parallel_iir.md)

## Source

| | |
|---|---|
| Upstream | `vortex/model/engine.py:131-260` |
| Vendored copy | `vortex_kernels/reference/engine_ref.py:146-275` |
| Called from | `HyenaCascade.parallel_forward` (model.py) — twice per layer (see below) |

## Overall purpose

**Dispatch the right depthwise 1D convolution for whichever Hyena flavor
this layer is.** Decides between three conv implementations based on
filter length and whether a custom kernel was injected:

- `fir_fn != F.conv1d` → **deprecated branch** — call whatever the
  caller passed in (legacy hook for custom conv implementations)
- `fir_length >= 128` → **HCM branch** — FFT-based conv via
  [`fftconv_func`](fftconv_func.md)
- `fir_length < 128` → **HCS branch** — direct depthwise conv1d

It also handles the **gated cascade** structure: when `gate=True`, the
input is split into three streams `(x2, x1, v)`, the conv is applied to
`x1 * v`, and the output is gated by `x2` post-conv. That gating
sequence is what makes HCS a "gated short conv" rather than just a
plain depthwise filter.

## The two call patterns

This is the gotcha that takes a beat to internalize. `parallel_fir`
gets called **twice per Hyena layer** with different shape conventions
and different roles:

```
HyenaCascade.parallel_forward(u_layer_input)
    │
    │ [u shape: (B, L, D)]
    │
    ├─► engine.parallel_fir(...)       Call #1: HCS FEATURIZER
    │     gate=False                                 ↑
    │     fir_length=3 (short_filter_length)        │
    │     dim_last=True                              │ projects layer input
    │     u: (B, L, 3*D)                             │ through a short
    │     weight: (3*D, 1, 3)                        │ depthwise conv
    │     output: (B, 3*D, L)
    │
    │     [output gets split into x2, x1, v by the cascade,
    │      then mixed and re-packed back as z_pre (B, 3*D, L)]
    │
    └─► engine.parallel_fir(...)       Call #2: HCS / HCM CASCADE
          gate=True                                  ↑
          fir_length=7 (HCS) or 128 (HCM)            │
          dim_last=False                              │ the gated long-ish
          u: (B, 3*D, L)                              │ conv that defines
          weight: (D, 1, 7) or (D, 1, 128)            │ HCS / HCM
          output: (B, D, L)
```

**The HCS adapter only intercepts call #2** (the cascade). Call #1
(the featurizer) stays as upstream `F.conv1d`. The predicate in
`patching.py` distinguishes them: `gate=True AND fir_length<128 AND
groups>1` matches *only* call #2 with HCS-shaped args.

## Realistic input shapes (evo2_7b HCS cascade — the case we patch)

```
INPUTS to parallel_fir (gate=True, fir_length=7, dim_last=False)
══════════════════════════════════════════════════════════════════

u       ( 1 , 12288 , 8192 )  bf16   ← pre-split cascade input
        └─┬─┘└──┬──┘└──┬──┘             3*D channels = (x2, x1, v) packed
          B   3*D=12288 L=8192

weight  ( 4096 , 1 , 7 )      bf16   ← HCS cascade filter
        └──┬──┘└┬┘└┬┘            (already repeat_interleaved from (g=256, 1, 7))
          D=4096 1 K=7

bias    ( 4096 , )            bf16   ← the D parameter (skip gain)
        └──┬──┘
         D=4096

groups = 256                          ← hyena_filter_groups for HCS in evo2_7b
fir_length = 7                        ← HCS cascade short conv length
column_split_hyena = True             ← evo2 default split mode
gate = True
dim_last = False
```

## Inputs explained

### `fir_fn`

The convolution function to apply. **Defaults to `F.conv1d`** unless
the caller passes something else. Vortex uses this as a legacy
"override the conv" hook — pass your own callable and the deprecated
branch fires instead of the standard `F.conv1d` path.

Mental model: this is the "strategy pattern" parameter. The strategies are:
- `F.conv1d` (default) → take the HCM or HCS branch based on filter length
- Any other callable → take the deprecated branch, call the callable directly

**Our HCS wire-up does NOT use this hook** (would lose access to `x2`
for the post-gate). We monkey-patch `parallel_fir` itself instead. See
[Wire-up](#what-changes-for-the-wire-up) below.

### `u` — input activations

Shape depends on `dim_last`:

| `dim_last` | Shape | Used by |
|---|---|---|
| `True` | `(B, L, channels)` | The HCS featurizer call (channels last) |
| `False` | `(B, channels, L)` | The HCS/HCM cascade call (channels in middle) |

Inside the function, `parallel_fir` permutes to `(B, channels, L)`
before the conv if `dim_last=True`. PyTorch's `F.conv1d` expects
channel-second.

### `weight` — depthwise filter

For the **featurizer call**: shape `(3*D, 1, short_filter_length=3)`.
For the **cascade call** (gate=True): shape `(D, 1, fir_inner_filter_length)`,
which is 7 for HCS or 128 for HCM. The `1` is the depthwise marker (one
input channel per group, where `groups=channels`).

### `bias` — the skip-gain / additive bias

For the featurizer: the optional `self.short_filter_bias`. Usually
`None` in evo2.

For the cascade: this is the **`D` parameter** (skip-gain) you saw in
[`fftconv_func`](fftconv_func.md#d--per-channel-skip-gain-the-parameter-not-the-channel-count).
Added after the conv via `z = z + bias[None, :, None]`.

⚠️ Naming reminder: `bias` here is the variable name. The actual tensor
passed is `self.D` from `HyenaCascade.__init__`. The function-local
name `bias` is a holdover from generic conv1d nomenclature.

### `L` — sequence length

Recomputed inside the function from `u.shape` based on `dim_last`. The
caller passes it but the function trusts its own measurement.

### `dims` — bundled shape constants

A tuple `(hidden_size, num_attention_heads, hidden_size_per_attention_head, state_size, hyena_filter_groups)`. Used only by the `column_split_hyena=True`
branch of the gate-prep.

### `groups` — filter-sharing count

**This is the most confusing parameter.** It does NOT control whether
the conv is depthwise — the conv is ALWAYS depthwise (line 218:
`groups=u.shape[1]`). What `groups` controls is:

- `None` (featurizer call) → no effect; the conv uses `u.shape[1]` as
  groups
- `>1` (cascade call, value = `hyena_filter_groups` = 256 for HCS) →
  a marker that says "this layer's filter was originally `(g, 1, hl)`
  per-group, then `repeat_interleave`d up to `(D, 1, hl)` by the
  cascade before being passed in"

**Why this matters for the wire-up**: our HCS adapter uses `groups`
as the signal to know "this is the cascade case where we should reach
for CGCG", AND it needs to invert the `repeat_interleave` to get back
to `(g, 1, hl)` for CGCG's per-group filter contract. So `groups`
isn't just routing — it's the de-expand factor: `g = groups`,
`dg = D // groups = 4096 // 256 = 16`, weight slice
`weight[::dg]` recovers the `(g, 1, hl)` original.

### `gated_bias` — bias multiplication mode

- `False` (HCS) → bias is added: `z = z + bias[None, :, None]`
- `True` (HCM with fir_inner_filter_length≥128) → bias is multiplicatively
  applied: `z = z + bias[None, :, None] * u` (where `u = x1 * v`)

Don't worry about this for HCS — it's always `False`.

### `column_split_hyena` — split mode

- `False` → simple `u.split([D, D, D], dim=1)` to get `x2, x1, v`
- `True` (evo2 default) → reshape-then-split that respects attention-head
  structure (the `column_split` helper in `vortex/model/utils.py`)

Both produce the same logical `(x2, x1, v)` tuple, just different
memory layouts. Your adapter must handle both (or assert one).

### `dim_last` — input layout

Discussed above. `True` = `(B, L, channels)`, `False` = `(B, channels, L)`.
The function permutes internally.

### `fir_length` — filter length, the dispatch key

- `< 128` → HCS branch (direct conv1d)
- `>= 128` → HCM branch (FFT-conv via `fftconv_func`)

This is the **main dispatch axis**. Different from `weight.shape[-1]`
in the sense that the caller passes the *intended* length even if
weight is padded.

### `gate` — gated cascade flag

- `False` (featurizer) → skip the split / multiply / post-gate steps;
  treat `u` as a plain input and apply conv directly
- `True` (cascade) → do the full gated dance:
  ```
  x2, x1, v = u.split(...)
  u = x1 * v
  ... conv ...
  z = x2 * z  ← post-gate
  ```

### `inference_params` — streaming inference handle

If non-`None`, the function captures the last `fir_length - 1` samples
of `u` as `fir_state` for the next streaming step. For parallel prefill
(our path), this is always `None` and `fir_state` is returned as `None`.

### `prefill_mode` — unused in current code

Parameter declared but not read anywhere in the function body.
Vestigial.

### `padding_mask` — sequence-mask tensor

If a `torch.Tensor`, mask `z` after the conv: `z = z * padding_mask[:, None]`.
For autoregressive inference this is usually `None`.

## Background concepts

### Depthwise conv1d in PyTorch

Standard `F.conv1d(input, weight, ..., groups=G)`:
- `input` shape: `(B, in_channels, L)`
- `weight` shape: `(out_channels, in_channels / groups, K)`
- `groups=in_channels=out_channels` → **depthwise**: each input channel
  has its own filter, channels never mix

Visually for depthwise:

```
                       INPUT (B, D, L)
                       │
                       ▼  apply D independent filters
              ┌────────┼────────┬─ ── ── ┐
              │        │        │        │
              filter   filter   filter   filter
              k[0]     k[1]     k[2]     k[D-1]
              │        │        │        │
              ▼        ▼        ▼        ▼
            out[0]   out[1]   out[2]   out[D-1]
              └────────┬────────┘────────┘
                       ▼
                    OUTPUT (B, D, L)
                    (channels independent;
                     no mixing happened)
```

The line `groups=u.shape[1]` in parallel_fir (line 218 of upstream) is
what enforces this — `groups` equals the channel count, so each channel
is its own group of size 1.

### `dim_last` — why the layout switching?

PyTorch's `F.conv1d` *requires* channel-second layout `(B, C, L)`. But
the rest of the Hyena cascade uses `(B, L, C)` (channels-last) because
that's what the linear projections produce. So `parallel_fir`:

1. Receives `u` in whichever layout the caller chose
2. Permutes to `(B, C, L)` before conv if needed
3. Returns `z` in `(B, C, L)` regardless of input layout (the caller
   re-permutes as needed)

The two call sites pass different `dim_last`:
- Featurizer (model.py:260) passes `dim_last=True` — input is `(B, L, 3*D)`
- Cascade (model.py:298) passes `dim_last=False` — input is `(B, 3*D, L)`

### Gated cascade — the `x2 * conv(x1*v, h)` pattern

The Hyena operator uses this structure:

```
u_in
 │
 ▼
3-way split → x2, x1, v   (each (B, D, L))
 │              │  │
 │              └──┴── elementwise multiply → u = x1 * v
 │                                            │
 │                                            ▼
 │                                         long depthwise conv
 │                                         u → h ⊛ u
 │                                            │
 │                                         post-conv result z
 │                                            │
 │  ←── elementwise multiply by x2 ──────────┘
 ▼
z * x2 → output
```

This is the CGCG pattern (Conv-Gate-Conv-Gate) from the original
Hyena paper. Two gates (`x1*v` pre-conv, `x2*z` post-conv) sandwich
the long convolution. It's what gives Hyena its expressivity — the
gates are *data-dependent*, the conv is *fixed* (per-layer learnable
weights, fixed at inference).

## Step-by-step with shape tracking (HCS cascade path)

Walking the gate=True, fir_length=7 case. Other branches and the
featurizer case follow in their own sections below.

### Step 1: Resolve L

```python
L = u.shape[1] if dim_last else u.shape[2]
```

```
SHAPES at entry (cascade call, dim_last=False)
  u  ( 1 , 12288 , 8192 )   bf16
  L = u.shape[2] = 8192
```

**Purpose**: Trust your own measurement of L rather than the caller's
argument. The caller-passed `L` exists to support cases where the
weight is longer than the sequence (e.g., HCL where `weight.shape[-1] = L_max`
but you're only using the first L samples).

### Step 2: Gate prep (gate=True only)

```python
if gate:
    hidden_size, num_attention_heads, hidden_size_per_attention_head, _, _ = dims
    if column_split_hyena:
        x2, x1, v = column_split(u, num_attention_heads, hidden_size_per_attention_head)
    else:
        x2, x1, v = u.split([hidden_size, hidden_size, hidden_size], dim=1)
    if self.hyena_flip_x1x2:
        x1, x2 = x2, x1
    u = x1 * v
```

```
SHAPES (evo2 has column_split_hyena=True)
  u                ( 1 , 12288 , 8192 )   bf16
  After column_split:
    x2, x1, v each ( 1 , 4096  , 8192 )   bf16
  After u = x1 * v:
    u              ( 1 , 4096  , 8192 )   bf16   ← rebound name
```

**Purpose**: This is the **first gate** of the CGCG pattern. The split
into three streams separates "what to filter" (the `x1 * v`
combination) from "what to gate the output by" (the `x2` stream). Note
that **`u` is rebound** — after this block, the variable name `u`
refers to `x1 * v`, not the original input. `x2` is held in scope for
step 5.

### Step 3: Branch dispatch — HCS path

```python
elif fir_length >= 128:
    # HCM branch (skipped for HCS)
else:
    if dim_last:
        u = u.permute(0, 2, 1)  # ← skipped, we're dim_last=False already

    if groups is None:
        g = u.shape[1]
    else:
        g = groups   # = 256 for HCS

    z = fir_fn(
        u.to(torch.float32),       # cast bf16 → fp32 for conv stability
        weight.to(torch.float32),  # same
        bias=None,
        stride=1,
        padding=fir_length - 1,    # = 6 for HCS (left-pad for causal)
        groups=u.shape[1],         # = 4096 (D) — ALWAYS depthwise
    )[..., :L]
```

```
SHAPES
  u (in fp32 cast)     ( 1 , 4096 , 8192 )   fp32
  weight (in fp32)     ( 4096 , 1 , 7 )      fp32
  F.conv1d output      ( 1 , 4096 , 8192+6 ) fp32   ← padded by fir_length-1=6
  After [..., :L]      ( 1 , 4096 , 8192 )   fp32   ← trim to seqlen
```

**Purpose**: Apply the depthwise conv. Key details:
- **`groups=u.shape[1]`** (line 218 upstream) — *always* depthwise,
  regardless of the `groups` parameter. The `groups` parameter is a
  marker for the caller's intent (group-sharing factor) but the conv
  itself never groups channels.
- **`padding=fir_length-1`** with the `[..., :L]` slice gives a
  **causal** conv: output `z[t]` depends only on `u[t-K+1..t]`. The
  left-pad shifts the filter so position `t` sees the past, and the
  right-side overflow is trimmed.
- **fp32 cast** for numerical stability (matches the autocast scope's
  expectation that the conv runs in fp32).

### Step 4: Cast back + bias-add

```python
    z = z.to(u.dtype)

    if bias is not None:
        if gated_bias:
            z = z + bias[None, :, None] * u
        else:
            z = z + bias[None, :, None]
```

```
SHAPES
  z                  ( 1 , 4096 , 8192 )   bf16   ← cast back from fp32
  bias               ( 4096 , )            bf16   (the D parameter)
  bias[None,:,None]  ( 1 , 4096 , 1 )      bf16
  z + bias[None,:,None]
                     ( 1 , 4096 , 8192 )   bf16   ← broadcast over B and L
```

**Purpose**: Skip-gain term. Same role as `D.unsqueeze(-1) * u` in
[`fftconv_func`](fftconv_func.md#step-6-skip-residual) but with an
*additive* skip rather than a multiplicative one (HCS uses
`gated_bias=False`).

### Step 5: Padding mask + post-gate

```python
if type(padding_mask) == torch.Tensor:
    z = z * padding_mask[:, None]

if gate:
    z = x2 * z
```

```
SHAPES
  padding_mask (if Tensor)  ( 1 , 8192 )           bool
  padding_mask[:, None]     ( 1 , 1 , 8192 )       bool
  z * padding_mask[:, None] ( 1 , 4096 , 8192 )    bf16   (zeros out padded positions)

  After post-gate:
  x2                        ( 1 , 4096 , 8192 )    bf16   (held from step 2)
  z = x2 * z                ( 1 , 4096 , 8192 )    bf16
```

**Purpose**:
- The padding mask zeros out positions beyond the actual sequence
  length (autoregressive inference usually has no padding, but the
  branch exists for batch-prefill).
- The **post-gate** `z = x2 * z` is the **second gate** of the CGCG
  pattern. It closes the gating sandwich: `x2 * conv(x1*v, weight)`.

### Step 6: FIR state for streaming inference

```python
if inference_params is not None:
    fir_state = u[..., -fir_length + 1:]
else:
    fir_state = None

return z, fir_state
```

```
SHAPES
  inference_params is None (parallel prefill, our path):
    fir_state         None
  inference_params is not None (streaming):
    u[..., -6:]       ( 1 , 4096 , 6 )   bf16   (last K-1 samples)

  Return:
    z                 ( 1 , 4096 , 8192 ) bf16
    fir_state         None or (1, D, K-1)
```

**Purpose**: Capture the last `K-1` samples of the (post-gate) input so
that during streaming inference, the next call can pick up where this
one left off. Used by `step_fir` in subsequent decoding steps. **Our
HCS adapter only fires when `inference_params is None`**, so this state
machinery is dead code for our wire-up.

## The three dispatch branches in detail

### Branch 1: deprecated (`fir_fn != F.conv1d`)

```python
if fir_fn != torch.nn.functional.conv1d:
    if dim_last:
        u = u.permute(0, 2, 1)
    z = fir_fn(u)[:, :L]
```

The "escape hatch" for callers who want to provide their own conv. The
custom callable takes `u` (already permuted to channel-second), returns
`(B, L_or_more, channels)`, and gets sliced to `L`. **Not used by HCS,
HCM, or HCL** in current evo2 configs — those all use `fir_fn=F.conv1d`
which routes to the other branches.

**Why we can't use this hook**: it never sees `x2`. The gating happens
before the dispatch (`u = x1 * v`), and the deprecated branch doesn't
have access to the original `x2` for post-gating. Our HCS adapter
needs `x2`, so we patch one level up.

### Branch 2: HCM (`fir_length >= 128`)

```python
elif fir_length >= 128:
    with torch.autocast("cuda"):
        z = fftconv_func(
            u.to(torch.float32),
            weight[:, :, :L].to(torch.float32),
            bias,
            None,                # dropout_mask
            gelu=False,
            bidirectional=False,
            print_activations=self.print_activations,
            groups=groups,
            layer_idx=self.layer_idx,
        )
        z = z.to(u.dtype)
```

```
SHAPES (HCM cascade: gate=True, fir_length=128)
  u                  ( 1 , 4096 , 8192 )    bf16   (after step 2's u = x1*v)
  u.float()          ( 1 , 4096 , 8192 )    fp32
  weight             ( 4096 , 1 , 128 )     bf16
  weight[:, :, :L]   ( 4096 , 1 , 128 )     bf16   (slice is no-op when K <= L)
  weight.float()     ( 4096 , 1 , 128 )     fp32
  z = fftconv_func(...)
                     ( 1 , 4096 , 8192 )    fp32
  z = z.to(bf16)     ( 1 , 4096 , 8192 )    bf16
```

This is the **only call site for `fftconv_func`** in the whole vortex
codebase. See [fftconv_func](fftconv_func.md) for the full FFT-conv
walkthrough.

Phase 3 (HCM) replaces this branch — either by wiring CGCG@128 if
feasibility checks pass, or by fusing the elementwise epilogues around
cuFFT.

### Branch 3: HCS (`fir_length < 128`, the else)

The one walked through above in steps 3–5. Direct depthwise `F.conv1d`
with `groups=u.shape[1]` (always depthwise) and `padding=fir_length-1`
(causal left-pad). **This is the branch our HCS adapter replaces**
for the cascade case.

## The featurizer call (gate=False) — quick tour

The first call from `parallel_forward` (model.py:260) uses very
different shape conventions. The walkthrough is the same dispatch
logic but with different bindings:

```
Featurizer call signature
─────────────────────────
gate=False                          ← no split, no post-gate
fir_length=3 (short_filter_length)  ← falls into the HCS branch
groups=None                         ← featurizer doesn't share filters
column_split_hyena=True             ← ignored (gate=False)
dim_last=True                       ← input is (B, L, channels)
u: (B, L, 3*D) = (1, 8192, 12288)   bf16
weight: (3*D, 1, 3)                 bf16
bias: self.short_filter_bias (usually None)
```

The execution skips step 2 (no gate prep), and step 3 does:

```
1. Resolve L = u.shape[1] = 8192  (because dim_last=True)
2. (No gate prep — skipped)
3. Dispatch: fir_fn = F.conv1d, fir_length=3 < 128 → HCS branch
4. Permute (B, L, 3*D) → (B, 3*D, L) = (1, 12288, 8192)
5. F.conv1d with groups=12288 (depthwise), padding=2
   Output: (1, 12288, 8194)[..., :8192] = (1, 12288, 8192) fp32
6. Cast back to bf16, optional bias add
7. (No post-gate — skipped, gate=False)
8. inference_params is None → fir_state = None
9. Return ((1, 12288, 8192) bf16, None)
```

**Our HCS adapter does NOT intercept this call.** The predicate
`gate=True AND fir_length<128 AND groups>1` doesn't match (gate=False
fails the first clause). The featurizer continues to use F.conv1d.

## End-to-end shape pipeline (HCS cascade path)

```
u    (1, 12288, 8192) bf16
       │ gate prep: column_split + flip
       ▼
  x2 (1, 4096, 8192) bf16  ──────────────┐
  x1 (1, 4096, 8192) bf16                │
  v  (1, 4096, 8192) bf16                │
       │ x1 * v                          │
       ▼                                 │
  u  (1, 4096, 8192) bf16                │
       │ cast to fp32                    │
       ▼                                 │
       (1, 4096, 8192) fp32              │
       │                                 │
       │  weight (4096, 1, 7) fp32       │
       │     │                           │
       └──┬──┘                           │
          ▼ F.conv1d, groups=4096        │
         padding=6                       │
       (1, 4096, 8198) fp32              │
          │ [..., :8192] trim            │
          ▼                              │
  z    (1, 4096, 8192) fp32              │
          │ cast to bf16                 │
          ▼                              │
       (1, 4096, 8192) bf16              │
          │ + bias[None, :, None]         │
          ▼                              │
  z    (1, 4096, 8192) bf16              │
          │ × padding_mask (no-op if None)│
          ▼                              │
  z    (1, 4096, 8192) bf16              │
          │ × x2  ←──────────────────────┘
          ▼
  z    (1, 4096, 8192) bf16
          │
          ▼
  RETURN (z, fir_state=None)
```

## Per-step purpose, in one line each

| Step | Code | Purpose (HCS cascade) |
|---|---|---|
| 1 | `L = u.shape[...]` | Trust your own measurement of L |
| 2 (gate) | `x2, x1, v = u.split(...); u = x1 * v` | First gate of CGCG — separate stream-to-filter from stream-to-gate-by |
| 3 (HCS branch) | `z = F.conv1d(u.float(), w.float(), groups=D, padding=K-1)[..., :L]` | Depthwise causal conv in fp32 |
| 4 | `z = z.to(bf16); z += bias[None, :, None]` | Cast back + skip-gain add |
| 5 (post-gate) | `z = x2 * z` | Second gate of CGCG — close the gating sandwich |
| 6 | `fir_state = u[..., -K+1:] or None` | Capture streaming state (no-op for parallel prefill) |

## What changes for the wire-up

The HCS adapter (`hcs_dispatch` in `vortex_kernels/ops/hcs_interface.py`)
**replaces this whole method when the predicate matches**, doing all
six steps in one CGCG kernel call.

```
Predicate: gate=True AND fir_length<128 AND groups>1
  └─ matches: HCS cascade
  ├─ doesn't match: featurizer (gate=False)
  ├─ doesn't match: HCM cascade (fir_length=128)
  └─ doesn't match: HCL (different method — parallel_iir)
```

When the predicate matches, the adapter:

1. Replicates step 2's gate prep (splits `u` into `x2, x1, v`)
2. **De-expands** weight: `weight[::dg]` recovers `(g, 1, K)` from `(D, 1, K)`
   (the inverse of `repeat_interleave` that happened in
   `HyenaCascade.parallel_forward`)
3. Reshapes activations to CGCG's `(bs, l, g, dg)` layout
4. Calls `TwoPassChunkedGateConvGate.apply(x=v, B=x1, C=x2, h=h_grouped)`
   — this does steps 3 + 5 (and effectively step 4's bias if `bias` is
   passed through) in **one kernel**
5. Reshapes output back to `(B, D, L)` and applies bias-add + padding_mask + fir_state

When the predicate doesn't match, fall through to upstream `parallel_fir`
verbatim. **Zero behavior change for non-HCS-cascade calls.**

See `IMPLEMENTATION_PLAN.md` Phase 2.2 for the full adapter skeleton.

## Where `parallel_fir` is used

**Defined** in two places (one is your vendored copy):

| Path | Line | Role |
|---|---|---|
| `vortex/model/engine.py` | 131 | Upstream method on `HyenaInferenceEngine` |
| `vortex_kernels/reference/engine_ref.py` | 146 | Vendored verbatim copy |

**Called** from `HyenaCascade.parallel_forward` in `model.py`, **twice
per HCS/HCM layer**:

| Path | Line | Purpose | Predicate args |
|---|---|---|---|
| `vortex/model/model.py` | ~260 | Featurizer (input projection conv) | `gate=False, fir_length=3, groups=None, dim_last=True` |
| `vortex/model/model.py` | ~298 | Cascade (HCS or HCM inner conv) | `gate=True, fir_length=7 or 128, groups=hyena_filter_groups, dim_last=False` |

HCL layers also have a parallel_forward, but it calls `parallel_iir`
(not `parallel_fir`) at line ~324 for the long IIR conv — different
method, different walkthrough ([Target 3](parallel_iir.md)).

## Cross-references

- [fftconv_func](fftconv_func.md) — Target 1 — what gets called from
  the HCM branch (line ~210 of engine_ref)
- [parallel_iir](parallel_iir.md) — Target 3 — the sibling method that
  handles HCL via FFT-conv in `parallel_iir`'s body
- [HyenaCascade.parallel_forward](hyena_cascade_parallel_forward.md) —
  Target 5 — what calls `parallel_fir` (twice per layer)
- [gcg_fwd_ref_corrected](gcg_fwd_ref_corrected.md) — Target 7 — the
  CGCG reference math that proves our adapter replaces this method
  correctly

## Open questions / to-revisit later

- **`column_split_hyena` impact on the adapter**: the
  `column_split` helper produces tensors with potentially non-contiguous
  strides. Need to check whether CGCG requires contiguous input (likely
  yes — it asserts `is_contiguous()` in `fwd.py`). If non-contiguous,
  the adapter must `.contiguous()` before reshape.
- **HCM cascade routing**: HCM hits this same method with
  `fir_length=128`, taking the HCM branch (call to `fftconv_func`).
  Phase 3 will decide whether to ALSO intercept HCM via this method
  (predicate `gate=True AND fir_length>=128`) or to patch
  `engine.fftconv_func` directly. Cleaner to patch the FFT function
  if HCM goes Option B (fusion), simpler to patch here if HCM goes
  Option A (CGCG).
- **`fir_fn` parameter use**: would a non-deprecated future PR want to
  use the `fir_fn` slot as a clean wire-up hook? Probably yes — if
  we propose adding a `use_triton_hcs` flag upstream, threading our
  kernel through `fir_fn` would be the least-invasive change to
  `engine.py`. Worth raising in the PR discussion.

---

*Last updated: 2026-05-12 (Phase 1 read-through, target 2/12)*
