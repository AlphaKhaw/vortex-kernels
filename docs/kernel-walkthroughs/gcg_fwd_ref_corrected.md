# `gcg_fwd_ref_corrected` — the AHA: CGCG ≡ HCS

> **Tour target 7 of 12** · [index](README.md) · prev: [compute_filter_pure](compute_filter_pure.md) · next: [TwoPassChunkedGateConvGate](two_pass_chunked_gate_conv_gate.md)

## Source

| | |
|---|---|
| Upstream | `vortex/ops/hyena_se/ref_fwd.py:80-127` |
| Mirrored | `vortex_kernels/ops/hyena_se/ref_fwd.py:80-127` (with `hyena_ops.utils` import fix) |
| Used as | Pure-PyTorch oracle for testing the CGCG Triton kernel |

## Overall purpose — THE AHA

**`gcg_fwd_ref_corrected` is the pure-PyTorch reference implementation
of the CGCG kernel's math.** It computes:

```
y = C * conv(B * x, h)
```

That's it. **3 operations: elementwise multiply, depthwise conv, elementwise
multiply.** No Triton, no chunks, no Toeplitz tricks. Just torch ops.

### Why this is the AHA

Look at the math: `y = C * conv(B * x, h)`. Now rename:
- `x` → `v`
- `B` → `x1`
- `C` → `x2`
- `h` → `weight`

You get: **`y = x2 * conv(x1 * v, weight)`** — which is **exactly** the
HCS cascade gated math from
[`parallel_fir`](parallel_fir.md#step-2-gate-prep-gatetrue-only) gate=True
branch:

```
parallel_fir gate=True flow:
  x2, x1, v = u.split(...)
  u = x1 * v                                    ← B*x in CGCG
  z = F.conv1d(u, weight, groups=D, padding=K-1)  ← conv(B*x, h) in CGCG
  z = z + bias[None, :, None]                   ← skip-add (separate)
  z = x2 * z                                    ← C*y in CGCG
```

The two implementations are computing **identical math** under
different variable names. They both implement the Hyena CGCG
(Conv-Gate-Conv-Gate) operator.

### What this means

```
                  HCS gated cascade math
                  ──────────────────────
                  y = x2 * conv(x1*v, weight)
                          │
                          │ rename
                          ▼
                  y = C * conv(B*x, h)
                          │
                          ├─► gcg_fwd_ref_corrected (this function)
                          │   — pure PyTorch reference, ~30 lines
                          │
                          ├─► TwoPassChunkedGateConvGate.forward
                          │   — autograd wrapper, calls the Triton kernel
                          │
                          └─► two_pass_fwd_grouped + Triton @jit kernels
                              — the actual chunked CGCG implementation
                              with Toeplitz block multiplications
```

vortex ships **all three layers** of this stack in
`vortex/ops/hyena_se/`. They were vendored from the "savanna" project,
the package names were partly renamed but never fully fixed (the
broken `hyena_ops.*` imports we fixed), and `engine.py` was never
updated to wire CGCG up as an HCS dispatch option.

**Your HCS adapter (Phase 2) is the missing wire-up.**

## Realistic input shapes (evo2_7b HCS cascade)

CGCG's inputs are in the `(bs, l, g, dg)` layout — different from
engine.py's `(B, D, L)`. The conversion is what your adapter has to
do.

```
INPUTS to gcg_fwd_ref_corrected (CGCG layout)
═══════════════════════════════════════════════

x     ( 1 , 8192 , 256 , 16 )   fp32   ← activations to be filtered (= v in HCS)
      └─┬─┘└──┬──┘└─┬─┘└──┬─┘
        bs    l    g=256 dg=16
        batch seq groups channels-per-group

B     ( 1 , 8192 , 256 , 16 )   fp32   ← pre-gate (= x1 in HCS)
C     ( 1 , 8192 , 256 , 16 )   fp32   ← post-gate (= x2 in HCS)

h     ( 256 , 1 , 7 )           fp32   ← filter, PER-GROUP not per-channel
      └─┬─┘└┬┘└┬┘                       (this is self.h in HCS, unexpanded)
       g=256 1 hl=7

Total channels: g * dg = 256 * 16 = 4096 = D (hidden_size)
```

The clever thing about `(bs, l, g, dg)`:
- It exposes the **filter-group structure** explicitly in the tensor
  layout — channels are nested inside groups
- All `dg` channels within a group share the same filter `h[g]`
- This matches the Triton kernel's tile-launching scheme (one program
  per `(bs, l_tile, g_tile, dg_tile)`)

## Inputs explained

### `x` — input activations
Shape `(bs, l, g, dg)` fp32. In HCS terms: this is `v`, the gating
"value" stream. Why fp32? CGCG's chunked Toeplitz math needs the
precision; the Triton kernel forces fp32 inside even when input is
fp16/bf16.

### `B` — pre-gate
Shape `(bs, l, g, dg)` fp32. In HCS terms: this is `x1`, the
multiplicatively-applied pre-gate. The product `Bx = B * x` is what
gets convolved.

### `C` — post-gate
Shape `(bs, l, g, dg)` fp32. In HCS terms: this is `x2`, the
multiplicatively-applied post-gate. The conv output is multiplied
by `C` at the very end.

### `h` — filter
Shape `(g, 1, hl)` fp32 — **per-group, not per-channel**. The `1` is a
PyTorch conv1d convention (same as you've seen for `(D, 1, K)` in
engine.py). The key fact: there are only `g=256` distinct filters, NOT
`D=4096`. Every `dg=16` consecutive channels share one filter.

This is the format CGCG natively expects. **Engine.py's `parallel_fir`
gets the filter in the *expanded* `(D, 1, hl)` form because
`HyenaCascade.parallel_forward` does `repeat_interleave(D//g, 0)` first.**
Your adapter has to undo this.

### `use_causal_conv` — fast path for short filters
When `hl <= 3` AND `causal_conv1d_fn` is installed, use Mamba's
causal_conv1d kernel instead of `F.conv1d`. For HCS at `hl=7`, this
path is disabled (line 103: `if hl > 3: use_causal_conv = False`).

### `interleave` — filter expansion mode
Two ways to expand `(g, 1, hl) → (D, 1, hl)`:

- `interleave=True` (default, line 113): `h.repeat_interleave(dg, dim=0)`
  — block-grouped. Channels `0..dg-1` use filter 0, channels `dg..2*dg-1`
  use filter 1, etc.
- `interleave=False` (line 115): `h.repeat(dg, 1, 1)` — striped. Channel
  `0` uses filter 0, channel `1` uses filter 1, ..., channel `g` uses
  filter 0 again.

Vortex's `HyenaCascade.parallel_forward` uses `repeat_interleave`
(block-grouped), so `interleave=True` is the correct choice for HCS.

## Step-by-step with shape tracking

### Step 1: Read shapes

```python
bs, l, g, dg = x.shape
hl = h.shape[-1]
if hl > 3:
    use_causal_conv = False
d = g * dg
```

```
SHAPES
  x        ( 1, 8192, 256, 16 ) fp32
  bs, l, g, dg = 1, 8192, 256, 16
  hl = 7
  d = g * dg = 256 * 16 = 4096
  use_causal_conv = False (hl=7 > 3)
```

**Purpose**: Compute the flattened channel count `d = g * dg` (which
equals D=4096) since the underlying F.conv1d works on `(B, d, L)` not
`(B, L, g, dg)`.

### Step 2: First gate — `Bx = B * x`

```python
Bx = B * x
Bx_l_last = Bx.permute(0, 2, 3, 1)  # b, g, dg, l
```

```
SHAPES
  B         ( 1, 8192, 256, 16 ) fp32
  x         ( 1, 8192, 256, 16 ) fp32
  Bx = B*x  ( 1, 8192, 256, 16 ) fp32   ← elementwise, same shape

  Bx_l_last = Bx.permute(0, 2, 3, 1)
              ( 1, 256, 16, 8192 ) fp32  ← (bs, g, dg, l)
                └─┬─┘
                  permute moves l to the end so conv1d can use it
```

**Purpose**: The first gate of CGCG. Same operation as `u = x1 * v` in
`parallel_fir`'s gate-prep. The permute reshuffles to channel-second
layout (after the upcoming flatten).

### Step 3: Flatten + expand filter

```python
Bx_l_last_flattened = Bx_l_last.reshape(bs, -1, l)  # b, d, l

if interleave:
    h_grouped = h.repeat_interleave(dg, dim=0)  # d, 1, hl
else:
    h_grouped = h.repeat(dg, 1, 1)              # d, 1, hl
```

```
SHAPES
  Bx_l_last_flattened  ( 1, 4096, 8192 ) fp32   ← (bs, g*dg=d, l)
                       collapsed g and dg into one channel axis

  h                    ( 256, 1, 7 ) fp32
  h.repeat_interleave(16, dim=0)
                       ( 4096, 1, 7 ) fp32      ← per-channel filter
                       block-grouped:
                         filters 0..15 (16 copies of h[0])
                         filters 16..31 (16 copies of h[1])
                         ...
                         filters 4080..4095 (16 copies of h[255])
```

**Purpose**: F.conv1d wants `(B, channels, L)` for input and
`(channels, 1, K)` for depthwise filter. Step 3 collapses the
`(g, dg)` structure into a flat channel axis and expands the filter
to match.

This expansion is **exactly the `repeat_interleave(D//g, 0)`** that
`HyenaCascade.parallel_forward` does at line 262. Your HCS adapter's
de-expand step (`weight[::dg]`) is the inverse of this expansion.

### Step 4: Depthwise conv

```python
y_l_last_flattened = F.conv1d(
    Bx_l_last_flattened, h_grouped, groups=d, stride=1, padding=hl - 1,
)[..., : -hl + 1]
```

```
SHAPES
  Input:
    Bx_l_last_flattened ( 1, 4096, 8192 )      fp32
    h_grouped           ( 4096, 1, 7 )         fp32
    groups=4096                                ← depthwise
    padding=6                                  ← causal left-pad
  F.conv1d output:
    raw                 ( 1, 4096, 8198 )      fp32   ← 8192 + 2*padding/2 (but actually padding adds 2*6 then ends up...)

  Wait, more carefully:
    F.conv1d adds `padding` zeros on EACH SIDE of the input
    Output length = L + 2*padding - K + 1 = 8192 + 12 - 7 + 1 = 8198
  After [..., :-hl+1] slice (drops the right hl-1 = 6 positions):
    sliced              ( 1, 4096, 8192 ) fp32   ← back to L
```

**Purpose**: Depthwise conv (same math as `parallel_fir`'s HCS branch).
Note the slightly different slicing convention here vs `parallel_fir`:
- `parallel_fir`: `padding=fir_length - 1`, then `[..., :L]`
- `gcg_fwd_ref_corrected`: `padding=hl - 1`, then `[..., :-hl + 1]`

Both produce a causal conv but trim from different ends. **Slightly
different left/right padding convention** — worth checking whether
this introduces a 1-step phase shift versus `parallel_fir`. (Likely
both give equivalent causal output; F.conv1d with `padding=K-1` and
either slice convention produces a length-L causal conv with the
filter aligned so output[t] depends on input[t-K+1..t].)

### Step 5: Unflatten + permute back

```python
y_l_last = y_l_last_flattened.reshape(bs, g, dg, l)
y = y_l_last.permute(0, 3, 1, 2)
```

```
SHAPES
  y_l_last_flattened  ( 1, 4096, 8192 )      fp32
  y_l_last (reshape)  ( 1, 256, 16, 8192 )   fp32   ← restore (g, dg) structure
  y = .permute(0,3,1,2)
                       ( 1, 8192, 256, 16 )   fp32   ← (bs, l, g, dg)
```

**Purpose**: Convert back to CGCG's native `(bs, l, g, dg)` layout for
the final gate-multiply with C.

### Step 6: Second gate — `return C * y`

```python
return C * y
```

```
SHAPES
  C   ( 1, 8192, 256, 16 ) fp32
  y   ( 1, 8192, 256, 16 ) fp32
  C*y ( 1, 8192, 256, 16 ) fp32   ← elementwise, same shape
```

**Purpose**: The second gate of CGCG. Same operation as `z = x2 * z` in
`parallel_fir`'s post-gate step.

## End-to-end shape pipeline

```
x  (1, 8192, 256, 16) fp32     ─┐
B  (1, 8192, 256, 16) fp32     ─┤
                                │ B * x (elementwise, first gate)
                                ▼
Bx (1, 8192, 256, 16) fp32
                                │ permute(0, 2, 3, 1)
                                ▼
   (1, 256, 16, 8192) fp32
                                │ reshape(bs, -1, l)
                                ▼
   (1, 4096, 8192) fp32
                                │  depthwise F.conv1d, groups=4096
   h (256, 1, 7) fp32           │  with h_grouped (per-channel expand)
        │                       │
        │ repeat_interleave(16) │
        ▼                       │
   h_grouped (4096, 1, 7) ──────┤
                                ▼
   (1, 4096, 8198) fp32         ← post-conv with padding
                                │ [..., :-6] trim
                                ▼
   (1, 4096, 8192) fp32
                                │ reshape(bs, g, dg, l)
                                ▼
   (1, 256, 16, 8192) fp32
                                │ permute(0, 3, 1, 2)
                                ▼
y  (1, 8192, 256, 16) fp32     ─┐
C  (1, 8192, 256, 16) fp32     ─┤
                                │ C * y (elementwise, second gate)
                                ▼
   (1, 8192, 256, 16) fp32
                                │
                                ▼
RETURN (1, 8192, 256, 16) fp32
```

## The renaming you need to internalize

```
┌─────────────────────────────────────────────────────┐
│         CGCG name  ←→  HCS name (engine.py)         │
├─────────────────────────────────────────────────────┤
│    x          ←→   v       (gating value)           │
│    B          ←→   x1      (pre-gate)               │
│    C          ←→   x2      (post-gate)              │
│    h          ←→   weight  (cascade filter)         │
│                                                     │
│  Layout:                                            │
│    (bs, l, g, dg)  ←→  (B, D, L) with D = g * dg    │
│                                                     │
│  Filter shape:                                      │
│    (g, 1, hl)  ←→  (g, 1, hl) before repeat         │
│                    (D, 1, hl) after repeat          │
└─────────────────────────────────────────────────────┘
```

If you commit this table to memory, the adapter is ~mechanical:

```python
# adapter sketch (conceptual)
def hcs_dispatch(self, fir_fn, u, weight, bias, L, dims, groups, ...):
    # 1. Split u → x2, x1, v (the gate prep)
    # 2. Recover per-group filter:    h = weight[::D//g]   (inverse of repeat_interleave)
    # 3. Reshape to (bs, l, g, dg):   x = v.permute().reshape(...)
    #                                  B = x1.permute().reshape(...)
    #                                  C = x2.permute().reshape(...)
    # 4. Call CGCG:                    y = TwoPassChunkedGateConvGate.apply(x, B, C, h, ...)
    # 5. Reshape back to (B, D, L):    return y.reshape(...).permute(...)
```

The mapping table is the bridge. Once you internalize it, the
adapter is just plumbing.

## Per-step purpose, in one line each

| Step | Code | Purpose |
|---|---|---|
| 1 | `bs, l, g, dg = x.shape; d = g * dg` | Read shapes; flatten d for conv1d |
| 2 | `Bx = B * x; Bx.permute(0, 2, 3, 1)` | First gate; reshape to (bs, g, dg, l) |
| 3 | `Bx.reshape(bs, -1, l); h.repeat_interleave(dg, 0)` | Flatten channels; expand filter per-channel |
| 4 | `F.conv1d(..., groups=d, padding=hl-1)[..., :-hl+1]` | Causal depthwise conv |
| 5 | `reshape(bs, g, dg, l); permute(0, 3, 1, 2)` | Restore (bs, l, g, dg) layout |
| 6 | `return C * y` | Second gate (post-gate multiply) |

## What you do with this function

`gcg_fwd_ref_corrected` is **your oracle** for testing. The HCS
correctness test (`tests/test_hcs_correctness.py`) does:

```python
y_ref = gcg_fwd_ref_corrected(x, B, C, h)             # pure PyTorch on CPU/GPU
y_kernel = TwoPassChunkedGateConvGate.apply(x, B, C, h, ...)  # Triton kernel
assert (y_ref - y_kernel).abs().max() < 1e-2
```

Then the e2e correctness test does an additional layer of validation:

```python
y_engine = parallel_fir_via_HCS_adapter(z_pre, weight, ...)   # engine path via adapter
# convert (B, D, L) ↔ (bs, l, g, dg) and check
y_ref_via_engine_layout = gcg_fwd_ref_corrected(x, B, C, h)
assert numerically_close(y_engine, reshape(y_ref_via_engine_layout))
```

Two oracles, two tests, two confidence levels. The first verifies that
**CGCG kernel math is correct**; the second verifies that **the adapter
plumbing is correct**.

## Cross-references

- [parallel_fir](parallel_fir.md) — Target 2 — the engine method
  whose gate=True+HCS branch implements the same math under different
  variable names
- [TwoPassChunkedGateConvGate](two_pass_chunked_gate_conv_gate.md) —
  Target 8 — the autograd wrapper around the Triton kernel; computes
  the same `C * conv(B*x, h)` math but chunked + with backward
- [two_pass_fwd_grouped](two_pass_fwd_grouped.md) — Target 9 — the
  kernel launcher with the shape contracts CGCG expects

## Open questions / to-revisit later

- **The `[..., :-hl + 1]` vs `[..., :L]` slicing**: `parallel_fir` uses
  `[..., :L]` after `padding=fir_length-1`; this function uses
  `[..., :-hl + 1]`. Both should give equivalent causal output but
  worth a numerical sanity-check during adapter development —
  off-by-one in trim direction is a common bug.
- **`use_causal_conv` fast path for `hl <= 3`**: enabled only when
  Mamba's `causal_conv1d_fn` is installed. Not relevant for HCS
  (`hl=7`) but worth knowing exists. If you wired CGCG for the HCS
  featurizer call (`hl=3, gate=False`), this would fire.
- **Interleave vs stripe filter expansion**: `gcg_fwd_ref_corrected`
  defaults to `interleave=True` (matching `HyenaCascade.parallel_forward`).
  Confirm in Phase 2.0 smoke test that your adapter passes the right
  expansion mode (or omits it since `interleave=True` is the default).

---

*Last updated: 2026-05-12 (Phase 1 read-through, target 7/12 — the AHA target)*
