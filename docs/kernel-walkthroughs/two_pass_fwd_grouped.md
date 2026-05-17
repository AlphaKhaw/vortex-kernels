# `two_pass_fwd_grouped` — the CGCG kernel launcher

> **Tour target 9 of 12** · [index](README.md) · prev: [TwoPassChunkedGateConvGate](two_pass_chunked_gate_conv_gate.md) · next: [upstream hcs_interface.py](upstream_hcs_interface.md)
>
> **Depth: light** — data contract only.

## Source

| | |
|---|---|
| Upstream | `vortex/ops/hyena_se/fwd.py:17-310` (legacy) + `313-533` (refactor) |
| Mirrored | `vortex_kernels/ops/hyena_se/fwd.py` (no edits needed) |
| Called by | `TwoPassChunkedGateConvGate.forward` (line 87 in interface.py) |

## Overall purpose

**The Python-side launcher for the CGCG Triton kernels.** Two functions
live in this file:

- `two_pass_fwd_grouped` (line 17, legacy v1/v2 path)
- `two_pass_fwd_grouped_refactor` (line 313, the cleaner refactor; what
  your adapter ultimately calls via `use_refactor_path=True`)

Both do the same shape-validation + grid-computation + Triton-kernel-launch
dance. The differences are minor (different return tuple, slightly
different internal kernel selection). Your wire-up uses the refactor;
treat the legacy as reference.

## What it does at a glance

```
┌────────────────────────────────────────────────────────────┐
│  two_pass_fwd_grouped_refactor(x, B, C, h, ...)             │
│                                                            │
│  1. Validate input shapes (assertions)                     │
│  2. Reshape (bs, l, g, dg) → (bs, l, d=g*dg)               │
│  3. Pick kernel: autotune vs manual config                 │
│  4. Compute launch grid (1D or persistent)                 │
│  5. Allocate output + intermediate buffers (y, y2, bx)     │
│  6. Launch the Triton @jit kernel                          │
│  7. Reshape y back to (bs, l, g, dg)                       │
│  8. Return (bx, y2, y) or (y, T, T_hat, y2, bx_lag)        │
└────────────────────────────────────────────────────────────┘
```

This is all glue code. The interesting math happens **inside** the
Triton @jit kernel (`_two_pass_fwd_refactor_kernel` in
`fwd_kernels.py`, ~800 lines of Triton not on this tour).

## Shape contract — the bit that matters for your adapter

```
Inputs to two_pass_fwd_grouped_refactor:
  x  ( bs , l  , g , dg )    fp32   bs * l * g * dg = bs * l * d
  B  ( bs , l  , g , dg )    fp32   same as x
  C  ( bs , l  , g , dg )    fp32   same as x
  h  ( g  , 1  , hl )         fp32   per-group filter, NOT per-channel

Constraints (asserted in the launcher):
  - x.shape == B.shape == C.shape    ← same shape, all three streams
  - h.shape[0] == g                  ← one filter per group
  - h.shape[1] == 1                  ← single in-channel-per-group (depthwise)
  - dg >= 16                         ← required for tensor-core use
  - x, B, C, h ALL contiguous        ← Triton needs contiguous layouts
  - dg % BLOCK_D == 0                ← BLOCK_D divides dg (default 32 divides 32)
  - hl <= CHUNK_SIZE                 ← filter must fit in one chunk (128 default)
  - if hl < 128 and seqlen > 1024: CHUNK_SIZE >= 128

Output:
  y  ( bs , l  , g , dg )    fp32   same shape as x
```

For evo2_7b HCS:
- `bs=1, l=8192, g=256, dg=16, hl=7`
- `dg=16` exactly meets the `>= 16` floor
- `hl=7` well under `CHUNK_SIZE=128`
- All constraints satisfied ✓

For evo2_7b HCM at the boundary:
- `bs=1, l=8192, g=128, dg=32, hl=128`
- `dg=32 >= 16` ✓
- **`hl=128` == `CHUNK_SIZE=128`** ← right at the boundary
- Phase 3.0 needs to verify: does `hl <= CHUNK_SIZE` pass for `hl == CHUNK_SIZE`?
  (Yes per the `<=` constraint, but the docstring at line 144 says
  `assert CHUNK_SIZE >= 128 for hl < 128 and seqlen > 1024` — different
  edge case. Verify empirically.)

## Where do you change anything in this file?

**Nowhere.**

Your wire-up doesn't modify `two_pass_fwd_grouped` or `two_pass_fwd_grouped_refactor`. The flow is:

```
hcs_dispatch (your adapter, in vortex_kernels/ops/hcs_interface.py)
        │
        │ calls
        ▼
TwoPassChunkedGateConvGate.apply(...)  (interface.py, target 8)
        │
        │ calls (with use_refactor_path=True)
        ▼
two_pass_fwd_grouped_refactor(...)  ← THIS FILE — unchanged
        │
        │ launches
        ▼
_two_pass_fwd_refactor_kernel (fwd_kernels.py, ~800 lines of Triton)
        │
        │ runs on GPU
        ▼
GPU computes y = C * conv(B*x, h)
```

The launcher's job is plumbing — kernel selection, grid sizing, buffer
allocation. None of it needs your attention.

## The kernel-selection logic

For your reference, the two relevant paths (lines 408–423 of the
refactor function):

```python
if CHUNK_SIZE is not None and BLOCK_D is not None and autotune:
    print("WARNING: ...")
    autotune = False                # explicit configs disable autotune

if autotune:
    kernel = _two_pass_fwd_refactor_autotuned
    # ← let Triton find the best CHUNK_SIZE / BLOCK_D / num_warps via @triton.autotune
else:
    assert all([CHUNK_SIZE, BLOCK_D]), "Must specify all of CHUNK_SIZE, BLOCK_D, CHUNK_TILES_PER_PROGRAM"
    kernel = _two_pass_fwd_refactor_kernel
    # ← use the hand-configured kernel with the values you passed
```

The autotuned variant runs once-per-shape to find the best config, then
caches. The hand-configured variant uses your values verbatim. Your
wire-up passes `autotune=False` with the default config dataclass —
predictable behavior for first GPU smoke test.

## Buffer allocations (the cost you pay per call)

```python
if y is None:
    y = torch.zeros_like(x)           # output buffer
if return_y2:
    y2 = torch.empty_like(x)          # backward intermediate
else:
    y2 = None
if return_bx:
    bx = torch.zeros_like(x)          # B*x intermediate (for backward reuse)
else:
    bx = None
```

`y, y2, bx` are each `(bs, l, g, dg) fp32 = bs * l * D * 4` bytes.

For evo2_7b HCS at L=8192:
- D=4096, fp32, L=8192 → each buffer is ~128 MiB
- Forward allocates 3 (y, y2, bx) → ~384 MiB per layer per call

For inference, `y2` and `bx` are **wasted** (backward never fires). Out
of your control here — the launcher always allocates them when forward
will be followed by a possible backward. The `inference_only` skip is
a Phase 4+ polish for upstream PR follow-up.

## Cross-references

- [TwoPassChunkedGateConvGate](two_pass_chunked_gate_conv_gate.md) —
  Target 8 — the autograd Function that calls this launcher
- [gcg_fwd_ref_corrected](gcg_fwd_ref_corrected.md) — Target 7 — the
  pure-PyTorch reference whose math this launcher's underlying kernel
  computes

## What you can ignore

- The 100+ commented-out block at the bottom (lines ~536-705) — debug
  scaffolding from kernel development
- The legacy `two_pass_fwd_grouped` (lines 17-310) — older path with
  v1/v2 split; your wire-up uses the refactor
- The TMA path — referenced but not wired through this file
- The `kernel.cache` lookup logic in autotune mode (lines 293–310) —
  internal Triton machinery for getting the chosen config back

## TL;DR — what to remember

1. **You don't touch this file.** Upstream code that just works once
   the broken imports are fixed (which you've already done).
2. **Shape contract**: `(bs, l, g, dg)` for activations, `(g, 1, hl)`
   for filter, fp32, contiguous, `dg >= 16`, `hl <= CHUNK_SIZE`.
3. **Buffer cost**: ~384 MiB allocated per layer per call at HCS
   evo2_7b L=8192 (`y + y2 + bx`). Wasted ~256 MiB on `y2 + bx` for
   inference. Polish later.

---

*Last updated: 2026-05-12 (Phase 1 read-through, target 9/12, light)*
