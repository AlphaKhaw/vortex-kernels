# `TwoPassChunkedGateConvGate` — autograd wrapper for the CGCG kernel

> **Tour target 8 of 12** · [index](README.md) · prev: [gcg_fwd_ref_corrected](gcg_fwd_ref_corrected.md) · next: [two_pass_fwd_grouped](two_pass_fwd_grouped.md)

## Source

| | |
|---|---|
| Upstream | `vortex/ops/hyena_se/interface.py:47-167` |
| Mirrored | `vortex_kernels/ops/hyena_se/interface.py:47-167` (with `hyena_ops.*` import fixes) |
| Used by | Phase 2 HCS adapter (`hcs_dispatch`); also exposed via `two_pass_chunked_gate_conv_gate` wrapper for direct callers |

## Overall purpose

**Orchestrate the CGCG Triton kernel call** — handle config defaults,
contiguity, kernel selection (refactor vs original), and the autograd
save/restore for backward.

This class is `torch.autograd.Function` subclass. It does **two
things**:

1. **`forward()`** — accept `(x, B, C, h)` plus optional config; pick the
   right Triton launcher (refactor or v2); ensure inputs are contiguous;
   call the kernel; save intermediates for backward; return `y`.
2. **`backward()`** — read the saved intermediates; call the backward
   Triton kernel; return gradients for `(x, B, C, h)`.

For inference (your wire-up), only `forward` matters. Backward exists
because the same code was used for training in the savanna project,
but our wire-up never invokes it.

## Where this sits in the stack

```
┌───────────────────────────────────────────────────────────────┐
│  hcs_dispatch  ─────────►  TwoPassChunkedGateConvGate.apply   │
│  (Phase 2 adapter)         (this class — the autograd entry)  │
└───────────────────────────┬───────────────────────────────────┘
                            │ inside .forward():
                            │  - default configs if not passed
                            │  - contiguity check
                            │  - pick kernel: refactor vs v2 vs TMA
                            ▼
                  two_pass_fwd_grouped_refactor (default)
                  OR two_pass_fwd_grouped (legacy)
                  (see target 9 — the launcher)
                            │
                            ▼  launches the actual Triton @jit kernel
                  _two_pass_fwd_refactor_kernel (or v1/v2)
                  (in fwd_kernels.py — not in the tour;
                   ~800 lines of Triton)
                            │
                            ▼
                  GPU computes:
                    y = C * conv(B*x, h)   chunk-by-chunk via Toeplitz
                            │
                            ▼
                  Returns (bx_lag, y2, y) to interface.py
                            │
                            ▼
                  forward() unpacks, saves for backward, returns y
```

## Realistic invocation (what your HCS adapter does)

```python
from vortex_kernels.ops.hyena_se.interface import (
    TwoPassChunkedGateConvGate,
    DefaultTwoPassChunkedGateConvGateFwdConfig,
    DefaultTwoPassChunkedGateConvGateBwdConfig,
)

# Adapter call (inside hcs_dispatch):
y = TwoPassChunkedGateConvGate.apply(
    x,                                           # (bs, l, g, dg) fp32
    B,                                           # (bs, l, g, dg) fp32
    C,                                           # (bs, l, g, dg) fp32
    h,                                           # (g, 1, hl) fp32
    "default",                                   # schedule
    True,                                        # use_refactor_path
    False,                                       # autotune
    DefaultTwoPassChunkedGateConvGateFwdConfig(),  # fwd_kernel_cfg
    DefaultTwoPassChunkedGateConvGateBwdConfig(),  # bwd_kernel_cfg
)
```

The call uses `.apply` (not direct construction) because that's how
`torch.autograd.Function` subclasses are invoked — PyTorch registers
the call with the autograd engine so `backward()` gets wired up.

## Inputs explained — the call signature

```python
def forward(
    ctx,
    x: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    h: torch.Tensor,
    schedule: str = "default",
    use_refactor_path: bool = True,
    autotune: bool = False,
    fwd_kernel_cfg: FwdKernelConfig = None,
    bwd_kernel_cfg: BwdKernelConfig = None,
):
```

### `ctx` — autograd's save/restore context

`torch.autograd.Function.forward` is a *static method*; `ctx` is the
first arg, used to stash tensors and metadata for the matching
`backward()` call. We use it at line 114: `ctx.save_for_backward(...)`.

### `x, B, C` — the three feature streams

Per [target 7](gcg_fwd_ref_corrected.md), the renamed HCS streams:
`x ↔ v`, `B ↔ x1`, `C ↔ x2`. Each shape `(bs, l, g, dg)` fp32.

### `h` — the filter
Shape `(g, 1, hl)` fp32, per-group. **Not pre-expanded** to `(D, 1, hl)` —
the kernel handles the per-group structure internally.

### `schedule` — kernel launch schedule

One of:
- `"default"` (the only supported option for v2 / refactor) — launches a
  1D grid where each program processes one tile
- `"persistent"` — would launch `min(NUM_SM, total_tiles)` programs for
  CTA reuse, but `NotImplementedError` for v2

Use `"default"`. The arg exists for future schedule options.

### `use_refactor_path` — kernel version selector

- `True` (default, recommended): use `two_pass_fwd_grouped_refactor` —
  a newer kernel that returns `(bx_lag, y2, y)` directly
- `False`: use the legacy `two_pass_fwd_grouped` — returns
  `(y, T, T_hat, y2, bx_lag)` with optional Toeplitz outputs

The refactor is cleaner and is what's actively maintained. Use it.

### `autotune` — let Triton pick configs

When `True`, Triton runs a calibration on first call to pick
`CHUNK_SIZE`, `BLOCK_D`, `num_warps`, etc. Slow first invocation; later
invocations cache the chosen config.

Recommend `autotune=False` with hand-picked configs (the
`Default...Config` classes) for predictability during testing. Autotune
is for production deployment after the kernel is verified.

### `fwd_kernel_cfg` / `bwd_kernel_cfg` — kernel hyperparameters

Two dataclasses. `FwdKernelConfig` carries chunk/block sizes; `BwdKernelConfig`
adds backward-specific options. Defaults at lines 27–44:

```python
@dataclass(eq=False)
class DefaultTwoPassChunkedGateConvGateFwdConfig(FwdKernelConfig):
    schedule: str = "default"
    autotune: bool = True
    CHUNK_SIZE: int = 128       # ← the L-axis chunk size
    BLOCK_D: int = 32           # ← the D-axis tile size
    THREADBLOCK_SWIZZLE: str = "row"

@dataclass(eq=False)
class DefaultTwoPassChunkedGateConvGateBwdConfig(BwdKernelConfig):
    ...
    CHUNK_SIZE: int = 128
    BLOCK_D: int = 32
    THREADBLOCK_SWIZZLE: str = "row"
    LOAD_TOEPLITZ: bool = False   # ← reuse forward's Toeplitz for backward?
    LOAD_BX_LAG: bool = False     # ← reuse forward's Bx_lag for backward?
```

**`CHUNK_SIZE = 128`** is the L-axis tile size. This is the **filter
length ceiling** — CGCG requires `filter_len ≤ CHUNK_SIZE`. For HCS at
`hl=7`, we're well under. For HCM at `hl=128`, we're AT the boundary —
Phase 3.0 investigation will check whether this works or needs a
custom config with `CHUNK_SIZE=256`.

**`BLOCK_D = 32`** is the D-axis tile size. The Triton kernel requires
`dg >= 16` to use tensor cores. With `dg=16` (HCS) the conv tiles fit
exactly; the kernel can use `tl.dot` for the chunked Toeplitz multiply.

## Step-by-step walkthrough of `forward()`

### Step 1: Default configs if not passed

```python
if fwd_kernel_cfg is None:
    fwd_kernel_cfg = DefaultTwoPassChunkedGateConvGateFwdConfig()
if bwd_kernel_cfg is None:
    bwd_kernel_cfg = DefaultTwoPassChunkedGateConvGateBwdConfig()
```

**Purpose**: Materialize default config dataclasses if the caller
didn't pass them. This means you can call `.apply(x, B, C, h)` without
worrying about configs and it'll use sensible defaults.

### Step 2: Read backward-related flags from `bwd_kernel_cfg`

```python
return_toeplitz = bwd_kernel_cfg.LOAD_TOEPLITZ
return_bx_lag = bwd_kernel_cfg.LOAD_BX_LAG
schedule = schedule if autotune else fwd_kernel_cfg.schedule
```

**Purpose**: The forward kernel can optionally return the *Toeplitz
matrices* and *lagged Bx* tensors to save the backward kernel from
recomputing them. The decision is made now (in forward) based on what
the matching backward kernel will need. Default for both: `False`
(backward recomputes — simpler but slower).

For inference (our wire-up), neither flag matters — backward never
fires.

### Step 3: Constraint check for Toeplitz reuse

```python
if return_toeplitz:
    bwd_kernel_cfg.CHUNK_SIZE = fwd_kernel_cfg.CHUNK_SIZE
```

**Purpose**: If forward returns Toeplitz, backward must use the **same
CHUNK_SIZE** (the Toeplitz matrices were sized for that chunk; backward
can't change). This line propagates the constraint.

Irrelevant for inference. Default `return_toeplitz=False`.

### Step 4: Kernel selection

```python
if not autotune and fwd_kernel_cfg.USE_TMA:
    raise NotImplementedError("Skip TMA for now")
else:
    kernel = two_pass_fwd_grouped_refactor if use_refactor_path else two_pass_fwd_grouped
```

**Purpose**: Pick which kernel launcher to use:

- TMA path (`fwd_kernel_cfg.USE_TMA = True`) — would route to a Hopper-
  specific kernel. Currently `NotImplementedError`. The TMA code exists in
  `_fwd_tma.py` but isn't wired here. Future optimization target for
  H100/H200.
- Refactor path (default, `use_refactor_path=True`) — `two_pass_fwd_grouped_refactor`
  from `fwd.py`. The newer, cleaner kernel launcher. **This is what
  your wire-up uses.**
- Legacy path (`use_refactor_path=False`) — `two_pass_fwd_grouped`.
  Older version with more return values and a v1/v2 split. Avoid.

### Step 5: Ensure contiguity

```python
x = x if x.is_contiguous() else x.contiguous()
B = B if B.is_contiguous() else B.contiguous()
C = C if C.is_contiguous() else C.contiguous()
h = h if h.is_contiguous() else h.contiguous()
```

**Purpose**: Triton kernels need contiguous inputs (they index by
strides; non-contiguous breaks the index math). The pattern
`tensor.contiguous() if not tensor.is_contiguous() else tensor` is a
no-op when already contiguous, allocates+copies when not.

**Important for your adapter**: after the `reshape().permute()` steps
to convert `(B, D, L) → (bs, l, g, dg)`, your tensors may be
non-contiguous. CGCG handles it (forces contiguity here), but the
implicit copy costs memory and time. Better to `.contiguous()` ONCE in
the adapter to avoid four separate copies.

### Step 6: Launch the kernel

```python
out = kernel(
    x,
    B,
    C,
    h,
    autotune=autotune,
    schedule=schedule,
    return_toeplitz=return_toeplitz,
    return_y2=True,                         # ← ALWAYS true (for backward)
    return_bx_lag=return_bx_lag,
    CHUNK_SIZE=fwd_kernel_cfg.CHUNK_SIZE,
    BLOCK_D=fwd_kernel_cfg.BLOCK_D,
    NUM_PIPELINE_STAGES=fwd_kernel_cfg.NUM_PIPELINE_STAGES,
    THREADBLOCK_SWIZZLE=fwd_kernel_cfg.THREADBLOCK_SWIZZLE,
    num_warps=fwd_kernel_cfg.num_warps,
    num_stages=fwd_kernel_cfg.num_stages,
    num_ctas=fwd_kernel_cfg.num_ctas,
    return_autotune_result=True if autotune else False,
)
```

**Purpose**: Invoke `two_pass_fwd_grouped_refactor` (or the legacy
variant). This is where the actual Triton compilation + GPU launch
happens. The kernel runs CGCG on the GPU in chunks of `CHUNK_SIZE`
along the L-axis.

`return_y2=True` is **hardcoded** — `y2` is the intermediate result
needed by the backward kernel, so forward always returns it. For
inference where backward never fires, the `y2` tensor is wasted
memory; that's the price of using a code path that supports training.

### Step 7: Unpack outputs

```python
if use_refactor_path:
    bx_lag, y2, y = out
    T, T_hat = None, None
else:
    y, T, T_hat, y2, bx_lag = out
```

**Purpose**: The refactor kernel returns `(bx_lag, y2, y)` (only three);
the legacy kernel returns `(y, T, T_hat, y2, bx_lag)`. Different return
orders for historical reasons. Both produce the same `y` (the final
output you care about).

`T` and `T_hat` are the Toeplitz matrices from the chunked conv. The
refactor path doesn't compute them as a side output (saves SMEM);
backward recomputes if needed.

### Step 8: Save for backward + return

```python
ctx.save_for_backward(x, B, C, h, T, T_hat, y2, bx_lag)
ctx.fwd_kernel_cfg = fwd_kernel_cfg
ctx.bwd_kernel_cfg = bwd_kernel_cfg

return y
```

**Purpose**:
- `save_for_backward(...)` stashes input + intermediate tensors for the
  matching `backward()` call. PyTorch's autograd uses these when `y` is
  used in a downstream `.backward()` chain.
- The config dataclasses are stashed on `ctx` directly (not via
  `save_for_backward`, which is for tensors only).
- Return `y` — the final `(bs, l, g, dg)` fp32 output.

**For inference**: `save_for_backward` runs but the saved tensors are
freed when `y` falls out of scope (no `.backward()` ever gets called).
Wasted ~4 MiB of allocator churn per call, not a correctness issue.

## `backward()` walkthrough (brief — not on our path)

```python
@staticmethod
def backward(ctx, dy: torch.Tensor):
    x, B, C, h, T, T_hat, y2, bx_lag = ctx.saved_tensors
    # ... ensure contiguity ...
    kernel = two_pass_bwd_grouped
    dx, dB, dC, dh = kernel(dy, x, B, C, h, y2=y2, T=T, T_hat=T_hat, bx_lag=bx_lag, ...)
    return (dx, dB, dC, dh, None, None, None, None)
```

The 8-tuple return matches the 8 forward args: gradients for the four
tensors (`x, B, C, h`), `None` for the four config args (not tensors,
no gradient).

`two_pass_bwd_grouped` lives in `vortex_kernels/ops/hyena_se/bwd.py`
(target 9's sibling file, ~530 lines of backward orchestration + ~1200
lines of Triton in `bwd_kernels.py`).

**Inference never calls this**; the entire backward stack is dead code
for our wire-up. But it's there if anyone wants to use CGCG for
training later.

## Per-step purpose, in one line each

| Step | Code | Purpose |
|---|---|---|
| 1 | `if fwd_kernel_cfg is None: ...` | Default configs |
| 2 | `return_toeplitz, return_bx_lag = ...` | Read backward-reuse flags |
| 3 | `bwd_kernel_cfg.CHUNK_SIZE = fwd_kernel_cfg.CHUNK_SIZE` | Constraint propagation for Toeplitz reuse |
| 4 | `kernel = two_pass_fwd_grouped_refactor` | Pick launcher (refactor vs legacy) |
| 5 | `x = x.contiguous() if not x.is_contiguous() else x` (×4) | Force contiguity for Triton |
| 6 | `out = kernel(x, B, C, h, ..., CHUNK_SIZE=128, BLOCK_D=32, ...)` | Launch the Triton kernel |
| 7 | `bx_lag, y2, y = out` | Unpack outputs |
| 8 | `ctx.save_for_backward(...); return y` | Stash for backward, return |

## What your adapter actually does with this

```python
# Inside hcs_dispatch (Phase 2.2):
def hcs_dispatch(self, fir_fn, u, weight, bias, L, dims, groups, ...):
    # ... split u into x2, x1, v ...
    # ... reshape (B, D, L) → (bs, l, g, dg) ...
    # ... de-expand weight from (D, 1, hl) → (g, 1, hl) ...

    # THIS LINE — the kernel call:
    y = TwoPassChunkedGateConvGate.apply(
        v_reshaped,           # x in CGCG
        x1_reshaped,          # B in CGCG
        x2_reshaped,          # C in CGCG
        h_grouped,            # h, per-group
        "default",
        True,                 # use refactor path
        False,                # no autotune for predictable testing
        None,                 # default fwd config (CHUNK_SIZE=128, BLOCK_D=32)
        None,                 # default bwd config (irrelevant for inference)
    )

    # ... reshape y back to (B, D, L) ...
    # ... apply optional bias-add, padding mask, etc. ...
    return z, fir_state
```

The autograd `.apply()` call is **one line**. Everything around it is
the layout-conversion plumbing.

## What you DON'T need to know for the wire-up

- The implementation details inside `two_pass_fwd_grouped` (target 9
  is a light walkthrough of its docstring; the actual Triton kernel
  in `fwd_kernels.py` is ~800 lines you can skip)
- The Toeplitz construction math (`utils.py::toeplitz` and
  `toeplitz_kernels.py`) — exists for reference/debug, kernel handles
  it internally
- The backward kernel (`bwd.py`, `bwd_kernels.py`, `_bwd_two_kernel.py`,
  `_bwd_tma.py`) — never invoked in inference
- TMA variants (`_fwd_tma.py`, `_bwd_tma.py`) — Hopper-specific
  optimization deferred

## Cross-references

- [gcg_fwd_ref_corrected](gcg_fwd_ref_corrected.md) — Target 7 — the
  pure-PyTorch reference that this kernel is a fast Triton implementation
  of
- [two_pass_fwd_grouped](two_pass_fwd_grouped.md) — Target 9 — the
  kernel launcher that this autograd Function calls
- [hyena_cascade_parallel_forward](hyena_cascade_parallel_forward.md) —
  Target 5 — the upstream call site whose `parallel_fir` call your
  adapter intercepts to reroute through this kernel

## Open questions / to-revisit later

- **`autotune` for production**: should the wire-up enable autotune?
  Pros: best perf per shape. Cons: first call latency. Recommendation
  for Phase 2 first GPU smoke test: `autotune=False` with hand-picked
  configs; flip to `autotune=True` for the microbench so we get the
  fastest numbers.
- **`USE_TMA` for Hopper**: TMA forward kernel exists at `_fwd_tma.py`
  but isn't wired through this interface. Future Phase 4+ polish — add
  a `USE_TMA=True` branch that routes to `two_pass_fwd_grouped_tma`
  when running on H100.
- **`return_y2=True` waste**: forward always returns `y2` for backward
  even when backward will never fire. Wastes ~4 MiB per call. Could
  add an `inference_only=True` flag that skips the `y2` allocation.
  Cosmetic polish; not blocking.
- **Default `BLOCK_D=32` vs `dg=16` for HCS**: `BLOCK_D` of 32 is the
  D-axis tile size; with HCS's `dg=16` the kernel processes 2 groups
  per tile. Need to verify autotuning would prefer `BLOCK_D=16` for
  HCS specifically. Phase 2.6 microbench can answer.

---

*Last updated: 2026-05-12 (Phase 1 read-through, target 8/12)*
