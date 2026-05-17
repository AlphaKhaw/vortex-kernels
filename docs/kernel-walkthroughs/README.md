# Kernel walkthroughs

Visual, shape-tracking walkthroughs of every function and method on the
HCS/HCM/HCL dispatch path. One markdown per symbol, in the order I'm
reading them via the revdiff stepwise tour (see `IMPLEMENTATION_PLAN.md`
Phase 1.1).

Each walkthrough has the same shape:

1. **Overall purpose** — one paragraph, "what is this doing and why"
2. **Source location** — `path:lines` so revdiff / your editor can jump
3. **Realistic input shapes** — evo2_7b numbers (`B=1, D=4096, L=8192`, etc.)
4. **Step-by-step with shape transitions** — every line of code with
   the tensor shape before and after
5. **Per-step purpose card** — one-line summary per step
6. **What changes for our wire-up** — what the HCS/HCM/HCL contribution
   modifies vs leaves verbatim
7. **Cross-references** — links to other walkthroughs the symbol calls
   into or gets called by

## Tour order and status

Two depth tiers (decided 2026-05-12 after target 1):

- **Full** — overall purpose + inputs explained + background concepts where
  unfamiliar + step-by-step with shape tracking + per-step purpose + wire-up
  impact + cross-refs. ~300–500 lines. Reserved for symbols you'll modify,
  call, or whose math has aha moments worth earning.
- **Light** — overall purpose + shape contract (in/out) + one shape pipeline
  diagram + call-graph map. ~50–100 lines. For symbols you only need the data
  contract on.

| # | Walkthrough | Source | Depth | Status |
|---|---|---|---|---|
| 1 | [fftconv_func](fftconv_func.md) | `engine_ref.py:51-96` | full | ✅ done |
| 2 | [parallel_fir](parallel_fir.md) | `engine_ref.py:146-275` | full | ✅ done |
| 3 | [parallel_iir](parallel_iir.md) | `engine_ref.py:277-419` | full | ✅ done |
| 4 | [HyenaCascade.\_\_init\_\_](hyena_cascade_init.md) | `vortex/model/model.py:126-213` | light | ✅ done |
| 5 | [HyenaCascade.parallel_forward](hyena_cascade_parallel_forward.md) | `vortex/model/model.py:221-317` | full | ✅ done |
| 6 | [compute_filter_pure](compute_filter_pure.md) | `model_ref.py:22-47` | full | ⏸ pending |
| 7 | [gcg_fwd_ref_corrected](gcg_fwd_ref_corrected.md) | `hyena_se/ref_fwd.py:80-127` | full | ✅ done |
| 8 | [TwoPassChunkedGateConvGate](two_pass_chunked_gate_conv_gate.md) | `hyena_se/interface.py:47-167` | full | ✅ done |
| 9 | [two_pass_fwd_grouped](two_pass_fwd_grouped.md) | `hyena_se/fwd.py:17-105` | light | ✅ done |
| 10 | [upstream hcs_interface.py (empty)](upstream_hcs_interface.md) | `vortex/ops/hcs_interface.py` | light | ⏸ pending |
| 11 | [our hcs_interface.py (scaffold)](our_hcs_interface.md) | `vortex_kernels/ops/hcs_interface.py` | light | ⏸ pending |
| 12 | [patching.py (wire-up)](patching.md) | `vortex_kernels/patching.py` | light | ⏸ pending |

6 full + 6 light. Reading time roughly: full ~10–15 min, light ~3–5 min.
Total tour: ~90 min of focused reading.

## How these get written

Each walkthrough is generated during the revdiff session by asking
Claude a deep question about the current target — visualizing the math,
tracking shapes, and explaining purpose. The answer gets adapted into
the markdown here. The session resumes; advance to the next target;
repeat.

The order matches the tour queue in
`~/.config/revdiff/session/<session-id>/queue.json`. If we add or
reorder targets mid-tour, the table above gets updated.

## How to read

- **First pass**: read in tour order top-to-bottom. Each walkthrough
  builds on the prior one's cross-references.
- **Reference pass**: use the table as an index; jump to whichever
  symbol you need to refresh on.
- **Before kernel work**: re-read targets 5 (`parallel_forward`),
  8 (`TwoPassChunkedGateConvGate`), and 9 (`two_pass_fwd_grouped`) —
  these are the three you'll be wiring together in Phase 2.2.
