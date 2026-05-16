"""
HCS adapter — fills the role of `vortex/ops/hcs_interface.py` (which is
0 bytes on upstream `cb229ae`).

Wires the existing CGCG kernel from `vortex_kernels/ops/hyena_se/` into
the HCS cascade short-conv call site in `engine.py::parallel_fir`'s
gate=True + fir_length<128 branch.

Math: `z = x2 * conv(x1*v, weight)` where `weight ∈ (g=256, 1, 7)`.
Maps to CGCG's `y = C * conv(B*x, h)` under `B↔x1, x↔v, C↔x2`.

The actual `hcs_dispatch` function will be written in Phase 2.2 — this
file is the placeholder so the layout mirrors upstream.

See IMPLEMENTATION_PLAN.md §"Phase 2.2 — Write the engine adapter".
"""

from typing import NoReturn


def hcs_dispatch(*args: object, **kwargs: object) -> NoReturn:
    """
    Replace HyenaInferenceEngine.parallel_fir for HCS-cascade calls.

    Signature will match the original parallel_fir method so the
    monkey-patch can fall through to upstream for non-HCS cases.

    Raises:
        NotImplementedError: Until the wire-up lands in Phase 2.2.
    """
    _ = args, kwargs
    raise NotImplementedError(
        "hcs_dispatch not yet implemented. See IMPLEMENTATION_PLAN.md Phase 2.2."
    )
