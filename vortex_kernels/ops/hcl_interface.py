"""
HCL adapter — fills the role of `vortex/ops/hcl_interface.py` (which is
0 bytes on upstream `cb229ae`).

Tiled `compute_filter` + tiled FFT-conv. The headline kernel: avoids
the `(D, state_size, L)` fp32 intermediate that OOMs at L=131k.

Unlike HCS/HCM, no upstream Triton precedent exists for HCL — CGCG's
filter-length constraint (`filter_len ≤ CHUNK_SIZE`) can't represent an
L-length filter. This kernel is written from scratch in Phase 4.

See IMPLEMENTATION_PLAN.md §"Phase 4 — HCL".
"""

from typing import NoReturn


def hcl_fft_conv(*args: object, **kwargs: object) -> NoReturn:
    """
    Replacement for the `long_fir_threshold is None` branch in
    `vortex.model.engine.HyenaInferenceEngine.parallel_iir`.

    Raises:
        NotImplementedError: Until Phase 4 lands.
    """
    _ = args, kwargs
    raise NotImplementedError(
        "hcl_fft_conv not yet implemented. See IMPLEMENTATION_PLAN.md Phase 4."
    )
