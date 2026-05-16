"""
HCM adapter — fills the role of `vortex/ops/hcm_interface.py` (which is
0 bytes on upstream `cb229ae`).

Two possible kernels, decided in Phase 3.0:
- Option A: reuse CGCG (same kernel as HCS) at `filter_length=128`
- Option B: Triton epilogues around cuFFT for the complex multiply and
  bias-residual fusion

Until Phase 3.0 runs the feasibility check on a GPU pod, this file
stays as a placeholder.

See IMPLEMENTATION_PLAN.md §"Phase 3 — HCM".
"""

from typing import NoReturn


def hcm_fft_conv(*args: object, **kwargs: object) -> NoReturn:
    """
    Drop-in replacement for `vortex.model.engine.fftconv_func`.

    Raises:
        NotImplementedError: Until Phase 3 lands.
    """
    _ = args, kwargs
    raise NotImplementedError(
        "hcm_fft_conv not yet implemented. See IMPLEMENTATION_PLAN.md Phase 3."
    )
