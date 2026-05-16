"""
Verbatim vortex reference implementations — correctness oracles only.

Pinned to vortex commit cb229ae. Update via re-vendoring (see
IMPLEMENTATION_PLAN.md Phase 1), never by editing in place.
"""

from .engine_ref import (
    HyenaInferenceEngine,
    adjust_filter_shape_for_broadcast,
    fftconv_func,
)
from .model_ref import compute_filter_pure

__all__ = [
    "HyenaInferenceEngine",
    "adjust_filter_shape_for_broadcast",
    "compute_filter_pure",
    "fftconv_func",
]
