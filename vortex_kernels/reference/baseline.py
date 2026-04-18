"""Verbatim reference copies of vortex functions — ground truth for correctness tests.

Do not modify these. They are the numerical oracle. If vortex's engine.py changes
upstream, update these copies in a separate commit and re-run the full test suite.

Sourced from vortex/model/engine.py at commit <pinned-on-setup>.
"""
from __future__ import annotations

# TODO: copy verbatim once Lambda env is up and vortex is pinned:
#   - fftconv_func
#   - the hot path of HyenaInferenceEngine.parallel_iir (fft_size = 2*L branch)
#   - adjust_filter_shape_for_broadcast
# Intentionally left empty — pinning against a specific vortex commit is
# Day-1 work on Lambda, not desk work.
