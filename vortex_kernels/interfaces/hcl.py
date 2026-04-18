"""HCL fused IIR FFT conv — fills the role of vortex/ops/hcl_interface.py.

Target: fused scale/multiply around cuFFT in the `long_fir_threshold is None`
branch of `HyenaInferenceEngine.parallel_iir`. Marginal value when FlashFFTConv
is available (since that single-call path already exists); useful as a pure-
Triton alternative for environments that can't compile FlashFFTConv.
"""

from __future__ import annotations


def hcl_fft_conv(*args, **kwargs):
    raise NotImplementedError("hcl_fft_conv not yet implemented. See docs/issue_draft.md scope.")
