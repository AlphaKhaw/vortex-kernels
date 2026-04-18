"""HCM fused FFT convolution — fills the role of vortex/ops/hcm_interface.py.

Target: replace the unfused `fftconv_func` call in
`HyenaInferenceEngine.parallel_fir` when `fir_length >= 128` (the HCM path).
This path is unoptimized even with `use_flashfft=True` — FlashFFTConv is only
wired into `parallel_iir` (HCL), never `parallel_fir`. Biggest unclaimed win.
"""
from __future__ import annotations


def hcm_fft_conv(*args, **kwargs):
    raise NotImplementedError(
        "hcm_fft_conv not yet implemented. See docs/issue_draft.md scope."
    )
