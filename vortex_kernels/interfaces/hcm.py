"""
HCM fused FFT convolution — fills the role of vortex/ops/hcm_interface.py.

Target: replace the unfused fftconv_func call in
HyenaInferenceEngine.parallel_fir when fir_length >= 128 (the HCM path).
This path is unoptimized even with use_flashfft=True — FlashFFTConv is only
wired into parallel_iir (HCL), never parallel_fir. Biggest unclaimed win.
"""

from typing import NoReturn


def hcm_fft_conv(*args: object, **kwargs: object) -> NoReturn:
    """
    Fused FFT convolution for Hyena Medium-Range (HCM) layers.

    Drop-in replacement for vortex.model.engine.fftconv_func when invoked from
    HyenaInferenceEngine.parallel_fir with fir_length >= 128. Fuses scale,
    complex multiply, and skip-add around the cuFFT calls to reduce kernel
    launches from 6 to 4 per layer.

    Intended signature once implemented will match fftconv_func:
        (u, k, D, *, dropout_mask=None, gelu=True, k_rev=None, bidirectional=False)

    Raises:
        NotImplementedError: Always, until the kernel is written.
    """
    _ = args, kwargs
    raise NotImplementedError("hcm_fft_conv not yet implemented. See docs/issue_draft.md scope.")
