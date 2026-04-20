"""
HCL fused IIR FFT conv — fills the role of vortex/ops/hcl_interface.py.

Target: fused scale/multiply around cuFFT in the long_fir_threshold is None
branch of HyenaInferenceEngine.parallel_iir. Marginal value when FlashFFTConv
is available (since that single-call path already exists); useful as a pure-
Triton alternative for environments that can't compile FlashFFTConv.
"""

from typing import NoReturn


def hcl_fft_conv(*args: object, **kwargs: object) -> NoReturn:
    """
    Fused IIR FFT convolution for Hyena Long-Range (HCL) layers.

    Replacement for the default (non-FlashFFT) path in
    HyenaInferenceEngine.parallel_iir. Fuses the scale-and-multiply step
    around the two cuFFT calls (rfft on filter, fft on input, irfft on
    product) to eliminate intermediate complex tensor allocations.

    Raises:
        NotImplementedError: Always, until the kernel is written.
    """
    _ = args, kwargs
    raise NotImplementedError("hcl_fft_conv not yet implemented. See docs/issue_draft.md scope.")
