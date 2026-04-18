"""Monkey-patches vortex with vortex-kernels replacements.

Call sites are gated on individual kernels being implemented — the function
returns the list of patches actually applied. Safe to call multiple times
(idempotent) and safe when vortex is missing (logs and returns empty list).
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_ORIGINALS: dict[str, object] = {}


def patch_vortex() -> list[str]:
    applied: list[str] = []
    try:
        from vortex.model import engine  # noqa: F401
    except ImportError:
        logger.warning("vortex not installed — no patches applied")
        return applied

    # HCM: fused fftconv_func for parallel_fir (fir_length >= 128)
    try:
        from .interfaces.hcm import hcm_fft_conv  # noqa: F401

        # TODO: wire once hcm_fft_conv is implemented
        # _ORIGINALS["fftconv_func"] = engine.fftconv_func
        # engine.fftconv_func = _dispatch_with_hcm_fallback(hcm_fft_conv, _ORIGINALS["fftconv_func"])
        # applied.append("hcm")
    except NotImplementedError:
        pass

    # HCS: wire vortex.ops.hyena_x into parallel_fir default branch
    try:
        from .interfaces.hcs import hcs_conv  # noqa: F401

        # TODO: wire once hcs_conv is implemented
    except NotImplementedError:
        pass

    # HCL: fused scale/multiply around cuFFT in parallel_iir
    try:
        from .interfaces.hcl import hcl_fft_conv  # noqa: F401

        # TODO: wire once hcl_fft_conv is implemented
    except NotImplementedError:
        pass

    if applied:
        logger.info("vortex-kernels: patched vortex (%s)", ", ".join(applied))
    return applied


def unpatch_vortex() -> None:
    if not _ORIGINALS:
        return
    from vortex.model import engine

    for name, original in _ORIGINALS.items():
        if name == "fftconv_func":
            engine.fftconv_func = original  # type: ignore[assignment]
    _ORIGINALS.clear()
    logger.info("vortex-kernels: unpatched vortex")
