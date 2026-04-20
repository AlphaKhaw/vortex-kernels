"""
Monkey-patches vortex with vortex-kernels replacements.

Call sites are gated on individual kernels being implemented — the function
returns the list of patches actually applied. Safe to call multiple times
(idempotent) and safe when vortex is missing (logs and returns empty list).
"""

import logging

logger = logging.getLogger(__name__)

_ORIGINALS: dict[str, object] = {}


def patch_vortex() -> list[str]:
    """
    Apply available vortex-kernels patches to vortex.

    Safe to call multiple times (idempotent) and safe when vortex is not
    installed (logs a warning and returns an empty list).

    Returns:
        List of patch names actually applied (e.g., ["hcm"]). Empty list if
        vortex is absent or no kernels are yet implemented.
    """
    applied: list[str] = []
    try:
        from vortex.model import engine  # noqa: F401
    except ImportError:
        logger.warning("vortex not installed — no patches applied")
        return applied

    # Wiring added when each kernel lands in vortex_kernels.interfaces.*.
    # Pattern for HCM (applies to HCS/HCL by analogy):
    #     from .interfaces.hcm import hcm_fft_conv
    #     _ORIGINALS["fftconv_func"] = engine.fftconv_func
    #     engine.fftconv_func = _dispatch_with_hcm_fallback(
    #         hcm_fft_conv, _ORIGINALS["fftconv_func"]
    #     )
    #     applied.append("hcm")

    if applied:
        logger.info("vortex-kernels: patched vortex (%s)", ", ".join(applied))
    return applied


def unpatch_vortex() -> None:
    """
    Restore original vortex functions previously replaced by patch_vortex.

    No-op if patch_vortex has not been called or applied no patches.
    """
    if not _ORIGINALS:
        return
    from vortex.model import engine

    for name, original in _ORIGINALS.items():
        if name == "fftconv_func":
            engine.fftconv_func = original  # type: ignore[assignment]
    _ORIGINALS.clear()
    logger.info("vortex-kernels: unpatched vortex")
