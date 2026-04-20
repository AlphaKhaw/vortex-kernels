"""
Sanity-check the environment after setup_lambda.sh.

Exits non-zero if any required package fails to import or CUDA is unavailable.
Optional packages (flashfftconv, transformer_engine) only warn.
"""

import sys


def _check_torch() -> str | None:
    """
    Verify torch imports and CUDA is available.

    Returns:
        None on success, or an error message describing the failure.
    """
    try:
        import torch
    except ImportError as e:
        return f"torch import failed: {e}"
    print(f"torch:            {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        return "CUDA not available"
    props = torch.cuda.get_device_properties(0)
    print(f"  GPU:            {torch.cuda.get_device_name(0)}")
    print(f"  VRAM:           {props.total_memory / 1e9:.1f} GB")
    print(f"  CUDA ver:       {torch.version.cuda}")  # pyright: ignore[reportAttributeAccessIssue]
    return None


def _check_triton() -> str | None:
    """
    Verify triton is importable (Linux-only).

    Returns:
        None on success, or an error message describing the failure.
    """
    try:
        import triton  # pyright: ignore[reportMissingImports]
    except ImportError as e:
        return f"triton import failed: {e}"
    print(f"triton:           {triton.__version__}")
    return None


def _check_vortex() -> str | None:
    """
    Verify vortex imports and its FFT entry points are present.

    Returns:
        None on success, or an error message describing the failure.
    """
    try:
        import vortex
        from vortex.model.engine import HyenaInferenceEngine, fftconv_func  # noqa: F401
    except ImportError as e:
        return f"vortex import failed: {e}"
    print(f"vortex:           imported ({vortex.__file__})")
    return None


def _check_evo2() -> str | None:
    """
    Verify evo2 is importable.

    Returns:
        None on success, or an error message describing the failure.
    """
    try:
        import evo2  # pyright: ignore[reportMissingImports]  # noqa: F401
    except ImportError as e:
        return f"evo2 import failed: {e}"
    print("evo2:             imported")
    return None


def _check_vortex_kernels() -> str | None:
    """
    Verify this package itself is importable from the active environment.

    Returns:
        None on success, or an error message describing the failure.
    """
    try:
        import vortex_kernels
    except ImportError as e:
        return f"vortex_kernels import failed: {e}"
    print(f"vortex_kernels:   {vortex_kernels.__version__}")
    return None


def _check_flashfftconv_optional() -> None:
    """
    Prints whether FlashFFTConv is available; never fails.
    """
    try:
        import flashfftconv  # pyright: ignore[reportMissingImports]  # noqa: F401
    except ImportError:
        print("flashfftconv:     NOT available (Tier-2 bench will skip)")
        return
    print("flashfftconv:     available (Tier-2 bench enabled)")


def _check_transformer_engine_optional() -> None:
    """
    Prints whether Transformer Engine is available; never fails.
    """
    try:
        import transformer_engine  # pyright: ignore[reportMissingImports]  # noqa: F401
    except ImportError as e:
        print(f"transformer_engine: NOT available — {e}")
        print("  (only required for evo2_40b / evo2_20b / evo2_1b — 7B_base works without)")
        return
    print("transformer_engine: imported")


def main() -> int:
    """
    Check the import health of every dependency the project needs.

    Returns:
        0 if all required imports succeed and CUDA is available, 1 otherwise.
    """
    required_checks = (
        _check_torch,
        _check_triton,
        _check_vortex,
        _check_evo2,
        _check_vortex_kernels,
    )
    optional_checks = (
        _check_flashfftconv_optional,
        _check_transformer_engine_optional,
    )

    failures: list[str] = [msg for check in required_checks if (msg := check())]
    for opt_check in optional_checks:
        opt_check()

    if failures:
        print("\nFAILURES:")
        for f in failures:
            print(f"  - {f}")
        return 1

    print("\nOK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
