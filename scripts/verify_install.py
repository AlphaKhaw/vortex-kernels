"""Sanity-check the env after setup_lambda.sh. Exits non-zero on hard failures."""
from __future__ import annotations

import sys


def main() -> int:
    failures: list[str] = []

    try:
        import torch

        print(f"torch:            {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  GPU:            {torch.cuda.get_device_name(0)}")
            print(f"  VRAM:           {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            print(f"  CUDA ver:       {torch.version.cuda}")
        else:
            failures.append("CUDA not available")
    except ImportError as e:
        failures.append(f"torch import failed: {e}")

    try:
        import triton

        print(f"triton:           {triton.__version__}")
    except ImportError as e:
        failures.append(f"triton import failed: {e}")

    try:
        import vortex
        from vortex.model.engine import HyenaInferenceEngine, fftconv_func  # noqa: F401

        print(f"vortex:           imported ({vortex.__file__})")
    except ImportError as e:
        failures.append(f"vortex import failed: {e}")

    try:
        import evo2  # noqa: F401

        print(f"evo2:             imported")
    except ImportError as e:
        failures.append(f"evo2 import failed: {e}")

    try:
        import flashfftconv  # noqa: F401

        print(f"flashfftconv:     available (Tier-2 bench enabled)")
    except ImportError:
        print(f"flashfftconv:     NOT available (Tier-2 bench will skip)")

    try:
        import transformer_engine  # noqa: F401

        print(f"transformer_engine: imported")
    except ImportError as e:
        print(f"transformer_engine: NOT available — {e}")
        print("  (only required for evo2_40b / evo2_20b / evo2_1b — 7B_base works without)")

    try:
        import vortex_kernels

        print(f"vortex_kernels:   {vortex_kernels.__version__}")
    except ImportError as e:
        failures.append(f"vortex_kernels import failed: {e}")

    if failures:
        print("\nFAILURES:")
        for f in failures:
            print(f"  - {f}")
        return 1

    print("\nOK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
