"""Import-level smoke tests. Run on any platform (no GPU required)."""
from __future__ import annotations

import pytest


def test_vortex_kernels_importable():
    import vortex_kernels

    assert vortex_kernels.__version__


def test_patch_is_idempotent():
    import vortex_kernels

    a = vortex_kernels.patch_vortex()
    b = vortex_kernels.patch_vortex()
    assert a == b


def test_interfaces_raise_until_implemented():
    from vortex_kernels.interfaces import hcl, hcm, hcs

    for fn in (hcm.hcm_fft_conv, hcs.hcs_conv, hcl.hcl_fft_conv):
        with pytest.raises(NotImplementedError):
            fn()
