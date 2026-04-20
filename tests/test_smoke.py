"""
Import-level smoke tests. Run on any platform (no GPU required).
"""

import pytest
import torch


def test_vortex_kernels_importable():
    """
    Package imports cleanly and exposes a version string.
    """
    import vortex_kernels

    assert vortex_kernels.__version__


def test_patch_is_idempotent():
    """
    Calling patch_vortex twice returns the same list of applied patches.
    """
    import vortex_kernels

    a = vortex_kernels.patch_vortex()
    b = vortex_kernels.patch_vortex()
    assert a == b


def test_patch_unpatch_roundtrip():
    """
    patch_vortex followed by unpatch_vortex completes without raising.
    """
    import vortex_kernels

    vortex_kernels.patch_vortex()
    vortex_kernels.unpatch_vortex()


def test_interfaces_raise_until_implemented():
    """
    Every interface stub raises NotImplementedError until its kernel lands.
    """
    from vortex_kernels.interfaces import hcl, hcm, hcs

    for fn in (hcm.hcm_fft_conv, hcs.hcs_conv, hcl.hcl_fft_conv):
        with pytest.raises(NotImplementedError):
            fn()


def test_autouse_seed_fixture_ran():
    """
    Confirm the autouse _seed_everything fixture runs before each test.

    If the fixture fired, torch's RNG is seeded to 42 at test entry; generating
    tensors, reseeding to 42, and regenerating must yield identical values.
    """
    observed = torch.randn(3).tolist()
    torch.manual_seed(42)
    expected = torch.randn(3).tolist()
    assert observed == expected, "_seed_everything autouse fixture did not run"


@pytest.mark.gpu
def test_device_fixture_returns_cuda(device: torch.device):
    """
    The `device` fixture yields cuda:0 on GPU machines; skipped on CPU.
    """
    assert device.type == "cuda"
