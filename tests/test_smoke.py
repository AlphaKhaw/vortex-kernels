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
