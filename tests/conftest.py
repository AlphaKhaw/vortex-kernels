# pyright: reportUnusedFunction=none
# conftest.py fixtures and hooks are invoked by pytest's registry, not by name
# references in source — basedpyright can't see through it.

import pytest
import torch


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """
    Auto-skip tests marked @pytest.mark.gpu when no CUDA device is present.

    Args:
        config (pytest.Config): Pytest config object, unused but required by
            the hook signature.
        items (list[pytest.Item]): Collected test items; mutated in place to
            attach a skip marker when CUDA is unavailable.
    """
    _ = config  # required by pytest hook signature; unused here
    if torch.cuda.is_available():
        return
    skip_gpu = pytest.mark.skip(reason="requires CUDA device")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip_gpu)


@pytest.fixture(scope="session")
def device() -> torch.device:
    """
    Provide a CUDA device for GPU-requiring tests; skip the test if absent.

    Returns:
        torch.device pointing at cuda:0.
    """
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA device")
    return torch.device("cuda:0")


@pytest.fixture(autouse=True)
def _seed_everything():
    """
    Seed torch CPU (and CUDA if available) RNGs to 42 before every test.

    Runs automatically via autouse=True — pytest invokes it through its
    fixture registry, not by direct call. Needed so torch.randn and friends
    produce deterministic values inside tests.
    """
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
