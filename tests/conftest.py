from __future__ import annotations

import pytest
import torch


def pytest_collection_modifyitems(config, items):
    if torch.cuda.is_available():
        return
    skip_gpu = pytest.mark.skip(reason="requires CUDA device")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip_gpu)


@pytest.fixture(scope="session")
def device() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA device")
    return torch.device("cuda:0")


@pytest.fixture(autouse=True)
def _seed_everything():
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
