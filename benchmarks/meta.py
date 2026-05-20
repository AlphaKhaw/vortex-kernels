"""
Shared run-metadata helpers for the benchmarks in this repo.

Every emitted JSON wraps its measurements inside a `run_meta` envelope so
that 4090 and H100 results stay traceable to the (gpu, software stack, repo
SHA) tuple that produced them.
"""

import platform
import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
import triton

REPO_ROOT: Path = Path(__file__).resolve().parent.parent

_GPU_NAME_PATTERNS: tuple[tuple[str, str], ...] = (
    ("4090", "rtx-4090"),
    ("h100", "h100"),
    ("a100", "a100"),
    ("3090", "rtx-3090"),
    ("l40", "l40"),
)


def gpu_slug() -> str:
    """
    Return a short filesystem-safe tag for the active CUDA device.

    Returns:
        str: A canonical short tag (e.g. `rtx-4090`, `h100`), a sanitised
        device name as fallback, or `cpu` when no CUDA device is visible.
    """
    if not torch.cuda.is_available():
        return "cpu"
    name: str = torch.cuda.get_device_name(0).lower()
    for needle, tag in _GPU_NAME_PATTERNS:
        if needle in name:
            return tag
    sanitised: str = re.sub(r"[^a-z0-9]+", "-", name).strip("-")
    return sanitised or "unknown-gpu"


def _git_sha(path: Path) -> str | None:
    """
    Return the short HEAD SHA at path, or None when not a git repo.
    """
    try:
        out = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "--short", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return None
    return out.stdout.strip() or None


def _driver_version() -> str | None:
    """
    Return the NVIDIA driver version from nvidia-smi, or None if unavailable.
    """
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return None
    lines: list[str] = out.stdout.strip().splitlines()
    return lines[0] if lines and lines[0] else None


def run_meta(config: dict[str, Any] | None = None) -> dict[str, Any]:
    """
    Build the canonical run-metadata envelope.

    Args:
        config (dict[str, Any] | None): Run-specific configuration merged into
            the envelope under the `config` key.

    Returns:
        dict[str, Any]: Envelope ready to serialise as JSON.
    """
    cuda_available: bool = torch.cuda.is_available()
    gpu_name: str = torch.cuda.get_device_name(0) if cuda_available else "cpu"
    capability: str | None = (
        ".".join(str(x) for x in torch.cuda.get_device_capability(0)) if cuda_available else None
    )
    vortex_root: Path = REPO_ROOT.parent / "vortex"
    return {
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "gpu": gpu_name,
        "gpu_slug": gpu_slug(),
        "compute_capability": capability,
        "cuda_version": torch.version.cuda,
        "driver_version": _driver_version(),
        "torch_version": torch.__version__,
        "triton_version": triton.__version__,
        "vortex_sha": _git_sha(vortex_root) if vortex_root.exists() else None,
        "vortex_kernels_sha": _git_sha(REPO_ROOT),
        "platform": platform.platform(),
        "config": config or {},
    }


def default_results_root() -> Path:
    """
    Return the per-GPU results root (results/<gpu_slug>/).
    """
    return REPO_ROOT / "results" / gpu_slug()
