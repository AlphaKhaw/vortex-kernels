"""
Shared run-metadata helpers for every benchmark and profiler in this repo.

Every emitted result JSON wraps its measurements inside a ``run_meta``
envelope so downstream comparison scripts can join across runs (4090 vs H100,
historical vs current commit, configs A vs B). See ``docs/BENCHMARKING.md``
for the full schema.
"""

import platform
import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

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
    Returns a filesystem-safe short tag for the active CUDA device.

    Picks one of the canonical tags (``rtx-4090``, ``h100``, ``a100``,
    ``rtx-3090``, ``l40``) when the device name matches, otherwise sanitises
    the full device name. Used as the default top-level ``results/``
    subdirectory so 4090 and H100 artefacts never collide.

    Returns:
        str: A short, filesystem-safe identifier for the active GPU, or
        ``"cpu"`` when no CUDA device is available.
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
    Returns the short HEAD SHA at ``path``, or ``None`` when not a git repo.

    Args:
        path (Path): Filesystem path to check.

    Returns:
        str | None: The short SHA, or ``None`` if ``path`` is not a git
        working tree or git is unavailable.
    """
    try:
        out: subprocess.CompletedProcess[str] = subprocess.run(
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
    Returns the NVIDIA driver version via nvidia-smi, or ``None`` if unavailable.

    Returns:
        str | None: The driver version, or ``None`` if unavailable.
    """
    try:
        out: subprocess.CompletedProcess[str] = subprocess.run(
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


def _triton_version() -> str | None:
    """
    Returns ``triton.__version__`` if importable, else ``None``.
    """
    try:
        import triton  # pyright: ignore[reportMissingImports]
    except ImportError:
        return None
    version: str = triton.__version__
    return version


def run_meta(config: dict[str, Any] | None = None) -> dict[str, Any]:
    """
    Returns the canonical run-metadata envelope.

    The envelope captures the (gpu, software-stack, repo-SHA, config) tuple
    that uniquely identifies a measurement. Every benchmark stores this
    alongside its results so that re-runs are traceable and 4090 vs H100
    comparisons are unambiguous.

    Args:
        config (dict[str, Any] | None): Run-specific configuration (model,
                                        seq_len, kernel flags, etc.) merged into the
                                        envelope under ``config``.
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
        "triton_version": _triton_version(),
        "vortex_sha": _git_sha(vortex_root) if vortex_root.exists() else None,
        "vortex_kernels_sha": _git_sha(REPO_ROOT),
        "platform": platform.platform(),
        "config": config or {},
    }


def default_results_root() -> Path:
    """
    Returns the per-GPU results root (``results/<gpu_slug>/``).
    """
    return REPO_ROOT / "results" / gpu_slug()
