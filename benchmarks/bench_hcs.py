"""
Microbenchmark for the HCS depthwise conv Triton kernel.

Compares the fused Triton kernel (`vortex.ops.hcs_interface.hcs_depthwise_conv`)
against `F.conv1d` (cuDNN) on the depthwise causal conv that defines an HCS
layer's time mixing. Both run in fp32 at evo2_7b shapes (D=4096, fir_length=7).

Results are written to `results/microbench/hcs.json` for commit alongside the
kernel.
"""

import argparse
import json
from collections.abc import Callable
from pathlib import Path

import torch
import torch.nn.functional as F
import triton
from vortex.ops.hcs_interface import hcs_depthwise_conv

from benchmarks.meta import default_results_root, run_meta

# evo2_7b HCS layer: hidden_size 4096, hcs_filter_length 7.
_D: int = 4096
_FIR_LENGTH: int = 7
_SEQ_LENS: list[int] = [32, 64, 128, 256, 512, 1024, 2048, 4096]


def _conv1d(u: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """
    The cuDNN baseline: depthwise causal conv via F.conv1d, trimmed to L.

    Args:
        u (torch.Tensor): Input tensor of shape (1, D, L).
        weight (torch.Tensor): Filter tensor of shape (D, 1, fir_length).

    Returns:
        torch.Tensor: Output tensor of shape (1, D, L).
    """
    L = u.shape[-1]
    return F.conv1d(
        u,
        weight,
        bias=None,
        stride=1,
        padding=_FIR_LENGTH - 1,
        groups=u.shape[1],
    )[..., :L]


def _median_ms(fn: Callable[[], object]) -> float:
    """
    Median wall-clock milliseconds of fn via triton.testing.do_bench.

    Args:
        fn (Callable[[], object]): Function to benchmark.

    Returns:
        float: Median wall-clock milliseconds.
    """
    return triton.testing.do_bench(fn)


def _bench_one(L: int) -> dict[str, float | int]:
    """
    Benchmark the Triton kernel and the cuDNN baseline at sequence length L.

    Args:
        L (int): Sequence length.

    Returns:
        dict[str, float | int]: Timings and peak-memory counters for one L.
    """
    torch.manual_seed(0)
    # torch.randn defaults to float32 -- the dtype the HCS conv runs in.
    u: torch.Tensor = torch.randn(1, _D, L, device="cuda")
    weight: torch.Tensor = torch.randn(_D, 1, _FIR_LENGTH, device="cuda")

    max_diff: float = (hcs_depthwise_conv(u, weight) - _conv1d(u, weight)).abs().max().item()

    torch.cuda.reset_peak_memory_stats()
    triton_ms: float = _median_ms(lambda: hcs_depthwise_conv(u, weight))
    triton_peak_bytes: int = torch.cuda.max_memory_allocated()

    torch.cuda.reset_peak_memory_stats()
    cudnn_ms: float = _median_ms(lambda: _conv1d(u, weight))
    cudnn_peak_bytes: int = torch.cuda.max_memory_allocated()

    return {
        "seq_len": L,
        "triton_ms": round(triton_ms, 5),
        "cudnn_ms": round(cudnn_ms, 5),
        "speedup": round(cudnn_ms / triton_ms, 3),
        "triton_peak_bytes": triton_peak_bytes,
        "cudnn_peak_bytes": cudnn_peak_bytes,
        "max_diff": max_diff,
    }


def main() -> None:
    """
    Run the HCS microbench across sequence lengths and write the JSON report.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Path to write the JSON report. Defaults to "
            "results/<gpu>/microbench/hcs.json with <gpu> auto-detected."
        ),
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("bench_hcs requires a CUDA device")

    output: Path = args.output or default_results_root() / "microbench" / "hcs.json"

    rows: list[dict[str, float | int]] = [_bench_one(L) for L in _SEQ_LENS]
    report = {
        "run_meta": run_meta(
            config={
                "kernel": "hcs_depthwise_conv",
                "dtype": "float32",
                "hidden_size": _D,
                "fir_length": _FIR_LENGTH,
                "seq_lens": _SEQ_LENS,
            }
        ),
        "results": rows,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")

    print(f"{'seq_len':>8} {'triton_ms':>11} {'cudnn_ms':>10} {'speedup':>9} {'max_diff':>10}")
    for row in rows:
        print(
            f"{row['seq_len']:>8} {row['triton_ms']:>11.5f} {row['cudnn_ms']:>10.5f} "
            f"{row['speedup']:>8.3f}x {row['max_diff']:>10.2e}"
        )
    print(f"\nwrote {output}")


if __name__ == "__main__":
    main()
