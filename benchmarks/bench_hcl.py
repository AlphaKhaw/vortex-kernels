"""
Microbenchmark for the HCL kernels.

Compares the stock HCL conv -- compute_filter's (D, state_size, L) reduction
plus the parallel_iir FFT branch -- against the fused path: _hcl_compute_filter
(the tiled filter build) followed by hcl_fft_conv. Reports latency and, the
headline, peak GPU memory: the stock filter build materialises a
(D, state_size, L) fp32 intermediate that OOMs evo2_7b at L=131k; the tiled
kernel never builds it.

Results are written to results/microbench/hcl.json for commit alongside the
kernels.
"""

import argparse
import json
import platform
from collections.abc import Callable
from pathlib import Path

import torch
import triton
from vortex.ops.hcl_interface import _hcl_compute_filter, hcl_fft_conv

# evo2_7b HCL layer: hidden_size 4096, state_size 16.
_D: int = 4096
_S: int = 16
_SEQ_LENS: list[int] = [2048, 8192, 32768]


def _median_ms(fn: Callable[[], object]) -> float:
    """
    Median wall-clock milliseconds of fn via triton.testing.do_bench.

    Args:
        fn (Callable[[], object]): Function to benchmark.

    Returns:
        float: Median wall-clock milliseconds.
    """
    return triton.testing.do_bench(fn)  # pyright: ignore[reportReturnType]


def _stock(
    residues: torch.Tensor,
    log_poles: torch.Tensor,
    t: torch.Tensor,
    x1v: torch.Tensor,
    x2: torch.Tensor,
    D: torch.Tensor,
    L: int,
    fft_size: int,
) -> torch.Tensor:
    """
    The stock HCL conv: the full compute_filter reduction, the FFT branch, the
    post-conv -- the path use_hcl_kernel replaces.
    """
    h = (residues[..., None] * (log_poles[..., None] * t).exp()).sum(1)[None]
    big_h = torch.fft.rfft(h, n=fft_size) / fft_size
    X = torch.fft.fft(x1v, n=fft_size)[..., : big_h.shape[-1]]
    y = torch.fft.irfft(X * big_h, n=fft_size, norm="forward")[..., :L]
    return (y + x1v * D.unsqueeze(-1)) * x2


def _kernel(
    residues: torch.Tensor,
    log_poles: torch.Tensor,
    t: torch.Tensor,
    x1v: torch.Tensor,
    x2: torch.Tensor,
    D: torch.Tensor,
    L: int,
    fft_size: int,
) -> torch.Tensor:
    """
    The fused HCL conv: the tiled _hcl_compute_filter followed by hcl_fft_conv.
    """
    h = _hcl_compute_filter(residues, log_poles, t)[None]
    return hcl_fft_conv(h, x1v, x2, D, L, fft_size)


def _peak_gb(fn: Callable[[], object]) -> float:
    """
    Peak CUDA allocator GiB for a single call of fn.

    Args:
        fn (Callable[[], object]): Function to measure.

    Returns:
        float: Peak allocated GiB since the reset.
    """
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    fn()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 2**30


def _bench_one(L: int) -> dict[str, float]:
    """
    Benchmark the stock HCL conv and the fused kernels at sequence length L.

    Args:
        L (int): Sequence length.

    Returns:
        dict[str, float]: Latency, peak memory and the correctness gap.
    """
    torch.manual_seed(0)
    fft_size = 2 * L
    # log_poles small and negative -> a stable, decaying modal filter.
    residues = torch.randn(_D, _S, device="cuda")
    log_poles = -torch.rand(_D, _S, device="cuda") * (8.0 / L)
    t = torch.arange(L, dtype=torch.float32, device="cuda")
    x1v = torch.randn(1, _D, L, device="cuda")
    x2 = torch.randn(1, _D, L, device="cuda")
    bias = torch.randn(_D, device="cuda")

    def stock() -> torch.Tensor:
        return _stock(residues, log_poles, t, x1v, x2, bias, L, fft_size)

    def kernel() -> torch.Tensor:
        return _kernel(residues, log_poles, t, x1v, x2, bias, L, fft_size)

    max_diff: float = (kernel() - stock()).abs().max().item()
    stock_ms: float = _median_ms(stock)
    kernel_ms: float = _median_ms(kernel)
    stock_gb: float = _peak_gb(stock)
    kernel_gb: float = _peak_gb(kernel)

    return {
        "seq_len": L,
        "stock_ms": round(stock_ms, 5),
        "kernel_ms": round(kernel_ms, 5),
        "speedup": round(stock_ms / kernel_ms, 3),
        "stock_peak_gb": round(stock_gb, 3),
        "kernel_peak_gb": round(kernel_gb, 3),
        "mem_ratio": round(stock_gb / kernel_gb, 2),
        "max_diff": max_diff,
    }


def main() -> None:
    """
    Run the HCL microbench across sequence lengths and write the JSON report.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/microbench/hcl.json"),
        help="Path to write the JSON report.",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("bench_hcl requires a CUDA device")

    rows: list[dict[str, float]] = []
    report = {
        "kernel": "hcl_fft_conv + _hcl_compute_filter",
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "triton": triton.__version__,
        "platform": platform.platform(),
        "dtype": "float32",
        "hidden_size": _D,
        "state_size": _S,
        "results": rows,
    }

    for L in _SEQ_LENS:
        rows.append(_bench_one(L))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")

    print(
        f"{'seq_len':>8} {'stock_ms':>10} {'kernel_ms':>11} {'speedup':>9} "
        f"{'stock_GB':>10} {'kernel_GB':>11} {'mem_ratio':>11} {'max_diff':>10}"
    )
    for row in rows:
        print(
            f"{row['seq_len']:>8} {row['stock_ms']:>10.4f} {row['kernel_ms']:>11.4f} "
            f"{row['speedup']:>8.3f}x {row['stock_peak_gb']:>10.3f} "
            f"{row['kernel_peak_gb']:>11.3f} {row['mem_ratio']:>10.2f}x "
            f"{row['max_diff']:>10.2e}"
        )
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
