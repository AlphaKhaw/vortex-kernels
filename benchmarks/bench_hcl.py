"""
Microbenchmark for the HCL kernels, including the OOM crossover.

Compares the stock HCL conv -- compute_filter's (D, state_size, L) reduction
plus the parallel_iir FFT branch -- against the fused path: _hcl_compute_filter
(the tiled filter build) followed by hcl_fft_conv. Reports latency and, the
headline, peak GPU memory.

The stock filter build materialises a (D, state_size, L) fp32 intermediate
that grows linearly with the sequence length and eventually OOMs; the tiled
kernel never builds it. The benchmark sweeps a range of sequence lengths and
records the crossover -- the shortest L at which the stock path OOMs while the
kernel still completes. A stock path that OOMs is caught and recorded as "OOM"
rather than aborting the sweep.

Results are written to results/microbench/hcl.json for commit alongside the
kernels.
"""

import argparse
import json
from collections.abc import Callable
from pathlib import Path

import torch
import triton
from vortex.ops.hcl_interface import _hcl_compute_filter, hcl_fft_conv

from benchmarks.meta import default_results_root, run_meta

# evo2_7b HCL layer: hidden_size 4096, state_size 16.
_D: int = 4096
_S: int = 16
# Sweeps from short layers up past the stock filter build's OOM point so the
# crossover is bracketed; the kernel path is expected to survive every length.
_SEQ_LENS: list[int] = [2048, 8192, 32768, 65536, 98304, 131072]


def _median_ms(fn: Callable[[], object]) -> float:
    """
    Median wall-clock milliseconds of fn via triton.testing.do_bench.

    Args:
        fn (Callable[[], object]): Function to benchmark.

    Returns:
        float: Median wall-clock milliseconds.
    """
    return triton.testing.do_bench(fn)


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


def _measure(fn: Callable[[], torch.Tensor]) -> tuple[float, float] | None:
    """
    Measure latency and peak memory for fn, or report an OOM.

    Args:
        fn (Callable[[], torch.Tensor]): The conv path to measure.

    Returns:
        tuple[float, float] | None: (median ms, peak GiB), or None when the
        path runs out of GPU memory.
    """
    try:
        ms = _median_ms(fn)
        gb = _peak_gb(fn)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return None
    return ms, gb


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


def _bench_one(L: int) -> dict[str, object]:
    """
    Benchmark the stock HCL conv and the fused kernels at sequence length L.

    The stock path is measured inside an OOM guard: past the crossover length
    its (D, state_size, L) filter intermediate no longer fits, and the row
    records "OOM" for the stock columns while still reporting the kernel.

    Args:
        L (int): Sequence length.

    Returns:
        dict[str, object]: Latency, peak memory, the speedup/ratio where both
        paths fit, and the correctness gap.
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

    # Measure the kernel first, then free its cached blocks so the stock path
    # gets a fragmentation-free shot -- the OOM crossover must be a true
    # capacity limit, not an allocator artefact.
    kernel_stats = _measure(kernel)
    torch.cuda.empty_cache()
    stock_stats = _measure(stock)

    max_diff: float | None = None
    speedup: float | None = None
    mem_ratio: float | None = None
    if kernel_stats is not None and stock_stats is not None:
        speedup = round(stock_stats[0] / kernel_stats[0], 3)
        mem_ratio = round(stock_stats[1] / kernel_stats[1], 2)
        try:
            max_diff = (kernel() - stock()).abs().max().item()
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()

    return {
        "seq_len": L,
        "stock_ms": round(stock_stats[0], 5) if stock_stats is not None else "OOM",
        "kernel_ms": round(kernel_stats[0], 5) if kernel_stats is not None else "OOM",
        "speedup": speedup,
        "stock_peak_gb": round(stock_stats[1], 3) if stock_stats is not None else "OOM",
        "kernel_peak_gb": round(kernel_stats[1], 3) if kernel_stats is not None else "OOM",
        "mem_ratio": mem_ratio,
        "max_diff": max_diff,
    }


def _cell(value: object, width: int, numfmt: str = "") -> str:
    """
    Right-align a table cell, formatting numbers and passing sentinels through.

    Args:
        value (object): A numeric measurement, None, or a string sentinel.
        width (int): Column width.
        numfmt (str): Format spec applied to numeric values only.

    Returns:
        str: The right-aligned cell text.
    """
    if value is None:
        text = "-"
    elif isinstance(value, int | float):
        text = format(value, numfmt)
    else:
        text = str(value)
    return text.rjust(width)


def main() -> None:
    """
    Run the HCL microbench across sequence lengths and write the JSON report.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Path to write the JSON report. Defaults to "
            "results/<gpu>/microbench/hcl.json with <gpu> auto-detected."
        ),
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("bench_hcl requires a CUDA device")

    output: Path = args.output or default_results_root() / "microbench" / "hcl.json"

    rows: list[dict[str, object]] = []
    for L in _SEQ_LENS:
        rows.append(_bench_one(L))
        torch.cuda.empty_cache()

    crossover = next(
        (r["seq_len"] for r in rows if r["stock_ms"] == "OOM" and r["kernel_ms"] != "OOM"),
        None,
    )

    report = {
        "run_meta": run_meta(
            config={
                "kernel": "hcl_fft_conv + _hcl_compute_filter",
                "dtype": "float32",
                "hidden_size": _D,
                "state_size": _S,
                "seq_lens": _SEQ_LENS,
            }
        ),
        "oom_crossover_seq_len": crossover,
        "results": rows,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")

    print(
        _cell("seq_len", 8)
        + " "
        + _cell("stock_ms", 10)
        + " "
        + _cell("kernel_ms", 11)
        + " "
        + _cell("speedup", 9)
        + " "
        + _cell("stock_GB", 10)
        + " "
        + _cell("kernel_GB", 11)
        + " "
        + _cell("mem_ratio", 11)
        + " "
        + _cell("max_diff", 11)
    )
    for row in rows:
        print(
            _cell(row["seq_len"], 8)
            + " "
            + _cell(row["stock_ms"], 10, ".4f")
            + " "
            + _cell(row["kernel_ms"], 11, ".4f")
            + " "
            + _cell(row["speedup"], 9, ".3f")
            + " "
            + _cell(row["stock_peak_gb"], 10, ".3f")
            + " "
            + _cell(row["kernel_peak_gb"], 11, ".3f")
            + " "
            + _cell(row["mem_ratio"], 11, ".2f")
            + " "
            + _cell(row["max_diff"], 11, ".2e")
        )

    if crossover is not None:
        print(
            f"\nOOM crossover: the stock filter build OOMs at L={crossover}, "
            f"where the kernel still completes -- the memory unlock."
        )
    else:
        print("\nno OOM crossover in this sweep -- the stock path fit every L")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
