"""
Microbenchmark for the HCM fused FFT-conv.

Compares the fused path (`vortex.ops.hcm_interface.hcm_fft_conv`) against the
stock `fftconv_func` on the FFT convolution that defines an HCM layer's time
mixing. Both keep cuFFT for the three transforms; hcm_fft_conv fuses the
elementwise glue into two Triton kernels. Both run in fp32 at evo2_7b shapes
(D=4096, fir_length=128).

Results are written to `results/microbench/hcm.json` for commit alongside the
kernel.
"""

import argparse
import json
from collections.abc import Callable
from pathlib import Path

import torch
import triton
from vortex.model.engine import fftconv_func
from vortex.ops.hcm_interface import hcm_fft_conv

from benchmarks.meta import default_results_root, run_meta

# evo2_7b HCM layer: hidden_size 4096, hcm filter length 128.
_D: int = 4096
_FIR_LENGTH: int = 128
_SEQ_LENS: list[int] = [2048, 8192, 32768, 65536]


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
    Benchmark hcm_fft_conv and the stock fftconv_func at sequence length L.

    Args:
        L (int): Sequence length.

    Returns:
        dict[str, float | int]: Timings and peak-memory counters for one L.
    """
    torch.manual_seed(0)
    # torch.randn defaults to float32 -- the dtype the HCM FFT-conv runs in.
    u: torch.Tensor = torch.randn(1, _D, L, device="cuda")
    weight: torch.Tensor = torch.randn(_D, 1, _FIR_LENGTH, device="cuda")
    bias: torch.Tensor = torch.randn(_D, device="cuda")

    def _fused() -> torch.Tensor:
        return hcm_fft_conv(u, weight, bias, None, gelu=False, bidirectional=False)

    def _stock() -> torch.Tensor:
        return fftconv_func(u, weight, bias, None, gelu=False, bidirectional=False)

    max_diff: float = (_fused() - _stock()).abs().max().item()

    torch.cuda.reset_peak_memory_stats()
    triton_ms: float = _median_ms(_fused)
    triton_peak_bytes: int = torch.cuda.max_memory_allocated()

    torch.cuda.reset_peak_memory_stats()
    fftconv_ms: float = _median_ms(_stock)
    fftconv_peak_bytes: int = torch.cuda.max_memory_allocated()

    return {
        "seq_len": L,
        "triton_ms": round(triton_ms, 5),
        "fftconv_ms": round(fftconv_ms, 5),
        "speedup": round(fftconv_ms / triton_ms, 3),
        "triton_peak_bytes": triton_peak_bytes,
        "fftconv_peak_bytes": fftconv_peak_bytes,
        "max_diff": max_diff,
    }


def main() -> None:
    """
    Run the HCM microbench across sequence lengths and write the JSON report.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Path to write the JSON report. Defaults to "
            "results/<gpu>/microbench/hcm.json with <gpu> auto-detected."
        ),
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("bench_hcm requires a CUDA device")

    output: Path = args.output or default_results_root() / "microbench" / "hcm.json"

    rows: list[dict[str, float | int]] = [_bench_one(L) for L in _SEQ_LENS]
    report = {
        "run_meta": run_meta(
            config={
                "kernel": "hcm_fft_conv",
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

    print(f"{'seq_len':>8} {'triton_ms':>11} {'fftconv_ms':>12} {'speedup':>9} {'max_diff':>10}")
    for row in rows:
        print(
            f"{row['seq_len']:>8} {row['triton_ms']:>11.5f} {row['fftconv_ms']:>12.5f} "
            f"{row['speedup']:>8.3f}x {row['max_diff']:>10.2e}"
        )
    print(f"\nwrote {output}")


if __name__ == "__main__":
    main()
