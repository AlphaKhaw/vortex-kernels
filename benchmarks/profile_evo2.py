"""
Profile Evo2 inference to identify kernel-level bottlenecks.

Produces:
- Chrome trace per (model, seq_len) — load in chrome://tracing
- Top-ops text table sorted by cuda_time_total
- JSON summary with category breakdown (fft / gemm / elementwise / conv / ...)

Usage:
    python benchmarks/profile_evo2.py --model evo2_7b_base --seq-lens 8192 32768
"""

import argparse
import json
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile, record_function

CATEGORIES = {
    "fft": ["aten::_fft", "aten::fft_", "cufft"],
    "conv": ["aten::conv1d", "aten::_convolution"],
    "gemm": ["aten::mm", "aten::matmul", "aten::linear", "aten::addmm", "aten::bmm"],
    "elementwise": ["aten::mul", "aten::add", "aten::gelu", "aten::silu"],
    "norm": ["aten::rms_norm", "aten::layer_norm", "aten::native_layer_norm"],
    "attention": ["aten::scaled_dot_product_attention", "flash_attn"],
}


def run_profile(
    model_name: str, seq_len: int, output_dir: Path, num_runs: int
) -> dict[str, object]:
    """
    Profile a single forward pass of Evo2 and write artifacts to output_dir.

    Args:
        model_name (str): Evo2 model ID (e.g., "evo2_7b_base").
        seq_len (int): Input sequence length in tokens.
        output_dir (Path): Directory to write chrome trace, top-ops table, and
                           JSON summary. Created if missing.
        num_runs (int): Number of timed forward passes to run after warmup.

    Returns:
        Summary dict with keys: model, seq_len, num_runs, total_cuda_ms,
                                avg_forward_ms, by_category_ms, by_category_pct.
    """
    from evo2 import Evo2  # pyright: ignore[reportMissingImports]

    print(f"\n=== {model_name} @ seq_len={seq_len} ===")
    model = Evo2(model_name)
    input_ids = torch.randint(1, 5, (1, seq_len), dtype=torch.int, device="cuda:0")

    print("warmup...")
    with torch.no_grad():
        for _ in range(2):
            _ = model(input_ids)
    torch.cuda.synchronize()

    print("profiling...")
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = output_dir / f"trace_{model_name}_L{seq_len}.json"

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_flops=True,
    ) as prof:
        for i in range(num_runs):
            with record_function(f"forward_run_{i}"), torch.no_grad():
                _ = model(input_ids)
            torch.cuda.synchronize()

    prof.export_chrome_trace(str(trace_path))
    print(f"  chrome trace: {trace_path}")

    key_avgs = prof.key_averages()
    top_table_path = output_dir / f"top_ops_{model_name}_L{seq_len}.txt"
    top_table_path.write_text(key_avgs.table(sort_by="cuda_time_total", row_limit=25))
    print(f"  top ops:      {top_table_path}")

    totals: dict[str, float] = dict.fromkeys(CATEGORIES, 0.0)
    total_cuda_us = 0.0
    for evt in key_avgs:
        total_cuda_us += evt.cuda_time_total
        for cat, patterns in CATEGORIES.items():
            if any(p in evt.key for p in patterns):
                totals[cat] += evt.cuda_time_total
                break

    by_category_ms = {k: v / 1000 for k, v in totals.items()}
    by_category_pct = (
        {k: 100 * v / total_cuda_us for k, v in totals.items()} if total_cuda_us > 0 else {}
    )
    summary: dict[str, object] = {
        "model": model_name,
        "seq_len": seq_len,
        "num_runs": num_runs,
        "total_cuda_ms": total_cuda_us / 1000,
        "avg_forward_ms": total_cuda_us / 1000 / num_runs,
        "by_category_ms": by_category_ms,
        "by_category_pct": by_category_pct,
    }

    (output_dir / f"summary_{model_name}_L{seq_len}.json").write_text(json.dumps(summary, indent=2))

    print(
        f"  total CUDA: {summary['total_cuda_ms']:.1f} ms "
        f"/ {summary['avg_forward_ms']:.1f} ms per forward"
    )
    for cat, pct in sorted(by_category_pct.items(), key=lambda x: -x[1]):
        print(f"    {cat:12s} {pct:5.1f}%  ({by_category_ms[cat]:.1f} ms)")
    return summary


def main() -> None:
    """
    Parse CLI arguments and profile Evo2 across every requested sequence length.

    Writes a combined_summary.json in the output directory aggregating the
    per-seq-length summaries returned by run_profile. Individual chrome traces
    and top-ops tables are written alongside.
    """
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model",
        default="evo2_7b_base",
        help="evo2_7b_base is safe on non-H100; evo2_7b may need TE",
    )
    p.add_argument("--seq-lens", nargs="+", type=int, default=[8192, 32768])
    p.add_argument("--output-dir", default="results/baseline_profile")
    p.add_argument("--num-runs", type=int, default=3)
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    summaries = [run_profile(args.model, sl, out_dir, args.num_runs) for sl in args.seq_lens]
    (out_dir / "combined_summary.json").write_text(json.dumps(summaries, indent=2))
    print("\ndone. load chrome traces at chrome://tracing")


if __name__ == "__main__":
    main()
