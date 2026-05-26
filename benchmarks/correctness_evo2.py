"""
End-to-end numerical correctness for the vortex-kernels HC{S,M,L} Triton path.

For each (model, seq_len) the script forwards a fixed-seed random input
twice on the same loaded checkpoint — once with all kernels OFF (stock
vortex math) and once with the requested kernel set ON — and reports:

    max_abs_diff              | bound on per-logit numerical drift
    mean_abs_diff             | average per-logit drift across the tensor
    cosine_sim_last_token     | semantic agreement at the autoregressive step
    cosine_sim_sequence_mean  | per-position cosine sim averaged over the
                                sequence (catches mid-sequence drift)
    argmax_match_rate         | fraction of positions where argmax(logits)
                                matches between stock and triton — the
                                cleanest answer to "would I sample the same
                                token here?"

The toggle mechanism mirrors `benchmarks/profile_evo2.py`: each
`HyenaInferenceEngine` exposes `use_{hcs,hcm,hcl}_kernel` flags that
branch into the Triton path when true and the stock path when false.

Artifacts under --output-dir (default: results/<gpu_slug>/correctness/, or
results/<gpu_slug>/<model>/correctness/ for non-default models):
    correctness.json   one record per (model, seq_len) plus run_meta

Usage:
    pixi run python -m benchmarks.correctness_evo2
    pixi run python -m benchmarks.correctness_evo2 --seq-lens 8192 32768
"""

import argparse
import gc
import json
from pathlib import Path
from typing import Any

import torch
from evo2 import Evo2
from vortex.model.engine import HyenaInferenceEngine

from benchmarks.meta import model_results_root, run_meta


def _apply_triton_kernels(model: Any, enabled: set[str]) -> int:
    """
    Toggle use_{hcs,hcm,hcl}_kernel on every HyenaInferenceEngine.

    Mirrors `benchmarks/profile_evo2.py::_apply_triton_kernels` so the
    correctness sweep and the perf sweep agree on what 'kernels on' means.

    Args:
        model (Any): Loaded Evo2 model.
        enabled (set[str]): Kernel names to turn on (subset of hcs/hcm/hcl).
                            Empty set means run the stock vortex math.

    Returns:
        Number of HyenaInferenceEngine instances updated.
    """
    root = getattr(model, "model", model)
    touched = 0
    for module in root.modules():
        engine = getattr(module, "engine", None)
        if isinstance(engine, HyenaInferenceEngine):
            engine.use_hcs_kernel = "hcs" in enabled
            engine.use_hcm_kernel = "hcm" in enabled
            engine.use_hcl_kernel = "hcl" in enabled
            touched += 1
    return touched


def _logits_only(out: Any) -> torch.Tensor:
    """
    Peel nested tuples from an Evo2/StripedHyena forward return.

    Evo2 wraps StripedHyena's `(logits, inference_params_dict)` return in
    its own forward signature, producing a nested tuple whose exact shape
    varies across evo2 package versions; peeling tuples until a tensor is
    reached keeps this script portable across those versions.

    Args:
        out (Any): Forward return from a loaded Evo2 model.

    Returns:
        The logits tensor.

    Raises:
        TypeError: When peeling does not terminate at a tensor.
    """
    while isinstance(out, tuple):
        out = out[0]
    if not isinstance(out, torch.Tensor):
        raise TypeError(
            f"expected logits tensor at the end of the tuple chain, got {type(out).__name__}"
        )
    return out


def _compare(stock: torch.Tensor, triton: torch.Tensor) -> dict[str, float]:
    """
    Compute the four correctness metrics between two logits tensors.

    Comparisons run in fp32 so diff bounds are independent of the model's
    inference dtype (typically bf16).

    Args:
        stock (torch.Tensor): Logits from the kernels-off pass, shape
            `(batch, seq_len, vocab)`.
        triton (torch.Tensor): Logits from the kernels-on pass, same shape.

    Returns:
        Dict with max_abs_diff, mean_abs_diff, cosine_sim_last_token,
        cosine_sim_sequence_mean, and argmax_match_rate.
    """
    assert stock.shape == triton.shape, f"shape mismatch: {stock.shape} vs {triton.shape}"
    a = stock.float()
    b = triton.float()
    diff = (a - b).abs()
    cos_last = torch.nn.functional.cosine_similarity(a[:, -1, :], b[:, -1, :], dim=-1).mean()
    cos_seq = torch.nn.functional.cosine_similarity(a, b, dim=-1).mean()
    argmax_match = (a.argmax(dim=-1) == b.argmax(dim=-1)).float().mean()
    return {
        "max_abs_diff": diff.max().item(),
        "mean_abs_diff": diff.mean().item(),
        "cosine_sim_last_token": cos_last.item(),
        "cosine_sim_sequence_mean": cos_seq.item(),
        "argmax_match_rate": argmax_match.item(),
    }


@torch.no_grad()
def check_one(
    model: Any,
    model_name: str,
    seq_len: int,
    seed: int,
    enabled: set[str],
) -> dict[str, Any]:
    """
    Forward stock vs triton for one (already-loaded model, seq_len) pair.

    Args:
        model (Any): Pre-loaded Evo2 model. Toggled in place between passes;
                     weights are not mutated.
        model_name (str): Evo2 model ID, recorded in the result.
        seq_len (int): Input sequence length in tokens.
        seed (int): Torch RNG seed for the random DNA input.
        enabled (set[str]): Kernel names to turn on for the triton pass.

    Returns:
        Comparison record ready to embed in the results JSON.
    """
    print(f"\n=== {model_name} @ seq_len={seq_len} ===", flush=True)
    torch.manual_seed(seed)
    input_ids = torch.randint(1, 5, (1, seq_len), dtype=torch.int, device="cuda:0")

    print("  pass 1/2: all kernels OFF (stock baseline)", flush=True)
    _apply_triton_kernels(model, enabled=set())
    stock_logits = _logits_only(model(input_ids)).detach()

    print(f"  pass 2/2: kernels ON = {sorted(enabled)}", flush=True)
    _apply_triton_kernels(model, enabled=enabled)
    triton_logits = _logits_only(model(input_ids)).detach()
    # Surface any kernel error at the forward boundary, not later inside _compare.
    torch.cuda.synchronize()

    cmp = _compare(stock_logits, triton_logits)
    print(
        f"  max_abs_diff={cmp['max_abs_diff']:.4e} "
        f"mean_abs_diff={cmp['mean_abs_diff']:.4e} "
        f"cos_last={cmp['cosine_sim_last_token']:.6f} "
        f"cos_seq={cmp['cosine_sim_sequence_mean']:.6f} "
        f"argmax_match={cmp['argmax_match_rate']:.6f}",
        flush=True,
    )

    return {
        "model": model_name,
        "seq_len": seq_len,
        "seed": seed,
        "enabled_kernels": sorted(enabled),
        "logits_shape": list(stock_logits.shape),
        "logits_dtype": str(stock_logits.dtype),
        **cmp,
    }


def main() -> None:
    """
    CLI entry point. See module docstring for usage examples.
    """
    parser = argparse.ArgumentParser(
        description="End-to-end numerical correctness check for vortex-kernels.",
    )
    parser.add_argument("--models", nargs="+", default=["evo2_7b"])
    parser.add_argument("--seq-lens", type=int, nargs="+", default=[8192, 32768])
    parser.add_argument("--enabled", nargs="+", default=["hcs", "hcm", "hcl"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA device required.")

    valid = {"hcs", "hcm", "hcl"}
    enabled = set(args.enabled)
    unknown = enabled - valid
    if unknown:
        raise SystemExit(
            f"--enabled: unknown kernel(s) {sorted(unknown)}; choose from {sorted(valid)}"
        )

    out_dir = args.output_dir or (model_results_root(args.models) / "correctness")
    out_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    for model_name in args.models:
        print(f"\nloading Evo2('{model_name}')...", flush=True)
        model = Evo2(model_name)
        try:
            results.extend(
                check_one(
                    model=model,
                    model_name=model_name,
                    seq_len=seq_len,
                    seed=args.seed,
                    enabled=enabled,
                )
                for seq_len in args.seq_lens
            )
        finally:
            del model
            gc.collect()
            torch.cuda.empty_cache()

    payload = {
        "run_meta": run_meta(
            config={
                "models": args.models,
                "seq_lens": args.seq_lens,
                "seed": args.seed,
                "enabled_kernels": sorted(enabled),
                "comparison": "stock_baseline_vs_kernels_on",
            },
        ),
        "results": results,
    }
    out_path = out_dir / "correctness.json"
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
