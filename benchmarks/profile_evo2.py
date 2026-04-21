"""
Profile Evo2 inference with per-layer attribution, op-category breakdown,
per-run timing statistics, and publication-ready plots.

Two orthogonal views are produced for every (model, seq_len):

1. Per-layer view — every top-level block's forward is wrapped in a named
   `record_function` scope before profiling, so the chrome trace and the
   aggregated key_averages contain one entry per (layer_idx, kind). Blocks
   are classified as hcl / hcm / hcs / attn / other; any block that lands
   in 'other' is surfaced loudly with its class name so the classifier can
   be extended. No layer is silently dropped.

2. Op-category view — leaf aten ops are bucketed into
   fft / conv / gemm / attention / norm / elementwise / cast_copy / reshape
   / other. The 'other' bucket is always reported so percentages sum to 100%.

Artifacts per (model, seq_len) under --output-dir:
    trace_<m>_L<sl>.json             chrome trace (load at chrome://tracing)
    top_ops_<m>_L<sl>.txt            top 40 aten ops by cuda_time_total
    layer_breakdown_<m>_L<sl>.json   per-layer classification + cuda_ms
    summary_<m>_L<sl>.json           full RunSummary (all numbers below)
    plots/per_layer_<m>_L<sl>.png
    plots/layer_kinds_<m>_L<sl>.png
    plots/op_categories_<m>_L<sl>.png

Aggregate artifacts under --output-dir:
    combined_summary.json
    plots/stacked_by_layer_kind.png
    report.md

Usage:
    # Pixi (Lambda — the default environment already includes the bench feature
    # so matplotlib + evo2 + CUDA deps are all present):
    pixi run profile
    pixi run profile --seq-lens 4096 16384 65536

    # uv equivalent (sync extras first so matplotlib + evo2 are installed):
    uv sync --extra bench --extra evo2
    uv run python benchmarks/profile_evo2.py --models evo2_7b_base --seq-lens 8192 32768

    # Sweep multiple variants of the same parameter size in one invocation:
    pixi run profile --models evo2_7b_base evo2_7b --seq-lens 8192 32768

This script imports evo2 and matplotlib at module load — it is not inspectable
on hosts without those packages installed. That is intentional: the script
only runs on a CUDA-equipped profiling host (Lambda A100 / H100 in practice).
"""

import argparse
import json
import os
import statistics
from collections import defaultdict
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt  # pyright: ignore[reportMissingImports]
import torch
import torch.nn as nn
from evo2 import Evo2  # pyright: ignore[reportMissingImports]
from torch.profiler import ProfilerActivity, profile, record_function

OP_CATEGORIES: dict[str, tuple[str, ...]] = {
    "fft": ("aten::_fft", "aten::fft_", "cufft"),
    "conv": ("aten::conv1d", "aten::_convolution", "cudnn_conv"),
    "gemm": (
        "aten::mm",
        "aten::matmul",
        "aten::linear",
        "aten::addmm",
        "aten::bmm",
        "gemm",
        "cutlass",
    ),
    "attention": ("aten::scaled_dot_product_attention", "flash_attn", "_flash"),
    "norm": (
        "aten::rms_norm",
        "aten::layer_norm",
        "aten::native_layer_norm",
        "aten::group_norm",
    ),
    "elementwise": (
        "aten::mul",
        "aten::add",
        "aten::gelu",
        "aten::silu",
        "aten::div",
        "aten::sub",
        "aten::sqrt",
        "aten::rsqrt",
        "aten::neg",
        "aten::exp",
    ),
    "cast_copy": (
        "aten::to",
        "aten::_to_copy",
        "aten::copy_",
        "aten::contiguous",
        "cudaMemcpy",
    ),
    "reshape": (
        "aten::view",
        "aten::reshape",
        "aten::permute",
        "aten::transpose",
        "aten::cat",
        "aten::split",
        "aten::slice",
    ),
}

LAYER_KINDS: tuple[str, ...] = ("hcl", "hcm", "hcs", "attn", "other")

LAYER_COLORS: dict[str, str] = {
    "hcl": "#1f77b4",
    "hcm": "#d62728",
    "hcs": "#2ca02c",
    "attn": "#ff7f0e",
    "other": "#7f7f7f",
}

LAYER_LABEL_PREFIX = "vk_layer"


@dataclass
class LayerInfo:
    """
    Per-block record produced by the classifier + profiler.

    Attributes:
        idx (int): Block index in the model's block list.
        kind (str): One of hcl / hcm / hcs / attn / other.
        class_name (str): Python class name of the block, used for audit and
                          for extending the classifier when a block lands in 'other'.
        cuda_ms (float): CUDA time attributed to this block across all
                         profiled runs (filled after profiling).
    """

    idx: int
    kind: str
    class_name: str
    cuda_ms: float = 0.0


@dataclass
class RunSummary:
    """
    Everything we know after profiling one (model, seq_len) pair.

    Attributes:
        model (str): Evo2 model ID (e.g., "evo2_7b_base").
        seq_len (int): Input sequence length in tokens.
        num_runs (int): Number of timed forward passes.
        warmup (int): Number of untimed forward passes before profiling.
        total_cuda_ms (float): Leaf-op CUDA time summed across all runs.
        forward_ms_mean (float): Per-forward CUDA time, mean across runs.
        forward_ms_std (float): Per-forward CUDA time, sample std across runs.
        by_category_ms (dict[str, float]): CUDA ms per op category.
        by_category_pct (dict[str, float]): Percent-of-total per op category.
        by_layer_kind_ms (dict[str, float]): CUDA ms per layer kind.
        by_layer_kind_pct (dict[str, float]): Percent-of-total per layer kind.
        per_layer (list[dict]): Per-block records (LayerInfo as dicts).
        unmatched_cuda_ms (float): total_cuda_ms minus sum of block-attributed
                                   time — captures work outside any block
                                   (embedding, lm_head, final norm).
                                   Surfaced so readers can judge coverage.
        unknown_block_classes (list[str]): Class names of blocks that the
                                           classifier labelled 'other'; empty when
                                           classification is clean.
        artifacts (dict[str, str]): Named file paths emitted by this run.
    """

    model: str
    seq_len: int
    num_runs: int
    warmup: int
    total_cuda_ms: float
    forward_ms_mean: float
    forward_ms_std: float
    by_category_ms: dict[str, float]
    by_category_pct: dict[str, float]
    by_layer_kind_ms: dict[str, float]
    by_layer_kind_pct: dict[str, float]
    per_layer: list[dict[str, Any]]
    unmatched_cuda_ms: float
    unknown_block_classes: list[str]
    artifacts: dict[str, str] = field(default_factory=dict)


def _find_blocks(model: Any) -> tuple[list[nn.Module], str]:
    """
    Locate the flat list of top-level blocks inside an Evo2 model.

    Tries several known attribute paths; returns the first non-empty match.
    The exact path differs across vortex versions, so fall through several
    candidates rather than hard-coding one.

    Returns:
        (blocks, path_used) — `blocks` is the list of nn.Module; `path_used`
        is a human-readable attribute chain for logging.

    Raises:
        RuntimeError: If no candidate path yields a non-empty block list.
    """
    candidates: list[tuple[str, Callable[[Any], Any]]] = [
        ("model.backbone.blocks", lambda m: m.model.backbone.blocks),
        ("model.blocks", lambda m: m.model.blocks),
        ("backbone.blocks", lambda m: m.backbone.blocks),
        ("blocks", lambda m: m.blocks),
    ]
    for path, accessor in candidates:
        try:
            blocks = accessor(model)
        except AttributeError:
            continue
        if isinstance(blocks, list | nn.ModuleList) and len(blocks) > 0:
            return list(blocks), path
    raise RuntimeError(
        "Could not locate block list. Tried: "
        + ", ".join(p for p, _ in candidates)
        + f". Top-level model type: {type(model).__name__}. "
        "Inspect the model and extend _find_blocks()."
    )


def _classify_block(block: nn.Module) -> str:
    """
    Classify a block using the StripedHyena model config.

    Vortex's StripedHyena stores the authoritative layer-index lists on
    every block via `block.config`. Each block also carries its own
    `block.layer_idx`. Match the index against each list:

        block.config['hcs_layer_idxs']   -> hcs
        block.config['hcm_layer_idxs']   -> hcm
        block.config['hcl_layer_idxs']   -> hcl
        block.config['attn_layer_idxs']  -> attn

    For non-vortex models that happen to reuse this profiler, fall back to
    class-name heuristics at the end.

    Args:
        block (nn.Module): One entry from the model's block list.
    """
    layer_idx = getattr(block, "layer_idx", None)
    config = getattr(block, "config", None)
    if isinstance(layer_idx, int) and isinstance(config, dict):
        if layer_idx in config.get("hcs_layer_idxs", []):
            return "hcs"
        if layer_idx in config.get("hcm_layer_idxs", []):
            return "hcm"
        if layer_idx in config.get("hcl_layer_idxs", []):
            return "hcl"
        if layer_idx in config.get("attn_layer_idxs", []):
            return "attn"

    # Fallback for non-vortex models: class-name heuristics.
    cls = type(block).__name__.lower()
    if "attention" in cls or "mha" in cls:
        return "attn"
    if "short" in cls or "_se" in cls:
        return "hcs"
    if "medium" in cls or "_mr" in cls:
        return "hcm"
    if "long" in cls or "hyena" in cls:
        return "hcl"
    return "other"


def _dump_block_attrs(block: nn.Module, max_depth: int = 2) -> None:
    """
    Print a block's non-private, non-callable attributes for classifier debugging.

    Recurses into `filter` / `mixer` sub-modules up to max_depth so we can see
    what the operator sub-module exposes. Only runs when a block falls into
    'other' — gives a fast diagnostic loop for extending _classify_block.
    """

    def _walk(obj: Any, prefix: str, depth: int) -> None:
        for attr in sorted(a for a in dir(obj) if not a.startswith("_")):
            try:
                val = getattr(obj, attr)
            except Exception:
                continue
            if callable(val) and not isinstance(val, nn.Module):
                continue
            if isinstance(val, nn.Module):
                if depth < max_depth:
                    print(f"    {prefix}.{attr} ({type(val).__name__}):", flush=True)
                    _walk(val, f"{prefix}.{attr}", depth + 1)
                else:
                    print(f"    {prefix}.{attr} ({type(val).__name__})", flush=True)
                continue
            s = repr(val)
            if len(s) > 120:
                s = s[:120] + "..."
            print(f"    {prefix}.{attr} = {s}", flush=True)

    print(f"  DEBUG: probing first unclassified block ({type(block).__name__}):", flush=True)
    _walk(block, "block", 0)


def _wrap_block_forwards(
    blocks: list[nn.Module],
) -> tuple[list[LayerInfo], list[Callable[[], None]]]:
    """
    Wrap each block's forward in a named record_function scope.

    Each wrapped forward emits a profiler event whose key is
    `vk_layer:<idx>:<kind>` — that key is later looked up in
    `prof.key_averages()` to attribute CUDA time to the block.

    Returns:
        (infos, undos) — `infos` is one LayerInfo per block (cuda_ms filled
        later); `undos` is a list of zero-arg callables that restore the
        original forward methods. Always invoke the undos in a `finally`
        block so that the model is returned to its original state even on
        profiling failure.
    """
    infos: list[LayerInfo] = []
    undos: list[Callable[[], None]] = []
    for idx, block in enumerate(blocks):
        kind = _classify_block(block)
        infos.append(LayerInfo(idx=idx, kind=kind, class_name=type(block).__name__))

        original_forward = block.forward
        label = f"{LAYER_LABEL_PREFIX}:{idx:02d}:{kind}"

        def _wrapped(*args: Any, _orig: Any = original_forward, _lbl: str = label, **kwargs: Any):
            """
            Call the original block forward inside a named record_function scope.

            _orig and _lbl are default-arg-captured at closure-creation time to
            avoid the classic late-binding bug when this factory runs in a loop.
            """
            with record_function(_lbl):
                return _orig(*args, **kwargs)

        block.forward = _wrapped  # type: ignore[method-assign]
        undos.append(lambda b=block, o=original_forward: setattr(b, "forward", o))
    return infos, undos


def _evt_cuda_us(evt: Any) -> float:
    """
    Return an event's GPU time in microseconds, tolerating API rename.

    PyTorch ≥2.7 renamed FunctionEventAvg.cuda_time_total to device_time_total
    (part of the multi-accelerator rework — same attribute also covers XPU,
    MPS, etc.). The old name was removed, not aliased. Probe both so the
    script works across torch 2.6–2.8.
    """
    val = getattr(evt, "device_time_total", None)
    if val is None:
        val = getattr(evt, "cuda_time_total", 0.0)
    return float(val)


def _extract_per_layer_times(prof: Any, infos: list[LayerInfo]) -> None:
    """
    Fill each LayerInfo.cuda_ms from the profiler's record_function events.

    The record_function scope aggregates GPU time of every op launched
    inside it, so its device-time total is the block's full GPU time
    across all profiled runs.
    """
    key_avgs = prof.key_averages()
    for info in infos:
        label = f"{LAYER_LABEL_PREFIX}:{info.idx:02d}:{info.kind}"
        for evt in key_avgs:
            if evt.key == label:
                info.cuda_ms = _evt_cuda_us(evt) / 1000.0
                break


def _extract_per_run_ms(prof: Any, num_runs: int) -> list[float]:
    """
    Extract GPU time of each `forward_run_i` scope in ms.
    """
    key_avgs = prof.key_averages()
    per_run: list[float] = []
    for i in range(num_runs):
        label = f"forward_run_{i}"
        for evt in key_avgs:
            if evt.key == label:
                per_run.append(_evt_cuda_us(evt) / 1000.0)
                break
    return per_run


def _categorize_ops(prof: Any) -> tuple[dict[str, float], float]:
    """
    Bucket leaf aten ops into OP_CATEGORIES + 'other'.

    Skips our own record_function scopes (which would otherwise double-count
    their own children). Every unmatched op falls into 'other' so the
    percentages sum to 100%.

    Returns:
        (by_category_ms, total_cuda_ms).
    """
    totals: dict[str, float] = defaultdict(float)
    total_us = 0.0
    for evt in prof.key_averages():
        if evt.key.startswith(LAYER_LABEL_PREFIX) or evt.key.startswith("forward_run_"):
            continue
        evt_us = _evt_cuda_us(evt)
        total_us += evt_us
        matched = False
        for category, patterns in OP_CATEGORIES.items():
            if any(p in evt.key for p in patterns):
                totals[category] += evt_us
                matched = True
                break
        if not matched:
            totals["other"] += evt_us
    return {k: v / 1000.0 for k, v in totals.items()}, total_us / 1000.0


def _render_plots(summary: RunSummary, plots_dir: Path) -> dict[str, str]:
    """
    Render per-run plots. matplotlib is a hard dependency at module load.

    Three figures per run:
      - per_layer: one bar per block, coloured by kind
      - layer_kinds: aggregate CUDA ms per kind
      - op_categories: horizontal bar, sorted by ms
    """
    plots_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{summary.model}_L{summary.seq_len}"
    paths: dict[str, str] = {}

    fig, ax = plt.subplots(figsize=(12, 4))
    xs = [li["idx"] for li in summary.per_layer]
    ys = [li["cuda_ms"] for li in summary.per_layer]
    colors = [LAYER_COLORS.get(li["kind"], "#999999") for li in summary.per_layer]
    ax.bar(xs, ys, color=colors)
    ax.set_xlabel("Layer index")
    ax.set_ylabel("CUDA time (ms, summed over runs)")
    ax.set_title(f"{summary.model} @ seq_len={summary.seq_len} — per-layer CUDA time")
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=c, label=k) for k, c in LAYER_COLORS.items()
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=9)
    per_layer_path = plots_dir / f"per_layer_{tag}.png"
    fig.tight_layout()
    fig.savefig(per_layer_path, dpi=120)
    plt.close(fig)
    paths["per_layer_plot"] = str(per_layer_path)

    fig, ax = plt.subplots(figsize=(6, 4))
    kinds_present = [k for k in LAYER_KINDS if summary.by_layer_kind_ms.get(k, 0.0) > 0]
    values = [summary.by_layer_kind_ms[k] for k in kinds_present]
    ax.bar(kinds_present, values, color=[LAYER_COLORS[k] for k in kinds_present])
    for i, v in enumerate(values):
        pct = summary.by_layer_kind_pct.get(kinds_present[i], 0.0)
        ax.text(i, v, f"{v:.1f} ms\n{pct:.1f}%", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("CUDA time (ms)")
    ax.set_title(f"{summary.model} @ seq_len={summary.seq_len} — layer-kind totals")
    kinds_path = plots_dir / f"layer_kinds_{tag}.png"
    fig.tight_layout()
    fig.savefig(kinds_path, dpi=120)
    plt.close(fig)
    paths["layer_kinds_plot"] = str(kinds_path)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    sorted_cats = sorted(summary.by_category_ms.items(), key=lambda kv: -kv[1])
    cats = [c for c, _ in sorted_cats]
    vals = [v for _, v in sorted_cats]
    ax.barh(cats, vals, color="steelblue")
    for i, v in enumerate(vals):
        pct = summary.by_category_pct.get(cats[i], 0.0)
        ax.text(v, i, f" {v:.1f} ms ({pct:.1f}%)", va="center", fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("CUDA time (ms)")
    ax.set_title(f"{summary.model} @ seq_len={summary.seq_len} — op categories")
    cats_path = plots_dir / f"op_categories_{tag}.png"
    fig.tight_layout()
    fig.savefig(cats_path, dpi=120)
    plt.close(fig)
    paths["op_categories_plot"] = str(cats_path)

    return paths


def _render_combined_plot(summaries: list[RunSummary], plots_dir: Path) -> str | None:
    """
    Render a stacked bar across all (model, seq_len) runs, stacked by layer kind.

    Returns the plot path, or None if the input list is empty.
    """
    if not summaries:
        return None

    plots_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(max(6.0, 2.0 * len(summaries)), 5))
    labels = [f"{s.model}\nL={s.seq_len}" for s in summaries]
    bottoms = [0.0] * len(summaries)
    for kind in LAYER_KINDS:
        vals = [s.by_layer_kind_ms.get(kind, 0.0) for s in summaries]
        if sum(vals) == 0:
            continue
        ax.bar(labels, vals, bottom=bottoms, color=LAYER_COLORS[kind], label=kind)
        bottoms = [b + v for b, v in zip(bottoms, vals, strict=True)]
    ax.set_ylabel("CUDA time (ms, summed over runs)")
    ax.set_title("Evo2 inference CUDA time — by layer kind")
    ax.legend()
    path = plots_dir / "stacked_by_layer_kind.png"
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return str(path)


def _write_report(summaries: list[RunSummary], output_dir: Path) -> Path:
    """
    Write a human-readable markdown report summarising every run.
    """
    lines: list[str] = ["# Evo2 baseline profile", ""]
    lines.append(f"Profiled {len(summaries)} run(s). CUDA times are summed across runs.")
    lines.append("")

    for s in summaries:
        lines.append(f"## {s.model} @ seq_len={s.seq_len}")
        lines.append("")
        lines.append(
            f"- Forward pass: **{s.forward_ms_mean:.2f} ± {s.forward_ms_std:.2f} ms** "
            f"(n={s.num_runs} runs, warmup={s.warmup})"
        )
        lines.append(f"- Total leaf-op CUDA time: {s.total_cuda_ms:.1f} ms")
        attributed = sum(s.by_layer_kind_ms.values())
        lines.append(
            f"- Block coverage: {attributed:.1f} ms inside blocks, "
            f"{s.unmatched_cuda_ms:.1f} ms outside (embedding / lm_head / final norm)"
        )
        if s.unknown_block_classes:
            lines.append(
                f"- ⚠ Unknown block classes (classifier fell through): "
                f"{', '.join(sorted(set(s.unknown_block_classes)))}"
            )
        lines.append("")
        lines.append("### By layer kind")
        lines.append("")
        lines.append("| Kind | CUDA ms | % of total |")
        lines.append("|---|---:|---:|")
        for k in LAYER_KINDS:
            ms = s.by_layer_kind_ms.get(k, 0.0)
            if ms == 0:
                continue
            lines.append(f"| {k} | {ms:.1f} | {s.by_layer_kind_pct.get(k, 0.0):.1f}% |")
        lines.append("")
        lines.append("### By op category")
        lines.append("")
        lines.append("| Category | CUDA ms | % of total |")
        lines.append("|---|---:|---:|")
        for c, ms in sorted(s.by_category_ms.items(), key=lambda kv: -kv[1]):
            lines.append(f"| {c} | {ms:.1f} | {s.by_category_pct.get(c, 0.0):.1f}% |")
        lines.append("")
        lines.append("### Artifacts")
        lines.append("")
        for name, path in s.artifacts.items():
            lines.append(f"- `{name}`: {path}")
        lines.append("")

    path = output_dir / "report.md"
    path.write_text("\n".join(lines) + "\n")
    return path


def run_profile(
    model_name: str,
    seq_len: int,
    output_dir: Path,
    num_runs: int,
    warmup: int,
) -> RunSummary:
    """
    Profile a single (model, seq_len) pair end-to-end.

    Wraps every block's forward in a named record_function scope, runs
    warmup + timed forwards under the torch profiler, writes all artifacts
    to `output_dir`, and returns a fully populated RunSummary.

    Args:
        model_name (str): Evo2 model ID (e.g., "evo2_7b_base").
        seq_len (int): Input sequence length in tokens.
        output_dir (Path): Output directory; created if missing.
        num_runs (int): Number of timed forward passes.
        warmup (int): Number of untimed warmup forward passes.

    Returns:
        Populated RunSummary including every artifact path.
    """
    print(f"\n=== {model_name} @ seq_len={seq_len} ===", flush=True)
    print(
        f"  loading Evo2('{model_name}') — downloads ~14 GB on first call, "
        "then cached in HF_HOME...",
        flush=True,
    )
    model = Evo2(model_name)
    print("  model loaded; building input tensor...", flush=True)
    device = torch.device("cuda:0")
    input_ids = torch.randint(1, 5, (1, seq_len), dtype=torch.int, device=device)

    blocks, blocks_path = _find_blocks(model)
    infos, undos = _wrap_block_forwards(blocks)
    print(f"  blocks at {blocks_path}: {len(blocks)}")
    kind_counts: dict[str, int] = defaultdict(int)
    for info in infos:
        kind_counts[info.kind] += 1
    print(f"  kinds: {dict(kind_counts)}")
    unknown_classes = [info.class_name for info in infos if info.kind == "other"]
    if unknown_classes:
        print(
            f"  WARN: {len(unknown_classes)} block(s) classified as 'other': "
            f"{sorted(set(unknown_classes))}",
            flush=True,
        )
        print("         Extend _classify_block() in benchmarks/profile_evo2.py.", flush=True)
        first_other = next(
            (b for b, i in zip(blocks, infos, strict=True) if i.kind == "other"), None
        )
        if first_other is not None:
            _dump_block_attrs(first_other)

    # Force a one-block structure dump when the VK_DUMP_BLOCKS env var is set.
    # This is a diagnostic escape hatch for cases where the classifier returns
    # a single kind for every block (e.g. all 'hcl') — no WARN fires, so the
    # normal unknown-class path above doesn't help. Run with
    # `VK_DUMP_BLOCKS=1 pixi run profile ...` to emit the tree.
    if os.environ.get("VK_DUMP_BLOCKS") and blocks:
        first_non_attn = next(
            (b for b, i in zip(blocks, infos, strict=True) if i.kind != "attn"),
            blocks[0],
        )
        print(
            f"  VK_DUMP_BLOCKS: structure of first non-attn block (kind='{_classify_block(first_non_attn)}'):",
            flush=True,
        )
        _dump_block_attrs(first_non_attn)

    try:
        print(f"  warmup ({warmup} runs)...")
        with torch.no_grad():
            for _ in range(warmup):
                _ = model(input_ids)
        torch.cuda.synchronize()

        output_dir.mkdir(parents=True, exist_ok=True)
        trace_path = output_dir / f"trace_{model_name}_L{seq_len}.json"

        print(f"  profiling ({num_runs} runs)...")
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
        ) as prof:
            for i in range(num_runs):
                with record_function(f"forward_run_{i}"), torch.no_grad():
                    _ = model(input_ids)
                torch.cuda.synchronize()

        prof.export_chrome_trace(str(trace_path))
        top_ops_path = output_dir / f"top_ops_{model_name}_L{seq_len}.txt"
        key_avgs = prof.key_averages()
        try:
            top_ops_str = key_avgs.table(sort_by="device_time_total", row_limit=40)
        except (ValueError, KeyError, AssertionError):
            top_ops_str = key_avgs.table(sort_by="cuda_time_total", row_limit=40)
        top_ops_path.write_text(top_ops_str)

        _extract_per_layer_times(prof, infos)
        per_run_ms = _extract_per_run_ms(prof, num_runs)
        by_category_ms, total_cuda_ms = _categorize_ops(prof)
        by_category_pct = (
            {k: 100.0 * v / total_cuda_ms for k, v in by_category_ms.items()}
            if total_cuda_ms > 0
            else {}
        )

        by_layer_kind_ms: dict[str, float] = dict.fromkeys(LAYER_KINDS, 0.0)
        for info in infos:
            by_layer_kind_ms[info.kind] += info.cuda_ms
        attributed_ms = sum(by_layer_kind_ms.values())
        by_layer_kind_pct = (
            {k: 100.0 * v / total_cuda_ms for k, v in by_layer_kind_ms.items()}
            if total_cuda_ms > 0
            else {}
        )

        layer_breakdown_path = output_dir / f"layer_breakdown_{model_name}_L{seq_len}.json"
        layer_breakdown_path.write_text(json.dumps([asdict(info) for info in infos], indent=2))

        forward_ms_mean = statistics.fmean(per_run_ms) if per_run_ms else 0.0
        forward_ms_std = statistics.stdev(per_run_ms) if len(per_run_ms) > 1 else 0.0

        summary = RunSummary(
            model=model_name,
            seq_len=seq_len,
            num_runs=num_runs,
            warmup=warmup,
            total_cuda_ms=total_cuda_ms,
            forward_ms_mean=forward_ms_mean,
            forward_ms_std=forward_ms_std,
            by_category_ms=by_category_ms,
            by_category_pct=by_category_pct,
            by_layer_kind_ms=by_layer_kind_ms,
            by_layer_kind_pct=by_layer_kind_pct,
            per_layer=[asdict(info) for info in infos],
            unmatched_cuda_ms=max(total_cuda_ms - attributed_ms, 0.0),
            unknown_block_classes=unknown_classes,
            artifacts={
                "chrome_trace": str(trace_path),
                "top_ops": str(top_ops_path),
                "layer_breakdown": str(layer_breakdown_path),
            },
        )

        plots_dir = output_dir / "plots"
        summary.artifacts.update(_render_plots(summary, plots_dir))

        summary_path = output_dir / f"summary_{model_name}_L{seq_len}.json"
        summary_path.write_text(json.dumps(asdict(summary), indent=2))
        summary.artifacts["summary"] = str(summary_path)

        print(f"  forward: {forward_ms_mean:.1f} ± {forward_ms_std:.1f} ms")
        print(
            f"  total CUDA: {total_cuda_ms:.1f} ms  "
            f"(blocks: {attributed_ms:.1f} ms, outside: {summary.unmatched_cuda_ms:.1f} ms)"
        )
        print("  by layer kind:")
        for k in LAYER_KINDS:
            ms = by_layer_kind_ms.get(k, 0.0)
            if ms > 0:
                print(f"    {k:5s} {ms:8.1f} ms  {by_layer_kind_pct.get(k, 0.0):5.1f}%")
        print("  by op category:")
        for c, ms in sorted(by_category_ms.items(), key=lambda kv: -kv[1]):
            print(f"    {c:12s} {ms:8.1f} ms  {by_category_pct.get(c, 0.0):5.1f}%")

        return summary
    finally:
        for undo in undos:
            undo()


def main() -> None:
    """
    Parse CLI arguments and profile the Cartesian product of models × seq_lens.

    Writes per-(model, seq_len) artifacts via run_profile, then aggregates
    into combined_summary.json, a stacked plot across all runs, and a
    markdown report.
    """
    # This print runs only after every top-level import has finished — on a
    # cold network volume that can be 2-5 min. Seeing this line means the
    # heavy imports are done and argparse is about to run.
    print("profile_evo2: module imports complete, parsing args...", flush=True)
    p = argparse.ArgumentParser()
    p.add_argument(
        "--models",
        nargs="+",
        default=["evo2_7b_base"],
        help=(
            "One or more Evo2 model IDs to sweep. evo2_7b_base is the safest "
            "default (bf16, any Ampere+). evo2_7b uses transformer_engine and "
            "is optimised for H100 FP8; it runs on A100 in bf16 mode too."
        ),
    )
    p.add_argument("--seq-lens", nargs="+", type=int, default=[8192, 32768])
    p.add_argument("--output-dir", default="results/baseline_profile")
    p.add_argument("--num-runs", type=int, default=5)
    p.add_argument("--warmup", type=int, default=3)
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    summaries: list[RunSummary] = [
        run_profile(model_name, seq_len, out_dir, args.num_runs, args.warmup)
        for model_name in args.models
        for seq_len in args.seq_lens
    ]

    (out_dir / "combined_summary.json").write_text(
        json.dumps([asdict(s) for s in summaries], indent=2)
    )
    combined_plot = _render_combined_plot(summaries, out_dir / "plots")
    report_path = _write_report(summaries, out_dir)

    print(f"\nreport:          {report_path}")
    if combined_plot:
        print(f"combined plot:   {combined_plot}")
    print("chrome traces:   load at chrome://tracing")


if __name__ == "__main__":
    main()
