"""
Cross-GPU, cross-config progression comparator.

Reads the combined_summary.json files under each results/<gpu>/{baseline_profile,
progression/<flagset>}/ tree and prints a unified table for the article and PR.
Picks up every GPU directory under results/ automatically and every
progression config (hcs / hcs_hcm / final). Drops a markdown table into
results/comparison.md by default.

Usage:
    pixi run python -m benchmarks.compare_progression
    pixi run python -m benchmarks.compare_progression --gpus rtx-4090 h100
    pixi run python -m benchmarks.compare_progression --output /tmp/cmp.md
"""

import argparse
import json
from pathlib import Path

from pydantic import BaseModel

from benchmarks.meta import REPO_ROOT

CONFIG_ORDER: tuple[str, ...] = ("base", "hcs", "hcs_hcm", "final")
CONFIG_LABEL: dict[str, str] = {
    "base": "base",
    "hcs": "+HCS",
    "hcs_hcm": "+HCS+HCM",
    "final": "+HCS+HCM+HCL",
}


class Row(BaseModel):
    """
    One (gpu, config, seq_len) measurement extracted from a combined_summary.

    Attributes:
        gpu (str): Filesystem-safe GPU tag (e.g., 'rtx-4090', 'h100').
        config (str): Kernel-flag set name; one of base / hcs / hcs_hcm / final.
        seq_len (int): Input sequence length in tokens.
        forward_ms (float): Mean forward CUDA time across N runs.
        forward_std (float): Sample std across N runs.
        peak_gb (float): Peak GPU memory allocated during the timed runs.
        launches (int): Total CUDA kernel launches captured by the profiler.
    """

    gpu: str
    config: str
    seq_len: int
    forward_ms: float
    forward_std: float
    peak_gb: float
    launches: int


def _load_summary(path: Path) -> list[dict] | None:
    """
    Return the 'summaries' list from a combined_summary.json, or None when
    the file does not exist.
    """
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    return payload.get("summaries", [])


def _collect(gpu_dir: Path) -> list[Row]:
    """
    Walk one results/<gpu>/ tree and return every progression row found.

    Args:
        gpu_dir (Path): A results/<gpu_slug>/ directory.

    Returns:
        list[Row]: All discovered measurements, base first then by config order.
    """
    rows: list[Row] = []
    gpu = gpu_dir.name

    sources: list[tuple[str, Path]] = [
        ("base", gpu_dir / "baseline_profile" / "combined_summary.json"),
        ("base", gpu_dir / "progression" / "base" / "combined_summary.json"),
    ]
    sources.extend(
        (cfg, gpu_dir / "progression" / cfg / "combined_summary.json")
        for cfg in ("hcs", "hcs_hcm", "final")
    )

    seen: set[tuple[str, str, int]] = set()
    for cfg, path in sources:
        summaries = _load_summary(path)
        if summaries is None:
            continue
        for s in summaries:
            key = (gpu, cfg, s["seq_len"])
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                Row(
                    gpu=gpu,
                    config=cfg,
                    seq_len=s["seq_len"],
                    forward_ms=s["forward_ms_mean"],
                    forward_std=s.get("forward_ms_std", 0.0),
                    peak_gb=s.get("peak_memory_bytes", 0) / 1e9,
                    launches=s.get("kernel_launch_count", 0),
                )
            )
    rows.sort(key=lambda r: (CONFIG_ORDER.index(r.config), r.seq_len))
    return rows


def _markdown_table(rows: list[Row]) -> str:
    """
    Format rows as a single markdown table, one row per (gpu, config, seq_len).
    """
    if not rows:
        return "_(no measurements found)_\n"

    base_lookup: dict[tuple[str, int], Row] = {
        (r.gpu, r.seq_len): r for r in rows if r.config == "base"
    }

    header = (
        "| GPU | Config | seq_len | forward ms | ± std | peak GB | launches | speedup vs base |\n"
        "|---|---|---:|---:|---:|---:|---:|---:|\n"
    )
    lines = [header]
    for r in rows:
        base = base_lookup.get((r.gpu, r.seq_len))
        speedup = f"{base.forward_ms / r.forward_ms:.2f}x" if base else "-"
        lines.append(
            f"| {r.gpu} | {CONFIG_LABEL[r.config]} | {r.seq_len} | "
            f"{r.forward_ms:.1f} | {r.forward_std:.1f} | "
            f"{r.peak_gb:.2f} | {r.launches} | {speedup} |\n"
        )
    return "".join(lines)


def _per_gpu_summary(rows: list[Row]) -> str:
    """
    Return a per-GPU base->final summary line for each seq_len.
    """
    by_gpu: dict[str, list[Row]] = {}
    for r in rows:
        by_gpu.setdefault(r.gpu, []).append(r)

    out: list[str] = []
    for gpu in sorted(by_gpu):
        gpu_rows = by_gpu[gpu]
        base = {r.seq_len: r for r in gpu_rows if r.config == "base"}
        final = {r.seq_len: r for r in gpu_rows if r.config == "final"}
        seq_lens = sorted(set(base) | set(final))
        out.append(f"\n### {gpu}\n")
        for L in seq_lens:
            b = base.get(L)
            f = final.get(L)
            if b and f:
                speedup = b.forward_ms / f.forward_ms
                mem_ratio = b.peak_gb / f.peak_gb if f.peak_gb else float("inf")
                out.append(
                    f"- L={L}: base {b.forward_ms:.0f} ms / {b.peak_gb:.1f} GB "
                    f"-> final {f.forward_ms:.0f} ms / {f.peak_gb:.1f} GB  "
                    f"({speedup:.2f}x faster, {mem_ratio:.2f}x mem)\n"
                )
            elif f and not b:
                out.append(
                    f"- L={L}: base OOM; final {f.forward_ms:.0f} ms / "
                    f"{f.peak_gb:.1f} GB  (memory unlock)\n"
                )
            elif b and not f:
                out.append(
                    f"- L={L}: base {b.forward_ms:.0f} ms / {b.peak_gb:.1f} GB; final missing\n"
                )
    return "".join(out)


def main() -> None:
    """
    Parse CLI flags, collect rows from each requested GPU, write the report.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gpus",
        nargs="*",
        default=None,
        help="GPU slugs to include (default: every results/<gpu>/ that exists).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "results" / "comparison.md",
        help="Where to write the markdown report.",
    )
    args = parser.parse_args()

    results_root = REPO_ROOT / "results"
    if args.gpus:
        gpu_dirs = [results_root / g for g in args.gpus]
    else:
        gpu_dirs = sorted(p for p in results_root.iterdir() if p.is_dir())

    all_rows: list[Row] = []
    for gpu_dir in gpu_dirs:
        if not gpu_dir.is_dir():
            print(f"  skipping {gpu_dir}: not a directory")
            continue
        rows = _collect(gpu_dir)
        if rows:
            print(f"  {gpu_dir.name}: {len(rows)} rows")
            all_rows.extend(rows)
        else:
            print(f"  {gpu_dir.name}: no combined_summary.json files found")

    md_table = _markdown_table(all_rows)
    summary = _per_gpu_summary(all_rows)

    report = f"# Progression comparison\n\n{md_table}\n## Per-GPU base -> final\n{summary}\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report)
    print(f"\nwrote {args.output}")
    print(report)


if __name__ == "__main__":
    main()
