# H100 sign-off

The dev box (RTX 4090) proves correctness. The H100 captures the headline
numbers the article and the upstream PR quote — autotune picks different
configs per GPU arch, and H100 is the hardware Evo2 users run on.

Budget: ~1 hour at ~$3/hr on 1x H100 80GB SXM secure-cloud.

## Pod config

| | |
|---|---|
| GPU | 1x H100 80GB SXM (PCIe is fine as a fallback) |
| Tier | Secure Cloud (non-preemptible) |
| Container disk | 100 GB |
| Network volume | None — kills Python import time |
| SSH | On |

Upload your laptop SSH pubkey to the provider before booting the pod. Keys
added after boot don't take effect until Stop/Start.

## Bring-up

```bash
# From your laptop
ssh-add ~/.ssh/id_ed25519
ssh -A root@<pod-ip> -p <port>            # -A forwards your GitHub auth

# Inside the pod
nvidia-smi                                # H100-SXM5-80GB, driver 550+, CUDA 12.x
apt-get update && apt-get install -y tmux
ssh -T git@github.com                     # confirm agent forwarding reached GitHub

cd ~ && git clone <vortex-kernels url> vortex-kernels
cd ~/vortex-kernels && bash scripts/setup_vm.sh   # ~10 min; clones the fork too
source ~/.bashrc
pixi run python -c "import vortex; print(vortex.__file__)"   # editable-install check
```

## Sign-off sweep

Run inside `tmux` so a dropped SSH doesn't kill the session. Detach with
`Ctrl-b d`, reattach with `tmux a`.

```bash
tmux new -s signoff

pixi run test-gpu                                 # correctness

pixi run python -m benchmarks.bench_hcs           # microbenches
pixi run python -m benchmarks.bench_hcm
pixi run python -m benchmarks.bench_hcl

# Progression matrix.
# OOMs are caught gracefully per (model, seq_len) and the sweep continues;
# rows that OOM are recorded as SKIPPED rather than aborting. base, +HCS,
# and +HCS+HCM all OOM in HCL's compute_filter at L=131k, so stop them at
# L=65k. Only the final config exercises the higher seq_lens.
SHORT_SEQ_LENS="8192 32768 65536"
LONG_SEQ_LENS="8192 32768 65536 131072 196608 262144"

pixi run profile --seq-lens $SHORT_SEQ_LENS                       # base
pixi run profile --triton hcs --seq-lens $SHORT_SEQ_LENS          # +HCS
pixi run profile --triton hcs,hcm --seq-lens $SHORT_SEQ_LENS      # +HCS+HCM
pixi run profile --triton hcs,hcm,hcl --seq-lens $LONG_SEQ_LENS   # final (push past 131k)
```

Microbenches auto-route to `results/h100/microbench/`. The profiler routes
to `results/h100/baseline_profile/` (when `--triton` is empty) or
`results/h100/progression/<flagset>/` (`base`, `hcs`, `hcs_hcm`, `final`).

`@triton.autotune`'s first compile per shape is slow — warmup absorbs it.

### Why these seq_lens — memory-budget justification

The HCL `compute_filter` materializes a `(D=4096, state_size=16, L)` fp32
tensor. That sets a hard ceiling on the stock path: filter alone is `L * 2 ^
18` bytes, growing linearly.

| L | Filter alone | Stock peak (4090 model-level) | Stock peak (4090 microbench) | Status on H100 80 GB |
|---|---:|---:|---:|---|
| 8 192 | 1 GiB | 18.0 GB | 4.3 GB | runs |
| 32 768 | 4 GiB | 32.6 GB | 17.0 GB | runs |
| 65 536 | 8 GiB | OOM on 4090 | 34.0 GB | **last L the stock path fits** |
| 131 072 | 16 GiB | n/a | OOM in microbench (>80 GB needed) | **stock OOMs** (the unlock point) |

Reason for stopping base / +HCS / +HCS+HCM at L=65k: every one of them runs
through the same stock `compute_filter` (the HCL flag is what diverts to
the tiled kernel). At L=131k the filter materialisation alone needs 16 GiB
on top of the rest of the working set, which the microbench already proves
OOMs at 80 GB. Running them at L=131k would burn ~5 minutes per config on
guaranteed-OOM rows that add no information beyond "stock OOMs at 131k",
which the microbench has already proven.

### Why push final past 131k

The HCL kernel sidesteps the filter materialisation entirely (tiles over
L), so the final config's ceiling is set by the *next* O(L) cost — the
per-block activations stacked across 32 blocks. Extrapolating from the
4090 final-config measurements (which captured `peak_memory_bytes`):

| L | final peak (4090) | linear extrapolation to H100 model | headroom on 80 GB |
|---|---:|---:|---:|
| 8 192  | 15.3 GB | ~15 GB | 65 GB |
| 32 768 | 22.7 GB | ~23 GB | 57 GB |
| 65 536 | 32.1 GB | ~32 GB | 48 GB |
| 131 072 | — | ~50 GB (predicted) | ~30 GB |
| 196 608 | — | ~70 GB (predicted) | ~10 GB — **on the edge** |
| 262 144 | — | ~110 GB (predicted) | **OOM** |

L=196608 picked because (a) it's `3 * 2^16`, FFT-friendly for cuFFT, and
(b) the linear-extrapolation prediction lands just below the 80 GB wall —
the most informative single probe. L=262144 is included as a guaranteed
OOM marker so the report explicitly states "this is the wall, not just an
arbitrary stopping point". If 196608 runs but 262144 OOMs, we have a
defensible "single-H100 ceiling is between 196k and 262k tokens" claim.

### Expected outcomes per row (the article story)

| Config | L=8192 | L=32k | L=65k | L=131k | L=196k | L=262k |
|---|---|---|---|---|---|---|
| base | runs | runs | runs | (not run) | (not run) | (not run) |
| +HCS | runs | runs | runs | (not run) | (not run) | (not run) |
| +HCS+HCM | runs | runs | runs | (not run) | (not run) | (not run) |
| **final** | runs | runs | runs | **runs (the unlock)** | **probe — most likely runs** | **probe — most likely OOM** |

The OOM marker at L=262k *and* the success at L=196k together carry the
strongest claim: the HCL kernel takes single-H100 inference from a ~65k
token wall (stock) to ~200k+ tokens (final), an order-of-magnitude shift
in usable context length without changing GPU. If L=196608 OOMs as well,
the headline degrades gracefully to "131k is the single-H100 ceiling"
which is still 2× the stock path.

## Commit + tear-down

```bash
git add results/h100/
git commit -m "feat: add the H100 sign-off measurement sweep"
git push origin main
```

`trace_*.json` files are gitignored — they stay on the pod. Summaries,
plots, layer breakdowns, top-ops, and `report.md` are tracked.

Stop or terminate the pod in the provider UI. Terminate wipes the disk and
stops billing.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `import vortex` resolves to site-packages, not `../vortex` | Editable install lost. `pixi run pip install -e ../vortex` |
| `pixi install` fails — `../vortex` not found | Re-run `setup_vm.sh`, which clones the fork as a sibling |
| SSH prompts for password | Pubkey not registered before boot. Add it, Stop/Start |
| `git clone` → Permission denied (publickey) | Agent forwarding failed. Generate a pod-local key, add at github.com/settings/ssh/new |
| First kernel call very slow | `@triton.autotune` benchmarking configs. `TRITON_PRINT_AUTOTUNING=1` to watch |
| OOM at L=131k with HCL flag off | Expected — the bug the HCL kernel fixes |
| OOM at L=131k with HCL flag on | Filter materialisation isn't the only culprit. Profile with `torch.cuda.memory._record_memory_history()` |
