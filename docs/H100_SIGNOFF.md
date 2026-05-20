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

# Progression matrix — push to L=131072 on every config.
# OOMs are caught gracefully per (model, seq_len) and the sweep continues;
# rows that OOM are recorded as SKIPPED rather than aborting.
SEQ_LENS="8192 32768 65536 131072"
pixi run profile --seq-lens $SEQ_LENS                          # base
pixi run profile --triton hcs --seq-lens $SEQ_LENS             # +HCS
pixi run profile --triton hcs,hcm --seq-lens $SEQ_LENS         # +HCS+HCM
pixi run profile --triton hcs,hcm,hcl --seq-lens $SEQ_LENS     # final
```

Microbenches auto-route to `results/h100/microbench/`. The profiler routes
to `results/h100/baseline_profile/` (when `--triton` is empty) or
`results/h100/progression/<flagset>/` (`base`, `hcs`, `hcs_hcm`, `final`).

`@triton.autotune`'s first compile per shape is slow — warmup absorbs it.

### Expected outcomes per row (the article story)

| Config | L=8192 | L=32k | L=65k | L=131k |
|---|---|---|---|---|
| base | runs | runs | likely runs (~65 GB) | **OOM** (32 GiB filter + model > 80 GB) |
| +HCS | same as base | same | same | same OOM |
| +HCS+HCM | same as base | same | same | same OOM |
| **final** | runs | runs | runs | **runs (the unlock)** |

The base/+HCS/+HCS+HCM OOMs at L=131k are the *desired* result: they show
the stock filter materialisation is what walls inference, and the HCL kernel
is what removes the wall. Both at the microbench level (already captured in
`bench_hcl`) and at the full-model level (this sweep).

L=262144 is intentionally skipped: even with the HCL kernel removing the
filter materialisation, per-block activation memory at D=4096, 32 blocks,
BF16 reaches ~70 GB on its own, which plus weights exceeds the H100's
80 GB. Guaranteed OOM; the L=131k row is the meaningful ceiling.

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
