# Progression comparison

Headline forward-pass times on evo2_7b, run with five timed forwards
after three warmups; CV% is `std / mean * 100` so a single column tells
the reader where the deltas sit relative to run-to-run jitter.

| GPU | Config | seq_len | forward ms | ± std | CV% | peak GB | launches | speedup vs base |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| h100 | base | 8192 | 382.2 | 7.2 | 1.9% | 18.08 | 28085 | 1.00x |
| h100 | base | 32768 | 2065.6 | 5.9 | 0.3% | 32.64 | 29087 | 1.00x |
| h100 | base | 65536 | 6653.3 | 18.0 | 0.3% | 52.04 | 35126 | 1.00x |
| h100 | +HCS | 8192 | 374.2 | 6.2 | 1.7% | 18.08 | 27804 | 1.02x |
| h100 | +HCS | 32768 | 2059.1 | 1.9 | 0.1% | 32.64 | 28906 | 1.00x |
| h100 | +HCS | 65536 | 6564.8 | 25.5 | 0.4% | 52.04 | 34930 | 1.01x |
| h100 | +HCS+HCM | 8192 | 374.4 | 9.1 | 2.4% | 18.08 | 27586 | 1.02x |
| h100 | +HCS+HCM | 32768 | 1909.4 | 2.4 | 0.1% | 32.64 | 28704 | 1.08x |
| h100 | +HCS+HCM | 65536 | 6319.7 | 19.7 | 0.3% | 52.04 | 34454 | 1.05x |
| h100 | +HCS+HCM+HCL | 8192 | 288.3 | 9.4 | 3.3% | 15.34 | 26564 | 1.33x |
| h100 | +HCS+HCM+HCL | 32768 | 1397.1 | 2.8 | 0.2% | 21.64 | 26237 | 1.48x |
| h100 | +HCS+HCM+HCL | 65536 | 3843.4 | 4.7 | 0.1% | 32.18 | 29715 | 1.73x |
| h100 | +HCS+HCM+HCL | 131072 | 12701.9 | 9.5 | 0.1% | 51.13 | 37708 | - |
| rtx-4090 | base | 8192 | 1163.0 | 26.5 | 2.3% | 18.03 | 26865 | 1.00x |
| rtx-4090 | base | 32768 | 6297.6 | 22.9 | 0.4% | 32.58 | 28896 | 1.00x |
| rtx-4090 | +HCS | 8192 | 1152.4 | 23.2 | 2.0% | 18.03 | 26681 | 1.01x |
| rtx-4090 | +HCS | 32768 | 6288.9 | 5.3 | 0.1% | 32.58 | 28714 | 1.00x |
| rtx-4090 | +HCS+HCM | 8192 | 1153.4 | 26.4 | 2.3% | 18.03 | 26446 | 1.01x |
| rtx-4090 | +HCS+HCM | 32768 | 6370.8 | 3.4 | 0.1% | 32.58 | 28501 | 0.99x [^rtx-hcm-noise] |
| rtx-4090 | +HCS+HCM+HCL | 8192 | 953.8 | 29.8 | 3.1% | 15.28 | 25459 | 1.22x |
| rtx-4090 | +HCS+HCM+HCL | 32768 | 4837.2 | 2.6 | 0.1% | 22.66 | 26082 | 1.30x |
| rtx-4090 | +HCS+HCM+HCL | 65536 | 11917.6 | 9.9 | 0.1% | 32.13 | 28015 | - |

## Per-GPU base -> final

### h100
- L=8192: base 382 ms / 18.1 GB -> final 288 ms / 15.3 GB  (1.33x faster, 1.18x mem)
- L=32768: base 2066 ms / 32.6 GB -> final 1397 ms / 21.6 GB  (1.48x faster, 1.51x mem)
- L=65536: base 6653 ms / 52.0 GB -> final 3843 ms / 32.2 GB  (1.73x faster, 1.62x mem)
- L=131072: base OOM; final 12702 ms / 51.1 GB  (memory unlock)

### rtx-4090
- L=8192: base 1163 ms / 18.0 GB -> final 954 ms / 15.3 GB  (1.22x faster, 1.18x mem)
- L=32768: base 6298 ms / 32.6 GB -> final 4837 ms / 22.7 GB  (1.30x faster, 1.44x mem)
- L=65536: base OOM; final 11918 ms / 32.1 GB  (memory unlock)

## Where the speedup comes from

HCL is responsible for ~89% of the end-to-end saving at L=65536 on
H100 (2476 of 2810 ms). HCM contributes ~9%; HCS contributes <2%.
HCS and HCM remain in the PR as correctness-preserving plumbing and
because their microbench wins (HCS: up to 2.65x vs cuDNN at L=4096;
HCM: ~1.1x vs FFTConv) are robust in isolation even when their
in-model footprint is small.

Roughly three-quarters of forward-pass CUDA time at L=8192 lives
outside the HC{S,M,L}/attn blocks — embedding, LM head, final norm.
This caps the achievable end-to-end speedup at short context and is
why the L=8192 numbers do not approach the per-kernel microbench
speedups.

## Numerical correctness

End-to-end logit agreement between the stock vortex math (all kernels
off) and the full Triton path (HCS+HCM+HCL on) is verified on H100 in
the model's native inference dtype at the same seq lengths as the
perf sweep. See `results/<gpu_slug>/correctness/correctness.json` for
the per-(model, seq_len) `max_abs_diff`, `mean_abs_diff`, and cosine
similarity of last-token logits. The same Triton kernel source is
compiled for SM 8.9 (Ada) and SM 9.0 (Hopper); the RTX-4090 rows in
the table above are perf measurements only.

The microbenches (`results/<gpu_slug>/microbench/*.json`) use fp32 as
a fixed-precision sanity floor for per-kernel attribution against the
stock reference. They are not a substitute for the end-to-end
correctness artifact above, which runs in the actual inference dtype.

## Footnotes

[^rtx-hcm-noise]: On RTX-4090 the +HCS+HCM row at L=32768 lands ~1%
above the stock baseline (+73 ms vs ±23 ms combined run-to-run sigma,
so a ~3 sigma gap rather than pure noise). Likely dispatch overhead
on Ada masks the small HCM kernel-only win at this sequence length;
the +HCS+HCM+HCL row at the same L recovers to 1.30x because HCL
dominates. HCM remains in the PR for correctness parity and because
it lifts net performance once HCL also runs.
