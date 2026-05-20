# Progression comparison

| GPU | Config | seq_len | forward ms | ± std | peak GB | launches | speedup vs base |
|---|---|---:|---:|---:|---:|---:|---:|
| rtx-4090 | base | 8192 | 1163.0 | 26.5 | 18.03 | 26865 | 1.00x |
| rtx-4090 | base | 32768 | 6297.6 | 22.9 | 32.58 | 28896 | 1.00x |
| rtx-4090 | +HCS | 8192 | 1152.4 | 23.2 | 18.03 | 26681 | 1.01x |
| rtx-4090 | +HCS | 32768 | 6288.9 | 5.3 | 32.58 | 28714 | 1.00x |
| rtx-4090 | +HCS+HCM | 8192 | 1153.4 | 26.4 | 18.03 | 26446 | 1.01x |
| rtx-4090 | +HCS+HCM | 32768 | 6370.8 | 3.4 | 32.58 | 28501 | 0.99x |
| rtx-4090 | +HCS+HCM+HCL | 8192 | 953.8 | 29.8 | 15.28 | 25459 | 1.22x |
| rtx-4090 | +HCS+HCM+HCL | 32768 | 4837.2 | 2.6 | 22.66 | 26082 | 1.30x |
| rtx-4090 | +HCS+HCM+HCL | 65536 | 11917.6 | 9.9 | 32.13 | 28015 | - |

## Per-GPU base -> final

### rtx-4090
- L=8192: base 1163 ms / 18.0 GB -> final 954 ms / 15.3 GB  (1.22x faster, 1.18x mem)
- L=32768: base 6298 ms / 32.6 GB -> final 4837 ms / 22.7 GB  (1.30x faster, 1.44x mem)
- L=65536: base OOM; final 11918 ms / 32.1 GB  (memory unlock)
