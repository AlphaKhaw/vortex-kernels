# Evo2 baseline profile

Profiled 2 run(s). CUDA times are summed across runs.

## evo2_7b @ seq_len=8192

- Forward pass: **1162.97 ± 26.46 ms** (n=5 runs, warmup=3)
- Total leaf-op CUDA time: 22331.4 ms
- Block coverage: 5809.2 ms inside blocks, 16522.2 ms outside (embedding / lm_head / final norm)

### By layer kind

| Kind | CUDA ms | % of total |
|---|---:|---:|
| hcl | 2307.0 | 10.3% |
| hcm | 1503.2 | 6.7% |
| hcs | 1244.1 | 5.6% |
| attn | 754.9 | 3.4% |

### By op category

| Category | CUDA ms | % of total |
|---|---:|---:|
| gemm | 11402.5 | 51.1% |
| other | 5900.8 | 26.4% |
| cast_copy | 1666.7 | 7.5% |
| elementwise | 1188.2 | 5.3% |
| conv | 1170.3 | 5.2% |
| fft | 825.2 | 3.7% |
| attention | 104.2 | 0.5% |
| reshape | 73.3 | 0.3% |

### Artifacts

- `chrome_trace`: /home/alpha/Projects/vortex-kernels/results/rtx-4090/baseline_profile/trace_evo2_7b_L8192.json
- `top_ops`: /home/alpha/Projects/vortex-kernels/results/rtx-4090/baseline_profile/top_ops_evo2_7b_L8192.txt
- `layer_breakdown`: /home/alpha/Projects/vortex-kernels/results/rtx-4090/baseline_profile/layer_breakdown_evo2_7b_L8192.json
- `per_layer_plot`: /home/alpha/Projects/vortex-kernels/results/rtx-4090/baseline_profile/plots/per_layer_evo2_7b_L8192.png
- `layer_kinds_plot`: /home/alpha/Projects/vortex-kernels/results/rtx-4090/baseline_profile/plots/layer_kinds_evo2_7b_L8192.png
- `op_categories_plot`: /home/alpha/Projects/vortex-kernels/results/rtx-4090/baseline_profile/plots/op_categories_evo2_7b_L8192.png
- `summary`: /home/alpha/Projects/vortex-kernels/results/rtx-4090/baseline_profile/summary_evo2_7b_L8192.json

## evo2_7b @ seq_len=32768

- Forward pass: **6297.63 ± 22.92 ms** (n=5 runs, warmup=3)
- Total leaf-op CUDA time: 121693.6 ms
- Block coverage: 31464.8 ms inside blocks, 90228.9 ms outside (embedding / lm_head / final norm)

### By layer kind

| Kind | CUDA ms | % of total |
|---|---:|---:|
| hcl | 13708.6 | 11.3% |
| hcm | 8090.1 | 6.6% |
| hcs | 5353.9 | 4.4% |
| attn | 4312.2 | 3.5% |

### By op category

| Category | CUDA ms | % of total |
|---|---:|---:|
| gemm | 52422.7 | 43.1% |
| other | 39357.2 | 32.3% |
| fft | 8003.6 | 6.6% |
| cast_copy | 7792.4 | 6.4% |
| elementwise | 7036.5 | 5.8% |
| conv | 5410.7 | 4.4% |
| attention | 1323.6 | 1.1% |
| reshape | 346.9 | 0.3% |

### Artifacts

- `chrome_trace`: /home/alpha/Projects/vortex-kernels/results/rtx-4090/baseline_profile/trace_evo2_7b_L32768.json
- `top_ops`: /home/alpha/Projects/vortex-kernels/results/rtx-4090/baseline_profile/top_ops_evo2_7b_L32768.txt
- `layer_breakdown`: /home/alpha/Projects/vortex-kernels/results/rtx-4090/baseline_profile/layer_breakdown_evo2_7b_L32768.json
- `per_layer_plot`: /home/alpha/Projects/vortex-kernels/results/rtx-4090/baseline_profile/plots/per_layer_evo2_7b_L32768.png
- `layer_kinds_plot`: /home/alpha/Projects/vortex-kernels/results/rtx-4090/baseline_profile/plots/layer_kinds_evo2_7b_L32768.png
- `op_categories_plot`: /home/alpha/Projects/vortex-kernels/results/rtx-4090/baseline_profile/plots/op_categories_evo2_7b_L32768.png
- `summary`: /home/alpha/Projects/vortex-kernels/results/rtx-4090/baseline_profile/summary_evo2_7b_L32768.json
