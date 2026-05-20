# Evo2 baseline profile

Profiled 2 run(s). CUDA times are summed across runs.

## evo2_7b @ seq_len=8192

- Forward pass: **1152.41 ± 23.19 ms** (n=5 runs, warmup=3)
- Total leaf-op CUDA time: 22221.2 ms
- Block coverage: 5756.4 ms inside blocks, 16464.8 ms outside (embedding / lm_head / final norm)

### By layer kind

| Kind | CUDA ms | % of total |
|---|---:|---:|
| hcl | 2304.6 | 10.4% |
| hcm | 1489.4 | 6.7% |
| hcs | 1207.1 | 5.4% |
| attn | 755.3 | 3.4% |

### By op category

| Category | CUDA ms | % of total |
|---|---:|---:|
| gemm | 11406.1 | 51.3% |
| other | 5828.2 | 26.2% |
| cast_copy | 1663.4 | 7.5% |
| elementwise | 1189.0 | 5.4% |
| conv | 1131.6 | 5.1% |
| fft | 825.2 | 3.7% |
| attention | 104.3 | 0.5% |
| reshape | 73.4 | 0.3% |

### Artifacts

- `chrome_trace`: /home/alpha/Projects/vortex-kernels/results/rtx-4090/progression/hcs/trace_evo2_7b_L8192.json
- `top_ops`: /home/alpha/Projects/vortex-kernels/results/rtx-4090/progression/hcs/top_ops_evo2_7b_L8192.txt
- `layer_breakdown`: /home/alpha/Projects/vortex-kernels/results/rtx-4090/progression/hcs/layer_breakdown_evo2_7b_L8192.json
- `per_layer_plot`: /home/alpha/Projects/vortex-kernels/results/rtx-4090/progression/hcs/plots/per_layer_evo2_7b_L8192.png
- `layer_kinds_plot`: /home/alpha/Projects/vortex-kernels/results/rtx-4090/progression/hcs/plots/layer_kinds_evo2_7b_L8192.png
- `op_categories_plot`: /home/alpha/Projects/vortex-kernels/results/rtx-4090/progression/hcs/plots/op_categories_evo2_7b_L8192.png
- `summary`: /home/alpha/Projects/vortex-kernels/results/rtx-4090/progression/hcs/summary_evo2_7b_L8192.json

## evo2_7b @ seq_len=32768

- Forward pass: **6288.88 ± 5.30 ms** (n=5 runs, warmup=3)
- Total leaf-op CUDA time: 121437.6 ms
- Block coverage: 31421.0 ms inside blocks, 90016.5 ms outside (embedding / lm_head / final norm)

### By layer kind

| Kind | CUDA ms | % of total |
|---|---:|---:|
| hcl | 13745.6 | 11.3% |
| hcm | 8108.5 | 6.7% |
| hcs | 5253.0 | 4.3% |
| attn | 4314.0 | 3.6% |

### By op category

| Category | CUDA ms | % of total |
|---|---:|---:|
| gemm | 52428.2 | 43.2% |
| other | 39187.0 | 32.3% |
| fft | 8118.9 | 6.7% |
| cast_copy | 7791.3 | 6.4% |
| elementwise | 7037.4 | 5.8% |
| conv | 5202.2 | 4.3% |
| attention | 1325.6 | 1.1% |
| reshape | 346.9 | 0.3% |

### Artifacts

- `chrome_trace`: /home/alpha/Projects/vortex-kernels/results/rtx-4090/progression/hcs/trace_evo2_7b_L32768.json
- `top_ops`: /home/alpha/Projects/vortex-kernels/results/rtx-4090/progression/hcs/top_ops_evo2_7b_L32768.txt
- `layer_breakdown`: /home/alpha/Projects/vortex-kernels/results/rtx-4090/progression/hcs/layer_breakdown_evo2_7b_L32768.json
- `per_layer_plot`: /home/alpha/Projects/vortex-kernels/results/rtx-4090/progression/hcs/plots/per_layer_evo2_7b_L32768.png
- `layer_kinds_plot`: /home/alpha/Projects/vortex-kernels/results/rtx-4090/progression/hcs/plots/layer_kinds_evo2_7b_L32768.png
- `op_categories_plot`: /home/alpha/Projects/vortex-kernels/results/rtx-4090/progression/hcs/plots/op_categories_evo2_7b_L32768.png
- `summary`: /home/alpha/Projects/vortex-kernels/results/rtx-4090/progression/hcs/summary_evo2_7b_L32768.json
