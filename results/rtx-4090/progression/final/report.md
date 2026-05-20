# Evo2 baseline profile

Profiled 3 run(s). CUDA times are summed across runs.

## evo2_7b @ seq_len=8192

- Forward pass: **953.80 ± 29.76 ms** (n=5 runs, warmup=3)
- Total leaf-op CUDA time: 20058.4 ms
- Block coverage: 4763.1 ms inside blocks, 15295.3 ms outside (embedding / lm_head / final norm)

### By layer kind

| Kind | CUDA ms | % of total |
|---|---:|---:|
| hcl | 1472.1 | 7.3% |
| hcm | 1416.4 | 7.1% |
| hcs | 1167.4 | 5.8% |
| attn | 707.1 | 3.5% |

### By op category

| Category | CUDA ms | % of total |
|---|---:|---:|
| gemm | 11611.1 | 57.9% |
| other | 4562.9 | 22.7% |
| cast_copy | 1656.9 | 8.3% |
| conv | 1084.9 | 5.4% |
| fft | 620.7 | 3.1% |
| elementwise | 345.9 | 1.7% |
| attention | 104.9 | 0.5% |
| reshape | 71.1 | 0.4% |

### Artifacts

- `chrome_trace`: /home/alpha/Projects/vortex-kernels/results/rtx-4090/progression/final/trace_evo2_7b_L8192.json
- `top_ops`: /home/alpha/Projects/vortex-kernels/results/rtx-4090/progression/final/top_ops_evo2_7b_L8192.txt
- `layer_breakdown`: /home/alpha/Projects/vortex-kernels/results/rtx-4090/progression/final/layer_breakdown_evo2_7b_L8192.json
- `per_layer_plot`: /home/alpha/Projects/vortex-kernels/results/rtx-4090/progression/final/plots/per_layer_evo2_7b_L8192.png
- `layer_kinds_plot`: /home/alpha/Projects/vortex-kernels/results/rtx-4090/progression/final/plots/layer_kinds_evo2_7b_L8192.png
- `op_categories_plot`: /home/alpha/Projects/vortex-kernels/results/rtx-4090/progression/final/plots/op_categories_evo2_7b_L8192.png
- `summary`: /home/alpha/Projects/vortex-kernels/results/rtx-4090/progression/final/summary_evo2_7b_L8192.json

## evo2_7b @ seq_len=32768

- Forward pass: **4837.20 ± 2.59 ms** (n=5 runs, warmup=3)
- Total leaf-op CUDA time: 101790.4 ms
- Block coverage: 24166.8 ms inside blocks, 77623.6 ms outside (embedding / lm_head / final norm)

### By layer kind

| Kind | CUDA ms | % of total |
|---|---:|---:|
| hcl | 7278.4 | 7.2% |
| hcm | 6792.7 | 6.7% |
| hcs | 5052.6 | 5.0% |
| attn | 5043.1 | 5.0% |

### By op category

| Category | CUDA ms | % of total |
|---|---:|---:|
| gemm | 50838.2 | 49.9% |
| other | 27739.8 | 27.3% |
| cast_copy | 7558.1 | 7.4% |
| fft | 6937.2 | 6.8% |
| conv | 4995.9 | 4.9% |
| attention | 1867.0 | 1.8% |
| elementwise | 1535.0 | 1.5% |
| reshape | 319.2 | 0.3% |

### Artifacts

- `chrome_trace`: /home/alpha/Projects/vortex-kernels/results/rtx-4090/progression/final/trace_evo2_7b_L32768.json
- `top_ops`: /home/alpha/Projects/vortex-kernels/results/rtx-4090/progression/final/top_ops_evo2_7b_L32768.txt
- `layer_breakdown`: /home/alpha/Projects/vortex-kernels/results/rtx-4090/progression/final/layer_breakdown_evo2_7b_L32768.json
- `per_layer_plot`: /home/alpha/Projects/vortex-kernels/results/rtx-4090/progression/final/plots/per_layer_evo2_7b_L32768.png
- `layer_kinds_plot`: /home/alpha/Projects/vortex-kernels/results/rtx-4090/progression/final/plots/layer_kinds_evo2_7b_L32768.png
- `op_categories_plot`: /home/alpha/Projects/vortex-kernels/results/rtx-4090/progression/final/plots/op_categories_evo2_7b_L32768.png
- `summary`: /home/alpha/Projects/vortex-kernels/results/rtx-4090/progression/final/summary_evo2_7b_L32768.json

## evo2_7b @ seq_len=65536

- Forward pass: **11917.63 ± 9.88 ms** (n=5 runs, warmup=3)
- Total leaf-op CUDA time: 260358.3 ms
- Block coverage: 59544.0 ms inside blocks, 200814.3 ms outside (embedding / lm_head / final norm)

### By layer kind

| Kind | CUDA ms | % of total |
|---|---:|---:|
| hcl | 16846.0 | 6.5% |
| hcm | 16739.4 | 6.4% |
| hcs | 12275.7 | 4.7% |
| attn | 13683.0 | 5.3% |

### By op category

| Category | CUDA ms | % of total |
|---|---:|---:|
| gemm | 113168.5 | 43.5% |
| other | 81886.7 | 31.5% |
| cast_copy | 21779.9 | 8.4% |
| fft | 16797.5 | 6.5% |
| conv | 15729.6 | 6.0% |
| attention | 7327.2 | 2.8% |
| elementwise | 3125.7 | 1.2% |
| reshape | 543.2 | 0.2% |

### Artifacts

- `chrome_trace`: /home/alpha/Projects/vortex-kernels/results/rtx-4090/progression/final/trace_evo2_7b_L65536.json
- `top_ops`: /home/alpha/Projects/vortex-kernels/results/rtx-4090/progression/final/top_ops_evo2_7b_L65536.txt
- `layer_breakdown`: /home/alpha/Projects/vortex-kernels/results/rtx-4090/progression/final/layer_breakdown_evo2_7b_L65536.json
- `per_layer_plot`: /home/alpha/Projects/vortex-kernels/results/rtx-4090/progression/final/plots/per_layer_evo2_7b_L65536.png
- `layer_kinds_plot`: /home/alpha/Projects/vortex-kernels/results/rtx-4090/progression/final/plots/layer_kinds_evo2_7b_L65536.png
- `op_categories_plot`: /home/alpha/Projects/vortex-kernels/results/rtx-4090/progression/final/plots/op_categories_evo2_7b_L65536.png
- `summary`: /home/alpha/Projects/vortex-kernels/results/rtx-4090/progression/final/summary_evo2_7b_L65536.json
