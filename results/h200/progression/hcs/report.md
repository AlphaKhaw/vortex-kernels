# Evo2 baseline profile

Profiled 4 run(s). CUDA times are summed across runs.

## evo2_7b @ seq_len=8192

- Forward pass: **346.95 ± 4.36 ms** (n=5 runs, warmup=3)
- Total leaf-op CUDA time: 6545.1 ms
- Block coverage: 1731.4 ms inside blocks, 4813.7 ms outside (embedding / lm_head / final norm)

### By layer kind

| Kind | CUDA ms | % of total |
|---|---:|---:|
| hcl | 745.8 | 11.4% |
| hcm | 416.8 | 6.4% |
| hcs | 352.6 | 5.4% |
| attn | 216.2 | 3.3% |

### By op category

| Category | CUDA ms | % of total |
|---|---:|---:|
| other | 2708.6 | 41.4% |
| gemm | 2011.0 | 30.7% |
| cast_copy | 551.5 | 8.4% |
| conv | 514.5 | 7.9% |
| elementwise | 419.8 | 6.4% |
| fft | 244.3 | 3.7% |
| attention | 48.0 | 0.7% |
| reshape | 47.4 | 0.7% |

### Artifacts

- `chrome_trace`: /root/vortex-kernels/results/h200/progression/hcs/trace_evo2_7b_L8192.json
- `top_ops`: /root/vortex-kernels/results/h200/progression/hcs/top_ops_evo2_7b_L8192.txt
- `layer_breakdown`: /root/vortex-kernels/results/h200/progression/hcs/layer_breakdown_evo2_7b_L8192.json
- `per_layer_plot`: /root/vortex-kernels/results/h200/progression/hcs/plots/per_layer_evo2_7b_L8192.png
- `layer_kinds_plot`: /root/vortex-kernels/results/h200/progression/hcs/plots/layer_kinds_evo2_7b_L8192.png
- `op_categories_plot`: /root/vortex-kernels/results/h200/progression/hcs/plots/op_categories_evo2_7b_L8192.png
- `summary`: /root/vortex-kernels/results/h200/progression/hcs/summary_evo2_7b_L8192.json

## evo2_7b @ seq_len=32768

- Forward pass: **1921.66 ± 11.10 ms** (n=5 runs, warmup=3)
- Total leaf-op CUDA time: 37130.2 ms
- Block coverage: 9596.5 ms inside blocks, 27533.6 ms outside (embedding / lm_head / final norm)

### By layer kind

| Kind | CUDA ms | % of total |
|---|---:|---:|
| hcl | 4080.2 | 11.0% |
| hcm | 2228.1 | 6.0% |
| hcs | 1718.3 | 4.6% |
| attn | 1569.9 | 4.2% |

### By op category

| Category | CUDA ms | % of total |
|---|---:|---:|
| other | 16296.3 | 43.9% |
| gemm | 10784.5 | 29.0% |
| conv | 2587.3 | 7.0% |
| cast_copy | 2489.9 | 6.7% |
| elementwise | 2387.4 | 6.4% |
| fft | 1539.8 | 4.1% |
| attention | 835.7 | 2.3% |
| reshape | 209.3 | 0.6% |

### Artifacts

- `chrome_trace`: /root/vortex-kernels/results/h200/progression/hcs/trace_evo2_7b_L32768.json
- `top_ops`: /root/vortex-kernels/results/h200/progression/hcs/top_ops_evo2_7b_L32768.txt
- `layer_breakdown`: /root/vortex-kernels/results/h200/progression/hcs/layer_breakdown_evo2_7b_L32768.json
- `per_layer_plot`: /root/vortex-kernels/results/h200/progression/hcs/plots/per_layer_evo2_7b_L32768.png
- `layer_kinds_plot`: /root/vortex-kernels/results/h200/progression/hcs/plots/layer_kinds_evo2_7b_L32768.png
- `op_categories_plot`: /root/vortex-kernels/results/h200/progression/hcs/plots/op_categories_evo2_7b_L32768.png
- `summary`: /root/vortex-kernels/results/h200/progression/hcs/summary_evo2_7b_L32768.json

## evo2_7b @ seq_len=65536

- Forward pass: **6105.49 ± 13.70 ms** (n=5 runs, warmup=3)
- Total leaf-op CUDA time: 118472.3 ms
- Block coverage: 30499.5 ms inside blocks, 87972.8 ms outside (embedding / lm_head / final norm)

### By layer kind

| Kind | CUDA ms | % of total |
|---|---:|---:|
| hcl | 14474.6 | 12.2% |
| hcm | 6159.7 | 5.2% |
| hcs | 4201.6 | 3.5% |
| attn | 5663.6 | 4.8% |

### By op category

| Category | CUDA ms | % of total |
|---|---:|---:|
| other | 56885.7 | 48.0% |
| gemm | 26622.3 | 22.5% |
| elementwise | 9003.8 | 7.6% |
| cast_copy | 8316.9 | 7.0% |
| conv | 7039.6 | 5.9% |
| fft | 6225.0 | 5.3% |
| attention | 3920.1 | 3.3% |
| reshape | 458.9 | 0.4% |

### Artifacts

- `chrome_trace`: /root/vortex-kernels/results/h200/progression/hcs/trace_evo2_7b_L65536.json
- `top_ops`: /root/vortex-kernels/results/h200/progression/hcs/top_ops_evo2_7b_L65536.txt
- `layer_breakdown`: /root/vortex-kernels/results/h200/progression/hcs/layer_breakdown_evo2_7b_L65536.json
- `per_layer_plot`: /root/vortex-kernels/results/h200/progression/hcs/plots/per_layer_evo2_7b_L65536.png
- `layer_kinds_plot`: /root/vortex-kernels/results/h200/progression/hcs/plots/layer_kinds_evo2_7b_L65536.png
- `op_categories_plot`: /root/vortex-kernels/results/h200/progression/hcs/plots/op_categories_evo2_7b_L65536.png
- `summary`: /root/vortex-kernels/results/h200/progression/hcs/summary_evo2_7b_L65536.json

## evo2_7b @ seq_len=131072

- Forward pass: **21920.67 ± 64.29 ms** (n=5 runs, warmup=3)
- Total leaf-op CUDA time: 447912.5 ms
- Block coverage: 109535.7 ms inside blocks, 338376.8 ms outside (embedding / lm_head / final norm)

### By layer kind

| Kind | CUDA ms | % of total |
|---|---:|---:|
| hcl | 59129.8 | 13.2% |
| hcm | 18944.7 | 4.2% |
| hcs | 12264.5 | 2.7% |
| attn | 19196.9 | 4.3% |

### By op category

| Category | CUDA ms | % of total |
|---|---:|---:|
| other | 233284.9 | 52.1% |
| gemm | 67403.2 | 15.0% |
| cast_copy | 40307.9 | 9.0% |
| elementwise | 38582.1 | 8.6% |
| conv | 35025.9 | 7.8% |
| fft | 17053.7 | 3.8% |
| attention | 15077.1 | 3.4% |
| reshape | 1177.7 | 0.3% |

### Artifacts

- `chrome_trace`: /root/vortex-kernels/results/h200/progression/hcs/trace_evo2_7b_L131072.json
- `top_ops`: /root/vortex-kernels/results/h200/progression/hcs/top_ops_evo2_7b_L131072.txt
- `layer_breakdown`: /root/vortex-kernels/results/h200/progression/hcs/layer_breakdown_evo2_7b_L131072.json
- `per_layer_plot`: /root/vortex-kernels/results/h200/progression/hcs/plots/per_layer_evo2_7b_L131072.png
- `layer_kinds_plot`: /root/vortex-kernels/results/h200/progression/hcs/plots/layer_kinds_evo2_7b_L131072.png
- `op_categories_plot`: /root/vortex-kernels/results/h200/progression/hcs/plots/op_categories_evo2_7b_L131072.png
- `summary`: /root/vortex-kernels/results/h200/progression/hcs/summary_evo2_7b_L131072.json

