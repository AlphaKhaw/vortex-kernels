# Evo2 baseline profile

Profiled 4 run(s). CUDA times are summed across runs.

## evo2_7b @ seq_len=8192

- Forward pass: **275.31 ± 6.87 ms** (n=5 runs, warmup=3)
- Total leaf-op CUDA time: 5687.2 ms
- Block coverage: 1373.2 ms inside blocks, 4314.0 ms outside (embedding / lm_head / final norm)

### By layer kind

| Kind | CUDA ms | % of total |
|---|---:|---:|
| hcl | 416.1 | 7.3% |
| hcm | 394.4 | 6.9% |
| hcs | 347.5 | 6.1% |
| attn | 215.2 | 3.8% |

### By op category

| Category | CUDA ms | % of total |
|---|---:|---:|
| other | 2266.3 | 39.8% |
| gemm | 1986.7 | 34.9% |
| cast_copy | 547.1 | 9.6% |
| conv | 511.6 | 9.0% |
| fft | 150.9 | 2.7% |
| elementwise | 128.6 | 2.3% |
| attention | 50.6 | 0.9% |
| reshape | 45.4 | 0.8% |

### Artifacts

- `chrome_trace`: /root/vortex-kernels/results/h200/progression/final/trace_evo2_7b_L8192.json
- `top_ops`: /root/vortex-kernels/results/h200/progression/final/top_ops_evo2_7b_L8192.txt
- `layer_breakdown`: /root/vortex-kernels/results/h200/progression/final/layer_breakdown_evo2_7b_L8192.json
- `per_layer_plot`: /root/vortex-kernels/results/h200/progression/final/plots/per_layer_evo2_7b_L8192.png
- `layer_kinds_plot`: /root/vortex-kernels/results/h200/progression/final/plots/layer_kinds_evo2_7b_L8192.png
- `op_categories_plot`: /root/vortex-kernels/results/h200/progression/final/plots/op_categories_evo2_7b_L8192.png
- `summary`: /root/vortex-kernels/results/h200/progression/final/summary_evo2_7b_L8192.json

## evo2_7b @ seq_len=32768

- Forward pass: **1344.60 ± 3.40 ms** (n=5 runs, warmup=3)
- Total leaf-op CUDA time: 27980.9 ms
- Block coverage: 6708.0 ms inside blocks, 21272.9 ms outside (embedding / lm_head / final norm)

### By layer kind

| Kind | CUDA ms | % of total |
|---|---:|---:|
| hcl | 1947.6 | 7.0% |
| hcm | 1804.6 | 6.4% |
| hcs | 1558.5 | 5.6% |
| attn | 1397.3 | 5.0% |

### By op category

| Category | CUDA ms | % of total |
|---|---:|---:|
| other | 11017.4 | 39.4% |
| gemm | 10070.0 | 36.0% |
| cast_copy | 2353.0 | 8.4% |
| conv | 2152.5 | 7.7% |
| fft | 977.4 | 3.5% |
| attention | 638.3 | 2.3% |
| elementwise | 558.1 | 2.0% |
| reshape | 214.4 | 0.8% |

### Artifacts

- `chrome_trace`: /root/vortex-kernels/results/h200/progression/final/trace_evo2_7b_L32768.json
- `top_ops`: /root/vortex-kernels/results/h200/progression/final/top_ops_evo2_7b_L32768.txt
- `layer_breakdown`: /root/vortex-kernels/results/h200/progression/final/layer_breakdown_evo2_7b_L32768.json
- `per_layer_plot`: /root/vortex-kernels/results/h200/progression/final/plots/per_layer_evo2_7b_L32768.png
- `layer_kinds_plot`: /root/vortex-kernels/results/h200/progression/final/plots/layer_kinds_evo2_7b_L32768.png
- `op_categories_plot`: /root/vortex-kernels/results/h200/progression/final/plots/op_categories_evo2_7b_L32768.png
- `summary`: /root/vortex-kernels/results/h200/progression/final/summary_evo2_7b_L32768.json

## evo2_7b @ seq_len=65536

- Forward pass: **3621.05 ± 10.29 ms** (n=5 runs, warmup=3)
- Total leaf-op CUDA time: 77714.6 ms
- Block coverage: 18073.1 ms inside blocks, 59641.6 ms outside (embedding / lm_head / final norm)

### By layer kind

| Kind | CUDA ms | % of total |
|---|---:|---:|
| hcl | 5430.8 | 7.0% |
| hcm | 4698.7 | 6.0% |
| hcs | 3714.5 | 4.8% |
| attn | 4229.1 | 5.4% |

### By op category

| Category | CUDA ms | % of total |
|---|---:|---:|
| other | 31674.9 | 40.8% |
| gemm | 23839.5 | 30.7% |
| cast_copy | 7351.9 | 9.5% |
| conv | 6291.3 | 8.1% |
| fft | 4419.0 | 5.7% |
| attention | 2482.8 | 3.2% |
| elementwise | 1209.5 | 1.6% |
| reshape | 445.7 | 0.6% |

### Artifacts

- `chrome_trace`: /root/vortex-kernels/results/h200/progression/final/trace_evo2_7b_L65536.json
- `top_ops`: /root/vortex-kernels/results/h200/progression/final/top_ops_evo2_7b_L65536.txt
- `layer_breakdown`: /root/vortex-kernels/results/h200/progression/final/layer_breakdown_evo2_7b_L65536.json
- `per_layer_plot`: /root/vortex-kernels/results/h200/progression/final/plots/per_layer_evo2_7b_L65536.png
- `layer_kinds_plot`: /root/vortex-kernels/results/h200/progression/final/plots/layer_kinds_evo2_7b_L65536.png
- `op_categories_plot`: /root/vortex-kernels/results/h200/progression/final/plots/op_categories_evo2_7b_L65536.png
- `summary`: /root/vortex-kernels/results/h200/progression/final/summary_evo2_7b_L65536.json

## evo2_7b @ seq_len=131072

- Forward pass: **11990.40 ± 40.93 ms** (n=5 runs, warmup=3)
- Total leaf-op CUDA time: 276795.9 ms
- Block coverage: 59893.6 ms inside blocks, 216902.3 ms outside (embedding / lm_head / final norm)

### By layer kind

| Kind | CUDA ms | % of total |
|---|---:|---:|
| hcl | 15259.2 | 5.5% |
| hcm | 14266.4 | 5.2% |
| hcs | 11027.6 | 4.0% |
| attn | 19340.3 | 7.0% |

### By op category

| Category | CUDA ms | % of total |
|---|---:|---:|
| other | 127435.1 | 46.0% |
| gemm | 58056.9 | 21.0% |
| cast_copy | 31962.5 | 11.5% |
| conv | 27726.9 | 10.0% |
| attention | 15158.8 | 5.5% |
| fft | 12377.4 | 4.5% |
| elementwise | 2984.4 | 1.1% |
| reshape | 1093.9 | 0.4% |

### Artifacts

- `chrome_trace`: /root/vortex-kernels/results/h200/progression/final/trace_evo2_7b_L131072.json
- `top_ops`: /root/vortex-kernels/results/h200/progression/final/top_ops_evo2_7b_L131072.txt
- `layer_breakdown`: /root/vortex-kernels/results/h200/progression/final/layer_breakdown_evo2_7b_L131072.json
- `per_layer_plot`: /root/vortex-kernels/results/h200/progression/final/plots/per_layer_evo2_7b_L131072.png
- `layer_kinds_plot`: /root/vortex-kernels/results/h200/progression/final/plots/layer_kinds_evo2_7b_L131072.png
- `op_categories_plot`: /root/vortex-kernels/results/h200/progression/final/plots/op_categories_evo2_7b_L131072.png
- `summary`: /root/vortex-kernels/results/h200/progression/final/summary_evo2_7b_L131072.json

