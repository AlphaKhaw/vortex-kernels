# Evo2 baseline profile

Profiled 4 run(s). CUDA times are summed across runs.

## evo2_7b @ seq_len=8192

- Forward pass: **288.34 ± 9.35 ms** (n=5 runs, warmup=3)
- Total leaf-op CUDA time: 5995.1 ms
- Block coverage: 1438.5 ms inside blocks, 4556.6 ms outside (embedding / lm_head / final norm)

### By layer kind

| Kind | CUDA ms | % of total |
|---|---:|---:|
| hcl | 458.6 | 7.6% |
| hcm | 412.0 | 6.9% |
| hcs | 355.6 | 5.9% |
| attn | 212.2 | 3.5% |

### By op category

| Category | CUDA ms | % of total |
|---|---:|---:|
| other | 2429.8 | 40.5% |
| gemm | 1951.1 | 32.5% |
| cast_copy | 631.2 | 10.5% |
| conv | 547.9 | 9.1% |
| fft | 190.6 | 3.2% |
| elementwise | 149.8 | 2.5% |
| attention | 47.5 | 0.8% |
| reshape | 47.2 | 0.8% |

### Artifacts

- `chrome_trace`: /root/vortex-kernels/results/h100/progression/final/trace_evo2_7b_L8192.json
- `top_ops`: /root/vortex-kernels/results/h100/progression/final/top_ops_evo2_7b_L8192.txt
- `layer_breakdown`: /root/vortex-kernels/results/h100/progression/final/layer_breakdown_evo2_7b_L8192.json
- `per_layer_plot`: /root/vortex-kernels/results/h100/progression/final/plots/per_layer_evo2_7b_L8192.png
- `layer_kinds_plot`: /root/vortex-kernels/results/h100/progression/final/plots/layer_kinds_evo2_7b_L8192.png
- `op_categories_plot`: /root/vortex-kernels/results/h100/progression/final/plots/op_categories_evo2_7b_L8192.png
- `summary`: /root/vortex-kernels/results/h100/progression/final/summary_evo2_7b_L8192.json

## evo2_7b @ seq_len=32768

- Forward pass: **1397.10 ± 2.83 ms** (n=5 runs, warmup=3)
- Total leaf-op CUDA time: 29163.4 ms
- Block coverage: 6970.3 ms inside blocks, 22193.0 ms outside (embedding / lm_head / final norm)

### By layer kind

| Kind | CUDA ms | % of total |
|---|---:|---:|
| hcl | 2081.5 | 7.1% |
| hcm | 1925.1 | 6.6% |
| hcs | 1604.9 | 5.5% |
| attn | 1358.9 | 4.7% |

### By op category

| Category | CUDA ms | % of total |
|---|---:|---:|
| other | 11609.9 | 39.8% |
| gemm | 9980.5 | 34.2% |
| cast_copy | 2688.6 | 9.2% |
| conv | 2248.6 | 7.7% |
| fft | 1165.9 | 4.0% |
| elementwise | 656.2 | 2.3% |
| attention | 599.6 | 2.1% |
| reshape | 214.1 | 0.7% |

### Artifacts

- `chrome_trace`: /root/vortex-kernels/results/h100/progression/final/trace_evo2_7b_L32768.json
- `top_ops`: /root/vortex-kernels/results/h100/progression/final/top_ops_evo2_7b_L32768.txt
- `layer_breakdown`: /root/vortex-kernels/results/h100/progression/final/layer_breakdown_evo2_7b_L32768.json
- `per_layer_plot`: /root/vortex-kernels/results/h100/progression/final/plots/per_layer_evo2_7b_L32768.png
- `layer_kinds_plot`: /root/vortex-kernels/results/h100/progression/final/plots/layer_kinds_evo2_7b_L32768.png
- `op_categories_plot`: /root/vortex-kernels/results/h100/progression/final/plots/op_categories_evo2_7b_L32768.png
- `summary`: /root/vortex-kernels/results/h100/progression/final/summary_evo2_7b_L32768.json

## evo2_7b @ seq_len=65536

- Forward pass: **3843.39 ± 4.68 ms** (n=5 runs, warmup=3)
- Total leaf-op CUDA time: 82595.3 ms
- Block coverage: 19184.1 ms inside blocks, 63411.2 ms outside (embedding / lm_head / final norm)

### By layer kind

| Kind | CUDA ms | % of total |
|---|---:|---:|
| hcl | 5978.4 | 7.2% |
| hcm | 5224.6 | 6.3% |
| hcs | 3833.6 | 4.6% |
| attn | 4147.5 | 5.0% |

### By op category

| Category | CUDA ms | % of total |
|---|---:|---:|
| other | 34013.3 | 41.2% |
| gemm | 23385.7 | 28.3% |
| cast_copy | 8303.9 | 10.1% |
| conv | 6755.1 | 8.2% |
| fft | 5851.9 | 7.1% |
| attention | 2409.5 | 2.9% |
| elementwise | 1433.3 | 1.7% |
| reshape | 442.7 | 0.5% |

### Artifacts

- `chrome_trace`: /root/vortex-kernels/results/h100/progression/final/trace_evo2_7b_L65536.json
- `top_ops`: /root/vortex-kernels/results/h100/progression/final/top_ops_evo2_7b_L65536.txt
- `layer_breakdown`: /root/vortex-kernels/results/h100/progression/final/layer_breakdown_evo2_7b_L65536.json
- `per_layer_plot`: /root/vortex-kernels/results/h100/progression/final/plots/per_layer_evo2_7b_L65536.png
- `layer_kinds_plot`: /root/vortex-kernels/results/h100/progression/final/plots/layer_kinds_evo2_7b_L65536.png
- `op_categories_plot`: /root/vortex-kernels/results/h100/progression/final/plots/op_categories_evo2_7b_L65536.png
- `summary`: /root/vortex-kernels/results/h100/progression/final/summary_evo2_7b_L65536.json

## evo2_7b @ seq_len=131072

- Forward pass: **12701.89 ± 9.55 ms** (n=5 runs, warmup=3)
- Total leaf-op CUDA time: 296742.4 ms
- Block coverage: 63453.8 ms inside blocks, 233288.7 ms outside (embedding / lm_head / final norm)

### By layer kind

| Kind | CUDA ms | % of total |
|---|---:|---:|
| hcl | 16952.7 | 5.7% |
| hcm | 15871.7 | 5.3% |
| hcs | 11718.1 | 3.9% |
| attn | 18911.3 | 6.4% |

### By op category

| Category | CUDA ms | % of total |
|---|---:|---:|
| other | 137235.6 | 46.2% |
| gemm | 56739.6 | 19.1% |
| cast_copy | 36629.5 | 12.3% |
| conv | 30958.4 | 10.4% |
| fft | 15749.3 | 5.3% |
| attention | 14760.6 | 5.0% |
| elementwise | 3559.3 | 1.2% |
| reshape | 1110.3 | 0.4% |

### Artifacts

- `chrome_trace`: /root/vortex-kernels/results/h100/progression/final/trace_evo2_7b_L131072.json
- `top_ops`: /root/vortex-kernels/results/h100/progression/final/top_ops_evo2_7b_L131072.txt
- `layer_breakdown`: /root/vortex-kernels/results/h100/progression/final/layer_breakdown_evo2_7b_L131072.json
- `per_layer_plot`: /root/vortex-kernels/results/h100/progression/final/plots/per_layer_evo2_7b_L131072.png
- `layer_kinds_plot`: /root/vortex-kernels/results/h100/progression/final/plots/layer_kinds_evo2_7b_L131072.png
- `op_categories_plot`: /root/vortex-kernels/results/h100/progression/final/plots/op_categories_evo2_7b_L131072.png
- `summary`: /root/vortex-kernels/results/h100/progression/final/summary_evo2_7b_L131072.json

