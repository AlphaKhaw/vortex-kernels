# Evo2 baseline profile

Profiled 3 run(s). CUDA times are summed across runs.

## evo2_7b @ seq_len=8192

- Forward pass: **382.17 ± 7.17 ms** (n=5 runs, warmup=3)
- Total leaf-op CUDA time: 7197.8 ms
- Block coverage: 1907.4 ms inside blocks, 5290.5 ms outside (embedding / lm_head / final norm)

### By layer kind

| Kind | CUDA ms | % of total |
|---|---:|---:|
| hcl | 827.5 | 11.5% |
| hcm | 446.7 | 6.2% |
| hcs | 405.9 | 5.6% |
| attn | 227.2 | 3.2% |

### By op category

| Category | CUDA ms | % of total |
|---|---:|---:|
| other | 3052.7 | 42.4% |
| gemm | 2010.3 | 27.9% |
| cast_copy | 651.1 | 9.0% |
| conv | 599.1 | 8.3% |
| elementwise | 495.3 | 6.9% |
| fft | 291.6 | 4.1% |
| reshape | 50.7 | 0.7% |
| attention | 47.1 | 0.7% |

### Artifacts

- `chrome_trace`: /root/vortex-kernels/results/h100/baseline_profile/trace_evo2_7b_L8192.json
- `top_ops`: /root/vortex-kernels/results/h100/baseline_profile/top_ops_evo2_7b_L8192.txt
- `layer_breakdown`: /root/vortex-kernels/results/h100/baseline_profile/layer_breakdown_evo2_7b_L8192.json
- `per_layer_plot`: /root/vortex-kernels/results/h100/baseline_profile/plots/per_layer_evo2_7b_L8192.png
- `layer_kinds_plot`: /root/vortex-kernels/results/h100/baseline_profile/plots/layer_kinds_evo2_7b_L8192.png
- `op_categories_plot`: /root/vortex-kernels/results/h100/baseline_profile/plots/op_categories_evo2_7b_L8192.png
- `summary`: /root/vortex-kernels/results/h100/baseline_profile/summary_evo2_7b_L8192.json

## evo2_7b @ seq_len=32768

- Forward pass: **2065.64 ± 5.93 ms** (n=5 runs, warmup=3)
- Total leaf-op CUDA time: 39650.6 ms
- Block coverage: 10315.9 ms inside blocks, 29334.7 ms outside (embedding / lm_head / final norm)

### By layer kind

| Kind | CUDA ms | % of total |
|---|---:|---:|
| hcl | 4555.9 | 11.5% |
| hcm | 2366.1 | 6.0% |
| hcs | 1839.0 | 4.6% |
| attn | 1554.9 | 3.9% |

### By op category

| Category | CUDA ms | % of total |
|---|---:|---:|
| other | 17735.0 | 44.7% |
| gemm | 10575.6 | 26.7% |
| conv | 2912.9 | 7.3% |
| cast_copy | 2856.7 | 7.2% |
| elementwise | 2775.5 | 7.0% |
| fft | 1763.1 | 4.4% |
| attention | 812.8 | 2.0% |
| reshape | 219.0 | 0.6% |

### Artifacts

- `chrome_trace`: /root/vortex-kernels/results/h100/baseline_profile/trace_evo2_7b_L32768.json
- `top_ops`: /root/vortex-kernels/results/h100/baseline_profile/top_ops_evo2_7b_L32768.txt
- `layer_breakdown`: /root/vortex-kernels/results/h100/baseline_profile/layer_breakdown_evo2_7b_L32768.json
- `per_layer_plot`: /root/vortex-kernels/results/h100/baseline_profile/plots/per_layer_evo2_7b_L32768.png
- `layer_kinds_plot`: /root/vortex-kernels/results/h100/baseline_profile/plots/layer_kinds_evo2_7b_L32768.png
- `op_categories_plot`: /root/vortex-kernels/results/h100/baseline_profile/plots/op_categories_evo2_7b_L32768.png
- `summary`: /root/vortex-kernels/results/h100/baseline_profile/summary_evo2_7b_L32768.json

## evo2_7b @ seq_len=65536

- Forward pass: **6653.29 ± 17.95 ms** (n=5 runs, warmup=3)
- Total leaf-op CUDA time: 128289.7 ms
- Block coverage: 33237.6 ms inside blocks, 95052.1 ms outside (embedding / lm_head / final norm)

### By layer kind

| Kind | CUDA ms | % of total |
|---|---:|---:|
| hcl | 16262.1 | 12.7% |
| hcm | 6875.2 | 5.4% |
| hcs | 4523.6 | 3.5% |
| attn | 5576.7 | 4.3% |

### By op category

| Category | CUDA ms | % of total |
|---|---:|---:|
| other | 62514.4 | 48.7% |
| gemm | 25981.9 | 20.3% |
| elementwise | 10239.4 | 8.0% |
| cast_copy | 9402.1 | 7.3% |
| fft | 7964.1 | 6.2% |
| conv | 7897.1 | 6.2% |
| attention | 3835.7 | 3.0% |
| reshape | 455.0 | 0.4% |

### Artifacts

- `chrome_trace`: /root/vortex-kernels/results/h100/baseline_profile/trace_evo2_7b_L65536.json
- `top_ops`: /root/vortex-kernels/results/h100/baseline_profile/top_ops_evo2_7b_L65536.txt
- `layer_breakdown`: /root/vortex-kernels/results/h100/baseline_profile/layer_breakdown_evo2_7b_L65536.json
- `per_layer_plot`: /root/vortex-kernels/results/h100/baseline_profile/plots/per_layer_evo2_7b_L65536.png
- `layer_kinds_plot`: /root/vortex-kernels/results/h100/baseline_profile/plots/layer_kinds_evo2_7b_L65536.png
- `op_categories_plot`: /root/vortex-kernels/results/h100/baseline_profile/plots/op_categories_evo2_7b_L65536.png
- `summary`: /root/vortex-kernels/results/h100/baseline_profile/summary_evo2_7b_L65536.json

