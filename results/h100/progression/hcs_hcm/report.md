# Evo2 baseline profile

Profiled 3 run(s). CUDA times are summed across runs.

## evo2_7b @ seq_len=8192

- Forward pass: **374.38 ± 9.06 ms** (n=5 runs, warmup=3)
- Total leaf-op CUDA time: 7057.9 ms
- Block coverage: 1868.2 ms inside blocks, 5189.7 ms outside (embedding / lm_head / final norm)

### By layer kind

| Kind | CUDA ms | % of total |
|---|---:|---:|
| hcl | 827.0 | 11.7% |
| hcm | 441.7 | 6.3% |
| hcs | 398.1 | 5.6% |
| attn | 201.4 | 2.9% |

### By op category

| Category | CUDA ms | % of total |
|---|---:|---:|
| other | 2955.4 | 41.9% |
| gemm | 2071.5 | 29.3% |
| cast_copy | 647.1 | 9.2% |
| conv | 528.1 | 7.5% |
| elementwise | 449.1 | 6.4% |
| fft | 315.8 | 4.5% |
| reshape | 51.7 | 0.7% |
| attention | 39.2 | 0.6% |

### Artifacts

- `chrome_trace`: /root/vortex-kernels/results/h100/progression/hcs_hcm/trace_evo2_7b_L8192.json
- `top_ops`: /root/vortex-kernels/results/h100/progression/hcs_hcm/top_ops_evo2_7b_L8192.txt
- `layer_breakdown`: /root/vortex-kernels/results/h100/progression/hcs_hcm/layer_breakdown_evo2_7b_L8192.json
- `per_layer_plot`: /root/vortex-kernels/results/h100/progression/hcs_hcm/plots/per_layer_evo2_7b_L8192.png
- `layer_kinds_plot`: /root/vortex-kernels/results/h100/progression/hcs_hcm/plots/layer_kinds_evo2_7b_L8192.png
- `op_categories_plot`: /root/vortex-kernels/results/h100/progression/hcs_hcm/plots/op_categories_evo2_7b_L8192.png
- `summary`: /root/vortex-kernels/results/h100/progression/hcs_hcm/summary_evo2_7b_L8192.json

## evo2_7b @ seq_len=32768

- Forward pass: **1909.38 ± 2.44 ms** (n=5 runs, warmup=3)
- Total leaf-op CUDA time: 36429.9 ms
- Block coverage: 9532.7 ms inside blocks, 26897.2 ms outside (embedding / lm_head / final norm)

### By layer kind

| Kind | CUDA ms | % of total |
|---|---:|---:|
| hcl | 4382.5 | 12.0% |
| hcm | 2187.7 | 6.0% |
| hcs | 1663.9 | 4.6% |
| attn | 1298.7 | 3.6% |

### By op category

| Category | CUDA ms | % of total |
|---|---:|---:|
| other | 15990.1 | 43.9% |
| gemm | 9820.6 | 27.0% |
| cast_copy | 2937.4 | 8.1% |
| conv | 2626.5 | 7.2% |
| elementwise | 2513.3 | 6.9% |
| fft | 1749.3 | 4.8% |
| attention | 592.2 | 1.6% |
| reshape | 200.5 | 0.6% |

### Artifacts

- `chrome_trace`: /root/vortex-kernels/results/h100/progression/hcs_hcm/trace_evo2_7b_L32768.json
- `top_ops`: /root/vortex-kernels/results/h100/progression/hcs_hcm/top_ops_evo2_7b_L32768.txt
- `layer_breakdown`: /root/vortex-kernels/results/h100/progression/hcs_hcm/layer_breakdown_evo2_7b_L32768.json
- `per_layer_plot`: /root/vortex-kernels/results/h100/progression/hcs_hcm/plots/per_layer_evo2_7b_L32768.png
- `layer_kinds_plot`: /root/vortex-kernels/results/h100/progression/hcs_hcm/plots/layer_kinds_evo2_7b_L32768.png
- `op_categories_plot`: /root/vortex-kernels/results/h100/progression/hcs_hcm/plots/op_categories_evo2_7b_L32768.png
- `summary`: /root/vortex-kernels/results/h100/progression/hcs_hcm/summary_evo2_7b_L32768.json

## evo2_7b @ seq_len=65536

- Forward pass: **6319.74 ± 19.67 ms** (n=5 runs, warmup=3)
- Total leaf-op CUDA time: 123876.3 ms
- Block coverage: 31571.2 ms inside blocks, 92305.0 ms outside (embedding / lm_head / final norm)

### By layer kind

| Kind | CUDA ms | % of total |
|---|---:|---:|
| hcl | 15032.6 | 12.1% |
| hcm | 6532.7 | 5.3% |
| hcs | 4201.1 | 3.4% |
| attn | 5804.8 | 4.7% |

### By op category

| Category | CUDA ms | % of total |
|---|---:|---:|
| other | 59844.1 | 48.3% |
| gemm | 25900.8 | 20.9% |
| cast_copy | 9228.6 | 7.4% |
| fft | 8476.4 | 6.8% |
| elementwise | 8221.7 | 6.6% |
| conv | 7846.1 | 6.3% |
| attention | 3847.3 | 3.1% |
| reshape | 511.4 | 0.4% |

### Artifacts

- `chrome_trace`: /root/vortex-kernels/results/h100/progression/hcs_hcm/trace_evo2_7b_L65536.json
- `top_ops`: /root/vortex-kernels/results/h100/progression/hcs_hcm/top_ops_evo2_7b_L65536.txt
- `layer_breakdown`: /root/vortex-kernels/results/h100/progression/hcs_hcm/layer_breakdown_evo2_7b_L65536.json
- `per_layer_plot`: /root/vortex-kernels/results/h100/progression/hcs_hcm/plots/per_layer_evo2_7b_L65536.png
- `layer_kinds_plot`: /root/vortex-kernels/results/h100/progression/hcs_hcm/plots/layer_kinds_evo2_7b_L65536.png
- `op_categories_plot`: /root/vortex-kernels/results/h100/progression/hcs_hcm/plots/op_categories_evo2_7b_L65536.png
- `summary`: /root/vortex-kernels/results/h100/progression/hcs_hcm/summary_evo2_7b_L65536.json

