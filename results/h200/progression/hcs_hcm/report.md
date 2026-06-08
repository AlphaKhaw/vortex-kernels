# Evo2 baseline profile

Profiled 4 run(s). CUDA times are summed across runs.

## evo2_7b @ seq_len=8192

- Forward pass: **348.03 ± 8.21 ms** (n=5 runs, warmup=3)
- Total leaf-op CUDA time: 6596.3 ms
- Block coverage: 1736.5 ms inside blocks, 4859.8 ms outside (embedding / lm_head / final norm)

### By layer kind

| Kind | CUDA ms | % of total |
|---|---:|---:|
| hcl | 752.8 | 11.4% |
| hcm | 418.6 | 6.3% |
| hcs | 362.0 | 5.5% |
| attn | 203.1 | 3.1% |

### By op category

| Category | CUDA ms | % of total |
|---|---:|---:|
| other | 2716.1 | 41.2% |
| gemm | 2066.4 | 31.3% |
| cast_copy | 566.1 | 8.6% |
| conv | 499.1 | 7.6% |
| elementwise | 389.3 | 5.9% |
| fft | 268.6 | 4.1% |
| reshape | 50.6 | 0.8% |
| attention | 40.2 | 0.6% |

### Artifacts

- `chrome_trace`: /root/vortex-kernels/results/h200/progression/hcs_hcm/trace_evo2_7b_L8192.json
- `top_ops`: /root/vortex-kernels/results/h200/progression/hcs_hcm/top_ops_evo2_7b_L8192.txt
- `layer_breakdown`: /root/vortex-kernels/results/h200/progression/hcs_hcm/layer_breakdown_evo2_7b_L8192.json
- `per_layer_plot`: /root/vortex-kernels/results/h200/progression/hcs_hcm/plots/per_layer_evo2_7b_L8192.png
- `layer_kinds_plot`: /root/vortex-kernels/results/h200/progression/hcs_hcm/plots/layer_kinds_evo2_7b_L8192.png
- `op_categories_plot`: /root/vortex-kernels/results/h200/progression/hcs_hcm/plots/op_categories_evo2_7b_L8192.png
- `summary`: /root/vortex-kernels/results/h200/progression/hcs_hcm/summary_evo2_7b_L8192.json

## evo2_7b @ seq_len=32768

- Forward pass: **1802.44 ± 10.29 ms** (n=5 runs, warmup=3)
- Total leaf-op CUDA time: 34640.6 ms
- Block coverage: 8998.5 ms inside blocks, 25642.1 ms outside (embedding / lm_head / final norm)

### By layer kind

| Kind | CUDA ms | % of total |
|---|---:|---:|
| hcl | 3999.7 | 11.5% |
| hcm | 2074.1 | 6.0% |
| hcs | 1618.3 | 4.7% |
| attn | 1306.4 | 3.8% |

### By op category

| Category | CUDA ms | % of total |
|---|---:|---:|
| other | 14982.0 | 43.2% |
| gemm | 10013.1 | 28.9% |
| cast_copy | 2587.3 | 7.5% |
| conv | 2503.2 | 7.2% |
| elementwise | 2236.4 | 6.5% |
| fft | 1518.7 | 4.4% |
| attention | 601.4 | 1.7% |
| reshape | 198.4 | 0.6% |

### Artifacts

- `chrome_trace`: /root/vortex-kernels/results/h200/progression/hcs_hcm/trace_evo2_7b_L32768.json
- `top_ops`: /root/vortex-kernels/results/h200/progression/hcs_hcm/top_ops_evo2_7b_L32768.txt
- `layer_breakdown`: /root/vortex-kernels/results/h200/progression/hcs_hcm/layer_breakdown_evo2_7b_L32768.json
- `per_layer_plot`: /root/vortex-kernels/results/h200/progression/hcs_hcm/plots/per_layer_evo2_7b_L32768.png
- `layer_kinds_plot`: /root/vortex-kernels/results/h200/progression/hcs_hcm/plots/layer_kinds_evo2_7b_L32768.png
- `op_categories_plot`: /root/vortex-kernels/results/h200/progression/hcs_hcm/plots/op_categories_evo2_7b_L32768.png
- `summary`: /root/vortex-kernels/results/h200/progression/hcs_hcm/summary_evo2_7b_L32768.json

## evo2_7b @ seq_len=65536

- Forward pass: **5807.58 ± 16.10 ms** (n=5 runs, warmup=3)
- Total leaf-op CUDA time: 114698.1 ms
- Block coverage: 29011.1 ms inside blocks, 85687.0 ms outside (embedding / lm_head / final norm)

### By layer kind

| Kind | CUDA ms | % of total |
|---|---:|---:|
| hcl | 13178.6 | 11.5% |
| hcm | 5870.4 | 5.1% |
| hcs | 4075.7 | 3.6% |
| attn | 5886.4 | 5.1% |

### By op category

| Category | CUDA ms | % of total |
|---|---:|---:|
| other | 54558.4 | 47.6% |
| gemm | 26502.0 | 23.1% |
| cast_copy | 8121.4 | 7.1% |
| conv | 7326.0 | 6.4% |
| elementwise | 7100.6 | 6.2% |
| fft | 6643.3 | 5.8% |
| attention | 3932.0 | 3.4% |
| reshape | 514.4 | 0.4% |

### Artifacts

- `chrome_trace`: /root/vortex-kernels/results/h200/progression/hcs_hcm/trace_evo2_7b_L65536.json
- `top_ops`: /root/vortex-kernels/results/h200/progression/hcs_hcm/top_ops_evo2_7b_L65536.txt
- `layer_breakdown`: /root/vortex-kernels/results/h200/progression/hcs_hcm/layer_breakdown_evo2_7b_L65536.json
- `per_layer_plot`: /root/vortex-kernels/results/h200/progression/hcs_hcm/plots/per_layer_evo2_7b_L65536.png
- `layer_kinds_plot`: /root/vortex-kernels/results/h200/progression/hcs_hcm/plots/layer_kinds_evo2_7b_L65536.png
- `op_categories_plot`: /root/vortex-kernels/results/h200/progression/hcs_hcm/plots/op_categories_evo2_7b_L65536.png
- `summary`: /root/vortex-kernels/results/h200/progression/hcs_hcm/summary_evo2_7b_L65536.json

## evo2_7b @ seq_len=131072

- Forward pass: **21350.09 ± 29.50 ms** (n=5 runs, warmup=3)
- Total leaf-op CUDA time: 448606.5 ms
- Block coverage: 106685.4 ms inside blocks, 341921.1 ms outside (embedding / lm_head / final norm)

### By layer kind

| Kind | CUDA ms | % of total |
|---|---:|---:|
| hcl | 57530.3 | 12.8% |
| hcm | 17340.8 | 3.9% |
| hcs | 12348.4 | 2.8% |
| attn | 19465.9 | 4.3% |

### By op category

| Category | CUDA ms | % of total |
|---|---:|---:|
| other | 232306.3 | 51.8% |
| gemm | 62082.1 | 13.8% |
| cast_copy | 46026.3 | 10.3% |
| conv | 39569.2 | 8.8% |
| elementwise | 35521.6 | 7.9% |
| fft | 17161.3 | 3.8% |
| attention | 15113.8 | 3.4% |
| reshape | 825.9 | 0.2% |

### Artifacts

- `chrome_trace`: /root/vortex-kernels/results/h200/progression/hcs_hcm/trace_evo2_7b_L131072.json
- `top_ops`: /root/vortex-kernels/results/h200/progression/hcs_hcm/top_ops_evo2_7b_L131072.txt
- `layer_breakdown`: /root/vortex-kernels/results/h200/progression/hcs_hcm/layer_breakdown_evo2_7b_L131072.json
- `per_layer_plot`: /root/vortex-kernels/results/h200/progression/hcs_hcm/plots/per_layer_evo2_7b_L131072.png
- `layer_kinds_plot`: /root/vortex-kernels/results/h200/progression/hcs_hcm/plots/layer_kinds_evo2_7b_L131072.png
- `op_categories_plot`: /root/vortex-kernels/results/h200/progression/hcs_hcm/plots/op_categories_evo2_7b_L131072.png
- `summary`: /root/vortex-kernels/results/h200/progression/hcs_hcm/summary_evo2_7b_L131072.json

