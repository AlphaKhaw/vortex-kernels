# Evo2 baseline profile

Profiled 2 run(s). CUDA times are summed across runs.

## evo2_7b @ seq_len=8192

- Forward pass: **1153.45 ± 26.37 ms** (n=5 runs, warmup=3)
- Total leaf-op CUDA time: 22536.0 ms
- Block coverage: 5761.3 ms inside blocks, 16774.6 ms outside (embedding / lm_head / final norm)

### By layer kind

| Kind | CUDA ms | % of total |
|---|---:|---:|
| hcl | 2403.3 | 10.7% |
| hcm | 1463.3 | 6.5% |
| hcs | 1226.2 | 5.4% |
| attn | 668.5 | 3.0% |

### By op category

| Category | CUDA ms | % of total |
|---|---:|---:|
| gemm | 11602.1 | 51.5% |
| other | 5947.2 | 26.4% |
| cast_copy | 1766.7 | 7.8% |
| conv | 1168.5 | 5.2% |
| elementwise | 1097.7 | 4.9% |
| fft | 790.7 | 3.5% |
| attention | 87.3 | 0.4% |
| reshape | 75.6 | 0.3% |

### Artifacts

- `chrome_trace`: /home/alpha/Projects/vortex-kernels/results/rtx-4090/progression/hcm_hcs/trace_evo2_7b_L8192.json
- `top_ops`: /home/alpha/Projects/vortex-kernels/results/rtx-4090/progression/hcm_hcs/top_ops_evo2_7b_L8192.txt
- `layer_breakdown`: /home/alpha/Projects/vortex-kernels/results/rtx-4090/progression/hcm_hcs/layer_breakdown_evo2_7b_L8192.json
- `per_layer_plot`: /home/alpha/Projects/vortex-kernels/results/rtx-4090/progression/hcm_hcs/plots/per_layer_evo2_7b_L8192.png
- `layer_kinds_plot`: /home/alpha/Projects/vortex-kernels/results/rtx-4090/progression/hcm_hcs/plots/layer_kinds_evo2_7b_L8192.png
- `op_categories_plot`: /home/alpha/Projects/vortex-kernels/results/rtx-4090/progression/hcm_hcs/plots/op_categories_evo2_7b_L8192.png
- `summary`: /home/alpha/Projects/vortex-kernels/results/rtx-4090/progression/hcm_hcs/summary_evo2_7b_L8192.json

## evo2_7b @ seq_len=32768

- Forward pass: **6370.84 ± 3.41 ms** (n=5 runs, warmup=3)
- Total leaf-op CUDA time: 124471.9 ms
- Block coverage: 31829.6 ms inside blocks, 92642.3 ms outside (embedding / lm_head / final norm)

### By layer kind

| Kind | CUDA ms | % of total |
|---|---:|---:|
| hcl | 14454.2 | 11.6% |
| hcm | 7739.6 | 6.2% |
| hcs | 5329.1 | 4.3% |
| attn | 4306.7 | 3.5% |

### By op category

| Category | CUDA ms | % of total |
|---|---:|---:|
| gemm | 53684.8 | 43.1% |
| other | 40400.8 | 32.5% |
| fft | 8821.6 | 7.1% |
| cast_copy | 7872.8 | 6.3% |
| elementwise | 6832.2 | 5.5% |
| conv | 5209.4 | 4.2% |
| attention | 1332.2 | 1.1% |
| reshape | 318.2 | 0.3% |

### Artifacts

- `chrome_trace`: /home/alpha/Projects/vortex-kernels/results/rtx-4090/progression/hcm_hcs/trace_evo2_7b_L32768.json
- `top_ops`: /home/alpha/Projects/vortex-kernels/results/rtx-4090/progression/hcm_hcs/top_ops_evo2_7b_L32768.txt
- `layer_breakdown`: /home/alpha/Projects/vortex-kernels/results/rtx-4090/progression/hcm_hcs/layer_breakdown_evo2_7b_L32768.json
- `per_layer_plot`: /home/alpha/Projects/vortex-kernels/results/rtx-4090/progression/hcm_hcs/plots/per_layer_evo2_7b_L32768.png
- `layer_kinds_plot`: /home/alpha/Projects/vortex-kernels/results/rtx-4090/progression/hcm_hcs/plots/layer_kinds_evo2_7b_L32768.png
- `op_categories_plot`: /home/alpha/Projects/vortex-kernels/results/rtx-4090/progression/hcm_hcs/plots/op_categories_evo2_7b_L32768.png
- `summary`: /home/alpha/Projects/vortex-kernels/results/rtx-4090/progression/hcm_hcs/summary_evo2_7b_L32768.json

