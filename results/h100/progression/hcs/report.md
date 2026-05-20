# Evo2 baseline profile

Profiled 3 run(s). CUDA times are summed across runs.

## evo2_7b @ seq_len=8192

- Forward pass: **374.16 ± 6.24 ms** (n=5 runs, warmup=3)
- Total leaf-op CUDA time: 6982.4 ms
- Block coverage: 1867.3 ms inside blocks, 5115.1 ms outside (embedding / lm_head / final norm)

### By layer kind

| Kind | CUDA ms | % of total |
|---|---:|---:|
| hcl | 827.0 | 11.8% |
| hcm | 451.0 | 6.5% |
| hcs | 373.9 | 5.4% |
| attn | 215.4 | 3.1% |

### By op category

| Category | CUDA ms | % of total |
|---|---:|---:|
| other | 2941.5 | 42.1% |
| gemm | 1977.0 | 28.3% |
| cast_copy | 633.2 | 9.1% |
| conv | 547.2 | 7.8% |
| elementwise | 493.5 | 7.1% |
| fft | 291.9 | 4.2% |
| reshape | 50.8 | 0.7% |
| attention | 47.3 | 0.7% |

### Artifacts

- `chrome_trace`: /root/vortex-kernels/results/h100/progression/hcs/trace_evo2_7b_L8192.json
- `top_ops`: /root/vortex-kernels/results/h100/progression/hcs/top_ops_evo2_7b_L8192.txt
- `layer_breakdown`: /root/vortex-kernels/results/h100/progression/hcs/layer_breakdown_evo2_7b_L8192.json
- `per_layer_plot`: /root/vortex-kernels/results/h100/progression/hcs/plots/per_layer_evo2_7b_L8192.png
- `layer_kinds_plot`: /root/vortex-kernels/results/h100/progression/hcs/plots/layer_kinds_evo2_7b_L8192.png
- `op_categories_plot`: /root/vortex-kernels/results/h100/progression/hcs/plots/op_categories_evo2_7b_L8192.png
- `summary`: /root/vortex-kernels/results/h100/progression/hcs/summary_evo2_7b_L8192.json

## evo2_7b @ seq_len=32768

- Forward pass: **2059.08 ± 1.90 ms** (n=5 runs, warmup=3)
- Total leaf-op CUDA time: 39371.2 ms
- Block coverage: 10282.8 ms inside blocks, 29088.5 ms outside (embedding / lm_head / final norm)

### By layer kind

| Kind | CUDA ms | % of total |
|---|---:|---:|
| hcl | 4588.0 | 11.7% |
| hcm | 2381.3 | 6.0% |
| hcs | 1744.3 | 4.4% |
| attn | 1569.1 | 4.0% |

### By op category

| Category | CUDA ms | % of total |
|---|---:|---:|
| other | 17557.1 | 44.6% |
| gemm | 10597.8 | 26.9% |
| cast_copy | 2843.4 | 7.2% |
| elementwise | 2803.0 | 7.1% |
| conv | 2744.1 | 7.0% |
| fft | 1777.6 | 4.5% |
| attention | 826.6 | 2.1% |
| reshape | 221.6 | 0.6% |

### Artifacts

- `chrome_trace`: /root/vortex-kernels/results/h100/progression/hcs/trace_evo2_7b_L32768.json
- `top_ops`: /root/vortex-kernels/results/h100/progression/hcs/top_ops_evo2_7b_L32768.txt
- `layer_breakdown`: /root/vortex-kernels/results/h100/progression/hcs/layer_breakdown_evo2_7b_L32768.json
- `per_layer_plot`: /root/vortex-kernels/results/h100/progression/hcs/plots/per_layer_evo2_7b_L32768.png
- `layer_kinds_plot`: /root/vortex-kernels/results/h100/progression/hcs/plots/layer_kinds_evo2_7b_L32768.png
- `op_categories_plot`: /root/vortex-kernels/results/h100/progression/hcs/plots/op_categories_evo2_7b_L32768.png
- `summary`: /root/vortex-kernels/results/h100/progression/hcs/summary_evo2_7b_L32768.json

## evo2_7b @ seq_len=65536

- Forward pass: **6564.82 ± 25.48 ms** (n=5 runs, warmup=3)
- Total leaf-op CUDA time: 126685.8 ms
- Block coverage: 32796.3 ms inside blocks, 93889.6 ms outside (embedding / lm_head / final norm)

### By layer kind

| Kind | CUDA ms | % of total |
|---|---:|---:|
| hcl | 15999.7 | 12.6% |
| hcm | 6876.2 | 5.4% |
| hcs | 4332.5 | 3.4% |
| attn | 5587.8 | 4.4% |

### By op category

| Category | CUDA ms | % of total |
|---|---:|---:|
| other | 61516.3 | 48.6% |
| gemm | 26004.8 | 20.5% |
| elementwise | 9968.5 | 7.9% |
| cast_copy | 9384.6 | 7.4% |
| fft | 7961.4 | 6.3% |
| conv | 7549.7 | 6.0% |
| attention | 3845.0 | 3.0% |
| reshape | 455.5 | 0.4% |

### Artifacts

- `chrome_trace`: /root/vortex-kernels/results/h100/progression/hcs/trace_evo2_7b_L65536.json
- `top_ops`: /root/vortex-kernels/results/h100/progression/hcs/top_ops_evo2_7b_L65536.txt
- `layer_breakdown`: /root/vortex-kernels/results/h100/progression/hcs/layer_breakdown_evo2_7b_L65536.json
- `per_layer_plot`: /root/vortex-kernels/results/h100/progression/hcs/plots/per_layer_evo2_7b_L65536.png
- `layer_kinds_plot`: /root/vortex-kernels/results/h100/progression/hcs/plots/layer_kinds_evo2_7b_L65536.png
- `op_categories_plot`: /root/vortex-kernels/results/h100/progression/hcs/plots/op_categories_evo2_7b_L65536.png
- `summary`: /root/vortex-kernels/results/h100/progression/hcs/summary_evo2_7b_L65536.json
