# Evo2 baseline profile

Profiled 4 run(s). CUDA times are summed across runs.

## evo2_7b @ seq_len=8192

- Forward pass: **351.94 ± 4.50 ms** (n=5 runs, warmup=3)
- Total leaf-op CUDA time: 6670.4 ms
- Block coverage: 1756.3 ms inside blocks, 4914.0 ms outside (embedding / lm_head / final norm)

### By layer kind

| Kind | CUDA ms | % of total |
|---|---:|---:|
| hcl | 744.2 | 11.2% |
| hcm | 415.8 | 6.2% |
| hcs | 380.5 | 5.7% |
| attn | 215.9 | 3.2% |

### By op category

| Category | CUDA ms | % of total |
|---|---:|---:|
| other | 2775.2 | 41.6% |
| gemm | 2019.0 | 30.3% |
| cast_copy | 560.3 | 8.4% |
| conv | 554.7 | 8.3% |
| elementwise | 421.4 | 6.3% |
| fft | 244.4 | 3.7% |
| attention | 48.0 | 0.7% |
| reshape | 47.3 | 0.7% |

### Artifacts

- `chrome_trace`: /root/vortex-kernels/results/h200/baseline_profile/trace_evo2_7b_L8192.json
- `top_ops`: /root/vortex-kernels/results/h200/baseline_profile/top_ops_evo2_7b_L8192.txt
- `layer_breakdown`: /root/vortex-kernels/results/h200/baseline_profile/layer_breakdown_evo2_7b_L8192.json
- `per_layer_plot`: /root/vortex-kernels/results/h200/baseline_profile/plots/per_layer_evo2_7b_L8192.png
- `layer_kinds_plot`: /root/vortex-kernels/results/h200/baseline_profile/plots/layer_kinds_evo2_7b_L8192.png
- `op_categories_plot`: /root/vortex-kernels/results/h200/baseline_profile/plots/op_categories_evo2_7b_L8192.png
- `summary`: /root/vortex-kernels/results/h200/baseline_profile/summary_evo2_7b_L8192.json

## evo2_7b @ seq_len=32768

- Forward pass: **1947.03 ± 11.74 ms** (n=5 runs, warmup=3)
- Total leaf-op CUDA time: 37714.1 ms
- Block coverage: 9723.5 ms inside blocks, 27990.6 ms outside (embedding / lm_head / final norm)

### By layer kind

| Kind | CUDA ms | % of total |
|---|---:|---:|
| hcl | 4111.6 | 10.9% |
| hcm | 2218.6 | 5.9% |
| hcs | 1799.7 | 4.8% |
| attn | 1593.6 | 4.2% |

### By op category

| Category | CUDA ms | % of total |
|---|---:|---:|
| other | 16648.2 | 44.1% |
| gemm | 10780.4 | 28.6% |
| conv | 2772.1 | 7.4% |
| cast_copy | 2512.6 | 6.7% |
| elementwise | 2405.9 | 6.4% |
| fft | 1536.1 | 4.1% |
| attention | 851.7 | 2.3% |
| reshape | 207.1 | 0.5% |

### Artifacts

- `chrome_trace`: /root/vortex-kernels/results/h200/baseline_profile/trace_evo2_7b_L32768.json
- `top_ops`: /root/vortex-kernels/results/h200/baseline_profile/top_ops_evo2_7b_L32768.txt
- `layer_breakdown`: /root/vortex-kernels/results/h200/baseline_profile/layer_breakdown_evo2_7b_L32768.json
- `per_layer_plot`: /root/vortex-kernels/results/h200/baseline_profile/plots/per_layer_evo2_7b_L32768.png
- `layer_kinds_plot`: /root/vortex-kernels/results/h200/baseline_profile/plots/layer_kinds_evo2_7b_L32768.png
- `op_categories_plot`: /root/vortex-kernels/results/h200/baseline_profile/plots/op_categories_evo2_7b_L32768.png
- `summary`: /root/vortex-kernels/results/h200/baseline_profile/summary_evo2_7b_L32768.json

## evo2_7b @ seq_len=65536

- Forward pass: **6044.05 ± 26.06 ms** (n=5 runs, warmup=3)
- Total leaf-op CUDA time: 117777.4 ms
- Block coverage: 30192.7 ms inside blocks, 87584.7 ms outside (embedding / lm_head / final norm)

### By layer kind

| Kind | CUDA ms | % of total |
|---|---:|---:|
| hcl | 13957.8 | 11.9% |
| hcm | 6160.3 | 5.2% |
| hcs | 4369.1 | 3.7% |
| attn | 5705.4 | 4.8% |

### By op category

| Category | CUDA ms | % of total |
|---|---:|---:|
| other | 56370.8 | 47.9% |
| gemm | 26614.8 | 22.6% |
| elementwise | 8480.1 | 7.2% |
| cast_copy | 8304.0 | 7.1% |
| conv | 7361.7 | 6.3% |
| fft | 6222.8 | 5.3% |
| attention | 3964.1 | 3.4% |
| reshape | 459.1 | 0.4% |

### Artifacts

- `chrome_trace`: /root/vortex-kernels/results/h200/baseline_profile/trace_evo2_7b_L65536.json
- `top_ops`: /root/vortex-kernels/results/h200/baseline_profile/top_ops_evo2_7b_L65536.txt
- `layer_breakdown`: /root/vortex-kernels/results/h200/baseline_profile/layer_breakdown_evo2_7b_L65536.json
- `per_layer_plot`: /root/vortex-kernels/results/h200/baseline_profile/plots/per_layer_evo2_7b_L65536.png
- `layer_kinds_plot`: /root/vortex-kernels/results/h200/baseline_profile/plots/layer_kinds_evo2_7b_L65536.png
- `op_categories_plot`: /root/vortex-kernels/results/h200/baseline_profile/plots/op_categories_evo2_7b_L65536.png
- `summary`: /root/vortex-kernels/results/h200/baseline_profile/summary_evo2_7b_L65536.json

## evo2_7b @ seq_len=131072

- Forward pass: **21819.54 ± 60.31 ms** (n=5 runs, warmup=3)
- Total leaf-op CUDA time: 445523.2 ms
- Block coverage: 109038.7 ms inside blocks, 336484.5 ms outside (embedding / lm_head / final norm)

### By layer kind

| Kind | CUDA ms | % of total |
|---|---:|---:|
| hcl | 58621.4 | 13.2% |
| hcm | 18582.8 | 4.2% |
| hcs | 12374.1 | 2.8% |
| attn | 19460.3 | 4.4% |

### By op category

| Category | CUDA ms | % of total |
|---|---:|---:|
| other | 231122.4 | 51.9% |
| gemm | 68239.6 | 15.3% |
| cast_copy | 40601.6 | 9.1% |
| elementwise | 38292.7 | 8.6% |
| conv | 34233.6 | 7.7% |
| fft | 16752.9 | 3.8% |
| attention | 15104.7 | 3.4% |
| reshape | 1175.6 | 0.3% |

### Artifacts

- `chrome_trace`: /root/vortex-kernels/results/h200/baseline_profile/trace_evo2_7b_L131072.json
- `top_ops`: /root/vortex-kernels/results/h200/baseline_profile/top_ops_evo2_7b_L131072.txt
- `layer_breakdown`: /root/vortex-kernels/results/h200/baseline_profile/layer_breakdown_evo2_7b_L131072.json
- `per_layer_plot`: /root/vortex-kernels/results/h200/baseline_profile/plots/per_layer_evo2_7b_L131072.png
- `layer_kinds_plot`: /root/vortex-kernels/results/h200/baseline_profile/plots/layer_kinds_evo2_7b_L131072.png
- `op_categories_plot`: /root/vortex-kernels/results/h200/baseline_profile/plots/op_categories_evo2_7b_L131072.png
- `summary`: /root/vortex-kernels/results/h200/baseline_profile/summary_evo2_7b_L131072.json

