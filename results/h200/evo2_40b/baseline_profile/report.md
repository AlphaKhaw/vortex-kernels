# Evo2 baseline profile

Profiled 2 run(s). CUDA times are summed across runs.

## evo2_40b @ seq_len=8192

- Forward pass: **2294.12 ± 30.91 ms** (n=5 runs, warmup=3)
- Total leaf-op CUDA time: 47448.4 ms
- Block coverage: 11464.4 ms inside blocks, 35984.0 ms outside (embedding / lm_head / final norm)

### By layer kind

| Kind | CUDA ms | % of total |
|---|---:|---:|
| hcl | 4650.6 | 9.8% |
| hcm | 2898.1 | 6.1% |
| hcs | 2468.6 | 5.2% |
| attn | 1447.0 | 3.0% |

### By op category

| Category | CUDA ms | % of total |
|---|---:|---:|
| other | 19752.9 | 41.6% |
| gemm | 19431.3 | 41.0% |
| conv | 2531.9 | 5.3% |
| cast_copy | 2148.9 | 4.5% |
| elementwise | 2010.6 | 4.2% |
| fft | 1178.3 | 2.5% |
| reshape | 216.0 | 0.5% |
| attention | 178.5 | 0.4% |

### Artifacts

- `chrome_trace`: /root/vortex-kernels/results/h200/evo2_40b/baseline_profile/trace_evo2_40b_L8192.json
- `top_ops`: /root/vortex-kernels/results/h200/evo2_40b/baseline_profile/top_ops_evo2_40b_L8192.txt
- `layer_breakdown`: /root/vortex-kernels/results/h200/evo2_40b/baseline_profile/layer_breakdown_evo2_40b_L8192.json
- `per_layer_plot`: /root/vortex-kernels/results/h200/evo2_40b/baseline_profile/plots/per_layer_evo2_40b_L8192.png
- `layer_kinds_plot`: /root/vortex-kernels/results/h200/evo2_40b/baseline_profile/plots/layer_kinds_evo2_40b_L8192.png
- `op_categories_plot`: /root/vortex-kernels/results/h200/evo2_40b/baseline_profile/plots/op_categories_evo2_40b_L8192.png
- `summary`: /root/vortex-kernels/results/h200/evo2_40b/baseline_profile/summary_evo2_40b_L8192.json

## evo2_40b @ seq_len=32768

- Forward pass: **13593.55 ± 72.92 ms** (n=5 runs, warmup=3)
- Total leaf-op CUDA time: 280492.6 ms
- Block coverage: 67937.8 ms inside blocks, 212554.8 ms outside (embedding / lm_head / final norm)

### By layer kind

| Kind | CUDA ms | % of total |
|---|---:|---:|
| hcl | 30754.2 | 11.0% |
| hcm | 14049.0 | 5.0% |
| hcs | 13187.6 | 4.7% |
| attn | 9947.0 | 3.5% |

### By op category

| Category | CUDA ms | % of total |
|---|---:|---:|
| other | 128218.0 | 45.7% |
| gemm | 92117.7 | 32.8% |
| elementwise | 18174.9 | 6.5% |
| cast_copy | 16067.9 | 5.7% |
| conv | 15122.0 | 5.4% |
| fft | 6016.3 | 2.1% |
| attention | 3865.1 | 1.4% |
| reshape | 910.8 | 0.3% |

### Artifacts

- `chrome_trace`: /root/vortex-kernels/results/h200/evo2_40b/baseline_profile/trace_evo2_40b_L32768.json
- `top_ops`: /root/vortex-kernels/results/h200/evo2_40b/baseline_profile/top_ops_evo2_40b_L32768.txt
- `layer_breakdown`: /root/vortex-kernels/results/h200/evo2_40b/baseline_profile/layer_breakdown_evo2_40b_L32768.json
- `per_layer_plot`: /root/vortex-kernels/results/h200/evo2_40b/baseline_profile/plots/per_layer_evo2_40b_L32768.png
- `layer_kinds_plot`: /root/vortex-kernels/results/h200/evo2_40b/baseline_profile/plots/layer_kinds_evo2_40b_L32768.png
- `op_categories_plot`: /root/vortex-kernels/results/h200/evo2_40b/baseline_profile/plots/op_categories_evo2_40b_L32768.png
- `summary`: /root/vortex-kernels/results/h200/evo2_40b/baseline_profile/summary_evo2_40b_L32768.json

