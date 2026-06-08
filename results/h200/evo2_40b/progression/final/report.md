# Evo2 baseline profile

Profiled 3 run(s). CUDA times are summed across runs.

## evo2_40b @ seq_len=8192

- Forward pass: **1903.97 ± 36.06 ms** (n=5 runs, warmup=3)
- Total leaf-op CUDA time: 42081.3 ms
- Block coverage: 9512.4 ms inside blocks, 32568.9 ms outside (embedding / lm_head / final norm)

### By layer kind

| Kind | CUDA ms | % of total |
|---|---:|---:|
| hcl | 2666.8 | 6.3% |
| hcm | 2535.2 | 6.0% |
| hcs | 2557.0 | 6.1% |
| attn | 1753.4 | 4.2% |

### By op category

| Category | CUDA ms | % of total |
|---|---:|---:|
| gemm | 19620.6 | 46.6% |
| other | 16464.2 | 39.1% |
| cast_copy | 2320.8 | 5.5% |
| conv | 2172.2 | 5.2% |
| fft | 586.1 | 1.4% |
| elementwise | 504.3 | 1.2% |
| reshape | 206.7 | 0.5% |
| attention | 206.4 | 0.5% |

### Artifacts

- `chrome_trace`: /root/vortex-kernels/results/h200/evo2_40b/progression/final/trace_evo2_40b_L8192.json
- `top_ops`: /root/vortex-kernels/results/h200/evo2_40b/progression/final/top_ops_evo2_40b_L8192.txt
- `layer_breakdown`: /root/vortex-kernels/results/h200/evo2_40b/progression/final/layer_breakdown_evo2_40b_L8192.json
- `per_layer_plot`: /root/vortex-kernels/results/h200/evo2_40b/progression/final/plots/per_layer_evo2_40b_L8192.png
- `layer_kinds_plot`: /root/vortex-kernels/results/h200/evo2_40b/progression/final/plots/layer_kinds_evo2_40b_L8192.png
- `op_categories_plot`: /root/vortex-kernels/results/h200/evo2_40b/progression/final/plots/op_categories_evo2_40b_L8192.png
- `summary`: /root/vortex-kernels/results/h200/evo2_40b/progression/final/summary_evo2_40b_L8192.json

## evo2_40b @ seq_len=32768

- Forward pass: **9106.80 ± 20.49 ms** (n=5 runs, warmup=3)
- Total leaf-op CUDA time: 210277.8 ms
- Block coverage: 45500.2 ms inside blocks, 164777.6 ms outside (embedding / lm_head / final norm)

### By layer kind

| Kind | CUDA ms | % of total |
|---|---:|---:|
| hcl | 12633.3 | 6.0% |
| hcm | 12748.1 | 6.1% |
| hcs | 11564.0 | 5.5% |
| attn | 8554.9 | 4.1% |

### By op category

| Category | CUDA ms | % of total |
|---|---:|---:|
| gemm | 87682.9 | 41.7% |
| other | 84110.7 | 40.0% |
| cast_copy | 15191.5 | 7.2% |
| conv | 14200.1 | 6.8% |
| fft | 4127.9 | 2.0% |
| attention | 2212.9 | 1.1% |
| elementwise | 2052.4 | 1.0% |
| reshape | 699.4 | 0.3% |

### Artifacts

- `chrome_trace`: /root/vortex-kernels/results/h200/evo2_40b/progression/final/trace_evo2_40b_L32768.json
- `top_ops`: /root/vortex-kernels/results/h200/evo2_40b/progression/final/top_ops_evo2_40b_L32768.txt
- `layer_breakdown`: /root/vortex-kernels/results/h200/evo2_40b/progression/final/layer_breakdown_evo2_40b_L32768.json
- `per_layer_plot`: /root/vortex-kernels/results/h200/evo2_40b/progression/final/plots/per_layer_evo2_40b_L32768.png
- `layer_kinds_plot`: /root/vortex-kernels/results/h200/evo2_40b/progression/final/plots/layer_kinds_evo2_40b_L32768.png
- `op_categories_plot`: /root/vortex-kernels/results/h200/evo2_40b/progression/final/plots/op_categories_evo2_40b_L32768.png
- `summary`: /root/vortex-kernels/results/h200/evo2_40b/progression/final/summary_evo2_40b_L32768.json

## evo2_40b @ seq_len=65536

- Forward pass: **26118.97 ± 44.20 ms** (n=5 runs, warmup=3)
- Total leaf-op CUDA time: 635177.3 ms
- Block coverage: 130528.3 ms inside blocks, 504649.0 ms outside (embedding / lm_head / final norm)

### By layer kind

| Kind | CUDA ms | % of total |
|---|---:|---:|
| hcl | 36799.9 | 5.8% |
| hcm | 34279.8 | 5.4% |
| hcs | 31136.4 | 4.9% |
| attn | 28312.1 | 4.5% |

### By op category

| Category | CUDA ms | % of total |
|---|---:|---:|
| other | 276954.9 | 43.6% |
| gemm | 194157.1 | 30.6% |
| cast_copy | 65063.1 | 10.2% |
| conv | 56411.9 | 8.9% |
| fft | 22039.9 | 3.5% |
| attention | 13346.8 | 2.1% |
| elementwise | 5264.2 | 0.8% |
| reshape | 1939.4 | 0.3% |

### Artifacts

- `chrome_trace`: /root/vortex-kernels/results/h200/evo2_40b/progression/final/trace_evo2_40b_L65536.json
- `top_ops`: /root/vortex-kernels/results/h200/evo2_40b/progression/final/top_ops_evo2_40b_L65536.txt
- `layer_breakdown`: /root/vortex-kernels/results/h200/evo2_40b/progression/final/layer_breakdown_evo2_40b_L65536.json
- `per_layer_plot`: /root/vortex-kernels/results/h200/evo2_40b/progression/final/plots/per_layer_evo2_40b_L65536.png
- `layer_kinds_plot`: /root/vortex-kernels/results/h200/evo2_40b/progression/final/plots/layer_kinds_evo2_40b_L65536.png
- `op_categories_plot`: /root/vortex-kernels/results/h200/evo2_40b/progression/final/plots/op_categories_evo2_40b_L65536.png
- `summary`: /root/vortex-kernels/results/h200/evo2_40b/progression/final/summary_evo2_40b_L65536.json

