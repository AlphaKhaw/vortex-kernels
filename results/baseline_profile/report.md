# Evo2 baseline profile

Profiled 4 run(s). CUDA times are summed across runs.

## evo2_7b @ seq_len=2048

- Forward pass: **79.86 ± 1.99 ms** (n=5 runs, warmup=3)
- Total leaf-op CUDA time: 1440.0 ms
- Block coverage: 398.6 ms inside blocks, 1041.3 ms outside (embedding / lm_head / final norm)

### By layer kind

| Kind | CUDA ms | % of total |
|---|---:|---:|
| hcl | 176.0 | 12.2% |
| hcm | 101.5 | 7.0% |
| hcs | 87.0 | 6.0% |
| attn | 34.2 | 2.4% |

### By op category

| Category | CUDA ms | % of total |
|---|---:|---:|
| other | 591.8 | 41.1% |
| gemm | 390.4 | 27.1% |
| cast_copy | 147.9 | 10.3% |
| conv | 132.5 | 9.2% |
| elementwise | 106.3 | 7.4% |
| fft | 56.1 | 3.9% |
| reshape | 11.3 | 0.8% |
| attention | 3.7 | 0.3% |

### Artifacts

- `chrome_trace`: results/baseline_profile/trace_evo2_7b_L2048.json
- `top_ops`: results/baseline_profile/top_ops_evo2_7b_L2048.txt
- `layer_breakdown`: results/baseline_profile/layer_breakdown_evo2_7b_L2048.json
- `per_layer_plot`: results/baseline_profile/plots/per_layer_evo2_7b_L2048.png
- `layer_kinds_plot`: results/baseline_profile/plots/layer_kinds_evo2_7b_L2048.png
- `op_categories_plot`: results/baseline_profile/plots/op_categories_evo2_7b_L2048.png
- `summary`: results/baseline_profile/summary_evo2_7b_L2048.json

## evo2_7b @ seq_len=8192

- Forward pass: **382.18 ± 2.19 ms** (n=5 runs, warmup=3)
- Total leaf-op CUDA time: 7272.9 ms
- Block coverage: 1907.8 ms inside blocks, 5365.2 ms outside (embedding / lm_head / final norm)

### By layer kind

| Kind | CUDA ms | % of total |
|---|---:|---:|
| hcl | 825.9 | 11.4% |
| hcm | 443.0 | 6.1% |
| hcs | 396.3 | 5.4% |
| attn | 242.5 | 3.3% |

### By op category

| Category | CUDA ms | % of total |
|---|---:|---:|
| other | 3073.0 | 42.3% |
| gemm | 2061.9 | 28.4% |
| cast_copy | 650.9 | 8.9% |
| conv | 600.1 | 8.3% |
| elementwise | 496.7 | 6.8% |
| fft | 291.5 | 4.0% |
| reshape | 51.5 | 0.7% |
| attention | 47.4 | 0.7% |

### Artifacts

- `chrome_trace`: results/baseline_profile/trace_evo2_7b_L8192.json
- `top_ops`: results/baseline_profile/top_ops_evo2_7b_L8192.txt
- `layer_breakdown`: results/baseline_profile/layer_breakdown_evo2_7b_L8192.json
- `per_layer_plot`: results/baseline_profile/plots/per_layer_evo2_7b_L8192.png
- `layer_kinds_plot`: results/baseline_profile/plots/layer_kinds_evo2_7b_L8192.png
- `op_categories_plot`: results/baseline_profile/plots/op_categories_evo2_7b_L8192.png
- `summary`: results/baseline_profile/summary_evo2_7b_L8192.json

## evo2_7b @ seq_len=32768

- Forward pass: **2064.46 ± 8.16 ms** (n=5 runs, warmup=3)
- Total leaf-op CUDA time: 39649.1 ms
- Block coverage: 10309.8 ms inside blocks, 29339.3 ms outside (embedding / lm_head / final norm)

### By layer kind

| Kind | CUDA ms | % of total |
|---|---:|---:|
| hcl | 4536.2 | 11.4% |
| hcm | 2362.8 | 6.0% |
| hcs | 1859.9 | 4.7% |
| attn | 1550.9 | 3.9% |

### By op category

| Category | CUDA ms | % of total |
|---|---:|---:|
| other | 17727.3 | 44.7% |
| gemm | 10603.3 | 26.7% |
| conv | 2909.0 | 7.3% |
| cast_copy | 2852.5 | 7.2% |
| elementwise | 2761.4 | 7.0% |
| fft | 1762.7 | 4.4% |
| attention | 813.9 | 2.1% |
| reshape | 218.9 | 0.6% |

### Artifacts

- `chrome_trace`: results/baseline_profile/trace_evo2_7b_L32768.json
- `top_ops`: results/baseline_profile/top_ops_evo2_7b_L32768.txt
- `layer_breakdown`: results/baseline_profile/layer_breakdown_evo2_7b_L32768.json
- `per_layer_plot`: results/baseline_profile/plots/per_layer_evo2_7b_L32768.png
- `layer_kinds_plot`: results/baseline_profile/plots/layer_kinds_evo2_7b_L32768.png
- `op_categories_plot`: results/baseline_profile/plots/op_categories_evo2_7b_L32768.png
- `summary`: results/baseline_profile/summary_evo2_7b_L32768.json

## evo2_7b @ seq_len=65536

- Forward pass: **6681.08 ± 11.84 ms** (n=5 runs, warmup=3)
- Total leaf-op CUDA time: 128889.9 ms
- Block coverage: 33376.5 ms inside blocks, 95513.5 ms outside (embedding / lm_head / final norm)

### By layer kind

| Kind | CUDA ms | % of total |
|---|---:|---:|
| hcl | 16392.7 | 12.7% |
| hcm | 6868.4 | 5.3% |
| hcs | 4522.4 | 3.5% |
| attn | 5592.9 | 4.3% |

### By op category

| Category | CUDA ms | % of total |
|---|---:|---:|
| other | 62900.9 | 48.8% |
| gemm | 25894.1 | 20.1% |
| elementwise | 10347.7 | 8.0% |
| cast_copy | 9486.9 | 7.4% |
| conv | 7985.1 | 6.2% |
| fft | 7967.4 | 6.2% |
| attention | 3852.7 | 3.0% |
| reshape | 455.1 | 0.4% |

### Artifacts

- `chrome_trace`: results/baseline_profile/trace_evo2_7b_L65536.json
- `top_ops`: results/baseline_profile/top_ops_evo2_7b_L65536.txt
- `layer_breakdown`: results/baseline_profile/layer_breakdown_evo2_7b_L65536.json
- `per_layer_plot`: results/baseline_profile/plots/per_layer_evo2_7b_L65536.png
- `layer_kinds_plot`: results/baseline_profile/plots/layer_kinds_evo2_7b_L65536.png
- `op_categories_plot`: results/baseline_profile/plots/op_categories_evo2_7b_L65536.png
- `summary`: results/baseline_profile/summary_evo2_7b_L65536.json

