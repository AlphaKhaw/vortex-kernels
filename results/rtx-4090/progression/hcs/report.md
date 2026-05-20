# Evo2 baseline profile

Profiled 1 run(s). CUDA times are summed across runs.

## evo2_7b @ seq_len=8192

- Forward pass: **1153.68 ± 22.88 ms** (n=5 runs, warmup=3)
- Total leaf-op CUDA time: 22263.2 ms
- Block coverage: 5762.7 ms inside blocks, 16500.5 ms outside (embedding / lm_head / final norm)

### By layer kind

| Kind | CUDA ms | % of total |
|---|---:|---:|
| hcl | 2306.5 | 10.4% |
| hcm | 1491.9 | 6.7% |
| hcs | 1208.9 | 5.4% |
| attn | 755.3 | 3.4% |

### By op category

| Category | CUDA ms | % of total |
|---|---:|---:|
| gemm | 11413.2 | 51.3% |
| other | 5843.6 | 26.2% |
| cast_copy | 1673.9 | 7.5% |
| elementwise | 1188.7 | 5.3% |
| conv | 1141.3 | 5.1% |
| fft | 825.1 | 3.7% |
| attention | 104.4 | 0.5% |
| reshape | 73.0 | 0.3% |

### Artifacts

- `chrome_trace`: results/progression/hcs/trace_evo2_7b_L8192.json
- `top_ops`: results/progression/hcs/top_ops_evo2_7b_L8192.txt
- `layer_breakdown`: results/progression/hcs/layer_breakdown_evo2_7b_L8192.json
- `per_layer_plot`: results/progression/hcs/plots/per_layer_evo2_7b_L8192.png
- `layer_kinds_plot`: results/progression/hcs/plots/layer_kinds_evo2_7b_L8192.png
- `op_categories_plot`: results/progression/hcs/plots/op_categories_evo2_7b_L8192.png
- `summary`: results/progression/hcs/summary_evo2_7b_L8192.json

