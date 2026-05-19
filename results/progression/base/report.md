# Evo2 baseline profile

Profiled 1 run(s). CUDA times are summed across runs.

## evo2_7b @ seq_len=8192

- Forward pass: **1163.30 ± 26.17 ms** (n=5 runs, warmup=3)
- Total leaf-op CUDA time: 22354.1 ms
- Block coverage: 5810.8 ms inside blocks, 16543.3 ms outside (embedding / lm_head / final norm)

### By layer kind

| Kind | CUDA ms | % of total |
|---|---:|---:|
| hcl | 2308.0 | 10.3% |
| hcm | 1504.0 | 6.7% |
| hcs | 1244.8 | 5.6% |
| attn | 754.0 | 3.4% |

### By op category

| Category | CUDA ms | % of total |
|---|---:|---:|
| gemm | 11390.6 | 51.0% |
| other | 5915.5 | 26.5% |
| cast_copy | 1677.5 | 7.5% |
| elementwise | 1188.2 | 5.3% |
| conv | 1180.2 | 5.3% |
| fft | 825.1 | 3.7% |
| attention | 104.1 | 0.5% |
| reshape | 73.0 | 0.3% |

### Artifacts

- `chrome_trace`: results/progression/base/trace_evo2_7b_L8192.json
- `top_ops`: results/progression/base/top_ops_evo2_7b_L8192.txt
- `layer_breakdown`: results/progression/base/layer_breakdown_evo2_7b_L8192.json
- `per_layer_plot`: results/progression/base/plots/per_layer_evo2_7b_L8192.png
- `layer_kinds_plot`: results/progression/base/plots/layer_kinds_evo2_7b_L8192.png
- `op_categories_plot`: results/progression/base/plots/op_categories_evo2_7b_L8192.png
- `summary`: results/progression/base/summary_evo2_7b_L8192.json

