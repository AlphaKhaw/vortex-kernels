# Evo2 baseline profile

Profiled 1 run(s). CUDA times are summed across runs.

## evo2_7b @ seq_len=8192

- Forward pass: **1149.85 ± 26.46 ms** (n=5 runs, warmup=3)
- Total leaf-op CUDA time: 22477.5 ms
- Block coverage: 5743.4 ms inside blocks, 16734.1 ms outside (embedding / lm_head / final norm)

### By layer kind

| Kind | CUDA ms | % of total |
|---|---:|---:|
| hcl | 2399.5 | 10.7% |
| hcm | 1458.6 | 6.5% |
| hcs | 1221.4 | 5.4% |
| attn | 663.8 | 3.0% |

### By op category

| Category | CUDA ms | % of total |
|---|---:|---:|
| gemm | 11520.2 | 51.3% |
| other | 5952.3 | 26.5% |
| cast_copy | 1777.6 | 7.9% |
| conv | 1177.2 | 5.2% |
| elementwise | 1097.3 | 4.9% |
| fft | 790.6 | 3.5% |
| attention | 87.0 | 0.4% |
| reshape | 75.4 | 0.3% |

### Artifacts

- `chrome_trace`: results/progression/hcs_hcm/trace_evo2_7b_L8192.json
- `top_ops`: results/progression/hcs_hcm/top_ops_evo2_7b_L8192.txt
- `layer_breakdown`: results/progression/hcs_hcm/layer_breakdown_evo2_7b_L8192.json
- `per_layer_plot`: results/progression/hcs_hcm/plots/per_layer_evo2_7b_L8192.png
- `layer_kinds_plot`: results/progression/hcs_hcm/plots/layer_kinds_evo2_7b_L8192.png
- `op_categories_plot`: results/progression/hcs_hcm/plots/op_categories_evo2_7b_L8192.png
- `summary`: results/progression/hcs_hcm/summary_evo2_7b_L8192.json

