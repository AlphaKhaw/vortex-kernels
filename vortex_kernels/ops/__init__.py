"""
Mirror of vortex/ops/ — kernels and interface files contributed to
upstream via the Phase 6 PR.

Directory layout matches upstream's `vortex/ops/` exactly so the
hand-port is `cp -r vortex_kernels/ops/* vortex-fork/vortex/ops/`.

This __init__.py is intentionally minimal. Upstream's
`vortex/ops/__init__.py` imports `attn_interface` which pulls
`flash_attn_2_cuda` (a CUDA-only C extension that fails on macOS). We
don't touch attention kernels, so we don't carry that import.
"""
