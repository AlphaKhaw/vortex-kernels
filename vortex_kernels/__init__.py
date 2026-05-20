"""
vortex-kernels — profiling and benchmark harness for the Vortex/Evo2 Triton
HC{S,M,L} kernel work.

The kernels themselves are developed in a vortex fork (branch
`triton-hc-kernels`, editable-installed as `vtx`). This package only carries
the harness version; benchmarks and tests live under `benchmarks/` and
`tests/`.
"""

from .version import __version__

__all__ = ["__version__"]
