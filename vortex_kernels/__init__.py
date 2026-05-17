"""
vortex-kernels — profiling and benchmark harness for the Vortex/Evo2 Triton
HC{S,M,L} kernel work.

The kernels themselves are developed in a vortex fork (branch
`triton-hc-kernels`, editable-installed as `vtx`) — see the vortex-kernels-dev
skill. This package carries the harness version and the frozen correctness
oracles in `reference/`.
"""

from .version import __version__

__all__ = ["__version__"]
