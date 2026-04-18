"""vortex-kernels: optimized inference kernels for Vortex/Evo2.

Third-party. Not affiliated with Arc Institute or the Vortex core team.
"""

from __future__ import annotations

import os

from .patching import patch_vortex, unpatch_vortex
from .version import __version__

if os.environ.get("VORTEX_KERNELS_NO_AUTOPATCH", "0") != "1":
    patch_vortex()

__all__ = ["__version__", "patch_vortex", "unpatch_vortex"]
