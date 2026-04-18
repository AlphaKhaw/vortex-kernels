"""HCS short depthwise conv — fills the role of vortex/ops/hcs_interface.py.

Target: wire `vortex/ops/hyena_x/triton_indirect_fwd.py` (already written,
forward-only, never called from engine.py) into the default branch of
`HyenaInferenceEngine.parallel_fir`. Integration-only PR, lowest risk.
"""

from __future__ import annotations


def hcs_conv(*args, **kwargs):
    raise NotImplementedError("hcs_conv not yet implemented. See docs/issue_draft.md scope.")
