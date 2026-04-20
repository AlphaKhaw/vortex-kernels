"""
HCS short depthwise conv — fills the role of vortex/ops/hcs_interface.py.

Target: wire vortex/ops/hyena_x/triton_indirect_fwd.py (already written,
forward-only, never called from engine.py) into the default branch of
HyenaInferenceEngine.parallel_fir. Integration-only PR, lowest risk.
"""

from typing import NoReturn


def hcs_conv(*args: object, **kwargs: object) -> NoReturn:
    """
    Short depthwise convolution for Hyena Short (HCS) layers.

    Re-exports vortex.ops.hyena_x.triton_indirect_fwd (already present on main
    but never imported from engine.py) so HyenaInferenceEngine.parallel_fir can
    dispatch to it instead of F.conv1d when the Triton kernel is available.

    Raises:
        NotImplementedError: Always, until the wrapper is wired up.
    """
    _ = args, kwargs
    raise NotImplementedError("hcs_conv not yet implemented. See docs/issue_draft.md scope.")
