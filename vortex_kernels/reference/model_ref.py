"""
Free-function adapter for HyenaCascade.compute_filter from
vortex/model/model.py @ cb229ae.

The math is byte-identical to upstream (verified in tests). The class
scaffolding around it in model.py — AttentionBlock, HyenaCascade,
ParallelGatedConvBlock, StripedHyena — pulls in transformer_engine and
flash_attn imports that fail on macOS, so we deliberately do NOT
vendor those classes. Re-vendoring is a 9-line diff against
.venv/lib/python3.12/site-packages/vortex/model/model.py:390-398.

Sourced from:
    .venv/lib/python3.12/site-packages/vortex/model/model.py
on 2026-05-10.

Original copyright: (c) 2024, Michael Poli.
"""

import torch


def compute_filter_pure(
    residues: torch.Tensor,
    log_poles: torch.Tensor,
    L: int,
    device: torch.device | str,
    filter_dtype: torch.dtype = torch.float32,
):
    """
    Free-function form of HyenaCascade.compute_filter — no self, no caching.

    Args:
        residues (torch.Tensor): Modal residues, shape (D, state_size) fp32.
        log_poles (torch.Tensor): Modal log-poles, shape (D, state_size, 1) fp32.
        L (int): Sequence length.
        device (torch.device | str): Target device.
        filter_dtype (torch.dtype): Compute dtype, default torch.float32.

    Returns:
        Tuple of (h, filter_dtype, log_poles, residues) matching the upstream
        HyenaCascade.compute_filter return signature. h is shape (1, D, L).
    """
    t = torch.arange(L, device=device)[None, None].to(filter_dtype)
    residues = residues.to(filter_dtype)
    log_poles = log_poles.to(filter_dtype)
    h = (residues[..., None] * (log_poles * t).exp()).sum(1)[None]
    return h, filter_dtype, log_poles, residues
