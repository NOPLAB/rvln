"""Checkpoint helpers shared by the AsyncVLA / OmniVLA backends.

Ports `remove_ddp_in_checkpoint` and `load_checkpoint` from
``external/OmniVLA/inference/run_omnivla.py:46-55`` (also identical to the
AsyncVLA copy). The on-disk filename pattern is ``<module>--<step>_checkpoint.pt``;
``pose_projector`` falls back to ``proprio_projector`` because OmniVLA's
release uses the latter name on disk while the code refers to the former.
"""
from __future__ import annotations

import os
from typing import Dict

import torch


def remove_ddp_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Strip the ``module.`` prefix that DDP adds to checkpoint keys."""
    return {
        (k[len('module.'):] if k.startswith('module.') else k): v
        for k, v in state_dict.items()
    }


def _resolve_path(module_name: str, path: str, step: int) -> str:
    primary = os.path.join(path, f'{module_name}--{step}_checkpoint.pt')
    if os.path.exists(primary):
        return primary
    if module_name == 'pose_projector':
        fallback = os.path.join(path, f'proprio_projector--{step}_checkpoint.pt')
        if os.path.exists(fallback):
            return fallback
    raise FileNotFoundError(
        f'no checkpoint for module={module_name!r} step={step} under {path!r}'
    )


def load_checkpoint(
    module_name: str,
    path: str,
    step: int,
    device: str = 'cpu',
) -> Dict[str, torch.Tensor]:
    """Load ``<module>--<step>_checkpoint.pt`` and strip DDP wrapping."""
    full = _resolve_path(module_name, path, step)
    state_dict = torch.load(full, map_location=device)
    return remove_ddp_prefix(state_dict)
