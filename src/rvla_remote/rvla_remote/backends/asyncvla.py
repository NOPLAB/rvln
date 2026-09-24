"""AsyncVLA cloud backend (Plan 2A).

Loads the OpenVLA-OFT backbone + ProprioProjector + L1RegressionActionHead +
Proj_Actiontokens from NHirose/AsyncVLA_release. Runs the full forward pass,
projects the action token hidden states with Proj_Actiontokens, and returns
the resulting `(NUM_ACTIONS_CHUNK=8, 1024)` tensor as the cloud->edge
ActionEmbedding payload. The edge consumes this with its own Edge_adapter
(see :class:`AsyncVLAEdgeAdapter` in rvla_edge.adapters.asyncvla).

The backbone loading and forward pass are shared with OmniVLA via
:class:`~rvla_remote.backends._openvla_oft.OpenVLAOFTBackendBase`
(the action_head it loads is unused at inference here — Plan 2A keeps the edge
in the loop); this subclass adds the AsyncVLA-only Proj_Actiontokens head.

Note: importing prismatic.models.small_head pulls vint_train at module load
time. The MBRA submodule's `train/` directory must be on PYTHONPATH (set by
``Dockerfile.asyncvla``).
"""
from __future__ import annotations

import logging

import torch

from ._checkpoints import load_checkpoint
from ._openvla_oft import OpenVLAOFTBackendBase
from .base import ModelInfoDict


_LOG = logging.getLogger(__name__)


class AsyncVLABackend(OpenVLAOFTBackendBase):
    """Cloud backend running the AsyncVLA forward pass + Proj_Actiontokens."""

    def __init__(
        self,
        *,
        vla_path: str,
        resume_step: int = 750000,
        device: str = 'cuda:0',
        dtype: torch.dtype = torch.bfloat16,
        num_images_in_input: int = 2,
    ) -> None:
        super().__init__(
            vla_path=vla_path,
            resume_step=resume_step,
            device=device,
            dtype=dtype,
            num_images_in_input=num_images_in_input,
        )

        from prismatic.models.small_head import Proj_Actiontokens

        # AsyncVLA's Proj_Actiontokens hidden_dim is 1024 (set in run_asyncvla.py:660).
        self._cloud_action_dim = 1024

        # AsyncVLA-only: cloud action projector that compresses
        # (B, NUM_ACTIONS_CHUNK*ACTION_DIM, llm_dim) -> (B, NUM_ACTIONS_CHUNK, 1024).
        _LOG.info('loading Proj_Actiontokens (step=%d)', resume_step)
        self._action_proj = Proj_Actiontokens(
            input_dim=self._vla.llm_dim,
            hidden_dim=self._vla.llm_dim,
            action_dim=self._cloud_action_dim,
        ).to(dtype).to(self._device).eval()
        self._action_proj.load_state_dict(
            load_checkpoint('action_proj', vla_path, resume_step, device=str(self._device)),
        )

    @property
    def _out_embed_dim(self) -> int:
        return self._cloud_action_dim

    def _project_actions(
        self, actions_hidden: torch.Tensor, modality_id: torch.Tensor,
    ) -> torch.Tensor:
        # Project the action hidden states down to the (NUM_ACTIONS_CHUNK, 1024)
        # tensor that the edge's Edge_adapter expects.
        return self._action_proj.predict_action(actions_hidden.detach(), modality_id)

    def model_info(self) -> ModelInfoDict:
        return ModelInfoDict(
            model_name='NHirose/AsyncVLA_release',
            model_version=f'asyncvla-step{self._resume_step}',
            num_tokens=self._num_actions_chunk,
            embed_dim=self._cloud_action_dim,
            device=str(self._device),
            ready=True,
        )
