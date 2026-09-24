"""OmniVLA-original cloud backend (Plan 2B Path 1).

Loads the OpenVLA-OFT backbone + ProprioProjector + L1RegressionActionHead from
NHirose/omnivla-original and serves the predicted action chunk
``(NUM_ACTIONS_CHUNK, ACTION_DIM)`` as the cloud->edge ActionEmbedding payload.

The edge does no further model work in Path 1 (see :class:`OmniVLAEdgeAdapter`
in rvln_edge.adapters.omnivla). This module is the inverse: all the
heavy compute lives here — shared with AsyncVLA via
:class:`~rvln_remote.backends._openvla_oft.OpenVLAOFTBackendBase`; only
the output head differs.
"""
from __future__ import annotations

import torch

from ._openvla_oft import OpenVLAOFTBackendBase
from .base import ModelInfoDict
from .omnivla_data_transform import METRIC_WAYPOINT_SPACING


class OmniVLABackend(OpenVLAOFTBackendBase):
    """Real-model backend running the OmniVLA-original forward pass on a GPU."""

    def __init__(
        self,
        *,
        vla_path: str,
        resume_step: int = 120000,
        device: str = 'cuda:0',
        dtype: torch.dtype = torch.bfloat16,
        num_images_in_input: int = 2,
        use_l1_regression: bool = True,
    ) -> None:
        if not use_l1_regression:
            raise NotImplementedError('use_l1_regression=False (diffusion head) not wired yet')
        self._use_l1_regression = use_l1_regression
        super().__init__(
            vla_path=vla_path,
            resume_step=resume_step,
            device=device,
            dtype=dtype,
            num_images_in_input=num_images_in_input,
        )

    @property
    def _out_embed_dim(self) -> int:
        return self._action_dim

    def _project_actions(
        self, actions_hidden: torch.Tensor, modality_id: torch.Tensor,
    ) -> torch.Tensor:
        # The L1 regression head decodes the hidden states straight to waypoints
        # (x, y, cos, sin) in waypoint-spacing units. Scale x/y to metres here so
        # the wire payload matches the Path 3 backend (omnivla_edge.py) and the
        # edge's path-only adapter can stay a pure pass-through
        # (run_omnivla.py:195 applies the same metric_waypoint_spacing).
        waypoints = self._action_head.predict_action(actions_hidden, modality_id)
        waypoints = waypoints.clone()
        waypoints[..., :2] *= METRIC_WAYPOINT_SPACING
        return waypoints

    def model_info(self) -> ModelInfoDict:
        return ModelInfoDict(
            model_name='NHirose/omnivla-original',
            model_version=f'omnivla-orig-step{self._resume_step}',
            num_tokens=self._num_actions_chunk,
            embed_dim=self._action_dim,
            device=str(self._device),
            ready=True,
        )
