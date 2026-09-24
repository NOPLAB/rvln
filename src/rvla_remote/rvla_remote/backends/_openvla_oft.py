"""Shared OpenVLA-OFT plumbing for the AsyncVLA / OmniVLA-original cloud backends.

Both backends load the same OpenVLA-OFT family checkpoint layout
(OpenVLAForActionPrediction_MMNv1 backbone + ProprioProjector +
L1RegressionActionHead) and run an identical forward pass up to the
action-token hidden states. They differ only in the head that turns those
hidden states into the wire payload:

- OmniVLA-original (Plan 2B Path 1): L1 regression head ->
  ``(NUM_ACTIONS_CHUNK, ACTION_DIM)`` absolute waypoints.
- AsyncVLA (Plan 2A): ``Proj_Actiontokens`` -> ``(NUM_ACTIONS_CHUNK, 1024)``
  features for the edge's Edge_adapter.

This base class owns everything up to that fork. Subclasses implement
:meth:`_project_actions` / :attr:`_out_embed_dim` / :meth:`model_info` and load
any extra heads after ``super().__init__()``.

AsyncVLA's ``data_transformer_asyncvla`` is byte-identical to OmniVLA's
``data_transformer_omnivla``, which is why a single ``build_inference_batch``
serves both.

Note: importing prismatic pulls vint_train at module load time in the AsyncVLA
image. The MBRA submodule's ``train/`` directory must be on PYTHONPATH (set by
``Dockerfile.asyncvla`` / ``Dockerfile.real``).
"""
from __future__ import annotations

import logging
import time
from abc import abstractmethod
from typing import Optional, Tuple

import numpy as np
import PIL.Image
import torch

from ._checkpoints import load_checkpoint
from .base import VLABackend
from .omnivla_data_transform import build_inference_batch, determine_modality_id


_LOG = logging.getLogger(__name__)


class OpenVLAOFTBackendBase(VLABackend):
    """Checkpoint loading + shared forward pass for OpenVLA-OFT-family backends."""

    def __init__(
        self,
        *,
        vla_path: str,
        resume_step: int,
        device: str = 'cuda:0',
        dtype: torch.dtype = torch.bfloat16,
        num_images_in_input: int = 2,
    ) -> None:
        from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
        from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction_MMNv1
        from prismatic.extern.hf.processing_prismatic import (
            PrismaticImageProcessor, PrismaticProcessor,
        )
        from prismatic.models.action_heads import L1RegressionActionHead_idcat
        from prismatic.models.backbones.llm.prompting import PurePromptBuilder
        from prismatic.models.projectors import ProprioProjector
        from prismatic.vla.action_tokenizer import ActionTokenizer
        from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK, POSE_DIM
        from transformers import (
            AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor,
        )

        self._vla_path = vla_path
        self._resume_step = resume_step
        self._device = torch.device(device)
        self._dtype = dtype
        self._num_images_in_input = num_images_in_input
        self._prompt_builder_cls = PurePromptBuilder

        self._action_dim = int(ACTION_DIM)
        self._num_actions_chunk = int(NUM_ACTIONS_CHUNK)
        self._pose_dim = int(POSE_DIM)

        # Register custom HF Auto classes (idempotent).
        AutoConfig.register('openvla', OpenVLAConfig, exist_ok=True)
        AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor, exist_ok=True)
        AutoProcessor.register(OpenVLAConfig, PrismaticProcessor, exist_ok=True)
        AutoModelForVision2Seq.register(
            OpenVLAConfig, OpenVLAForActionPrediction_MMNv1, exist_ok=True)

        _LOG.info('loading processor + backbone from %s', vla_path)
        self._processor = AutoProcessor.from_pretrained(vla_path, trust_remote_code=True)
        self._vla = AutoModelForVision2Seq.from_pretrained(
            vla_path,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        ).to(self._device)
        self._vla.vision_backbone.set_num_images_in_input(num_images_in_input)
        self._vla = self._vla.to(dtype=dtype, device=self._device).eval()

        _LOG.info('loading auxiliary heads (step=%d)', resume_step)
        self._pose_projector = ProprioProjector(
            llm_dim=self._vla.llm_dim, proprio_dim=self._pose_dim,
        ).to(self._device).eval()
        self._pose_projector.load_state_dict(
            load_checkpoint('pose_projector', vla_path, resume_step, device=str(self._device)),
        )

        # AsyncVLA loads this but serves Proj_Actiontokens output instead; it is
        # instantiated anyway to keep the checkpoint footprint honest and to
        # support a future "no edge model" fallback mirroring OmniVLA Path 1.
        self._action_head = L1RegressionActionHead_idcat(
            input_dim=self._vla.llm_dim,
            hidden_dim=self._vla.llm_dim,
            action_dim=self._action_dim,
        ).to(dtype).to(self._device).eval()
        self._action_head.load_state_dict(
            load_checkpoint('action_head', vla_path, resume_step, device=str(self._device)),
        )

        self._num_patches = (
            self._vla.vision_backbone.get_num_patches()
            * self._vla.vision_backbone.get_num_images_in_input()
        ) + 1  # +1 for goal_pose token

        self._action_tokenizer = ActionTokenizer(self._processor.tokenizer)
        _LOG.info(
            'OpenVLA-OFT backbone ready (num_patches=%d, NUM_ACTIONS_CHUNK=%d, '
            'ACTION_DIM=%d, POSE_DIM=%d)',
            self._num_patches, self._num_actions_chunk, self._action_dim, self._pose_dim,
        )

    # ------------------------------------------------------------------- hooks

    @property
    @abstractmethod
    def _out_embed_dim(self) -> int:
        """Last dim of the served `(num_tokens, embed_dim)` payload."""

    @abstractmethod
    def _project_actions(
        self, actions_hidden: torch.Tensor, modality_id: torch.Tensor,
    ) -> torch.Tensor:
        """Map the action-token hidden states to the served output tensor.

        ``actions_hidden`` is ``(1, NUM_ACTIONS_CHUNK * ACTION_DIM, llm_dim)``;
        ``modality_id`` is already cast to the model dtype/device. The result is
        reshaped by the caller to ``(NUM_ACTIONS_CHUNK, _out_embed_dim)``.
        """

    # ------------------------------------------------------------- shared pass

    @torch.no_grad()
    def warmup(self, num_iters: int = 1) -> None:
        dummy = PIL.Image.new('RGB', (224, 224))
        for _ in range(max(1, num_iters)):
            self.infer(
                current_image=dummy, past_image=dummy,
                lang_instruction='warmup',
                goal_image=None, goal_pose_xy_theta=(0.0, 0.0, 0.0),
            )

    @torch.no_grad()
    def infer(
        self,
        *,
        current_image: PIL.Image.Image,
        past_image: Optional[PIL.Image.Image] = None,  # unused: single-frame prompt
        lang_instruction: str,
        goal_image: Optional[PIL.Image.Image],
        goal_pose_xy_theta: Optional[Tuple[float, float, float]],
    ) -> Tuple[np.ndarray, dict]:
        from prismatic.training.train_utils import (
            get_current_action_mask, get_next_actions_mask,
        )

        t0 = time.monotonic()

        modality_id_int = determine_modality_id(
            has_lang=bool(lang_instruction),
            has_pose=goal_pose_xy_theta is not None,
            has_image_goal=goal_image is not None,
        )
        modality_id = torch.as_tensor([modality_id_int], dtype=torch.float32)

        batch = build_inference_batch(
            current_image=current_image,
            goal_image=goal_image,
            lang_instruction=lang_instruction,
            goal_pose_xy_theta=goal_pose_xy_theta,
            action_tokenizer=self._action_tokenizer,
            processor=self._processor,
            prompt_builder_cls=self._prompt_builder_cls,
            pose_dim=self._pose_dim,
            num_actions_chunk=self._num_actions_chunk,
            action_dim=self._action_dim,
        )

        with torch.autocast(self._device.type, dtype=self._dtype):
            output = self._vla(
                input_ids=batch['input_ids'].to(self._device),
                attention_mask=batch['attention_mask'].to(self._device),
                pixel_values=batch['pixel_values'].to(self._dtype).to(self._device),
                modality_id=modality_id.to(self._dtype).to(self._device),
                labels=batch['labels'].to(self._device),
                output_hidden_states=True,
                proprio=batch['goal_pose'].to(self._dtype).to(self._device),
                proprio_projector=self._pose_projector,
                use_film=False,
            )

        gt_token_ids = batch['labels'][:, 1:].to(self._device)
        cur_mask = get_current_action_mask(gt_token_ids)
        next_mask = get_next_actions_mask(gt_token_ids)

        last_hidden = output.hidden_states[-1]
        text_hidden = last_hidden[:, self._num_patches:-1]
        actions_hidden = (
            text_hidden[cur_mask | next_mask]
            .reshape(1, self._num_actions_chunk * self._action_dim, -1)
            .to(self._dtype)
        )

        projected = self._project_actions(
            actions_hidden, modality_id.to(self._dtype).to(self._device),
        )
        arr = (
            projected
            .detach()
            .reshape(self._num_actions_chunk, self._out_embed_dim)
            .to(torch.float32)
            .cpu()
            .numpy()
        )

        return arr, {
            'inference_ms': (time.monotonic() - t0) * 1000.0,
            'modality_id': modality_id_int,
        }
