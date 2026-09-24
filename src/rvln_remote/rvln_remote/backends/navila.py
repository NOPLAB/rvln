"""NaVILA language navigation backend using the authors' released model code.

The upstream policy predicts one short metric move, turn, or stop. Keep model
loading separate from the pure action conversion so malformed text fails closed.
"""
from __future__ import annotations

import math
import re
import time
from collections import deque
from typing import Optional, Tuple

import numpy as np
import PIL.Image

from .base import ModelInfoDict, VLABackend


_ACTION = re.compile(
    r'^\s*(?:the\s+)?next\s+action\s+is\s+'
    r'(?:move\s+forward\s+(\d+)\s*cm|turn\s+(left|right)\s+'
    r'(\d+)\s*degrees?|stop)\s*\.?\s*$',
    re.IGNORECASE,
)


def action_to_embedding(response: str) -> np.ndarray:
    """Convert one canonical NaVILA action to one metric (x,y,cos,sin) point."""
    match = _ACTION.fullmatch(response)
    if match is None:
        raise ValueError(f'unrecognized NaVILA action: {response!r}')
    if match.group(1) is None and match.group(2) is None:
        return np.array([[0.0, 0.0, 1.0, 0.0]], dtype=np.float32)
    if match.group(1) is not None:
        centimetres = int(match.group(1))
        if not 0 < centimetres <= 75:
            raise ValueError(f'NaVILA forward distance outside 1..75 cm: {centimetres}')
        return np.array([[centimetres / 100.0, 0.0, 1.0, 0.0]], dtype=np.float32)
    degrees = int(match.group(3))
    if not 0 < degrees <= 45:
        raise ValueError(f'NaVILA turn outside 1..45 degrees: {degrees}')
    yaw = math.radians(degrees) * (1 if match.group(2).lower() == 'left' else -1)
    return np.array([[0.0, 0.0, math.cos(yaw), math.sin(yaw)]], dtype=np.float32)


def sample_and_pad_images(images: list, num_frames: int) -> list:
    """Use the same black-frame padding and temporal sampling as upstream eval."""
    if num_frames < 2:
        raise ValueError('NaVILA requires at least two video frames')
    frames = list(images)
    if not frames:
        raise ValueError('no observation frames')
    while len(frames) < num_frames:
        frames.insert(0, PIL.Image.new('RGB', (512, 512)))
    indices = np.linspace(0, len(frames) - 1, num=num_frames - 1,
                          endpoint=False, dtype=int)
    return [frames[i] for i in indices] + [frames[-1]]


class NaVILABackend(VLABackend):
    """Run the official NaVILA checkpoint on a GPU and return one short action."""

    def __init__(self, *, checkpoint_dir: str, device: str = 'cuda:0') -> None:
        import torch
        from llava.conversation import SeparatorStyle, conv_templates
        from llava.constants import IMAGE_TOKEN_INDEX
        from llava.mm_utils import (KeywordsStoppingCriteria, process_images,
                                    tokenizer_image_token)
        from llava.model.builder import load_pretrained_model

        if not device.startswith('cuda') or not torch.cuda.is_available():
            raise ValueError('NaVILA upstream inference requires a CUDA device')
        model_name = checkpoint_dir.rstrip('/').split('/')[-1]
        tokenizer, model, processor, _ = load_pretrained_model(checkpoint_dir, model_name)
        self._device = device
        self._torch = torch
        self._tokenizer = tokenizer
        self._model = model.to(device).eval()
        self._processor = processor
        self._conv_templates = conv_templates
        self._separator_style = SeparatorStyle
        self._image_token_index = IMAGE_TOKEN_INDEX
        self._stopping_criteria = KeywordsStoppingCriteria
        self._process_images = process_images
        self._tokenizer_image_token = tokenizer_image_token
        self._checkpoint_dir = checkpoint_dir
        self._num_frames = int(model.config.num_video_frames)
        self._frames = deque(maxlen=256)
        self._instruction = None

    def warmup(self, num_iters: int = 1) -> None:
        # A warmup with a synthetic navigation instruction would affect history.
        # The first real observation initializes the episode instead.
        pass

    def infer(
        self, *, current_image: PIL.Image.Image,
        past_image: Optional[PIL.Image.Image] = None,
        lang_instruction: str,
        goal_image: Optional[PIL.Image.Image],
        goal_pose_xy_theta: Optional[Tuple[float, float, float]],
    ) -> Tuple[np.ndarray, dict]:
        if goal_image is not None or goal_pose_xy_theta is not None or not lang_instruction:
            raise ValueError('NaVILA accepts a nonempty language goal only')
        if lang_instruction != self._instruction:
            self._frames.clear()
            self._instruction = lang_instruction
        t0 = time.monotonic()
        self._frames.append(current_image.convert('RGB'))
        frames = sample_and_pad_images(list(self._frames), self._num_frames)
        history_tokens = '<image>\n' * (len(frames) - 1)
        question = (
            'Imagine you are a robot programmed for navigation tasks. You have been '
            f'given a video of historical observations {history_tokens}, and current '
            f'observation <image>\n. Your assigned task is: "{lang_instruction}" '
            'Analyze this series of images to decide your next action, which could be '
            'turning left or right by a specific degree, moving forward a certain '
            'distance, or stop if the task is completed.'
        )
        conv = self._conv_templates['llama_3'].copy()
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        image_tensor = self._process_images(frames, self._processor, self._model.config)
        image_tensor = image_tensor.to(self._device, dtype=self._torch.float16)
        input_ids = self._tokenizer_image_token(
            prompt, self._tokenizer, self._image_token_index, return_tensors='pt',
        ).unsqueeze(0).to(self._device)
        stop_str = conv.sep if conv.sep_style != self._separator_style.TWO else conv.sep2
        stopping = self._stopping_criteria([stop_str], self._tokenizer, input_ids)
        with self._torch.inference_mode():
            output_ids = self._model.generate(
                input_ids, images=image_tensor, do_sample=False, temperature=0.0,
                max_new_tokens=32, use_cache=True, stopping_criteria=[stopping],
                pad_token_id=self._tokenizer.eos_token_id,
            )
        response = self._tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        response = response.removesuffix(stop_str).strip()
        return action_to_embedding(response), {
            'inference_ms': (time.monotonic() - t0) * 1000.0,
        }

    def model_info(self) -> ModelInfoDict:
        return ModelInfoDict(
            model_name='AnjieCheng/NaVILA', model_version=self._checkpoint_dir,
            num_tokens=1, embed_dim=4, device=self._device, ready=True,
        )
