"""NaVIDA RGB language-navigation backend based on the official evaluation path."""
from __future__ import annotations

import base64
import importlib.util
import io
import math
import re
import time
from collections import deque
from typing import Optional, Tuple

import numpy as np
import PIL.Image

from .base import ModelInfoDict, VLABackend


_ANSWER = re.compile(r'<answer>(.*?)</answer>', re.IGNORECASE | re.DOTALL)
_ACTION = re.compile(
    r'^(?:move\s+)?forward(?:\s+(\d+)\s*cm)?$|^turn\s+(left|right)'
    r'(?:\s+(\d+)\s*degrees?)?$|^stop$',
    re.IGNORECASE,
)


def attention_implementation() -> str:
    """Use FlashAttention when installed, otherwise PyTorch's SDPA backend."""
    return 'flash_attention_2' if importlib.util.find_spec('flash_attn') else 'sdpa'


def action_to_embedding(response: str) -> np.ndarray:
    """Decode the first official discrete action into a short robot-frame Path."""
    tagged = _ANSWER.search(response)
    answer = tagged.group(1) if tagged else response
    first = answer.split(',', 1)[0].strip().rstrip('.')
    match = _ACTION.fullmatch(first)
    if match is None:
        raise ValueError(f'unrecognized NaVIDA action: {response!r}')
    if first.lower() == 'stop':
        return np.array([[0.0, 0.0, 1.0, 0.0]], dtype=np.float32)
    if match.group(1) is not None or first.lower().startswith(('forward', 'move forward')):
        distance = int(match.group(1) or 25)
        if not 0 < distance <= 75:
            raise ValueError(f'NaVIDA forward distance outside 1..75 cm: {distance}')
        return np.array([[distance / 100.0, 0.0, 1.0, 0.0]], dtype=np.float32)
    degrees = int(match.group(3) or 15)
    if not 0 < degrees <= 45:
        raise ValueError(f'NaVIDA turn outside 1..45 degrees: {degrees}')
    yaw = math.radians(degrees) * (1 if match.group(2).lower() == 'left' else -1)
    return np.array([[0.0, 0.0, math.cos(yaw), math.sin(yaw)]], dtype=np.float32)


def _data_url(image: PIL.Image.Image) -> str:
    buffer = io.BytesIO()
    image.convert('RGB').resize((308, 252)).save(buffer, format='JPEG')
    return 'data:image/jpeg;base64,' + base64.b64encode(buffer.getvalue()).decode('ascii')


def _sample_history(images: list, count: int = 8) -> list:
    if len(images) <= count:
        return images
    indices = [round(i * (len(images) - 1) / (count - 1)) for i in range(count)]
    return [images[i] for i in indices]


class NaVIDABackend(VLABackend):
    """Load the official Qwen2.5-VL checkpoint and infer one short action."""

    def __init__(self, *, checkpoint_dir: str, device: str = 'cuda:0') -> None:
        import torch
        from qwen_vl_utils import process_vision_info
        from transformers import (AutoProcessor, GenerationConfig,
                                  Qwen2_5_VLForConditionalGeneration)

        if not device.startswith('cuda') or not torch.cuda.is_available():
            raise ValueError('NaVIDA upstream inference requires a CUDA device')
        self._attention_impl = attention_implementation()
        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            checkpoint_dir, attn_implementation=self._attention_impl,
            torch_dtype=torch.bfloat16,
        ).to(device).eval()
        self._processor = AutoProcessor.from_pretrained(checkpoint_dir)
        self._processor.image_processor.max_pixels = 501760
        self._generation_config = GenerationConfig(
            do_sample=True, temperature=0.2, max_new_tokens=512, top_p=1.0,
            use_cache=True, repetition_penalty=1.05, num_return_sequences=1,
        )
        self._process_vision_info = process_vision_info
        self._torch = torch
        self._device = device
        self._checkpoint_dir = checkpoint_dir
        self._frames = deque(maxlen=10)
        self._instruction = None

    def warmup(self, num_iters: int = 1) -> None:
        pass

    def infer(
        self, *, current_image: PIL.Image.Image,
        past_image: Optional[PIL.Image.Image] = None,
        lang_instruction: str,
        goal_image: Optional[PIL.Image.Image],
        goal_pose_xy_theta: Optional[Tuple[float, float, float]],
    ) -> Tuple[np.ndarray, dict]:
        if goal_image is not None or goal_pose_xy_theta is not None or not lang_instruction:
            raise ValueError('NaVIDA accepts a nonempty language goal only')
        if lang_instruction != self._instruction:
            self._frames.clear()
            self._instruction = lang_instruction
        t0 = time.monotonic()
        self._frames.append(current_image.convert('RGB'))
        history = _sample_history(list(self._frames)[:-1]) or [self._frames[-1]]
        content = [
            {'type': 'text', 'text': 'Imagine you are a robot programmed for navigation tasks. '
             'You have been given a video of historical observations'},
        ]
        content.extend({'type': 'image', 'image': _data_url(frame)}
                       for frame in history)
        content.extend([
            {'type': 'text', 'text': 'and an image of the current observation'},
            {'type': 'image', 'image': _data_url(self._frames[-1])},
            {'type': 'text', 'text': f'. Your assigned task is: \'{lang_instruction}\'. '
             'Analyze this series of images to decide your next move, which could involve '
             'turning left or right by a specific degree or moving forward a certain '
             'distance.'},
        ])
        messages = [
            {'role': 'system', 'content': [
                {'type': 'text', 'text': 'You are a helpful assistant.'},
            ]},
            {'role': 'user', 'content': content},
        ]
        prompt = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        images, _ = self._process_vision_info(messages)
        inputs = self._processor(text=[prompt], images=images, return_tensors='pt',
                                 padding=True).to(self._device)
        with self._torch.inference_mode():
            output = self._model.generate(
                **inputs, generation_config=self._generation_config)
        prefix_len = inputs['input_ids'].shape[1]
        response = self._processor.batch_decode(
            output[:, prefix_len:], skip_special_tokens=True)[0].strip()
        return action_to_embedding(response), {
            'inference_ms': (time.monotonic() - t0) * 1000.0,
            'raw_response': response,
        }

    def model_info(self) -> ModelInfoDict:
        return ModelInfoDict(
            model_name='waynechu/NaVIDA',
            model_version=f'{self._checkpoint_dir} attention={self._attention_impl}',
            num_tokens=1, embed_dim=4, device=self._device, ready=True,
        )
