"""Habitat-free inference graph for the pinned VLN-CE CMA_PM_DA_Aug model.

Architecture follows VLN-CE 729d141 and Habitat-Lab v0.1.7 (both MIT).
Only the single-environment inference path is implemented. Official checkpoint
compatibility and numerical parity must be checked with the licensed weights.
"""
from __future__ import annotations

import math
import pickle
import types
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet50 as vision_resnet50


class DepthBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes: int, planes: int, stride: int = 1,
                 downsample: nn.Module | None = None):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(inplanes, planes, 1, bias=False), nn.GroupNorm(16, planes),
            nn.ReLU(True), nn.Conv2d(planes, planes, 3, stride, 1, bias=False),
            nn.GroupNorm(16, planes), nn.ReLU(True),
            nn.Conv2d(planes, planes * 4, 1, bias=False),
            nn.GroupNorm(16, planes * 4))
        self.downsample = downsample
        self.relu = nn.ReLU(True)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        residual = tensor if self.downsample is None else self.downsample(tensor)
        return self.relu(self.convs(tensor) + residual)


class DepthResNet50(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(1, 32, 7, 2, 3, bias=False),
                                   nn.GroupNorm(16, 32), nn.ReLU(True))
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        self.inplanes = 32
        self.layer1 = self._layer(32, 3)
        self.layer2 = self._layer(64, 4, 2)
        self.layer3 = self._layer(128, 6, 2)
        self.layer4 = self._layer(256, 3, 2)

    def _layer(self, planes: int, count: int, stride: int = 1) -> nn.Sequential:
        target = planes * DepthBottleneck.expansion
        downsample = None
        if stride != 1 or self.inplanes != target:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, target, 1, stride,
                                                  bias=False), nn.GroupNorm(16, target))
        blocks = [DepthBottleneck(self.inplanes, planes, stride, downsample)]
        self.inplanes = target
        blocks.extend(DepthBottleneck(target, planes) for _ in range(1, count))
        return nn.Sequential(*blocks)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = self.maxpool(self.conv1(tensor))
        for layer in (self.layer1, self.layer2, self.layer3, self.layer4):
            tensor = layer(tensor)
        return tensor


class DepthVisualEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.running_mean_and_var = nn.Sequential()
        self.backbone = DepthResNet50()
        self.compression = nn.Sequential(nn.Conv2d(1024, 128, 3, padding=1,
                                                    bias=False),
                                         nn.GroupNorm(1, 128), nn.ReLU(True))

    def forward(self, depth: torch.Tensor) -> torch.Tensor:
        depth = F.avg_pool2d(depth.permute(0, 3, 1, 2), 2)
        return self.compression(self.backbone(depth))


class DepthEncoder(nn.Module):
    output_shape = (192, 4, 4)

    def __init__(self):
        super().__init__()
        self.visual_encoder = DepthVisualEncoder()
        self.spatial_embeddings = nn.Embedding(16, 64)

    def forward(self, depth: torch.Tensor) -> torch.Tensor:
        tensor = self.visual_encoder(depth)
        batch, _, height, width = tensor.shape
        spatial = self.spatial_embeddings(
            torch.arange(height * width, device=tensor.device))
        spatial = spatial.view(1, 64, height, width)
        return torch.cat((tensor, spatial.expand(batch, -1, -1, -1)), dim=1)


class SpatialAvgPool(nn.Module):
    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return F.adaptive_avg_pool2d(tensor, (4, 4))


class RGBEncoder(nn.Module):
    output_shape = (2112, 4, 4)

    def __init__(self):
        super().__init__()
        backbone = vision_resnet50(weights=None)
        self.cnn = nn.Sequential(*list(backbone.children())[:-2])
        self.cnn.add_module('avgpool', SpatialAvgPool())
        self.spatial_embeddings = nn.Embedding(16, 64)

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        tensor = self.cnn(rgb.permute(0, 3, 1, 2).float() / 255.0)
        batch, _, height, width = tensor.shape
        spatial = self.spatial_embeddings(
            torch.arange(height * width, device=tensor.device))
        spatial = spatial.view(1, 64, height, width)
        return torch.cat((tensor, spatial.expand(batch, -1, -1, -1)), dim=1)


class InstructionEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder_rnn = nn.LSTM(50, 128, bidirectional=True)
        self.embedding_layer = nn.Embedding(2504, 50, padding_idx=0)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        lengths = (tokens != 0).sum(dim=1).cpu()
        embedded = self.embedding_layer(tokens)
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths, batch_first=True, enforce_sorted=False)
        output, _ = self.encoder_rnn(packed)
        return nn.utils.rnn.pad_packed_sequence(output, batch_first=True)[0].permute(
            0, 2, 1)


class StateEncoder(nn.Module):
    num_recurrent_layers = 1

    def __init__(self, input_size: int):
        super().__init__()
        self.rnn = nn.GRU(input_size, 512)

    def forward(self, tensor: torch.Tensor, state: torch.Tensor,
                keep_state: bool) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = state.transpose(0, 1)
        if not keep_state:
            hidden = torch.zeros_like(hidden)
        output, hidden = self.rnn(tensor.unsqueeze(0), hidden)
        return output.squeeze(0), hidden.transpose(0, 1)


class CMANet(nn.Module):
    def __init__(self):
        super().__init__()
        self.instruction_encoder = InstructionEncoder()
        self.depth_encoder = DepthEncoder()
        self.rgb_encoder = RGBEncoder()
        self.prev_action_embedding = nn.Embedding(5, 32)
        self.rgb_linear = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(),
                                        nn.Linear(2112, 256), nn.ReLU(True))
        self.depth_linear = nn.Sequential(nn.Flatten(), nn.Linear(192 * 16, 128),
                                          nn.ReLU(True))
        self.state_encoder = StateEncoder(128 + 256 + 32)
        self.rgb_kv = nn.Conv1d(2112, 512, 1)
        self.depth_kv = nn.Conv1d(192, 384, 1)
        self.state_q = nn.Linear(512, 256)
        self.text_k = nn.Conv1d(256, 256, 1)
        self.text_q = nn.Linear(256, 256)
        self.register_buffer('_scale', torch.tensor(1.0 / math.sqrt(256)))
        self.second_state_compress = nn.Sequential(nn.Linear(512 + 256 + 256 + 128 + 32,
                                                             512), nn.ReLU(True))
        self.second_state_encoder = StateEncoder(512)
        self.progress_monitor = nn.Linear(512, 1)

    def _attn(self, query: torch.Tensor, key: torch.Tensor,
              value: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        logits = torch.einsum('nc,nci->ni', query, key)
        if mask is not None:
            logits = logits - mask.float() * 1e8
        weights = F.softmax(logits * self._scale, dim=1)
        return torch.einsum('ni,nci->nc', weights, value)

    def forward(self, rgb: torch.Tensor, depth: torch.Tensor, tokens: torch.Tensor,
                previous_action: int | None, states: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor]:
        text = self.instruction_encoder(tokens)
        depth_spatial = self.depth_encoder(depth).flatten(2)
        rgb_spatial = self.rgb_encoder(rgb).flatten(2)
        action = 0 if previous_action is None else previous_action + 1
        previous = self.prev_action_embedding(torch.tensor([action], device=rgb.device))
        first_input = torch.cat((self.rgb_linear(rgb_spatial),
                                 self.depth_linear(depth_spatial), previous), dim=1)
        first, first_state = self.state_encoder(first_input, states[:, :1],
                                                previous_action is not None)
        attended_text = self._attn(self.state_q(first), self.text_k(text), text,
                                   (text == 0).all(dim=1))
        rgb_key, rgb_value = self.rgb_kv(rgb_spatial).split(256, dim=1)
        depth_key, depth_value = self.depth_kv(depth_spatial).split(256, dim=1)
        query = self.text_q(attended_text)
        attended_rgb = self._attn(query, rgb_key, rgb_value)
        attended_depth = self._attn(query, depth_key, depth_value)
        second_input = torch.cat((first, attended_text, attended_rgb,
                                  attended_depth, previous), dim=1)
        second_input = self.second_state_compress(second_input)
        second, second_state = self.second_state_encoder(
            second_input, states[:, 1:], previous_action is not None)
        return second, torch.cat((first_state, second_state), dim=1)


class CMAPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = CMANet()
        self.action_distribution = nn.Module()
        self.action_distribution.add_module('linear', nn.Linear(512, 4))

    def forward(self, rgb: torch.Tensor, depth: torch.Tensor, tokens: torch.Tensor,
                previous_action: int | None, state: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor]:
        features, state = self.net(rgb, depth, tokens, previous_action, state)
        return self.action_distribution.linear(features), state


class _DiscardedConfig(dict):
    """Receive pickled training settings without importing Habitat or YACS."""


def _load_state_dict(checkpoint: Path) -> dict:
    """Read official tensor weights with a narrowly restricted pickle loader."""
    from torch._weights_only_unpickler import _get_allowed_globals

    allowed = _get_allowed_globals()
    configuration = {'habitat.config.default.Config', 'yacs.config.CfgNode'}
    legacy_builtins = {'__builtin__.set': set}

    class RestrictedUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            key = f'{module}.{name}'
            if key in configuration:
                return _DiscardedConfig
            if key in legacy_builtins:
                return legacy_builtins[key]
            if key in allowed:
                return allowed[key]
            raise pickle.UnpicklingError(f'unsupported CMA checkpoint global: {key}')

    restricted = types.ModuleType('restricted_cma_pickle')
    restricted.Unpickler = RestrictedUnpickler
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False,
                         pickle_module=restricted)
    if not isinstance(payload, dict) or not isinstance(payload.get('state_dict'), dict):
        raise ValueError('CMA checkpoint has no state_dict')
    state = payload['state_dict']
    # The published CMA checkpoint predates this BatchNorm bookkeeping buffer.
    # It does not affect eval-mode inference; all learned tensors remain strict.
    state.setdefault('net.rgb_encoder.cnn.1.num_batches_tracked',
                     torch.zeros((), dtype=torch.long))
    return state


class CMAInference:
    """Policy-server backend with strict checkpoint loading and greedy actions."""

    def __init__(self, checkpoint: Path):
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.policy = CMAPolicy()
        state = _load_state_dict(checkpoint)
        self.policy.load_state_dict(state, strict=True)
        self.policy.to(self.device).eval()

    def reset(self) -> None:
        return None

    @torch.inference_mode()
    def act(self, rgb: np.ndarray, depth: np.ndarray, tokens: list[int],
            previous_action: int | None, state: torch.Tensor | None
            ) -> tuple[int, torch.Tensor]:
        if state is None:
            state = torch.zeros((1, 2, 512), device=self.device)
        observations = (
            torch.from_numpy(np.array(rgb, copy=True, order='C')).unsqueeze(0).to(
                self.device),
            torch.from_numpy(np.array(depth, copy=True, order='C')).unsqueeze(0).to(
                self.device),
            torch.tensor([tokens], dtype=torch.long, device=self.device),
        )
        logits, next_state = self.policy(*observations, previous_action, state)
        return int(logits.argmax(dim=-1).item()), next_state


def load_model(checkpoint: Path) -> CMAInference:
    """Factory for ``python -m isaac_r2r.policy_server --factory ...``."""
    return CMAInference(checkpoint)
