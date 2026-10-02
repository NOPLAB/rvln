"""Compare the Isaac CMA graph to pinned original source with identical weights.

This script uses public VLN-CE and Habitat-Lab source trees but no MP3D data or
pretrained checkpoint. Its small module shims replace training-only imports.
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torchvision.models

from isaac_r2r.cma_model import CMAPolicy


def package(name: str) -> types.ModuleType:
    result = types.ModuleType(name)
    result.__path__ = []
    sys.modules[name] = result
    parent, _, child = name.rpartition(".")
    if parent:
        setattr(sys.modules[parent], child, result)
    return result


def load(name: str, path: Path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    parent, _, child = name.rpartition(".")
    setattr(sys.modules[parent], child, result)
    spec.loader.exec_module(result)
    return result


def build_reference(vlnce: Path, habitat: Path):
    class Config:
        def __init__(self, **values):
            self.__dict__.update(values)

        def defrost(self):
            pass

        def freeze(self):
            pass

    class Space:
        def __init__(self, shape=None):
            self.shape = shape

    class DictSpace(Space):
        def __init__(self, spaces):
            self.spaces = spaces

    gym = package("gym")
    gym.Space = Space
    gym.spaces = types.SimpleNamespace(Dict=DictSpace)
    habitat_module = package("habitat")
    habitat_module.Config = Config
    package("habitat.core")
    package("habitat.core.simulator")
    sys.modules["habitat.core.simulator"].Observations = dict

    package("habitat_baselines")
    package("habitat_baselines.common")
    registry = package("habitat_baselines.common.baseline_registry")
    registry.baseline_registry = types.SimpleNamespace(register_policy=lambda cls: cls)
    package("habitat_baselines.rl")
    package("habitat_baselines.rl.ddppo")
    ddppo = package("habitat_baselines.rl.ddppo.policy")
    resnet_path = habitat / "habitat_baselines/rl/ddppo/policy/resnet.py"
    ddppo.resnet = load("habitat_baselines.rl.ddppo.policy.resnet", resnet_path)
    resnet_policy = package("habitat_baselines.rl.ddppo.policy.resnet_policy")
    encoder_path = habitat / "habitat_baselines/rl/ddppo/policy/resnet_policy.py"
    tree = ast.parse(encoder_path.read_text(encoding="utf-8"))
    encoder_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "ResNetEncoder"
    )
    scope = {
        "nn": nn,
        "torch": torch,
        "np": np,
        "spaces": gym.spaces,
        "resnet": ddppo.resnet,
        "F": torch.nn.functional,
        "Dict": dict,
    }
    exec(
        compile(ast.Module(body=[encoder_class], type_ignores=[]), str(encoder_path), "exec"),
        scope,
    )
    resnet_policy.ResNetEncoder = scope["ResNetEncoder"]
    package("habitat_baselines.rl.models")
    rnn_path = habitat / "habitat_baselines/rl/models/rnn_state_encoder.py"
    load("habitat_baselines.rl.models.rnn_state_encoder", rnn_path)
    package("habitat_baselines.rl.ppo")
    package("habitat_baselines.rl.ppo.policy").Net = nn.Module

    package("vlnce_baselines")
    package("vlnce_baselines.common")
    aux = package("vlnce_baselines.common.aux_losses")
    aux.AuxLosses = types.SimpleNamespace(is_active=lambda: False)
    package("vlnce_baselines.common.utils").single_frame_box_shape = lambda space: space
    package("vlnce_baselines.models")
    encoders = package("vlnce_baselines.models.encoders")
    load(
        "vlnce_baselines.models.encoders.instruction_encoder",
        vlnce / "vlnce_baselines/models/encoders/instruction_encoder.py",
    )
    encoders.resnet_encoders = load(
        "vlnce_baselines.models.encoders.resnet_encoders",
        vlnce / "vlnce_baselines/models/encoders/resnet_encoders.py",
    )
    policy_module = package("vlnce_baselines.models.policy")

    class ILPolicy(nn.Module):
        def __init__(self, net, dim_actions):
            super().__init__()
            self.net = net
            self.action_distribution = nn.Module()
            self.action_distribution.add_module("linear", nn.Linear(512, dim_actions))

    policy_module.ILPolicy = ILPolicy
    cma = load("vlnce_baselines.models.cma_policy", vlnce / "vlnce_baselines/models/cma_policy.py")
    model_config = Config(
        INSTRUCTION_ENCODER=Config(
            sensor_uuid="instruction",
            vocab_size=2504,
            use_pretrained_embeddings=False,
            fine_tune_embeddings=False,
            embedding_size=50,
            hidden_size=128,
            rnn_type="LSTM",
            bidirectional=True,
            final_state_only=False,
        ),
        DEPTH_ENCODER=Config(
            cnn_type="VlnResnetDepthEncoder",
            output_size=128,
            ddppo_checkpoint="NONE",
            backbone="resnet50",
            trainable=False,
        ),
        RGB_ENCODER=Config(cnn_type="TorchVisionResNet50", output_size=256, trainable=False),
        STATE_ENCODER=Config(hidden_size=512, rnn_type="GRU"),
        PROGRESS_MONITOR=Config(use=True),
        normalize_rgb=False,
        ablate_instruction=False,
        ablate_rgb=False,
        ablate_depth=False,
    )
    original_resnet50 = torchvision.models.resnet50
    torchvision.models.resnet50 = lambda *args, **kwargs: original_resnet50(weights=None)
    try:
        reference = cma.CMAPolicy(
            DictSpace({"depth": Space((256, 256, 1))}), types.SimpleNamespace(n=4), model_config
        )
    finally:
        torchvision.models.resnet50 = original_resnet50
    return reference


def main():
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vlnce", type=Path, required=True)
    parser.add_argument("--habitat", type=Path, required=True)
    args = parser.parse_args()
    reference = build_reference(args.vlnce, args.habitat)
    candidate = CMAPolicy()
    expected = reference.state_dict()
    actual = candidate.state_dict()
    missing = sorted(expected.keys() - actual.keys())
    extra = sorted(actual.keys() - expected.keys())
    shape_differences = {
        name: [list(expected[name].shape), list(actual[name].shape)]
        for name in expected.keys() & actual.keys()
        if expected[name].shape != actual[name].shape
    }
    if missing or extra or shape_differences:
        raise RuntimeError(
            json.dumps(
                {"missing": missing, "extra": extra, "shape_differences": shape_differences}
            )
        )
    reference.load_state_dict(actual, strict=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reference.to(device).eval()
    candidate.to(device).eval()
    compare_layers = (
        "instruction_encoder",
        "depth_encoder",
        "rgb_encoder.cnn.0",
        "rgb_encoder.cnn.1",
        "rgb_encoder.cnn.4",
        "rgb_encoder.cnn.5",
        "rgb_encoder.cnn.6",
        "rgb_encoder.cnn.7",
        "rgb_encoder.cnn",
        "rgb_encoder",
        "rgb_linear",
        "depth_linear",
        "state_encoder",
        "rgb_kv",
        "depth_kv",
        "second_state_compress",
        "second_state_encoder",
    )
    captured = {"reference": {}, "candidate": {}}

    def capture(side, name):
        def record(_module, _inputs, output):
            tensor = output[0] if isinstance(output, tuple) else output
            captured[side][name] = tensor.detach().clone()

        return record

    for name in compare_layers:
        reference.net.get_submodule(name).register_forward_hook(capture("reference", name))
        candidate.net.get_submodule(name).register_forward_hook(capture("candidate", name))
    torch.manual_seed(17)
    observations = {
        "rgb": torch.randint(0, 255, (1, 224, 224, 3), device=device),
        "depth": torch.rand((1, 256, 256, 1), device=device),
        "instruction": torch.tensor([[2, 3, 4, 0]], device=device),
    }
    states = torch.zeros((1, 2, 512), device=device)
    differences = []
    with torch.inference_mode():
        for previous_action in (None, 2):
            action = torch.tensor([[previous_action or 0]], device=device)
            mask = torch.tensor([[previous_action is not None]], device=device)
            features, ref_state = reference.net(observations, states, action, mask)
            ref_logits = reference.action_distribution.linear(features)
            logits, candidate_state = candidate(
                observations["rgb"],
                observations["depth"],
                observations["instruction"],
                previous_action,
                states,
            )
            differences.append(
                {
                    "logits": (ref_logits - logits).abs().max().item(),
                    "recurrent": (ref_state - candidate_state).abs().max().item(),
                }
            )
            states = ref_state
    if any(max(item.values()) > 1e-5 for item in differences):
        layers = {
            name: (captured["reference"][name] - captured["candidate"][name]).abs().max().item()
            for name in compare_layers
        }
        raise RuntimeError(f"CMA numerical parity failed: {differences}; layers={layers}")
    print(
        json.dumps({"reference_keys": len(expected), "device": str(device), "steps": differences})
    )


if __name__ == "__main__":
    main()
