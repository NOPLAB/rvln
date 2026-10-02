"""Check the Habitat-free CMA graph with synthetic weights and RGB-D inputs."""

import json
import sys
import tempfile
import threading
import types
from pathlib import Path
from http.server import ThreadingHTTPServer

import numpy as np
import torch
from PIL import Image

from bench.episode import Observation
from isaac_r2r.cma_model import CMAInference, CMAPolicy
from isaac_r2r.policy import R2RHttpPolicy
from isaac_r2r.policy_server import PolicySession, make_handler


expected_shapes = {
    "net.instruction_encoder.embedding_layer.weight": (2504, 50),
    "net.instruction_encoder.encoder_rnn.weight_ih_l0_reverse": (512, 50),
    "net.depth_encoder.visual_encoder.backbone.conv1.0.weight": (32, 1, 7, 7),
    "net.rgb_encoder.cnn.0.weight": (64, 3, 7, 7),
    "net.prev_action_embedding.weight": (5, 32),
    "net.state_encoder.rnn.weight_ih_l0": (1536, 416),
    "net.second_state_encoder.rnn.weight_ih_l0": (1536, 512),
    "action_distribution.linear.weight": (4, 512),
}
torch.manual_seed(7)
policy = CMAPolicy()
state = policy.state_dict()
for name, shape in expected_shapes.items():
    if tuple(state[name].shape) != shape:
        raise RuntimeError(f"unexpected CMA layer shape: {name}: {state[name].shape}")
with tempfile.TemporaryDirectory() as directory:
    checkpoint = Path(directory) / "synthetic-cma.pth"
    habitat = types.ModuleType("habitat")
    habitat.__path__ = []
    config_module = types.ModuleType("habitat.config")
    config_module.__path__ = []
    default_module = types.ModuleType("habitat.config.default")
    config_class = type("Config", (dict,), {"__module__": default_module.__name__})
    default_module.Config = config_class
    habitat.config = config_module
    config_module.default = default_module
    sys.modules.update(
        {
            "habitat": habitat,
            "habitat.config": config_module,
            "habitat.config.default": default_module,
        }
    )
    training_config = config_class({"MODEL": config_class({"policy_name": "CMAPolicy"})})
    training_config._immutable = True
    torch.save({"state_dict": state, "config": training_config}, checkpoint)
    for name in ("habitat.config.default", "habitat.config", "habitat"):
        sys.modules.pop(name)
    del policy, state
    backend = CMAInference(checkpoint)
    rgb = np.full((224, 224, 3), 60, dtype=np.uint8)
    depth = np.full((256, 256, 1), 0.2, dtype=np.float32)
    first, recurrent = backend.act(rgb, depth, [2, 3, 4], None, backend.reset())
    second, recurrent = backend.act(rgb, depth, [2, 3, 4], first, recurrent)
    repeated, _ = backend.act(rgb, depth, [2, 3, 4], None, backend.reset())
    if repeated != first or recurrent.shape != (1, 2, 512):
        raise RuntimeError("CMA inference did not reset its two recurrent layers")
    import io

    jpeg = io.BytesIO()
    Image.fromarray(np.full((256, 256, 3), 60, dtype=np.uint8)).save(jpeg, format="JPEG")
    observation = Observation(
        jpeg.getvalue(), np.full((256, 256), 2.0, dtype="<f4").tobytes(), 256, 256
    )
    session = PolicySession(backend, checkpoint)
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(session))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        client = R2RHttpPolicy(
            f"http://127.0.0.1:{server.server_port}", [2, 3, 4], session.policy_id
        )
        client.reset("synthetic-cma-episode")
        http_action = client.act("synthetic-cma-episode", 0, "Walk.", observation)
        if http_action not in ("stop", "forward", "left", "right"):
            raise RuntimeError(f"invalid HTTP CMA action: {http_action}")
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    print(
        json.dumps(
            {
                "checkpoint_bytes": checkpoint.stat().st_size,
                "parameter_keys": len(backend.policy.state_dict()),
                "first_action_id": first,
                "second_action_id": second,
                "repeated_action_id": repeated,
                "http_action": http_action,
                "recurrent_shape": list(recurrent.shape),
                "device": str(backend.device),
            }
        )
    )
