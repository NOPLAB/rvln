"""Stateful HTTP bridge between Isaac R2R episodes and a discrete policy.

The loaded backend owns its model and recurrent state representation. It must
return a new state with each action and must not use the benchmark goal.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import importlib
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from bench.artifacts import sha256

from isaac_r2r.cma_inputs import prepare_rgbd, validate_instruction_tokens


ACTIONS = ("stop", "forward", "left", "right")
MAX_REQUEST_BYTES = 2_000_000


class PolicySession:
    """One sequential episode with explicit reset and immutable policy ID."""

    def __init__(self, model, checkpoint: Path):
        if not checkpoint.is_file():
            raise ValueError(f"missing policy checkpoint: {checkpoint}")
        self.model = model
        self.policy_id = f"sha256:{sha256(checkpoint)}"
        self.lock = threading.Lock()
        self.episode_id = None
        self.next_frame = 0
        self.instruction = None
        self.tokens = None
        self.previous_action = None
        self.state = None
        self.stopped = False

    def reset(self, payload: dict) -> dict:
        episode_id = payload.get("episode_id")
        if not isinstance(episode_id, str) or not episode_id:
            raise ValueError("reset requires a nonempty episode_id")
        state = self.model.reset()
        self.episode_id = episode_id
        self.next_frame = 0
        self.instruction = None
        self.tokens = None
        self.previous_action = None
        self.state = state
        self.stopped = False
        return {"episode_id": episode_id, "policy_id": self.policy_id}

    def act(self, payload: dict) -> dict:
        episode_id = payload.get("episode_id")
        frame_id = payload.get("frame_id")
        if episode_id != self.episode_id or self.episode_id is None:
            raise ValueError("action episode differs from active reset")
        if type(frame_id) is not int or frame_id != self.next_frame or self.stopped:
            raise ValueError("action frame is stale, out of order, or after stop")
        instruction = payload.get("instruction")
        if not isinstance(instruction, str) or not instruction:
            raise ValueError("action requires an instruction")
        tokens = payload.get("instruction_tokens")
        if not isinstance(tokens, list):
            raise ValueError("action requires audited instruction_tokens")
        validate_instruction_tokens(tokens)
        if self.next_frame and (instruction != self.instruction or tokens != self.tokens):
            raise ValueError("instruction changed within an episode")
        try:
            jpeg = base64.b64decode(payload["jpeg_base64"], validate=True)
            depth = base64.b64decode(payload["depth_f32_base64"], validate=True)
        except (KeyError, TypeError, ValueError, binascii.Error) as error:
            raise ValueError("invalid RGB-D payload") from error
        rgb, normalized_depth = prepare_rgbd(
            jpeg, depth, payload.get("depth_width"), payload.get("depth_height")
        )
        action_id, next_state = self.model.act(
            rgb, normalized_depth, tokens, self.previous_action, self.state
        )
        if type(action_id) is not int or not 0 <= action_id < len(ACTIONS):
            raise ValueError(f"model returned an invalid action ID: {action_id!r}")
        self.instruction = instruction
        self.tokens = tokens.copy()
        self.previous_action = action_id
        self.state = next_state
        self.next_frame += 1
        self.stopped = action_id == 0
        return {"episode_id": episode_id, "frame_id": frame_id, "action": ACTIONS[action_id]}


def make_handler(session: PolicySession):
    """Build a localhost-only JSON handler around one serialized session."""

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if size < 1 or size > MAX_REQUEST_BYTES:
                    raise ValueError("invalid request size")
                payload = json.loads(self.rfile.read(size))
                if not isinstance(payload, dict):
                    raise ValueError("request must be a JSON object")
                with session.lock:
                    if self.path == "/reset":
                        result = session.reset(payload)
                    elif self.path == "/act":
                        result = session.act(payload)
                    else:
                        self.send_error(404)
                        return
            except (ValueError, KeyError, TypeError, OSError) as error:
                self.send_error(422, str(error))
                return
            body = json.dumps(result).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):
            pass

    return Handler


def load_backend(spec: str, checkpoint: Path):
    """Construct an independently supplied, checkpoint-loading model backend."""
    module_name, separator, factory_name = spec.partition(":")
    if not separator or not module_name or not factory_name:
        raise ValueError("--factory must be module:factory")
    factory = getattr(importlib.import_module(module_name), factory_name)
    return factory(checkpoint)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--factory",
        required=True,
        help="module:factory that loads the checkpoint and returns a model",
    )
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)
    model = load_backend(args.factory, args.checkpoint)
    session = PolicySession(model, args.checkpoint)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(session))
    print(json.dumps({"policy_id": session.policy_id, "port": server.server_port}), flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
