"""Serve one Slurm-reserved VLN backend to the local ROS edge over Tailscale HTTP."""
from __future__ import annotations

import argparse
import base64
import io
import json
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
from PIL import Image

from replay import build_backend


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--bind', required=True)
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--resume-step', type=int, default=120000)
    args = parser.parse_args()
    backend = build_backend(args.backend, args.checkpoint, args.device, args.resume_step)
    version = backend.model_info().model_version
    previous = None

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            if self.path != '/health':
                self.send_error(404)
                return
            self._send({'backend': args.backend, 'model_version': version})

        def do_POST(self) -> None:  # noqa: N802
            nonlocal previous
            if self.path != '/infer':
                self.send_error(404)
                return
            size = int(self.headers.get('Content-Length', '0'))
            if size < 1 or size > 2_000_000:
                self.send_error(413)
                return
            try:
                request = json.loads(self.rfile.read(size))
                jpeg = base64.b64decode(request['jpeg_base64'], validate=True)
                with Image.open(io.BytesIO(jpeg)) as source:
                    current = source.convert('RGB')
                text = str(request['text'])
                start = time.monotonic()
                embedding, reported = backend.infer(
                    current_image=current,
                    past_image=previous if previous is not None else current,
                    lang_instruction=text, goal_image=None, goal_pose_xy_theta=None,
                )
                previous = current
                output = np.asarray(embedding, dtype=np.float32)
                if output.ndim != 2 or not output.size or not np.isfinite(output).all():
                    raise ValueError(f'invalid embedding shape={output.shape}')
                payload = {'frame_id': int(request['frame_id']), 'embedding': output.tolist(),
                           'inference_ms': float(reported['inference_ms']),
                           'server_wall_ms': (time.monotonic() - start) * 1000.0,
                           'model_version': version}
                print(json.dumps({'frame_id': payload['frame_id'],
                                  'server_wall_ms': payload['server_wall_ms']}), flush=True)
                self._send(payload)
            except (KeyError, ValueError, TypeError, OSError) as exc:
                self.send_error(400, str(exc))

        def _send(self, payload: dict) -> None:
            body = json.dumps(payload).encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = HTTPServer((args.bind, args.port), Handler)
    print(f'ready backend={args.backend} bind={args.bind}:{args.port}', flush=True)
    server.serve_forever()


if __name__ == '__main__':
    main()
