"""Synthetic embedding service for the Isaac ROS live-pipeline smoke only."""
from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class Handler(BaseHTTPRequestHandler):
    allowed_client: str | None = None

    def log_message(self, _format: str, *_args) -> None:
        """Avoid blocking inference on synchronous shared-storage logging."""

    def _authorized(self) -> bool:
        if self.allowed_client is not None and self.client_address[0] != self.allowed_client:
            self.send_error(403)
            return False
        return True

    def do_GET(self) -> None:  # noqa: N802
        if not self._authorized():
            return
        if self.path != '/health':
            self.send_error(404)
            return
        self._send({'status': 'ready', 'model_version': 'synthetic-stub'})

    def do_POST(self) -> None:  # noqa: N802
        if not self._authorized():
            return
        size = int(self.headers.get('Content-Length', '0'))
        if size < 1 or size > 2_000_000:
            self.send_error(413)
            return
        try:
            request = json.loads(self.rfile.read(size))
        except (ValueError, UnicodeDecodeError):
            self.send_error(400)
            return
        if self.path == '/reset':
            self._send({'reset': True, 'model_version': 'synthetic-stub'})
        elif self.path == '/infer':
            if not request.get('jpeg_base64') or not request.get('text'):
                self.send_error(400)
                return
            self._send({'frame_id': int(request['frame_id']),
                        'embedding': [[1.0, 0.0, 0.0, 0.0]],
                        'inference_ms': 0.1, 'server_wall_ms': 0.1,
                        'model_version': 'synthetic-stub'})
        else:
            self.send_error(404)

    def _send(self, payload: dict) -> None:
        body = json.dumps(payload).encode('utf-8')
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8766)
    parser.add_argument('--bind', default='127.0.0.1')
    parser.add_argument('--allowed-client')
    args = parser.parse_args()
    Handler.allowed_client = args.allowed_client
    ThreadingHTTPServer((args.bind, args.port), Handler).serve_forever()


if __name__ == '__main__':
    main()
