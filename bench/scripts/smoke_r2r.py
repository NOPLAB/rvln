"""Run a scripted Isaac R2R integration check on the generated corridor USD."""
import argparse
import base64
import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--headless', action='store_true')
args = parser.parse_args()

bench = Path(__file__).resolve().parents[1]
directory = bench / 'runs/validation'
directory.mkdir(parents=True, exist_ok=True)
requests = []


class Policy(BaseHTTPRequestHandler):
    def do_POST(self):
        size = int(self.headers['Content-Length'])
        data = json.loads(self.rfile.read(size))
        if self.path == '/reset':
            response = {'episode_id': data['episode_id']}
        elif self.path == '/act':
            jpeg = base64.b64decode(data['jpeg_base64'])
            assert jpeg.startswith(b'\xff\xd8') and jpeg.endswith(b'\xff\xd9')
            if not requests:
                (directory / 'synthetic_first_frame.jpg').write_bytes(jpeg)
            requests.append({'frame_id': data['frame_id'], 'jpeg_bytes': len(jpeg)})
            response = {'episode_id': data['episode_id'],
                        'frame_id': data['frame_id'],
                        'action': 'forward' if data['frame_id'] < 10 else 'stop'}
        else:
            self.send_error(404)
            return
        body = json.dumps(response).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        pass


split = {'episodes': [{
    'episode_id': 'synthetic-corridor-1',
    'scene_id': 'mp3d/corridor/corridor.glb',
    'instruction': {'instruction_text': 'Move ahead to the red marker.'},
    'start_position': [0.0, 0.0, 0.0],
    'start_rotation': [0.0, -0.7071067811865475, 0.0, 0.7071067811865476],
    'goals': [{'position': [2.35, 0.0, 0.0]}],
    'info': {'geodesic_distance': 2.05},
}]}
scenes = {'scenes': {'corridor': {
    'usd': str(bench / 'assets/corridor.usd'),
    'isaac_from_habitat': [[1, 0, 0, 0], [0, 0, -1, 0],
                           [0, 1, 0, 0], [0, 0, 0, 1]],
}}}
split_path = directory / 'synthetic_split.json'
scenes_path = directory / 'synthetic_scenes.json'
split_path.write_text(json.dumps(split))
scenes_path.write_text(json.dumps(scenes))

server = ThreadingHTTPServer(('127.0.0.1', 0), Policy)
thread = threading.Thread(target=server.serve_forever, daemon=True)
thread.start()
try:
    output = directory / 'synthetic_result.json'
    output.unlink(missing_ok=True)
    command = [sys.executable, '-u', '-m', 'bench.cli']
    command.extend([
               'r2r', '--split', str(split_path), '--scenes', str(scenes_path),
               '--episode', 'synthetic-corridor-1', '--out', str(output),
               '--max-steps', '12',
               '--policy-url', f'http://127.0.0.1:{server.server_port}'])
    if args.headless:
        command.append('--headless')
    with (bench / 'runs/r2r_synthetic_success.log').open('w') as log:
        child = subprocess.run(command, cwd=bench, env=os.environ.copy(),
                               stdout=log, stderr=subprocess.STDOUT, timeout=300)
    result = json.loads(output.read_text())
    print(json.dumps({'exit': child.returncode, 'status': result['status'],
                      'success': result.get('success'), 'spl': result.get('spl'),
                      'actions': result.get('actions'),
                      'blocked_steps': result.get('blocked_steps'),
                      'requests': requests}), flush=True)
    if child.returncode or result['status'] != 'completed' or not requests:
        raise SystemExit(1)
finally:
    server.shutdown()
