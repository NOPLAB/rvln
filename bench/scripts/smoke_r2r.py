"""Run a scripted Isaac R2R integration check on the generated corridor USD."""
import argparse
import base64
import io
import json
import math
import os
import struct
import subprocess
import sys
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from PIL import Image

from isaac_r2r.cma_inputs import prepare_rgbd
from isaac_r2r.policy_server import PolicySession, make_handler


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--headless', action='store_true')
parser.add_argument('--batch', action='store_true',
                    help='exercise the full-split batch command on two synthetic episodes')
parser.add_argument('--cma-tokens', action='store_true',
                    help='transport aligned preprocessed tokens and prepare CMA RGB-D inputs')
parser.add_argument('--stateful-server', action='store_true',
                    help='run the policy server contract with a scripted model backend')
args = parser.parse_args()
if args.stateful_server and not (args.batch and args.cma_tokens):
    parser.error('--stateful-server requires --batch --cma-tokens')

bench = Path(__file__).resolve().parents[1]
directory = bench / 'runs/validation'
directory.mkdir(parents=True, exist_ok=True)
requests = []


class Policy(BaseHTTPRequestHandler):
    def do_POST(self):
        size = int(self.headers['Content-Length'])
        data = json.loads(self.rfile.read(size))
        if self.path == '/reset':
            response = {'episode_id': data['episode_id'],
                        'policy_id': 'scripted-smoke-v1'}
        elif self.path == '/act':
            jpeg = base64.b64decode(data['jpeg_base64'])
            assert jpeg.startswith(b'\xff\xd8') and jpeg.endswith(b'\xff\xd9')
            with Image.open(io.BytesIO(jpeg)) as rendered:
                assert rendered.size == (256, 256)
                if max(rendered.convert('RGB').getpixel((0, 0))) >= 30:
                    (directory / f'synthetic_corrupt_frame_{data["frame_id"]}.jpg').write_bytes(
                        jpeg)
                    self.send_error(422, 'synthetic sky color indicates RGB corruption')
                    return
            depth = base64.b64decode(data['depth_f32_base64'])
            assert (data['depth_width'], data['depth_height']) == (256, 256)
            assert len(depth) == 256 * 256 * 4
            assert any(value[0] > 0 for value in struct.iter_unpack('<f', depth))
            if args.cma_tokens:
                assert data['instruction_tokens'] == ([2, 3] if data['episode_id'].endswith('1')
                                                       else [2, 4])
                rgb, cma_depth = prepare_rgbd(jpeg, depth, 256, 256)
                assert rgb.shape == (224, 224, 3)
                assert cma_depth.shape == (256, 256, 1)
                assert 0 <= cma_depth.min() <= cma_depth.max() <= 1
            if not requests:
                (directory / 'synthetic_first_frame.jpg').write_bytes(jpeg)
            if data['frame_id'] in (3, 10):
                (directory / f'synthetic_frame_{data["frame_id"]}.jpg').write_bytes(jpeg)
            requests.append({'episode_id': data['episode_id'],
                             'frame_id': data['frame_id'], 'jpeg_bytes': len(jpeg),
                             'depth_bytes': len(depth)})
            response = {'episode_id': data['episode_id'],
                        'frame_id': data['frame_id'],
                        'action': 'forward' if data['frame_id'] < 13 else 'stop'}
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
    'start_position': [-1.0, 0.0, 0.0],
    'start_rotation': [0.0, -0.7071067811865475, 0.0, 0.7071067811865476],
    'goals': [{'position': [2.8, 0.0, 0.0]}],
    'info': {'geodesic_distance': 3.8},
}]}
assert math.dist(split['episodes'][0]['start_position'],
                 split['episodes'][0]['goals'][0]['position']) > 3.0
if args.batch:
    split['episodes'].append({**split['episodes'][0],
                              'episode_id': 'synthetic-corridor-2',
                              'instruction': {'instruction_text': 'Stay near the entrance.'},
                              'goals': [{'position': [-1.25, 0.0, 0.0]}],
                              'info': {'geodesic_distance': 0.25}})
scenes = {'scenes': {'corridor': {
    'usd': str(bench / 'assets/corridor.usd'),
    'isaac_from_habitat': [[1, 0, 0, 0], [0, 0, -1, 0],
                           [0, 1, 0, 0], [0, 0, 0, 1]],
}}}
split_path = directory / 'synthetic_split.json'
scenes_path = directory / 'synthetic_scenes.json'
split_path.write_text(json.dumps(split))
scenes_path.write_text(json.dumps(scenes))
if args.cma_tokens:
    policy_split = {'episodes': []}
    for index, episode in enumerate(split['episodes']):
        policy_split['episodes'].append({**episode, 'instruction': {
            **episode['instruction'], 'instruction_tokens': [2, 3 + index]}})
    policy_split_path = directory / 'synthetic_policy_split.json'
    policy_split_path.write_text(json.dumps(policy_split))

policy_id = 'scripted-smoke-v1'
if args.stateful_server:
    class StatefulScript:
        def reset(self):
            return 0

        def act(self, rgb, depth, tokens, previous_action, state):
            assert rgb.shape == (224, 224, 3)
            assert depth.shape == (256, 256, 1)
            assert rgb[0, 0].max() < 30
            assert 0 <= depth.min() <= depth.max() <= 1
            assert previous_action == (None if state == 0 else 1)
            assert tokens in ([2, 3], [2, 4])
            episode_id = ('synthetic-corridor-1' if tokens == [2, 3]
                          else 'synthetic-corridor-2')
            requests.append({'episode_id': episode_id, 'frame_id': state,
                             'depth_bytes': depth.nbytes})
            return (1 if state < 13 else 0), state + 1

    checkpoint = directory / 'synthetic_scripted_policy.ckpt'
    checkpoint.write_bytes(b'Isaac R2R scripted policy server smoke v1\n')
    session = PolicySession(StatefulScript(), checkpoint)
    policy_id = session.policy_id
    handler = make_handler(session)
else:
    handler = Policy
server = ThreadingHTTPServer(('127.0.0.1', 0), handler)
thread = threading.Thread(target=server.serve_forever, daemon=True)
thread.start()
try:
    output = directory / 'synthetic_result.json'
    if not args.batch:
        output.unlink(missing_ok=True)
    command = [sys.executable, '-u', '-m', 'bench.cli']
    if args.batch:
        batch_dir = directory / f'synthetic_batch_{uuid.uuid4().hex}'
        command.extend(['r2r-batch', '--split', str(split_path),
                        '--scenes', str(scenes_path), '--out-dir', str(batch_dir),
                        '--policy-id', policy_id, '--max-steps', '15'])
    else:
        command.extend(['r2r', '--split', str(split_path),
                        '--scenes', str(scenes_path),
                        '--episode', 'synthetic-corridor-1', '--out', str(output),
                        '--max-steps', '15'])
    command.extend(['--policy-url', f'http://127.0.0.1:{server.server_port}'])
    if not args.batch:
        command.extend(['--policy-id', policy_id])
    if args.cma_tokens:
        command.extend(['--policy-split', str(policy_split_path)])
    if args.headless:
        command.append('--headless')
    with (bench / 'runs/r2r_synthetic_success.log').open('w') as log:
        child = subprocess.run(command, cwd=bench, env=os.environ.copy(),
                               stdout=log, stderr=subprocess.STDOUT, timeout=300)
    result_path = batch_dir / 'summary.json' if args.batch else output
    if not result_path.is_file():
        detail_path = batch_dir / 'manifest.json' if args.batch else output
        detail = (json.loads(detail_path.read_text()) if detail_path.is_file() else {})
        print(json.dumps({'exit': child.returncode, 'result_missing': str(result_path),
                          'detail': detail, 'requests': requests}), flush=True)
        raise SystemExit(1)
    result = json.loads(result_path.read_text())
    status = (json.loads((batch_dir / 'manifest.json').read_text())['status']
              if args.batch else result['status'])
    print(json.dumps({'exit': child.returncode, 'status': status,
                      'success': result.get('success', result.get('success_rate')),
                      'spl': result.get('spl'), 'actions': result.get('actions'),
                      'blocked_steps': result.get('blocked_steps'),
                      'episodes': result.get('episodes'), 'requests': requests}), flush=True)
    expected_ids = {item['episode_id'] for item in split['episodes']}
    observed = {(item['episode_id'], item['frame_id']) for item in requests}
    expected = {(episode_id, frame_id) for episode_id in expected_ids
                for frame_id in range(14)}
    expected_success = 0.5 if args.batch else 1
    expected_spl = 0.5 if args.batch else 1
    if child.returncode or status != 'completed' or observed != expected or \
            len(requests) != len(expected) or \
            result.get('success', result.get('success_rate')) != expected_success or \
            not math.isclose(result['spl'], expected_spl):
        raise SystemExit(1)
finally:
    server.shutdown()
