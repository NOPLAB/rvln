"""Run one local Gazebo/Edge episode against a Slurm hosted inference server."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import signal
import subprocess
import sys
import time
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
import rclpy
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry, Path as RosPath
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rvln_msgs.msg import ActionEmbedding, GoalSpec, Observation
from std_srvs.srv import SetBool


class Episode(Node):
    """Bridge observations and collect the actual odometry and command trace."""

    def __init__(self, url: str, instruction: str, video: Path,
                 episode_id: str, goal_xy: list[float]) -> None:
        super().__init__('rvln_bench_episode')
        self.url = url.rstrip('/') + '/infer'
        self.instruction = instruction
        self.episode_id = episode_id
        self.goal_xy = goal_xy
        self.video = video
        self.video.parent.mkdir(parents=True, exist_ok=True)
        self.writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'mp4v'),
                                      2.0, (960, 480))
        if not self.writer.isOpened():
            raise RuntimeError(f'cannot open video writer: {video}')
        self.video_frames = 0
        best_effort = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(Observation, '/rvln/observation', self._observation, best_effort)
        self.create_subscription(Odometry, '/odom', self._odom, 10)
        self.create_subscription(ModelStates, '/model_states', self._state, 10)
        self.create_subscription(Twist, '/cmd_vel', self._command, 10)
        self.create_subscription(RosPath, '/rvln/predicted_path', self._path, 10)
        self.embedding_pub = self.create_publisher(
            ActionEmbedding, '/rvln/remote_embedding', best_effort)
        self.goal_pub = self.create_publisher(GoalSpec, '/rvln/goal', latched)
        self.motor = self.create_client(SetBool, '/motor_power')
        self.follower_stop = self.create_client(SetBool, '/rvln/follower_stop')
        self.trace = []
        self.inferences = []
        self.received = 0
        self.published = 0
        self.path_count = 0
        self.nonempty_paths = 0
        self.path_samples = []
        self.model_version = None
        self.last_cmd = [0.0, 0.0]
        self.last_cmd_at = 0.0
        self.last_pose = None
        self.last_odom_pose = None
        self.last_odom_velocity = [0.0, 0.0]
        self.started_at = None
        self.errors = []

    def _odom(self, msg: Odometry) -> None:
        self.last_odom_pose = (float(msg.pose.pose.position.x),
                               float(msg.pose.pose.position.y))
        self.last_odom_velocity = [float(msg.twist.twist.linear.x),
                                   float(msg.twist.twist.angular.z)]

    def _state(self, msg: ModelStates) -> None:
        if 'raspicat' not in msg.name:
            return
        pose = msg.pose[msg.name.index('raspicat')]
        q = pose.orientation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y),
                         1 - 2 * (q.y * q.y + q.z * q.z))
        self.last_pose = (float(pose.position.x), float(pose.position.y), yaw)
        if self.started_at is not None:
            now = time.monotonic() - self.started_at
            if not self.trace or now - self.trace[-1]['t_sec'] >= 0.09:
                self.trace.append({'t_sec': now, 'x': self.last_pose[0],
                                   'y': self.last_pose[1], 'yaw': yaw,
                                   'cmd_v': self.last_cmd[0], 'cmd_w': self.last_cmd[1]})

    def _command(self, msg: Twist) -> None:
        self.last_cmd = [float(msg.linear.x), float(msg.angular.z)]
        self.last_cmd_at = time.monotonic()

    def _path(self, msg: RosPath) -> None:
        self.path_count += 1
        if msg.poses:
            self.nonempty_paths += 1
            if len(self.path_samples) < 10:
                point = msg.poses[min(4, len(msg.poses) - 1)].pose
                self.path_samples.append({
                    't_sec': time.monotonic() - self.started_at
                    if self.started_at is not None else None,
                    'waypoints': len(msg.poses),
                    'frame_id': msg.header.frame_id,
                    'first_xy': [msg.poses[0].pose.position.x,
                                 msg.poses[0].pose.position.y],
                    'selected_xy': [point.position.x, point.position.y],
                    'selected_cos_sin': [point.orientation.w, point.orientation.z],
                })

    def _observation(self, msg: Observation) -> None:
        self.received += 1
        if self.started_at is not None:
            self._record_video(msg)
        if msg.goal.mode != GoalSpec.MODE_TEXT:
            self.errors.append('non-text observation')
            return
        if msg.goal.text != self.instruction:
            self.errors.append('observation instruction differs from published goal')
            return
        payload = {'frame_id': int(msg.frame_id), 'text': msg.goal.text,
                   'jpeg_base64': base64.b64encode(bytes(msg.image.data)).decode('ascii')}
        if self.last_pose is not None:
            payload['pose_xyyaw'] = list(self.last_pose)
            payload['velocity_vw'] = self.last_odom_velocity
        request = urllib.request.Request(
            self.url, json.dumps(payload).encode('utf-8'),
            {'Content-Type': 'application/json'}, method='POST')
        start = time.monotonic()
        try:
            with urllib.request.urlopen(request, timeout=8.0) as response:
                result = json.load(response)
            if result['frame_id'] != msg.frame_id:
                raise ValueError('frame ID mismatch')
            tokens = result['embedding']
            if not tokens or not tokens[0] or any(len(t) != len(tokens[0]) for t in tokens):
                raise ValueError('invalid embedding shape')
            output = ActionEmbedding()
            output.header.stamp = self.get_clock().now().to_msg()
            output.frame_id = msg.frame_id
            output.num_tokens = len(tokens)
            output.embed_dim = len(tokens[0])
            output.embedding = [float(v) for token in tokens for v in token]
            output.inference_ms = float(result['inference_ms'])
            output.model_version = result['model_version']
            if self.model_version is None:
                self.model_version = output.model_version
            elif self.model_version != output.model_version:
                raise ValueError('remote model version changed during episode')
            self.embedding_pub.publish(output)
            self.published += 1
            record = {'frame_id': int(msg.frame_id),
                      'round_trip_ms': (time.monotonic() - start) * 1000,
                      'server_wall_ms': result['server_wall_ms'],
                      'shape': [len(tokens), len(tokens[0])],
                      'diagnostics': result.get('diagnostics', {})}
            if len(tokens[0]) == 4:
                record['first_waypoint'] = tokens[0]
            else:
                record['feature_l2'] = float(np.linalg.norm(output.embedding))
            self.inferences.append(record)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            self.errors.append(str(exc))

    def _record_video(self, msg: Observation) -> None:
        frame = cv2.imdecode(np.frombuffer(bytes(msg.image.data), dtype=np.uint8),
                             cv2.IMREAD_COLOR)
        if frame is None:
            self.errors.append('cannot decode observation JPEG for video')
            return
        canvas = np.full((480, 960, 3), 245, dtype=np.uint8)
        canvas[:, :640] = cv2.resize(frame, (640, 480))
        now = time.monotonic() - self.started_at
        cv2.putText(canvas, f'{self.episode_id} t={now:.1f}s', (650, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (20, 20, 20), 2)
        cv2.putText(canvas, self.instruction[:29], (650, 67),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 1)

        def point(x: float, y: float) -> tuple[int, int]:
            return int(690 + 65 * x), int(290 - 65 * y)

        cv2.line(canvas, point(0, -2), point(0, 2), (200, 200, 200), 1)
        cv2.line(canvas, point(0, 0), point(4, 0), (200, 200, 200), 1)
        goal = point(*self.goal_xy)
        cv2.circle(canvas, goal, 9, (0, 0, 220), 2)
        points = [point(row['x'], row['y']) for row in self.trace]
        if len(points) >= 2:
            cv2.polylines(canvas, [np.asarray(points, dtype=np.int32)],
                          False, (230, 120, 0), 2)
        if self.last_pose is not None:
            robot = point(self.last_pose[0], self.last_pose[1])
            cv2.circle(canvas, robot, 6, (0, 150, 0), -1)
            distance = math.hypot(self.last_pose[0] - self.goal_xy[0],
                                  self.last_pose[1] - self.goal_xy[1])
            cv2.putText(canvas, f'goal distance {distance:.2f} m', (650, 425),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1)
        cv2.putText(canvas, f'cmd {self.last_cmd[0]:+.2f} m/s '
                    f'{self.last_cmd[1]:+.2f} rad/s', (650, 455),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (20, 20, 20), 1)
        self.writer.write(canvas)
        self.video_frames += 1

    def set_bool(self, client, value: bool, timeout: float = 10.0) -> bool:
        if not client.wait_for_service(timeout_sec=timeout):
            raise RuntimeError(f'service unavailable: {client.srv_name}')
        future = client.call_async(SetBool.Request(data=value))
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout)
        return bool(future.done() and future.result() and future.result().success)

    def publish_goal(self) -> None:
        goal = GoalSpec()
        goal.mode = GoalSpec.MODE_TEXT
        goal.text = self.instruction
        self.goal_pub.publish(goal)


def spin_until(node: Node, condition, deadline: float) -> bool:
    while rclpy.ok() and time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
        if condition():
            return True
    return False


def start_contacts(scene: str, output: Path):
    world = Path('/workspace/bench/worlds') / f'{scene}.world'
    obstacles = [model.attrib['name']
                 for model in ET.parse(world).getroot().findall('.//model')]
    if not obstacles:
        raise RuntimeError(f'no obstacle models in {world}')
    contact_file = output.with_suffix('.contacts.json')
    ready_file = output.with_suffix('.contacts.ready')
    ready_file.unlink(missing_ok=True)
    contact_file.unlink(missing_ok=True)
    log_file = output.with_suffix('.contacts.log')
    with log_file.open('w', encoding='utf-8') as log:
        process = subprocess.Popen(
            [sys.executable, '/workspace/bench/contact_logger.py',
             '--obstacles', ','.join(obstacles),
             '--out', str(contact_file), '--ready', str(ready_file)],
            stdout=log, stderr=subprocess.STDOUT)
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline and process.poll() is None:
        if ready_file.exists():
            return process, contact_file, log_file
        time.sleep(0.1)
    process.send_signal(signal.SIGINT)
    process.wait(timeout=10)
    raise RuntimeError(f'contact logger did not receive all sensors: {log_file}')


def finish_contacts(process, contact_file: Path) -> dict:
    if process.poll() is None:
        process.send_signal(signal.SIGINT)
    process.wait(timeout=10)
    if process.returncode != 0 or not contact_file.exists():
        raise RuntimeError(f'contact logger failed: exit={process.returncode}')
    result = json.loads(contact_file.read_text(encoding='utf-8'))
    if not all(count > 0 for count in result['samples'].values()):
        raise RuntimeError('one or more contact sensors produced no samples')
    if result.get('receive_errors', 0):
        raise RuntimeError(f'contact conversion errors: {result["receive_errors"]}')
    return result


def run(args: argparse.Namespace) -> dict:
    manifest = json.loads(args.manifest.read_text(encoding='utf-8'))
    episode = next(x for x in manifest['episodes']
                   if x['id'] == args.episode)
    world = Path('/workspace/bench/worlds') / f'{episode["scene"]}.world'
    actual_hash = hashlib.sha256(world.read_bytes()).hexdigest()
    if actual_hash != manifest['oracle']['world_sha256'][episode['scene']]:
        raise RuntimeError(f'world changed since shortest path calculation: {world}')
    reset = urllib.request.Request(args.url.rstrip('/') + '/reset', data=b'{}',
                                   method='POST')
    with urllib.request.urlopen(reset, timeout=120) as response:
        if not json.load(response)['reset']:
            raise RuntimeError('remote reset failed')
    rclpy.init()
    node = Episode(args.url, args.text or episode['text'], args.video,
                   episode['id'], episode['goal_xy'])
    stop_reason = 'startup_failure'
    confirmed_stop = False
    contact_process = None
    contact_data = None
    contact_file = None
    try:
        deadline = time.monotonic() + args.startup_timeout
        if not spin_until(node, lambda: node.last_pose is not None, deadline):
            raise RuntimeError('no Gazebo odometry')
        if not node.follower_stop.wait_for_service(timeout_sec=10):
            raise RuntimeError('no local follower stop service')
        if not node.set_bool(node.follower_stop, True):
            raise RuntimeError('cannot gate follower')
        contact_process, contact_file, _log = start_contacts(episode['scene'], args.out)
        node.publish_goal()
        if not spin_until(node, lambda: node.published > 0, deadline):
            raise RuntimeError('no remote embedding after goal')
        node.started_at = time.monotonic()
        x, y, yaw = node.last_pose
        node.trace.append({'t_sec': 0.0, 'x': x, 'y': y, 'yaw': yaw,
                           'cmd_v': 0.0, 'cmd_w': 0.0})
        if not node.set_bool(node.motor, True):
            raise RuntimeError('cannot enable simulator motor')
        if not node.set_bool(node.follower_stop, False):
            raise RuntimeError('cannot release follower')
        stop_reason = 'timeout'
        deadline = time.monotonic() + args.duration
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
            if node.last_pose is None:
                continue
            distance = math.hypot(node.last_pose[0] - episode['goal_xy'][0],
                                  node.last_pose[1] - episode['goal_xy'][1])
            if distance <= args.tolerance:
                stop_reason = 'goal_tolerance'
                break
            if len(node.errors) >= 5:
                stop_reason = 'remote_error'
                break
        node.set_bool(node.follower_stop, True)
        confirmed_stop = spin_until(
            node, lambda: time.monotonic() - node.last_cmd_at < 0.5
            and max(abs(v) for v in node.last_cmd) < 1e-3,
            time.monotonic() + 3.0)
        if stop_reason == 'goal_tolerance' and not confirmed_stop:
            stop_reason = 'unconfirmed_stop'
        if node.started_at is not None and node.last_pose is not None:
            x, y, yaw = node.last_pose
            node.trace.append({'t_sec': time.monotonic() - node.started_at,
                               'x': x, 'y': y, 'yaw': yaw,
                               'cmd_v': node.last_cmd[0], 'cmd_w': node.last_cmd[1]})
    except Exception as exc:  # preserve partial run for audit
        node.errors.append(str(exc))
    finally:
        try:
            node.set_bool(node.follower_stop, True, timeout=3)
            node.set_bool(node.motor, False, timeout=3)
        except Exception as exc:
            node.errors.append(f'shutdown: {exc}')
        if contact_process is not None:
            try:
                contact_data = finish_contacts(contact_process, contact_file)
            except Exception as exc:
                node.errors.append(f'contacts: {exc}')
        node.writer.release()
        node.destroy_node()
        rclpy.shutdown()
    events = contact_data['events'] if contact_data is not None else None
    if events is not None and node.started_at is not None:
        for event in events:
            event['t_sec'] = event['wall_monotonic_sec'] - node.started_at
    return {**episode, 'model': args.model, 'deployment': 'remote_gpu_local_edge',
            'oracle': manifest['oracle'],
            'pose_source': 'gazebo_model_states',
            'instruction_sent': node.instruction, 'trace': node.trace,
            'stop_reason': stop_reason, 'confirmed_stop': confirmed_stop,
            'collisions': len(events) if events is not None else None,
            'contact_events': events,
            'contact_samples': contact_data['samples'] if contact_data else None,
            'contact_log': contact_file.name if contact_data else None,
            'observations': node.received,
            'embeddings': node.published, 'paths': node.path_count,
            'nonempty_paths': node.nonempty_paths,
            'path_samples': node.path_samples,
            'model_version': node.model_version,
            'inferences': node.inferences, 'errors': node.errors,
            'video': args.video.name, 'video_frames': node.video_frames,
            'final_odom_xy': node.last_odom_pose}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--episode', required=True)
    parser.add_argument('--manifest', type=Path,
                        default=Path('/workspace/bench/episodes/pilot.json'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--video', type=Path, required=True)
    parser.add_argument('--text')
    parser.add_argument('--duration', type=float, default=45.0)
    parser.add_argument('--startup-timeout', type=float, default=90.0)
    parser.add_argument('--tolerance', type=float, default=0.30)
    args = parser.parse_args()
    result = run(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'model': args.model, 'episode': args.episode,
                      'stop_reason': result['stop_reason'],
                      'embeddings': result['embeddings'],
                      'video_frames': result['video_frames'],
                      'collisions': result['collisions'],
                      'final': result['trace'][-1] if result['trace'] else None,
                      'errors': result['errors']}), flush=True)


if __name__ == '__main__':
    main()
