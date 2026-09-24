"""Forward local ROS observations to a Slurm VLN server and publish embeddings."""
from __future__ import annotations

import argparse
import base64
import json
import time
import urllib.request

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rvln_msgs.msg import ActionEmbedding, Observation


class Bridge(Node):
    def __init__(self, url: str) -> None:
        super().__init__('rvln_bench_live_bridge')
        self.url = url.rstrip('/') + '/infer'
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.publisher = self.create_publisher(ActionEmbedding, '/rvln/remote_embedding', qos)
        self.create_subscription(Observation, '/rvln/observation', self.on_observation, qos)
        self.received = 0
        self.published = 0

    def on_observation(self, observation: Observation) -> None:
        self.received += 1
        if observation.goal.mode != 1:
            self.get_logger().error('live bridge currently requires text goals')
            return
        payload = {'frame_id': int(observation.frame_id), 'text': observation.goal.text,
                   'jpeg_base64': base64.b64encode(bytes(observation.image.data)).decode('ascii')}
        request = urllib.request.Request(
            self.url, json.dumps(payload).encode('utf-8'),
            {'Content-Type': 'application/json'}, method='POST')
        start = time.monotonic()
        try:
            with urllib.request.urlopen(request, timeout=4.0) as response:
                result = json.load(response)
        except (OSError, ValueError) as exc:
            self.get_logger().error(f'remote inference failed: {exc}')
            return
        if result['frame_id'] != observation.frame_id:
            self.get_logger().error('remote frame ID mismatch')
            return
        embedding = result['embedding']
        if not embedding or not embedding[0] or any(
                len(token) != len(embedding[0]) for token in embedding):
            self.get_logger().error('invalid remote embedding shape')
            return
        msg = ActionEmbedding()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.frame_id = observation.frame_id
        msg.num_tokens = len(embedding)
        msg.embed_dim = len(embedding[0])
        msg.embedding = [float(value) for token in embedding for value in token]
        msg.inference_ms = float(result['inference_ms'])
        msg.model_version = result['model_version']
        self.publisher.publish(msg)
        self.published += 1
        print(json.dumps({'frame_id': int(msg.frame_id), 'round_trip_ms':
                          (time.monotonic() - start) * 1000.0,
                          'server_wall_ms': result['server_wall_ms']}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', required=True)
    parser.add_argument('--duration', type=float, default=30.0)
    args = parser.parse_args()
    rclpy.init()
    bridge = Bridge(args.url)
    deadline = time.monotonic() + args.duration
    try:
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(bridge, timeout_sec=0.1)
    finally:
        bridge.destroy_node()
        rclpy.shutdown()
    print(json.dumps({'received': bridge.received, 'published': bridge.published}), flush=True)


if __name__ == '__main__':
    main()
