"""Record a deterministic-rate sequence from the local Gazebo camera."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from PIL import Image as PILImage
from rclpy.node import Node
from sensor_msgs.msg import Image


class Capture(Node):
    def __init__(self, output: Path, count: int, interval: float) -> None:
        super().__init__('rvln_bench_capture')
        self.output = output
        self.count = count
        self.interval = interval
        self.rows: list[dict] = []
        self.next_time = 0.0
        self.odom = None
        self.create_subscription(Image, '/camera/color/image_raw', self.on_image, 10)
        self.create_subscription(Odometry, '/odom', self.on_odom, 10)

    def on_odom(self, msg: Odometry) -> None:
        self.odom = msg

    def on_image(self, msg: Image) -> None:
        now = time.monotonic()
        if len(self.rows) >= self.count or now < self.next_time:
            return
        if msg.encoding not in ('rgb8', 'bgr8'):
            raise ValueError(f'unsupported camera encoding {msg.encoding!r}')
        pixels = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.step)
        pixels = pixels[:, :msg.width * 3].reshape(msg.height, msg.width, 3)
        if msg.encoding == 'bgr8':
            pixels = pixels[:, :, ::-1]
        name = f'{len(self.rows):04d}.jpg'
        PILImage.fromarray(pixels).save(self.output / name, quality=85)
        position = self.odom.pose.pose.position if self.odom is not None else None
        self.rows.append({
            'frame': name,
            'capture_monotonic_ns': time.monotonic_ns(),
            'ros_stamp_sec': msg.header.stamp.sec,
            'ros_stamp_nanosec': msg.header.stamp.nanosec,
            'width': msg.width,
            'height': msg.height,
            'odom_xy': [position.x, position.y] if position is not None else None,
        })
        self.next_time = now + self.interval


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--count', type=int, default=20)
    parser.add_argument('--interval', type=float, default=0.5)
    parser.add_argument('--timeout', type=float, default=90.0)
    args = parser.parse_args()
    if args.count < 1 or args.interval <= 0:
        parser.error('count and interval must be positive')
    args.out.mkdir(parents=True, exist_ok=True)
    rclpy.init()
    node = Capture(args.out, args.count, args.interval)
    deadline = time.monotonic() + args.timeout
    try:
        while len(node.rows) < args.count and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.2)
    finally:
        node.destroy_node()
        rclpy.shutdown()
    (args.out / 'capture.json').write_text(
        json.dumps({'schema': 1, 'frames': node.rows}, indent=2) + '\n',
        encoding='utf-8',
    )
    if len(node.rows) != args.count:
        raise RuntimeError(f'captured {len(node.rows)}/{args.count} frames')
    print(f'captured {len(node.rows)} frames in {args.out}')


if __name__ == '__main__':
    main()
