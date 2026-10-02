"""Drive the Isaac RVLN robot through a pilot world and verify contacts."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from importlib.util import find_spec
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world", type=Path, required=True)
    parser.add_argument("--robot-urdf", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--expected-obstacle", default="red_marker")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=False)
    package = find_spec("isaacsim")
    if package is None or not package.submodule_search_locations:
        raise RuntimeError("Isaac Sim is not installed")
    bundle = (
        Path(next(iter(package.submodule_search_locations)))
        / "exts"
        / "isaacsim.ros2.core"
        / "humble"
        / "rclpy"
    )
    sys.path.insert(0, str(bundle))

    import rclpy
    from geometry_msgs.msg import Twist
    from nav_msgs.msg import Odometry
    from rclpy.node import Node
    from std_srvs.srv import SetBool

    contacts = args.out_dir / "contacts.json"
    command = [
        sys.executable,
        "-m",
        "bench.cli",
        "rvln",
        "--world",
        str(args.world.resolve(strict=True)),
        "--robot-urdf",
        str(args.robot_urdf.resolve(strict=True)),
        "--physics-only",
        "--headless",
        "--max-seconds",
        "12",
        "--contact-out",
        str(contacts),
    ]
    with (args.out_dir / "isaac.log").open("w", encoding="utf-8") as log:
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        rclpy.init()
        node = Node("isaac_contact_smoke")
        positions = []
        node.create_subscription(
            Odometry,
            "/odom",
            lambda message: positions.append(
                (float(message.pose.pose.position.x), float(message.pose.pose.position.y))
            ),
            10,
        )
        publisher = node.create_publisher(Twist, "/cmd_vel", 10)
        motor = node.create_client(SetBool, "/motor_power")
        try:
            deadline = time.monotonic() + 90
            while not motor.wait_for_service(timeout_sec=0.2):
                if child.poll() is not None or time.monotonic() >= deadline:
                    raise RuntimeError("Isaac motor service did not become ready")
            request = SetBool.Request()
            request.data = True
            future = motor.call_async(request)
            rclpy.spin_until_future_complete(node, future, timeout_sec=10)
            if not future.done() or not future.result().success:
                raise RuntimeError("Isaac motor did not enable")
            command = Twist()
            command.linear.x = 0.4
            until = time.monotonic() + 8
            while time.monotonic() < until and child.poll() is None:
                publisher.publish(command)
                rclpy.spin_once(node, timeout_sec=0.03)
            publisher.publish(Twist())
            child.wait(timeout=90)
        finally:
            node.destroy_node()
            rclpy.shutdown()
            if child.poll() is None:
                child.terminate()
                child.wait(timeout=15)
    if not contacts.is_file():
        raise RuntimeError("Isaac did not write a contact report")
    report = json.loads(contacts.read_text(encoding="utf-8"))
    result = {
        "exit_code": child.returncode,
        "odom_samples": len(positions),
        "start_xy": positions[0] if positions else None,
        "end_xy": positions[-1] if positions else None,
        "collisions": report["collisions"],
        "obstacles": [event["obstacle"] for event in report["events"]],
    }
    (args.out_dir / "result.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result), flush=True)
    if child.returncode != 0 or not positions or args.expected_obstacle not in result["obstacles"]:
        raise RuntimeError(f"Isaac robot did not report {args.expected_obstacle} contact")


if __name__ == "__main__":
    main()
