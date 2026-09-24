#!/usr/bin/env python3
"""Send goals / motor commands to a running rvln edge (sim or real).

Mode-agnostic: it talks to whatever edge node is up (edge, cmd_vel, sim, or
edge-local) — the goal topic and /motor_power service are the same across them.
A bare ``--mode remote`` box has no edge node, so there is nothing here to drive.

Runs *inside* the ROS environment — it needs rclpy and the rvln_msgs
overlay on the path. From the host use the ``scripts/control.sh`` wrapper, which
execs this inside the running edge container with the overlays sourced.

Subcommands::

    motor on|off                       toggle raspimouse motor power
    goal pose X Y [THETA] [FRAME]      send a POSE goal (FRAME default: odom)
    goal text "go down the hallway"   send a TEXT (language) goal
    goal image /path/to/goal.jpg      send an IMAGE goal (path inside container)
    stop                               motor off (robot coasts to a halt)
    status                             print cmd_vel / cmd_vel_vla / sim_cmd_vel / odom once

The ``control.sh`` wrapper additionally handles ``logs [-f] [server|edge]`` on
the host (a ``docker logs`` shortcut for model-load progress + runtime output);
that never reaches this helper.

The edge stays idle (zero cmd_vel) until a goal arrives, and the raspimouse gates
cmd_vel -> sim_cmd_vel on motor power, so a typical first run is ``motor on``
followed by ``goal pose 2 0``. ``motor off`` (or ``stop``) releases the motors.
"""
from __future__ import annotations

import sys
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile

from geometry_msgs.msg import Twist
from std_srvs.srv import SetBool
from rvln_msgs.msg import GoalSpec

GOAL_TOPIC = '/rvln/goal'
MOTOR_SERVICE = '/motor_power'


def _set_motor(node: Node, on: bool) -> int:
    cli = node.create_client(SetBool, MOTOR_SERVICE)
    if not cli.wait_for_service(timeout_sec=5.0):
        node.get_logger().error(f'service {MOTOR_SERVICE} unavailable')
        return 1
    fut = cli.call_async(SetBool.Request(data=on))
    rclpy.spin_until_future_complete(node, fut, timeout_sec=5.0)
    res = fut.result()
    if res is None:
        node.get_logger().error('motor_power call timed out')
        return 1
    print(f'motor_power({on}): success={res.success} message="{res.message}"')
    return 0 if res.success else 1


def _publish_goal(node: Node, goal: GoalSpec) -> int:
    # transient_local so the edge receives it even if we publish-and-exit; a
    # volatile reader is compatible with a transient_local writer.
    qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
    pub = node.create_publisher(GoalSpec, GOAL_TOPIC, qos)
    # spin briefly so discovery + delivery actually happen before we drop the node
    deadline = time.time() + 1.5
    sent = False
    while time.time() < deadline:
        if pub.get_subscription_count() > 0 and not sent:
            pub.publish(goal)
            sent = True
        rclpy.spin_once(node, timeout_sec=0.1)
    if not sent:
        pub.publish(goal)  # no subscriber seen, fire anyway
        node.get_logger().warn(f'no subscriber on {GOAL_TOPIC}; published regardless')
    print(f'published goal mode={goal.mode} to {GOAL_TOPIC}')
    return 0


def _goal_pose(args: list[str]) -> GoalSpec:
    if len(args) < 2:
        raise SystemExit('usage: goal pose X Y [THETA] [FRAME]')
    x, y = float(args[0]), float(args[1])
    theta = float(args[2]) if len(args) > 2 else 0.0
    frame = args[3] if len(args) > 3 else 'odom'
    g = GoalSpec()
    g.mode = GoalSpec.MODE_POSE
    g.pose.header.frame_id = frame
    g.pose.pose.position.x = x
    g.pose.pose.position.y = y
    # yaw -> quaternion (z, w)
    import math
    g.pose.pose.orientation.z = math.sin(theta / 2.0)
    g.pose.pose.orientation.w = math.cos(theta / 2.0)
    return g


def _goal_text(args: list[str]) -> GoalSpec:
    if not args:
        raise SystemExit('usage: goal text "instruction"')
    g = GoalSpec()
    g.mode = GoalSpec.MODE_TEXT
    g.text = ' '.join(args)
    return g


def _goal_image(args: list[str]) -> GoalSpec:
    if not args:
        raise SystemExit('usage: goal image /path/to/goal.jpg')
    with open(args[0], 'rb') as fh:
        data = fh.read()
    g = GoalSpec()
    g.mode = GoalSpec.MODE_IMAGE
    g.image.format = 'jpeg'
    g.image.data = list(data)
    return g


def _status(node: Node) -> int:
    from nav_msgs.msg import Odometry
    seen: dict[str, str] = {}

    def grab(topic, msg_type, fmt):
        def cb(msg):
            seen[topic] = fmt(msg)
        return node.create_subscription(msg_type, topic, cb, 1)

    def twist_fmt(m):
        return f'lin.x={m.linear.x:.3f} ang.z={m.angular.z:.3f}'
    # /cmd_vel      -> real robot / edge-local; /cmd_vel_vla -> cmd_vel preview
    # mode (non-motor topic); /sim_cmd_vel + /odom -> Gazebo sim. Whichever the
    # running mode doesn't publish simply shows "(no message)".
    topics = ('/cmd_vel', '/cmd_vel_vla', '/sim_cmd_vel', '/odom')
    grab('/cmd_vel', Twist, twist_fmt)
    grab('/cmd_vel_vla', Twist, twist_fmt)
    grab('/sim_cmd_vel', Twist, twist_fmt)
    grab('/odom', Odometry,
         lambda m: f'x={m.pose.pose.position.x:.3f} y={m.pose.pose.position.y:.3f}')
    deadline = time.time() + 3.0
    while time.time() < deadline and len(seen) < len(topics):
        rclpy.spin_once(node, timeout_sec=0.1)
    for t in topics:
        print(f'{t:16s} {seen.get(t, "(no message)")}')
    return 0


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    rclpy.init()
    node = rclpy.create_node('rvln_control')
    try:
        cmd = argv[0]
        if cmd == 'motor':
            if len(argv) < 2 or argv[1] not in ('on', 'off'):
                raise SystemExit('usage: motor on|off')
            return _set_motor(node, argv[1] == 'on')
        if cmd == 'stop':
            return _set_motor(node, False)
        if cmd == 'status':
            return _status(node)
        if cmd == 'goal':
            if len(argv) < 2:
                raise SystemExit('usage: goal pose|text|image ...')
            kind, rest = argv[1], argv[2:]
            builder = {'pose': _goal_pose, 'text': _goal_text, 'image': _goal_image}.get(kind)
            if builder is None:
                raise SystemExit(f'unknown goal kind: {kind}')
            return _publish_goal(node, builder(rest))
        raise SystemExit(f'unknown command: {cmd}')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
