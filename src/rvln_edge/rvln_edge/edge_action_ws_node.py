"""Bridge Web/mobile WebSocket action chunks to nav_msgs/Path.

The JSON protocol mirrors ActionChunk and ControlAck; values use fp16+base64.
The path follower handles tracking and safe-stop (an empty Path). A watchdog
publishes one empty Path after the chunk timeout. The WebSocket thread only
updates a locked slot; a ROS timer publishes at a separate rate. The server
lives with the process, and websockets is imported only when it starts.
"""
from __future__ import annotations

import base64
import json
import threading
import time
from typing import Optional, Tuple

import rclpy
from nav_msgs.msg import Path
from rclpy.node import Node

from .action_path_bridge import ActionPathBridge, decode_path


def decode_chunk_msg(
    msg: dict,
    *,
    waypoint_spacing: float,
    frame_id: str = 'base_link',
) -> Tuple[Path, int, str]:
    """Convert a JSON action chunk to a Path, sender frame, and goal ID.

    Convert waypoint units to meters unless scaled_to_m is true. Malformed input
    raises ValueError for the caller to include in its acknowledgement.
    """
    if msg.get('type') != 'action_chunk':
        raise ValueError(f"unexpected type: {msg.get('type')!r}")
    try:
        num_tokens = int(msg['num_tokens'])
        embed_dim = int(msg['embed_dim'])
        raw = base64.b64decode(msg['values_fp16_b64'], validate=True)
    except (KeyError, TypeError, ValueError) as e:  # binascii.Error is a ValueError.
        raise ValueError(f'malformed action_chunk: {e}') from e
    path = decode_path(
        raw, num_tokens=num_tokens, embed_dim=embed_dim,
        scaled_to_m=bool(msg.get('scaled_to_m')),
        waypoint_spacing=waypoint_spacing, frame_id=frame_id,
    )
    return path, int(msg.get('frame_id', 0)), str(msg.get('goal_id', ''))


class EdgeActionWsNode(Node):
    """WebSocket server to nav_msgs/Path bridge with a watchdog."""

    def __init__(self) -> None:
        super().__init__('edge_action_ws')
        self.declare_parameter('host', '0.0.0.0')
        self.declare_parameter('port', 8765)
        self.declare_parameter('path_topic', '/rvln/predicted_path')
        self.declare_parameter('frame_id', 'base_link')
        self.declare_parameter('chunk_max_age_sec', 1.0)
        # Convert model output units to meters (0.1 m/unit for OmniVLA-edge).
        self.declare_parameter('waypoint_spacing', 0.1)
        self.declare_parameter('publish_rate_hz', 20.0)

        self._host: str = self.get_parameter('host').value
        self._port: int = self.get_parameter('port').value
        self._frame_id: str = self.get_parameter('frame_id').value
        self._max_age_sec: float = self.get_parameter('chunk_max_age_sec').value
        self._spacing: float = self.get_parameter('waypoint_spacing').value

        self._pub = self.create_publisher(
            Path, self.get_parameter('path_topic').value, 1)

        self._bridge = ActionPathBridge(frame_id=self._frame_id, max_age_sec=self._max_age_sec)

        rate = float(self.get_parameter('publish_rate_hz').value)
        self._timer = self.create_timer(1.0 / rate, self._on_timer)
        self._ws_thread: Optional[threading.Thread] = None

    # Receive on the WebSocket thread.

    def handle_message(self, text: str, now: Optional[float] = None) -> dict:
        """Handle one message and return a ControlAck-shaped dictionary."""
        now = time.monotonic() if now is None else now
        try:
            msg = json.loads(text)
            if not isinstance(msg, dict):
                raise ValueError('not a JSON object')
            path, frame_seq, goal_id = decode_chunk_msg(
                msg, waypoint_spacing=self._spacing, frame_id=self._frame_id)
        except ValueError as e:
            self.get_logger().warning(f'bad chunk: {e}')
            following = self._bridge.following
            return {'type': 'ack', 'frame_id': 0, 'following': following,
                    'status': f'error: {e}'}

        previous = self._bridge.receive(path, goal_id, now)
        if previous is not None:
            self.get_logger().info(f'goal changed: {previous!r} -> {goal_id!r}')
        return {'type': 'ack', 'frame_id': frame_seq, 'following': True,
                'status': 'ok'}

    # Publish on the ROS timer.

    def _on_timer(self) -> None:
        self._tick(time.monotonic())

    def _tick(self, now: float) -> None:
        """Publish the latest chunk and run the watchdog; callable in tests."""
        path, timed_out = self._bridge.tick(now)
        if path is not None:
            path.header.stamp = self.get_clock().now().to_msg()
            self._pub.publish(path)
        if timed_out:
            self.get_logger().warning(
                f'no chunk for > {self._max_age_sec:.1f}s -> empty Path for safe-stop')

    # ------------------------------------------------------------- WS server

    def start_server(self) -> None:
        """Start the WebSocket server in a daemon thread."""
        self._ws_thread = threading.Thread(
            target=self._serve_forever, name='edge_action_ws', daemon=True)
        self._ws_thread.start()

    def _serve_forever(self) -> None:
        import asyncio

        try:
            # Support the asyncio API (13+) and the legacy API (9-12).
            try:
                from websockets.asyncio.server import serve  # type: ignore
            except ImportError:
                from websockets.server import serve  # type: ignore
        except ImportError:
            self.get_logger().error(
                'python3-websockets is missing; WS server cannot start '
                '(pip install websockets)')
            return

        # The optional path parameter supports v9 and v14+ handler APIs.
        async def handler(websocket, path=None):  # noqa: ANN001
            peer = getattr(websocket, 'remote_address', '?')
            self.get_logger().info(f'client connected: {peer}')
            try:
                async for text in websocket:
                    ack = self.handle_message(text)
                    await websocket.send(json.dumps(ack))
            except Exception as e:  # ConnectionClosed is expected on disconnect.
                self.get_logger().debug(f'client {peer} closed: {e}')
            finally:
                self.get_logger().info(f'client disconnected: {peer}')

        async def serve_main() -> None:
            async with serve(handler, self._host, self._port):
                self.get_logger().info(
                    f'EdgeActionService(WS) listening on ws://{self._host}:{self._port}')
                await asyncio.Future()  # The daemon thread exits with the process.

        try:
            asyncio.run(serve_main())
        except Exception as e:
            self.get_logger().error(f'WS server died: {e}')


def main() -> None:
    rclpy.init()
    node = EdgeActionWsNode()
    node.start_server()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
