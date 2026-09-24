"""Bridge Flutter gRPC action chunks to nav_msgs/Path on the Pi.

Implements EdgeActionService.StreamActions from proto/edge_action.proto. The
path follower handles tracking and safe-stop (an empty Path). This matches the
WebSocket bridge apart from transport and message format. A watchdog publishes
one empty Path after the chunk timeout. The receive thread only updates a
locked slot; a ROS timer publishes at a separate rate. Import grpc at startup.
"""
from __future__ import annotations

import time
from concurrent import futures
from typing import Optional, Tuple

import rclpy
from nav_msgs.msg import Path
from rclpy.node import Node

from rvln_proto import edge_action_pb2
from .action_path_bridge import ActionPathBridge, decode_path


def decode_action_chunk(
    chunk: edge_action_pb2.ActionChunk,
    *,
    waypoint_spacing: float,
    frame_id: str = 'base_link',
) -> Tuple[Path, int, str]:
    """Convert a proto action chunk to a Path, sender frame, and goal ID.

    Convert waypoint units to meters unless scaled_to_m is true. Malformed input
    raises ValueError for the caller to include in its acknowledgement.
    """
    num_tokens = int(chunk.num_tokens)
    embed_dim = int(chunk.embed_dim)
    path = decode_path(
        chunk.values_fp16, num_tokens=num_tokens, embed_dim=embed_dim,
        scaled_to_m=chunk.scaled_to_m,
        waypoint_spacing=waypoint_spacing, frame_id=frame_id,
    )
    return path, int(chunk.frame_id), str(chunk.goal_id)


class EdgeActionGrpcNode(Node):
    """gRPC server to nav_msgs/Path bridge with a watchdog."""

    def __init__(self) -> None:
        super().__init__('edge_action_grpc')
        self.declare_parameter('host', '0.0.0.0')
        self.declare_parameter('port', 50061)
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
        self._grpc_server = None

    # Receive on the gRPC thread.

    def handle_chunk(
        self,
        chunk: edge_action_pb2.ActionChunk,
        now: Optional[float] = None,
    ) -> edge_action_pb2.ControlAck:
        """Handle one chunk and return its ControlAck."""
        now = time.monotonic() if now is None else now
        try:
            path, frame_seq, goal_id = decode_action_chunk(
                chunk, waypoint_spacing=self._spacing, frame_id=self._frame_id)
        except ValueError as e:
            self.get_logger().warning(f'bad chunk: {e}')
            following = self._bridge.following
            return edge_action_pb2.ControlAck(
                frame_id=int(chunk.frame_id), following=following,
                status=f'error: {e}')

        previous = self._bridge.receive(path, goal_id, now)
        if previous is not None:
            self.get_logger().info(f'goal changed: {previous!r} -> {goal_id!r}')
        # Follow dummy trajectories for non-motor cmd_vel preview tests.
        # Preview normally publishes to a non-motor topic.
        # Expose dummy output in the ack status during motor operation.
        status = 'ok' if chunk.from_model else 'ok-dummy'
        return edge_action_pb2.ControlAck(
            frame_id=frame_seq, following=True, status=status)

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

    # ------------------------------------------------------------- gRPC server

    def start_server(self) -> None:
        """Start the gRPC server using its own thread pool."""
        import grpc

        from rvln_proto import edge_action_pb2_grpc

        node = self

        class _Servicer(edge_action_pb2_grpc.EdgeActionServiceServicer):
            def StreamActions(self, request_iterator, context):  # noqa: N802
                peer = context.peer()
                node.get_logger().info(f'client connected: {peer}')
                try:
                    for chunk in request_iterator:
                        yield node.handle_chunk(chunk)
                finally:
                    node.get_logger().info(f'client disconnected: {peer}')

        server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
        edge_action_pb2_grpc.add_EdgeActionServiceServicer_to_server(
            _Servicer(), server)
        server.add_insecure_port(f'{self._host}:{self._port}')
        server.start()
        self._grpc_server = server
        self.get_logger().info(
            f'EdgeActionService(gRPC) listening on {self._host}:{self._port}')

    def stop_server(self) -> None:
        if self._grpc_server is not None:
            self._grpc_server.stop(grace=1.0)
            self._grpc_server = None


def main() -> None:
    rclpy.init()
    node = EdgeActionGrpcNode()
    node.start_server()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop_server()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
