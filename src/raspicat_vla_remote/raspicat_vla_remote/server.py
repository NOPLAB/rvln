"""ROS 2 topic bridge from compressed observations to VLA backends."""
from __future__ import annotations

import io
import logging

import numpy as np
import PIL.Image
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

from raspicat_vla_msgs.msg import ActionEmbedding, GoalSpec, Observation

from .backends.base import VLABackend


_LOG = logging.getLogger(__name__)


def _decode_image(data: bytes) -> PIL.Image.Image:
    """Decode JPEG, preserving the dummy backend's empty-image fallback."""
    if not data:
        return PIL.Image.new('RGB', (1, 1))
    try:
        return PIL.Image.open(io.BytesIO(data)).convert('RGB')
    except Exception as exc:  # noqa: BLE001
        _LOG.warning('image decode failed: %s; using 1x1 placeholder', exc)
        return PIL.Image.new('RGB', (1, 1))


def _goal_to_python(goal: GoalSpec):
    """Map the native ROS goal message to backend inputs."""
    if goal.mode == GoalSpec.MODE_POSE:
        p = goal.pose.pose.position
        q = goal.pose.pose.orientation
        yaw = float(np.arctan2(2 * (q.w * q.z + q.x * q.y),
                               1 - 2 * (q.y * q.y + q.z * q.z)))
        return (p.x, p.y, yaw), '', None
    if goal.mode == GoalSpec.MODE_TEXT:
        return None, goal.text, None
    if goal.mode == GoalSpec.MODE_IMAGE:
        return None, '', _decode_image(bytes(goal.image.data))
    raise ValueError(f'unknown goal mode {goal.mode}')


class VLAInferenceNode(Node):
    """Consume the newest observation and publish one correlated embedding."""

    def __init__(self, *, backend: VLABackend) -> None:
        super().__init__('vla_inference_node')
        self.declare_parameter('observation_topic', '/raspicat_vla/observation')
        self.declare_parameter('embedding_topic', '/raspicat_vla/remote_embedding')
        self._backend = backend
        self._past_image = None
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        self._publisher = self.create_publisher(
            ActionEmbedding, self.get_parameter('embedding_topic').value, qos)
        self._subscriber = self.create_subscription(
            Observation, self.get_parameter('observation_topic').value,
            self._on_observation, qos)

    def _on_observation(self, obs: Observation) -> None:
        current = _decode_image(bytes(obs.image.data))
        past = self._past_image if self._past_image is not None else current
        self._past_image = current
        try:
            pose, text, goal_image = _goal_to_python(obs.goal)
            result, metrics = self._backend.infer(
                current_image=current,
                past_image=past,
                lang_instruction=text,
                goal_image=goal_image,
                goal_pose_xy_theta=pose,
            )
            arr = np.asarray(result, dtype=np.float32)
            if arr.ndim < 2:
                raise ValueError(f'backend returned shape {arr.shape}, expected tokens x dim')
            arr = arr.reshape(-1, arr.shape[-1])
            msg = ActionEmbedding()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.frame_id = obs.frame_id
            msg.num_tokens, msg.embed_dim = arr.shape
            msg.embedding = arr.reshape(-1).tolist()
            msg.inference_ms = float(metrics.get('inference_ms', 0.0))
            msg.model_version = self._backend.model_info().model_version
            self._publisher.publish(msg)
            self.get_logger().info(
                f'frame={obs.frame_id} mode={obs.goal.mode} '
                f'inference={msg.inference_ms:.0f}ms shape={arr.shape}')
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f'inference failed for frame {obs.frame_id}: {exc}')
