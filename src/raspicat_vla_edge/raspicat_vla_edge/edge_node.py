"""VLA edge LifecycleNode — Plan 1 skeleton.

In Plan 1 the "edge adapter" is a stub that emits a fixed straight-ahead
path of length 1.0 m sampled at 0.1 m. Plan 2 replaces this stub with the
real Edge Adapter PyTorch model (model-specific, e.g. AsyncVLA / OmniVLA).
"""
from __future__ import annotations

import time
from typing import Optional

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from nav_msgs.msg import Path
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.lifecycle import LifecycleNode, TransitionCallbackReturn, State
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage, Image

from raspicat_vla_msgs.msg import (
    ActionEmbedding as ActionEmbeddingMsg,
    GoalSpec as GoalSpecMsg,
    Observation as ObservationMsg,
)

from .preprocess import resize_and_jpeg
from .embedding_cache import EmbeddingCache, CachedEmbedding
from .adapters.base import EdgeAdapter, EdgeGoal
from .camera_capture import V4L2CameraCapture
from .observation_state import CameraFrameStore, ObservationLedger


def _build_adapter(kind: str, *, params: dict) -> EdgeAdapter:
    """Construct the EdgeAdapter selected by the ``adapter_kind`` parameter.

    ``params`` is a dict of node parameters used by adapters that need extra
    config (e.g. AsyncVLA's Edge_adapter weights path).
    """
    if kind == 'stub':
        from .adapters.stub import StubAdapter
        return StubAdapter()
    if kind == 'omnivla':
        from .adapters.omnivla import OmniVLAEdgeAdapter
        return OmniVLAEdgeAdapter()
    if kind == 'omnivla_edge_local':
        from .adapters.omnivla_edge_local import OmniVLAEdgeLocalAdapter
        return OmniVLAEdgeLocalAdapter(
            weights_path=str(params.get(
                'omnivla_edge_weights_path', '/workspace/models/omnivla-edge/omnivla-edge.pth')),
            clip_type=str(params.get('omnivla_edge_clip_type', 'ViT-B/32')),
            device=str(params.get('omnivla_edge_device', 'cuda:0')),
        )
    if kind == 'asyncvla':
        from .adapters.asyncvla import AsyncVLAEdgeAdapter
        return AsyncVLAEdgeAdapter(
            weights_path=str(params.get('asyncvla_weights_path',
                             '/workspace/models/AsyncVLA_release')),
            resume_step=int(params.get('asyncvla_resume_step', 750000)),
            device=str(params.get('asyncvla_device', 'cpu')),
        )
    raise ValueError(
        f'unknown adapter_kind: {kind!r} '
        '(choices: stub|asyncvla|omnivla|omnivla_edge_local)'
    )


def _quat_to_yaw(qx: float, qy: float, qz: float, qw: float) -> float:
    """Yaw (rad) from a quaternion; 0.0 for a zero (uninitialized) quaternion."""
    if qx == 0.0 and qy == 0.0 and qz == 0.0 and qw == 0.0:
        return 0.0
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return float(np.arctan2(siny_cosp, cosy_cosp))


class VLAEdgeNode(LifecycleNode):

    def __init__(self) -> None:
        super().__init__('vla_edge_node')
        self._declare_parameters()
        self._bridge = CvBridge()
        self._camera_frames = CameraFrameStore()
        self._observations = ObservationLedger(max_frames=8)
        # Monotonic timestamp of the last action-tick diagnostic log line.
        self._last_diag_log_ns = 0
        # Duration of the most recent adapter inference, for the diag line.
        self._last_predict_ms = 0.0
        # In-process camera capture (used when the `camera_device` parameter is
        # set): the edge grabs the V4L2 device directly instead of subscribing
        # to a separate driver node's topic. One process fewer on the robot and
        # no 30 fps raw-Image DDS hop — both matter on a Pi where CPU contention
        # was starving the camera feed.
        self._camera = V4L2CameraCapture(
            self._camera_frames,
            lambda message: self.get_logger().warn(message, throttle_duration_sec=5.0),
        )
        # Callback groups: rclpy puts every callback in ONE mutually exclusive
        # group by default, so a CPU-heavy adapter inference in the action tick
        # blocks the image callback for its whole duration — on a Pi the camera
        # feed then reaches the node at <1 Hz and the freshness guard flaps
        # (surge-and-stop driving). Isolate the heavy tick so I/O keeps flowing
        # on the executor's other threads.
        self._heavy_group = MutuallyExclusiveCallbackGroup()
        self._io_group = MutuallyExclusiveCallbackGroup()
        self._cache: Optional[EmbeddingCache] = None
        self._observation_pub = None
        self._remote_embedding_sub = None
        self._adapter: Optional[EdgeAdapter] = None
        # True when the adapter runs the policy on-edge (no remote/cache).
        self._local_mode = False
        self._send_timer = None
        self._action_timer = None
        self._status_timer = None
        self._image_sub = None
        self._goal_sub = None
        self._path_pub = None
        self._embedding_pub = None
        self._status_pub = None

    # ----------------------------------------------------------------- params

    def _declare_parameters(self) -> None:
        self.declare_parameter('observation_topic', '/raspicat_vla/observation')
        self.declare_parameter('remote_embedding_topic', '/raspicat_vla/remote_embedding')
        self.declare_parameter('obs_publish_rate_hz', 2.0)
        self.declare_parameter('action_rate_hz', 10.0)
        self.declare_parameter('image_size', [224, 224])
        # A camera frame older than this is treated as "no image": stop
        # sending observations and safe-stop the action loop. Prevents blind
        # driving on a frozen frame when the camera driver dies mid-run.
        self.declare_parameter('image_max_age_sec', 2.0)
        self.declare_parameter('jpeg_quality', 85)
        self.declare_parameter('embedding_max_age_sec', 6.0)
        self.declare_parameter('embedding_hard_timeout_sec', 15.0)
        self.declare_parameter('goal_tolerance_m', 0.3)
        # A topic ending in '/compressed' is subscribed as CompressedImage
        # (JPEG) instead of raw Image — use it whenever the camera lives on
        # another host (raw 30 fps Image does not survive WiFi).
        self.declare_parameter('image_topic', '/camera/image_raw')
        # Non-empty = grab this V4L2 device in-process instead of subscribing
        # to image_topic. 0 / 0.0 leave the driver defaults untouched.
        self.declare_parameter('camera_device', '')
        self.declare_parameter('camera_width', 0)
        self.declare_parameter('camera_height', 0)
        self.declare_parameter('camera_fps', 0.0)
        self.declare_parameter('goal_topic', '/raspicat_vla/goal')
        self.declare_parameter('path_topic', '/raspicat_vla/predicted_path')
        self.declare_parameter('status_topic', '/raspicat_vla/status')
        self.declare_parameter('embedding_debug_topic', '/raspicat_vla/embedding')
        self.declare_parameter('publish_embedding_debug', True)
        self.declare_parameter('adapter_kind', 'stub')  # stub|asyncvla|omnivla|omnivla_edge_local
        # AsyncVLA edge knobs (only used when adapter_kind='asyncvla').
        self.declare_parameter('asyncvla_weights_path', '/workspace/models/AsyncVLA_release')
        self.declare_parameter('asyncvla_resume_step', 750000)
        self.declare_parameter('asyncvla_device', 'cpu')
        # OmniVLA-edge local knobs (only used when adapter_kind='omnivla_edge_local').
        self.declare_parameter(
            'omnivla_edge_weights_path', '/workspace/models/omnivla-edge/omnivla-edge.pth')
        self.declare_parameter('omnivla_edge_clip_type', 'ViT-B/32')
        self.declare_parameter('omnivla_edge_device', 'cuda:0')

    # ------------------------------------------------------------- lifecycle

    def on_configure(self, state: State) -> TransitionCallbackReturn:  # noqa: ARG002
        self.get_logger().info('on_configure')
        max_age = self.get_parameter('embedding_max_age_sec').value
        hard = self.get_parameter('embedding_hard_timeout_sec').value
        self._cache = EmbeddingCache(max_age_sec=float(max_age), hard_timeout_sec=float(hard))
        adapter_kind = str(self.get_parameter('adapter_kind').value)
        adapter_params = {
            'asyncvla_weights_path': self.get_parameter('asyncvla_weights_path').value,
            'asyncvla_resume_step': self.get_parameter('asyncvla_resume_step').value,
            'asyncvla_device': self.get_parameter('asyncvla_device').value,
            'omnivla_edge_weights_path': self.get_parameter('omnivla_edge_weights_path').value,
            'omnivla_edge_clip_type': self.get_parameter('omnivla_edge_clip_type').value,
            'omnivla_edge_device': self.get_parameter('omnivla_edge_device').value,
        }
        self._adapter = _build_adapter(adapter_kind, params=adapter_params)
        self._local_mode = bool(getattr(self._adapter, 'is_local', False))
        if not self._local_mode:
            qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
            self._observation_pub = self.create_publisher(
                ObservationMsg, self.get_parameter('observation_topic').value, qos)
            self._remote_embedding_sub = self.create_subscription(
                ActionEmbeddingMsg, self.get_parameter('remote_embedding_topic').value,
                self._on_embedding_received, qos, callback_group=self._io_group)
        self.get_logger().info(
            f'edge adapter_kind={adapter_kind!r} local_mode={self._local_mode}'
        )

        image_topic = self.get_parameter('image_topic').value
        goal_topic = self.get_parameter('goal_topic').value
        path_topic = self.get_parameter('path_topic').value
        status_topic = self.get_parameter('status_topic').value
        emb_topic = self.get_parameter('embedding_debug_topic').value

        camera_device = str(self.get_parameter('camera_device').value)
        if camera_device:
            # In-process capture: the edge owns the camera; no subscription.
            if not self._open_camera(camera_device):
                return TransitionCallbackReturn.FAILURE
            self.get_logger().info(
                f'grabbing camera {camera_device} in-process '
                f'(image_topic {image_topic!r} is NOT subscribed)'
            )
        else:
            # depth=1 + BEST_EFFORT: always hand the callback the newest frame.
            # A deeper queue re-delivers a backlog of stale frames whenever the
            # executor was busy, which defeats the freshness guard's purpose.
            image_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
            if image_topic.endswith('/compressed'):
                # JPEG transport: 20-50x less traffic than raw Image. Essential
                # when the camera sits on another host (e.g. the Pi's camera
                # feeding an edge on the Jetson over WiFi — raw 30 fps Image
                # saturates the link and frames stall for seconds).
                self._image_sub = self.create_subscription(
                    CompressedImage, image_topic, self._on_compressed_image,
                    image_qos, callback_group=self._io_group,
                )
            else:
                self._image_sub = self.create_subscription(
                    Image, image_topic, self._on_image, image_qos,
                    callback_group=self._io_group,
                )
        # Goals are latched: control.py publishes a single goal with
        # TRANSIENT_LOCAL durability and exits. A VOLATILE reader only gets
        # samples published *after* the match is established, so on a slow host
        # (Jetson discovery lag) the one-shot goal races the connection and is
        # silently dropped — the edge then sits in WAITING_REMOTE with
        # frame_counter=0 forever. Match the writer's durability so the last
        # goal is delivered on match regardless of timing.
        goal_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self._goal_sub = self.create_subscription(
            GoalSpecMsg, goal_topic, self._on_goal, goal_qos,
            callback_group=self._io_group,
        )
        self._path_pub = self.create_publisher(Path, path_topic, 10)
        self._status_pub = self.create_publisher(DiagnosticArray, status_topic, 10)
        if self.get_parameter('publish_embedding_debug').value:
            self._embedding_pub = self.create_publisher(ActionEmbeddingMsg, emb_topic, 10)

        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state: State) -> TransitionCallbackReturn:  # noqa: ARG002
        self.get_logger().info('on_activate')
        self._camera.start()
        act_rate = float(self.get_parameter('action_rate_hz').value)
        # In local mode there is no cloud: skip the ROS observation and the
        # observation-send loop; the action loop drives the local policy directly.
        if not self._local_mode:
            obs_rate = float(self.get_parameter('obs_publish_rate_hz').value)
            self._send_timer = self.create_timer(
                1.0 / obs_rate, self._send_observation_tick,
                callback_group=self._io_group,
            )
        # The action tick runs the adapter (CPU inference, possibly ~1 s on the
        # robot) — keep it off the I/O group so camera/goal/send callbacks are
        # never starved behind it.
        self._action_timer = self.create_timer(
            1.0 / act_rate, self._action_tick, callback_group=self._heavy_group,
        )
        self._status_timer = self.create_timer(
            1.0, self._publish_status, callback_group=self._io_group,
        )
        return super().on_activate(state)

    def on_deactivate(self, state: State) -> TransitionCallbackReturn:  # noqa: ARG002
        self.get_logger().info('on_deactivate')
        self._camera.stop()
        self._camera_frames.clear()
        for t in (self._send_timer, self._action_timer, self._status_timer):
            if t is not None:
                self.destroy_timer(t)
        self._send_timer = self._action_timer = self._status_timer = None
        return super().on_deactivate(state)

    def on_cleanup(self, state: State) -> TransitionCallbackReturn:  # noqa: ARG002
        self.get_logger().info('on_cleanup')
        self._camera.close()
        self._observations.reset()
        if self._remote_embedding_sub is not None:
            self.destroy_subscription(self._remote_embedding_sub)
            self._remote_embedding_sub = None
        if self._observation_pub is not None:
            self.destroy_publisher(self._observation_pub)
            self._observation_pub = None
        self._cache = None
        self._adapter = None
        self._local_mode = False
        # Destroy subscriptions and publishers so a subsequent configure
        # doesn't leak duplicates (lifecycle expects on_cleanup to invert
        # on_configure).
        if self._image_sub is not None:
            self.destroy_subscription(self._image_sub)
            self._image_sub = None
        if self._goal_sub is not None:
            self.destroy_subscription(self._goal_sub)
            self._goal_sub = None
        for pub_attr in ('_path_pub', '_status_pub', '_embedding_pub'):
            pub = getattr(self, pub_attr)
            if pub is not None:
                self.destroy_publisher(pub)
                setattr(self, pub_attr, None)
        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state: State) -> TransitionCallbackReturn:  # noqa: ARG002
        self.get_logger().info('on_shutdown')
        return TransitionCallbackReturn.SUCCESS

    # ---------------------------------------------------------- camera (v4l2)

    def _open_camera(self, device: str) -> bool:
        """Open the V4L2 device for in-process capture. False on failure.

        Called from on_configure; a failed open fails the transition, and the
        launch-side retry loop keeps re-running configure until the device is
        available.
        """
        opened = self._camera.open(
            device,
            width=int(self.get_parameter('camera_width').value),
            height=int(self.get_parameter('camera_height').value),
            fps=float(self.get_parameter('camera_fps').value),
        )
        if not opened:
            self.get_logger().error(f'cannot open camera device {device}')
        return opened

    # ----------------------------------------------------------- subscribers

    def _on_image(self, msg: Image) -> None:
        try:
            cv_img = self._bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f'cv_bridge failed: {exc}')
            return
        self._camera_frames.put(cv_img)

    def _on_compressed_image(self, msg: CompressedImage) -> None:
        buf = np.frombuffer(bytes(msg.data), dtype=np.uint8)
        bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if bgr is None:
            self.get_logger().warn(
                'compressed image decode failed', throttle_duration_sec=5.0)
            return
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        self._camera_frames.put(rgb)

    def _ros_goal_to_edge_goal(self, goal: GoalSpecMsg) -> Optional[EdgeGoal]:
        """Convert a GoalSpec ROS msg to the adapter-facing EdgeGoal.

        Used by on-edge adapters (omnivla_edge_local) that run the policy
        locally and need the raw goal, not the cloud embedding.
        """
        if goal.mode == GoalSpecMsg.MODE_POSE:
            q = goal.pose.pose.orientation
            theta = _quat_to_yaw(q.x, q.y, q.z, q.w)
            return EdgeGoal(
                mode='pose',
                pose_xy_theta=(
                    goal.pose.pose.position.x,
                    goal.pose.pose.position.y,
                    theta,
                ),
            )
        if goal.mode == GoalSpecMsg.MODE_TEXT:
            return EdgeGoal(mode='text', text=goal.text)
        if goal.mode == GoalSpecMsg.MODE_IMAGE:
            try:
                img = self._bridge.imgmsg_to_cv2(goal.image, desired_encoding='rgb8')
            except Exception as exc:  # noqa: BLE001
                self.get_logger().warn(f'goal image decode failed: {exc}')
                return None
            return EdgeGoal(mode='image', image_rgb=img)
        self.get_logger().warn(f'unknown goal mode {goal.mode}; not forwarding to adapter')
        return None

    def _on_goal(self, msg: GoalSpecMsg) -> None:
        self.get_logger().info(f'received goal mode={msg.mode}')
        self._observations.change_goal(
            msg,
            lambda floor: self._cache.invalidate(floor=floor) if self._cache else None,
        )
        if self._adapter is not None:
            edge_goal = self._ros_goal_to_edge_goal(msg)
            if edge_goal is not None:
                self._adapter.set_goal(edge_goal)

    # ------------------------------------------------------------ tick: send

    def _fresh_image(self) -> Optional[np.ndarray]:
        """Latest camera frame, or None when none arrived or it went stale.

        Staleness matters because both loops otherwise latch the last frame
        forever: a dead camera would keep feeding the same observation to the
        cloud and the same `cur` to the adapter, and the robot would blindly
        drive the model's constant output for that frozen frame.
        """
        max_age_ns = int(float(self.get_parameter('image_max_age_sec').value) * 1e9)
        img = self._camera_frames.fresh(max_age_ns)
        if img is None:
            self.get_logger().warn(
                f'camera frame stale (>{max_age_ns / 1e9:.1f}s old); treating as no image',
                throttle_duration_sec=2.0,
            )
        return img

    def _send_observation_tick(self) -> None:
        if self._observation_pub is None:
            return
        img = self._fresh_image()
        goal, generation = self._observations.snapshot_goal()
        if img is None or goal is None:
            return
        size = self.get_parameter('image_size').value
        quality = int(self.get_parameter('jpeg_quality').value)
        try:
            jpeg, w, h = resize_and_jpeg(img, target=(int(size[0]), int(size[1])), quality=quality)
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f'preprocess failed: {exc}')
            return
        frame_id = self._observations.record_sent(generation, img)
        if frame_id is None:
            return
        obs = ObservationMsg()
        obs.frame_id = frame_id
        obs.image.header.stamp = self.get_clock().now().to_msg()
        obs.image.format = 'jpeg'
        obs.image.data = jpeg
        obs.goal = goal
        self._observation_pub.publish(obs)

    # -------------------------------------------------- callback: embeddings

    def _on_embedding_received(self, emb: ActionEmbeddingMsg) -> None:
        if self._cache is None:
            return
        if emb.num_tokens * emb.embed_dim != len(emb.embedding):
            self.get_logger().warn(f'invalid embedding shape for frame {emb.frame_id}')
            return
        arr = np.asarray(emb.embedding, dtype=np.float32)
        obs_img = self._observations.take_reply_frame(int(emb.frame_id))
        cached = CachedEmbedding(
            frame_id=emb.frame_id,
            recv_time_ns=time.monotonic_ns(),
            embedding=arr,
            num_tokens=emb.num_tokens,
            embed_dim=emb.embed_dim,
            inference_ms=float(emb.inference_ms),
            model_version=emb.model_version,
            obs_image_rgb=obs_img,
        )
        self._cache.put(cached)
        if self._embedding_pub is not None:
            self._embedding_pub.publish(emb)

    # ---------------------------------------------------------- tick: action

    def _action_tick(self) -> None:
        """Publish a Path. Plan 1 stub: straight-ahead path of 1.0 m.

        Status-aware: WAITING_REMOTE / STALE → publish empty path so the
        follower stops driving. DEGRADED is treated as usable but logged.

        Note: the follower holds the last moving command for its
        ``hold_timeout_sec`` before zeroing (path_follower_node), so a *brief*
        empty-path gap coasts rather than hard-stopping; a sustained one still
        safe-stops.
        """
        if self._path_pub is None or self._adapter is None:
            return
        if self._local_mode:
            self._action_tick_local()
            return
        if self._cache is None:
            return
        status = self._cache.status()
        path = Path()
        path.header.frame_id = 'base_link'
        path.header.stamp = self.get_clock().now().to_msg()

        if status in (EmbeddingCache.STATUS_WAITING, EmbeddingCache.STATUS_STALE):
            # Empty path → follower emits zero Twist (safe-stop).
            self.get_logger().warn(
                f'embedding {status}; publishing empty path (safe-stop)',
                throttle_duration_sec=2.0,
            )
            self._path_pub.publish(path)
            return

        if status == EmbeddingCache.STATUS_DEGRADED:
            self.get_logger().warn('embedding age over max_age; running degraded')

        emb = self._cache.get_latest_raw()  # OK or DEGRADED
        cur = self._fresh_image()
        if cur is None:
            # No usable camera frame: safe-stop instead of acting blind.
            self._path_pub.publish(path)
            return
        # past = the frame the embedding was computed from (run_asyncvla.py's
        # past.png). The Edge_adapter compares it to `cur` to compensate for
        # what happened during the cloud round trip; falling back to `cur`
        # (no correlated frame) degrades to zero-latency behaviour.
        past = emb.obs_image_rgb if emb.obs_image_rgb is not None else cur
        try:
            t0 = time.monotonic()
            path = self._adapter.predict_path(
                embedding=np.asarray(emb.embedding, dtype=np.float32),
                embedding_shape=(1, int(emb.num_tokens), int(emb.embed_dim)),
                cur_image_rgb=cur,
                past_image_rgb=past,
                frame_id='base_link',
            )
            self._last_predict_ms = (time.monotonic() - t0) * 1000.0
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f'adapter.predict_path failed: {exc}; safe-stopping')
            path = Path()
            path.header.frame_id = 'base_link'
        self._log_action_diag(emb, path)
        path.header.stamp = self.get_clock().now().to_msg()
        self._path_pub.publish(path)

    def _log_action_diag(self, emb: CachedEmbedding, path: Path) -> None:
        """Throttled (1 s) one-line diagnostic of what the adapter produced.

        Meant to make field debugging possible from the launch log alone:
        embedding freshness + magnitude and the follower's target waypoint
        (index 4) with its bearing, so a biased model, a stale/degenerate
        embedding, or a follower-side problem can be told apart on the robot.
        """
        now_ns = time.monotonic_ns()
        if now_ns - self._last_diag_log_ns < 1_000_000_000:
            return
        self._last_diag_log_ns = now_ns
        age_ms = (now_ns - emb.recv_time_ns) / 1e6
        img_age_ms = self._camera_frames.age_ms(now_ns)
        arr = np.asarray(emb.embedding, dtype=np.float32)
        wp = 'none'
        if len(path.poses) > 4:
            p = path.poses[4].pose.position
            bearing = float(np.degrees(np.arctan2(p.y, p.x)))
            wp = f'({p.x:+.3f},{p.y:+.3f}) brg={bearing:+.1f}deg'
        self.get_logger().info(
            f'diag: emb frame={emb.frame_id} age={age_ms:.0f}ms '
            f'img_age={img_age_ms:.0f}ms pred={self._last_predict_ms:.0f}ms '
            f'std={arr.std():.4f} absmax={np.abs(arr).max():.3f} '
            f'past={"paired" if emb.obs_image_rgb is not None else "MISSING(cur)"} '
            f'wp4={wp}'
        )

    def _action_tick_local(self) -> None:
        """Action tick for on-edge adapters (no cloud / no embedding cache).

        Runs the local policy directly from the latest camera frame. The goal
        was handed to the adapter via ``set_goal`` in ``_on_goal``. Publishes an
        empty Path (safe-stop) until a frame is available; the adapter itself
        returns an empty Path until a goal arrives.
        """
        cur = self._fresh_image()
        path = Path()
        path.header.frame_id = 'base_link'
        if cur is None:
            path.header.stamp = self.get_clock().now().to_msg()
            self._path_pub.publish(path)
            return
        try:
            path = self._adapter.predict_path(
                embedding=None,
                embedding_shape=None,
                cur_image_rgb=cur,
                past_image_rgb=cur,
                frame_id='base_link',
            )
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f'local adapter.predict_path failed: {exc}; safe-stopping')
            path = Path()
            path.header.frame_id = 'base_link'
        path.header.stamp = self.get_clock().now().to_msg()
        self._path_pub.publish(path)

    # ----------------------------------------------------------- tick: status

    def _publish_status(self) -> None:
        if self._status_pub is None:
            return
        if self._local_mode:
            # No cloud/cache: readiness is "do we have an image and a goal".
            max_age_ns = int(float(self.get_parameter('image_max_age_sec').value) * 1e9)
            have_img = self._camera_frames.has_fresh(max_age_ns)
            have_goal = self._observations.has_goal
            status_str = 'OK' if (have_img and have_goal) else 'WAITING_REMOTE'
        elif self._cache is None:
            return
        else:
            status_str = self._cache.status()
        msg = DiagnosticArray()
        msg.header.stamp = self.get_clock().now().to_msg()
        ds = DiagnosticStatus()
        ds.name = 'vla_edge'
        ds.message = status_str
        ds.level = (
            DiagnosticStatus.OK
            if status_str == 'OK'
            else DiagnosticStatus.WARN
            if status_str in ('DEGRADED', 'WAITING_REMOTE')
            else DiagnosticStatus.ERROR
        )
        ds.values.append(KeyValue(
            key='frame_counter', value=str(self._observations.frame_counter)))
        msg.status.append(ds)
        self._status_pub.publish(msg)


def main() -> None:
    rclpy.init()
    node = VLAEdgeNode()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        node.destroy_node()
        rclpy.shutdown()
