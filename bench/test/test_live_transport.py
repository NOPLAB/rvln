"""ROS boundary regressions without an Isaac or ROS installation."""

import importlib.util
import io
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch


def vector():
    return NS(x=0.0, y=0.0, z=0.0)


def pose():
    return NS(position=vector(), orientation=NS(w=1.0, x=0.0, y=0.0, z=0.0))


def twist():
    return NS(linear=vector(), angular=vector())


def header():
    return NS(stamp=None, frame_id="")


class Node:
    def __init__(self, name):
        self.name = name
        self.subscriptions = []
        self.services = []
        self.publishers = {}

    def create_subscription(self, kind, topic, callback, qos):
        self.subscriptions.append((kind, topic, callback, qos))

    def create_service(self, kind, topic, callback):
        self.services.append((kind, topic, callback))

    def create_publisher(self, kind, topic, qos):
        publisher = NS(kind=kind, qos=qos, messages=[])
        publisher.publish = publisher.messages.append
        self.publishers[topic] = publisher
        return publisher


def runtime_modules():
    values = {
        "geometry_msgs.msg": dict(
            PoseStamped=lambda: NS(header=header(), pose=pose()),
            TransformStamped=lambda: NS(
                header=header(), transform=NS(translation=vector(), rotation=None)
            ),
            Twist=twist,
        ),
        "nav_msgs.msg": dict(
            Odometry=lambda: NS(header=header(), pose=NS(pose=pose()), twist=NS(twist=twist())),
            Path=NS,
        ),
        "rclpy.node": dict(Node=Node),
        "rclpy.qos": dict(
            QoSProfile=NS,
            ReliabilityPolicy=NS(BEST_EFFORT=1),
            DurabilityPolicy=NS(TRANSIENT_LOCAL=2),
        ),
        "rclpy.time": dict(Time=lambda **kw: NS(to_msg=lambda: kw)),
        "rosgraph_msgs.msg": dict(Clock=NS),
        "sensor_msgs.msg": dict(Image=lambda: NS(header=header())),
        "std_srvs.srv": dict(SetBool=NS(Request=NS)),
        "tf2_msgs.msg": dict(TFMessage=NS),
        "rvln_msgs.msg": dict(
            ActionEmbedding=lambda: NS(header=header()), GoalSpec=NS(MODE_TEXT=1), Observation=NS
        ),
        "rclpy": {},
        "cv2": {},
    }
    modules = {}
    for name, attributes in values.items():
        module = types.ModuleType(name)
        module.__dict__.update(attributes)
        modules[name] = module
    return modules


def load_boundary(name):
    path = Path(__file__).resolve().parents[1] / "src" / "isaac_rvln" / (name + ".py")
    spec = importlib.util.spec_from_file_location("_test_" + name, path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {**runtime_modules(), spec.name: module}):
        spec.loader.exec_module(module)
    return module


class LiveTransportTest(unittest.TestCase):
    def setUp(self):
        self.module = load_boundary("live_transport")
        self.node = object.__new__(self.module.Episode)
        self.node.url = "http://fixture/infer"
        self.node.instruction = "Go forward."
        self.node.received = self.node.published = 0
        self.node.started_at = None
        self.node.last_pose = (1.0, 2.0, 0.3)
        self.node.last_odom_velocity = [0.2, 0.0]
        self.node.errors = []
        self.node.inferences = []
        self.node.model_version = None
        self.node.embedding_pub = NS(messages=[])
        self.node.embedding_pub.publish = self.node.embedding_pub.messages.append
        self.node.get_clock = lambda: NS(now=lambda: NS(to_msg=lambda: "stamp"))
        self.message = NS(frame_id=9, goal=NS(mode=1, text="Go forward."), image=NS(data=b"jpeg"))

    def test_inference_wire_payload_shape_and_model_identity(self):
        result = {
            "frame_id": 9,
            "embedding": [[1, 2, 3, 4]],
            "inference_ms": 2.0,
            "model_version": "model-a",
            "server_wall_ms": 3.0,
        }
        requests = []

        def response(request, timeout):
            requests.append((request, timeout))
            return io.BytesIO(json.dumps(result).encode())

        with patch.object(self.module.urllib.request, "urlopen", side_effect=response):
            self.node._observation(self.message)
            self.assertEqual(self.node.published, 1)
            self.assertEqual(self.node.embedding_pub.messages[0].embedding, [1, 2, 3, 4])
            self.assertEqual(self.node.inferences[0]["first_waypoint"], [1, 2, 3, 4])
            result["model_version"] = "model-b"
            self.node._observation(self.message)
            result["model_version"] = "model-a"
            result["embedding"] = [[1], [2, 3]]
            self.node._observation(self.message)
        self.assertEqual((self.node.received, self.node.published), (3, 1))
        self.assertEqual(len(self.node.errors), 2)
        request, timeout = requests[0]
        self.assertEqual(
            (request.full_url, request.method, timeout), ("http://fixture/infer", "POST", 8.0)
        )
        self.assertEqual(
            json.loads(request.data),
            {
                "frame_id": 9,
                "text": "Go forward.",
                "jpeg_base64": "anBlZw==",
                "pose_xyyaw": [1.0, 2.0, 0.3],
                "velocity_vw": [0.2, 0.0],
            },
        )

    def test_run_preserves_partial_report_and_shutdown(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            world, usd, manifest = (root / name for name in ("world", "world.usd", "pilot"))
            world.write_bytes(b"world")
            usd.write_bytes(b"usd")
            manifest.write_text(
                json.dumps(
                    {
                        "episodes": [
                            {
                                "id": "one",
                                "scene": "room",
                                "text": "Go forward.",
                                "goal_xy": [1.0, 2.0],
                            }
                        ],
                        "oracle": {
                            "world_sha256": {
                                "room": self.module.hashlib.sha256(b"world").hexdigest()
                            }
                        },
                    }
                )
            )
            args = NS(
                manifest=manifest,
                episode="one",
                world_source=world,
                usd=usd,
                url="http://fixture",
                text=None,
                video=root / "video.mp4",
                startup_timeout=30,
                duration=5,
                tolerance=0.3,
                model="model",
                deployment="remote",
            )
            node = self.node
            node.trace = []
            node.path_count = node.nonempty_paths = node.video_frames = 0
            node.path_samples = []
            node.last_odom_pose = None
            node.writer = NS(release=lambda: actions.append("release"))
            node.destroy_node = lambda: actions.append("destroy")
            node.follower_stop, node.motor = object(), object()
            node.set_bool = lambda client, value, **kw: actions.append((client, value))
            actions = []
            self.module.rclpy.init = lambda: actions.append("init")
            self.module.rclpy.shutdown = lambda: actions.append("shutdown")
            with (
                patch.object(self.module, "Episode", return_value=node),
                patch.object(self.module, "spin_until", return_value=False),
                patch.object(
                    self.module.urllib.request,
                    "urlopen",
                    return_value=io.BytesIO(b'{"reset": true}'),
                ) as http,
            ):
                row = self.module.run(args)
            self.assertEqual(row["stop_reason"], "startup_failure")
            self.assertFalse(row["confirmed_stop"])
            self.assertIsNone(row["wall_started_at"])
            self.assertEqual(len(row["errors"]), 1)
            self.assertEqual(row["trace"], [])
            self.assertIsNone(row["collisions"])
            self.assertEqual(
                actions,
                [
                    "init",
                    (node.follower_stop, True),
                    (node.motor, False),
                    "release",
                    "destroy",
                    "shutdown",
                ],
            )
            request = http.call_args.args[0]
            self.assertEqual(
                (request.full_url, request.method, request.data),
                ("http://fixture/reset", "POST", b"{}"),
            )
            self.assertEqual(http.call_args.kwargs["timeout"], 120)
