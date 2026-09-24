"""Tests for the edge node's in-process camera capture (camera_device param).

No real V4L2 device in CI: the open-failure path and the capture-loop
bookkeeping are tested; the happy-path open is exercised on the robot.
"""
import numpy as np
import pytest
import rclpy
from rclpy.lifecycle import TransitionCallbackReturn

from raspicat_vla_edge.edge_node import VLAEdgeNode


@pytest.fixture(scope='module')
def ros_runtime():
    rclpy.init()
    yield
    rclpy.shutdown()


@pytest.fixture()
def node(ros_runtime):
    n = VLAEdgeNode()
    yield n
    n.destroy_node()


def test_configure_fails_when_camera_device_cannot_be_opened(node):
    node.set_parameters([
        rclpy.parameter.Parameter('camera_device', value='/dev/nonexistent-video99'),
    ])
    assert node.on_configure(None) == TransitionCallbackReturn.FAILURE
    assert node._camera._cap is None
    assert node._image_sub is None  # no fallback subscription in camera mode


def test_configure_without_camera_device_subscribes_to_image_topic(node):
    assert node.on_configure(None) == TransitionCallbackReturn.SUCCESS
    assert node._camera._cap is None
    assert node._image_sub is not None
    node.on_cleanup(None)


class _StubCap:
    """VideoCapture stand-in: yields one BGR frame, then stops the loop."""

    def __init__(self, camera):
        self._camera = camera
        self._reads = 0

    def read(self):
        self._reads += 1
        if self._reads > 1:
            self._camera._stop.set()
            return False, None
        frame_bgr = np.zeros((4, 4, 3), dtype=np.uint8)
        frame_bgr[..., 0] = 255  # blue plane in BGR
        return True, frame_bgr

    def release(self):
        pass


def test_camera_loop_stores_rgb_frame_and_stamp(node):
    node._camera._cap = _StubCap(node._camera)
    node._camera._stop.clear()

    node._camera.capture_loop()

    image = node._camera_frames.fresh(int(1e9))
    assert image is not None
    # BGR blue must land in the RGB blue channel (i.e. cvtColor happened).
    assert image[0, 0, 2] == 255
    assert image[0, 0, 0] == 0


def test_compressed_image_callback_decodes_jpeg_to_rgb(node):
    import cv2
    from sensor_msgs.msg import CompressedImage

    bgr = np.zeros((32, 32, 3), dtype=np.uint8)
    bgr[..., 0] = 255  # blue plane in BGR
    ok, jpeg = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
    assert ok
    msg = CompressedImage()
    msg.format = 'jpeg'
    msg.data = jpeg.tobytes()

    node._on_compressed_image(msg)

    image = node._camera_frames.fresh(int(1e9))
    assert image is not None
    # Blue in BGR must land in the RGB blue channel (allow JPEG loss).
    assert image[16, 16, 2] > 200
    assert image[16, 16, 0] < 50


def test_compressed_topic_selects_compressed_subscription(node):
    node.set_parameters([
        rclpy.parameter.Parameter('image_topic', value='/camera/image_raw/compressed'),
    ])
    assert node.on_configure(None) == TransitionCallbackReturn.SUCCESS
    from sensor_msgs.msg import CompressedImage
    assert node._image_sub is not None
    assert node._image_sub.msg_type is CompressedImage
    node.on_cleanup(None)
