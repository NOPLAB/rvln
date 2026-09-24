"""edge_action_ws_node のユニットテスト。

websockets には依存しない: JSON decode (`decode_chunk_msg`) と、
`handle_message` / `_tick` を合成時刻で直接叩いてウォッチドッグを検証する
(test_path_follower_hold.py と同じ流儀)。
"""
import base64
import json
from types import SimpleNamespace

import numpy as np
import pytest
import rclpy

from rvln_proto.conversions import float32_array_to_fp16_bytes

from rvln_edge.edge_action_ws_node import EdgeActionWsNode, decode_chunk_msg


@pytest.fixture(scope='module')
def ros_runtime():
    rclpy.init()
    yield
    rclpy.shutdown()


def _chunk_msg(waypoints=None, *, scaled_to_m=False, goal_id='text:door', frame_seq=7):
    wp = (np.arange(32, dtype=np.float32).reshape(8, 4)
          if waypoints is None else np.asarray(waypoints, dtype=np.float32))
    return {
        'type': 'action_chunk',
        'frame_id': frame_seq,
        'capture_time_ms': 0,
        'num_tokens': wp.shape[0],
        'embed_dim': wp.shape[1],
        'values_fp16_b64': base64.b64encode(
            float32_array_to_fp16_bytes(wp.ravel())).decode('ascii'),
        'scaled_to_m': scaled_to_m,
        'goal_id': goal_id,
    }


# ------------------------------------------------------------ decode_chunk_msg

def test_decode_scales_by_waypoint_spacing():
    path, frame_seq, goal_id = decode_chunk_msg(
        _chunk_msg(), waypoint_spacing=0.1, frame_id='base_link')
    assert frame_seq == 7
    assert goal_id == 'text:door'
    assert path.header.frame_id == 'base_link'
    assert len(path.poses) == 8
    # 2 行目 = (4, 5, 6, 7): x,y は spacing 単位 -> ×0.1、cos/sin はそのまま。
    p1 = path.poses[1].pose
    assert p1.position.x == pytest.approx(0.4, abs=1e-3)
    assert p1.position.y == pytest.approx(0.5, abs=1e-3)
    assert p1.orientation.w == pytest.approx(6.0, abs=1e-3)
    assert p1.orientation.z == pytest.approx(7.0, abs=1e-3)


def test_decode_scaled_to_m_uses_unity_spacing():
    path, _, _ = decode_chunk_msg(
        _chunk_msg(scaled_to_m=True), waypoint_spacing=0.1, frame_id='base_link')
    assert path.poses[1].pose.position.x == pytest.approx(4.0, abs=1e-2)


@pytest.mark.parametrize('mutate', [
    lambda m: m.update(type='observation'),
    lambda m: m.update(embed_dim=2),
    lambda m: m.update(num_tokens=0),
    lambda m: m.update(values_fp16_b64='!!not-base64!!'),
    lambda m: m.update(values_fp16_b64=base64.b64encode(b'\x00\x00').decode()),
    lambda m: m.pop('values_fp16_b64'),
])
def test_decode_rejects_malformed(mutate):
    msg = _chunk_msg()
    mutate(msg)
    with pytest.raises(ValueError):
        decode_chunk_msg(msg, waypoint_spacing=0.1, frame_id='base_link')


# ------------------------------------------------- handle_message / watchdog

def _make_node() -> tuple:
    node = EdgeActionWsNode()
    published = []
    node._pub = SimpleNamespace(publish=published.append)
    return node, published


def test_chunk_is_published_once_then_watchdog_stops(ros_runtime):
    node, published = _make_node()
    try:
        ack = node.handle_message(json.dumps(_chunk_msg()), now=0.0)
        assert ack['status'] == 'ok'
        assert ack['following'] is True
        assert ack['frame_id'] == 7

        # 最新 chunk は次の tick で 1 回だけ publish される。
        node._tick(0.05)
        assert len(published) == 1
        assert len(published[0].poses) == 8
        node._tick(0.10)
        assert len(published) == 1  # pending は消費済み・まだ新鮮

        # chunk_max_age_sec (既定 1.0s) 超過 -> 空 Path を 1 回だけ発行。
        node._tick(2.0)
        assert len(published) == 2
        assert len(published[1].poses) == 0
        node._tick(3.0)
        assert len(published) == 2  # 停止済み、連打しない
    finally:
        node.destroy_node()


def test_new_chunk_after_stale_resumes(ros_runtime):
    node, published = _make_node()
    try:
        node.handle_message(json.dumps(_chunk_msg()), now=0.0)
        node._tick(0.05)
        node._tick(2.0)  # -> safe-stop
        assert len(published[-1].poses) == 0

        node.handle_message(json.dumps(_chunk_msg(frame_seq=8)), now=2.5)
        node._tick(2.55)
        assert len(published[-1].poses) == 8
    finally:
        node.destroy_node()


def test_stale_pending_chunk_is_never_published(ros_runtime):
    node, published = _make_node()
    try:
        node.handle_message(json.dumps(_chunk_msg()), now=0.0)
        node._tick(2.0)
        assert len(published) == 1
        assert published[0].poses == []
    finally:
        node.destroy_node()


def test_malformed_message_acks_error_and_never_publishes(ros_runtime):
    node, published = _make_node()
    try:
        ack = node.handle_message('not json at all', now=0.0)
        assert ack['status'].startswith('error:')
        assert ack['following'] is False  # まだ何も追従していない

        ack = node.handle_message(json.dumps({'type': 'action_chunk'}), now=0.1)
        assert ack['status'].startswith('error:')

        node._tick(0.2)
        node._tick(5.0)
        assert published == []  # chunk が来ていないのでウォッチドッグも発火しない
    finally:
        node.destroy_node()


def test_goal_change_is_accepted(ros_runtime):
    node, published = _make_node()
    try:
        node.handle_message(json.dumps(_chunk_msg(goal_id='text:a')), now=0.0)
        node._tick(0.05)
        ack = node.handle_message(
            json.dumps(_chunk_msg(goal_id='pose:1.0,0.0,0.0', frame_seq=9)), now=0.1)
        assert ack['status'] == 'ok'
        node._tick(0.15)
        assert len(published) == 2
    finally:
        node.destroy_node()
