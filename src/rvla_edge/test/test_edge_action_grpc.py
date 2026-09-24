"""edge_action_grpc_node のユニットテスト。

grpc サーバ起動には依存しない: proto decode (`decode_action_chunk`) と、
`handle_chunk` / `_tick` を合成時刻で直接叩いてウォッチドッグを検証する
(test_edge_action_ws.py / test_path_follower_hold.py と同じ流儀)。
"""
from types import SimpleNamespace

import numpy as np
import pytest
import rclpy

from rvla_proto import edge_action_pb2
from rvla_proto.conversions import float32_array_to_fp16_bytes

from rvla_edge.edge_action_grpc_node import (
    EdgeActionGrpcNode,
    decode_action_chunk,
)


@pytest.fixture(scope='module')
def ros_runtime():
    rclpy.init()
    yield
    rclpy.shutdown()


def _chunk(waypoints=None, *, scaled_to_m=False, goal_id='text:door',
           frame_seq=7, from_model=True):
    wp = (np.arange(32, dtype=np.float32).reshape(8, 4)
          if waypoints is None else np.asarray(waypoints, dtype=np.float32))
    return edge_action_pb2.ActionChunk(
        frame_id=frame_seq,
        capture_time_ns=0,
        num_tokens=wp.shape[0],
        embed_dim=wp.shape[1],
        values_fp16=float32_array_to_fp16_bytes(wp.ravel()),
        scaled_to_m=scaled_to_m,
        goal_id=goal_id,
        from_model=from_model,
    )


# --------------------------------------------------------- decode_action_chunk

def test_decode_scales_by_waypoint_spacing():
    path, frame_seq, goal_id = decode_action_chunk(
        _chunk(), waypoint_spacing=0.1, frame_id='base_link')
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
    path, _, _ = decode_action_chunk(
        _chunk(scaled_to_m=True), waypoint_spacing=0.1, frame_id='base_link')
    assert path.poses[1].pose.position.x == pytest.approx(4.0, abs=1e-2)


@pytest.mark.parametrize('make_bad', [
    lambda: edge_action_pb2.ActionChunk(num_tokens=0, embed_dim=4),
    lambda: edge_action_pb2.ActionChunk(num_tokens=8, embed_dim=2),
    # byte 長と num_tokens*embed_dim の不一致。
    lambda: edge_action_pb2.ActionChunk(
        num_tokens=8, embed_dim=4, values_fp16=b'\x00\x00'),
])
def test_decode_rejects_malformed(make_bad):
    with pytest.raises(ValueError):
        decode_action_chunk(make_bad(), waypoint_spacing=0.1, frame_id='base_link')


# ---------------------------------------------------- handle_chunk / watchdog

def _make_node() -> tuple:
    node = EdgeActionGrpcNode()
    published = []
    node._pub = SimpleNamespace(publish=published.append)
    return node, published


def test_chunk_is_published_once_then_watchdog_stops(ros_runtime):
    node, published = _make_node()
    try:
        ack = node.handle_chunk(_chunk(), now=0.0)
        assert ack.status == 'ok'
        assert ack.following is True
        assert ack.frame_id == 7

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
        node.handle_chunk(_chunk(), now=0.0)
        node._tick(0.05)
        node._tick(2.0)  # -> safe-stop
        assert len(published[-1].poses) == 0

        node.handle_chunk(_chunk(frame_seq=8), now=2.5)
        node._tick(2.55)
        assert len(published[-1].poses) == 8
    finally:
        node.destroy_node()


def test_malformed_chunk_acks_error_and_never_publishes(ros_runtime):
    node, published = _make_node()
    try:
        bad = edge_action_pb2.ActionChunk(frame_id=3, num_tokens=8, embed_dim=2)
        ack = node.handle_chunk(bad, now=0.0)
        assert ack.status.startswith('error:')
        assert ack.following is False  # まだ何も追従していない
        assert ack.frame_id == 3

        node._tick(0.2)
        node._tick(5.0)
        assert published == []  # chunk が来ていないのでウォッチドッグも発火しない
    finally:
        node.destroy_node()


def test_dummy_chunk_is_followed_with_marked_status(ros_runtime):
    node, published = _make_node()
    try:
        ack = node.handle_chunk(_chunk(from_model=False), now=0.0)
        assert ack.status == 'ok-dummy'
        assert ack.following is True
        node._tick(0.05)
        assert len(published) == 1
    finally:
        node.destroy_node()


def test_goal_change_is_accepted(ros_runtime):
    node, published = _make_node()
    try:
        node.handle_chunk(_chunk(goal_id='text:a'), now=0.0)
        node._tick(0.05)
        ack = node.handle_chunk(
            _chunk(goal_id='pose:1.0,0.0,0.0', frame_seq=9), now=0.1)
        assert ack.status == 'ok'
        node._tick(0.15)
        assert len(published) == 2
    finally:
        node.destroy_node()
