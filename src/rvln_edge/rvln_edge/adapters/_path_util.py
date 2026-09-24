"""Shared waypoint-array -> nav_msgs/Path conversion for edge adapters."""
from __future__ import annotations

import numpy as np
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path


def trajectory_to_path(
    waypoints: np.ndarray,
    *,
    spacing: float = 1.0,
    frame_id: str = 'base_link',
) -> Path:
    """(T, ACTION_DIM>=4) packed as (x, y, cos, sin) -> nav_msgs/Path.

    x is forward, y is left (robot frame); x/y are multiplied by ``spacing``
    (metres per waypoint unit — pass 1.0 when they are already metres).
    Orientation maps (cos, sin) to the (w, z) of a yaw-only quaternion.
    Pure numpy — no torch — so it is unit testable without model weights.
    """
    wp = np.asarray(waypoints, dtype=np.float32)
    if wp.ndim != 2 or wp.shape[-1] < 4:
        raise ValueError(
            f'expected (T, ACTION_DIM>=4) packed as (x, y, cos, sin); got shape={wp.shape}'
        )
    path = Path()
    path.header.frame_id = frame_id
    for x, y, c, s in wp[:, :4]:
        ps = PoseStamped()
        ps.header.frame_id = frame_id
        ps.pose.position.x = float(x) * spacing
        ps.pose.position.y = float(y) * spacing
        ps.pose.orientation.z = float(s)
        ps.pose.orientation.w = float(c)
        path.poses.append(ps)
    return path
