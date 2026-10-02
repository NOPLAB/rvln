"""Raspicat-compatible contact logger backed by the mobile runtime."""

import time

from usim.ports.isaac.contacts import ObstacleContactLog as MobileContactLog


class ObstacleContactLog(MobileContactLog):
    def __init__(self, obstacles: set[str], clock=time.monotonic):
        super().__init__(obstacles, clock, robot_prim_path="/World/Raspicat")
