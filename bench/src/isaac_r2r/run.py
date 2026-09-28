"""Isaac Sim implementation of the common R2R episode simulator contract."""
from __future__ import annotations

import io
import json
import math
from pathlib import Path

from bench.episode import EpisodeRequest, EpisodeSimulator, Observation


def load_scene_record(path: Path, scene: str) -> dict:
    """Require an explicit, metric alignment for each converted MP3D scan."""
    import numpy as np

    record = json.loads(path.read_text(encoding='utf-8'))['scenes'][scene]
    usd = Path(record['usd']).expanduser()
    if not usd.is_absolute():
        raise ValueError('scene USD path must be absolute')
    usd = usd.resolve()
    if not usd.is_file() or usd.suffix.lower() not in ('.usd', '.usda', '.usdc'):
        raise ValueError(f'missing USD scene: {usd}')
    matrix = record['isaac_from_habitat']
    if len(matrix) != 4 or any(len(row) != 4 for row in matrix):
        raise ValueError('isaac_from_habitat must be a 4x4 matrix')
    transform = np.asarray(matrix, dtype=float)
    if not np.isfinite(transform).all() or not np.allclose(transform[3], [0, 0, 0, 1]):
        raise ValueError('invalid scene transform')
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4):
        raise ValueError('scene transform must preserve metric distance')
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-4):
        raise ValueError('scene transform must be a proper rotation')
    return {'usd': usd, 'matrix': transform, 'inverse': np.linalg.inv(transform)}


def transform_point(matrix, position):
    import numpy as np
    return (matrix @ np.asarray([*position, 1.0], dtype=float))[:3]


def initial_yaw(episode: dict, rotation) -> float:
    """Convert Habitat's camera-forward (-Z) direction into Isaac Z-up yaw."""
    import numpy as np
    x, y, z, w = (float(v) for v in episode['start_rotation'])
    forward = np.array([
        -2 * (x * z + w * y),
        -2 * (y * z - w * x),
        -(1 - 2 * (x * x + y * y)),
    ])
    direction = rotation @ forward
    if math.hypot(*direction[:2]) < 1e-5:
        raise ValueError('initial forward vector is vertical after conversion')
    return math.atan2(direction[1], direction[0])


class IsaacR2RSimulator(EpisodeSimulator):
    """Isaac renderer with discrete kinematic motion and collision raycasts."""

    def __init__(self, scene: dict, *, headless: bool):
        from isaacsim import SimulationApp
        self.app = SimulationApp({'headless': headless})
        try:
            import omni.usd
            from isaacsim.core.api import World
            from isaacsim.core.utils.stage import add_reference_to_stage
            from isaacsim.sensors.camera import Camera
            from pxr import Usd, UsdGeom, UsdPhysics

            self.scene = scene
            self.world = World(stage_units_in_meters=1.0)
            add_reference_to_stage(str(scene['usd']), '/World/Environment')
            self.camera = Camera(prim_path='/World/RVLN_Camera',
                                 resolution=(224, 224), frequency=2)
            self.world.scene.add(self.camera)
            self.world.reset()
            self.camera.initialize()
            stage = omni.usd.get_context().get_stage()
            if UsdGeom.GetStageUpAxis(stage) != UsdGeom.Tokens.z:
                raise ValueError('USD scene must use Z-up coordinates')
            if not math.isclose(UsdGeom.GetStageMetersPerUnit(stage), 1.0):
                raise ValueError('USD scene must use metres')
            scene_prim = stage.GetPrimAtPath('/World/Environment')
            colliders = sum(prim.HasAPI(UsdPhysics.CollisionAPI)
                            for prim in Usd.PrimRange(scene_prim))
            if colliders == 0:
                raise ValueError('USD scene has no PhysX colliders')
            self.position = None
            self.yaw = None
            self.blocked_steps = 0
        except BaseException:
            self.app.close()
            raise

    def reset(self, request: EpisodeRequest) -> list[float]:
        episode = request.payload
        self.position = transform_point(self.scene['matrix'], episode['start_position'])
        self.yaw = initial_yaw(episode, self.scene['matrix'][:3, :3])
        self.blocked_steps = 0
        return list(episode['start_position'])

    def observe(self) -> Observation:
        import numpy as np
        from PIL import Image

        orientation = np.array([math.cos(self.yaw / 2), 0, 0, math.sin(self.yaw / 2)])
        self.camera.set_world_pose(position=self.position + [0.0, 0.0, 1.25],
                                   orientation=orientation, camera_axes='world')
        self.world.step(render=True)
        rgb = np.asarray(self.camera.get_rgba())[:, :, :3]
        if rgb.shape != (224, 224, 3):
            raise RuntimeError(f'unexpected RGB shape {rgb.shape}')
        buffer = io.BytesIO()
        Image.fromarray(rgb.astype('uint8')).save(buffer, format='JPEG', quality=90)
        return Observation(buffer.getvalue())

    def apply(self, action: str) -> list[float]:
        if action in ('left', 'right'):
            self.yaw += math.radians(15) * (1 if action == 'left' else -1)
        elif action == 'forward':
            import carb
            import numpy as np
            from omni.physx import get_physx_scene_query_interface

            direction = np.array([math.cos(self.yaw), math.sin(self.yaw), 0.0])
            side = np.array([-direction[1], direction[0], 0.0])
            blocked = False
            for height in (0.15, 0.45):
                for offset in (-0.17, 0.17):
                    origin = self.position + side * offset + [0.0, 0.0, height]
                    hit = get_physx_scene_query_interface().raycast_closest(
                        carb.Float3(*origin), carb.Float3(*direction), 0.25)
                    blocked |= bool(hit['hit'])
            if blocked:
                self.blocked_steps += 1
            else:
                self.position = self.position + direction * 0.25
        else:
            raise ValueError(f'unsupported action: {action!r}')
        return transform_point(self.scene['inverse'], self.position).tolist()

    def close(self) -> None:
        self.app.close()
