"""Isaac Sim implementation of the common R2R episode simulator contract."""
from __future__ import annotations

import io
import json
import math
import time
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
    dome_intensity = float(record.get('dome_light_intensity', 0.0))
    if not math.isfinite(dome_intensity) or dome_intensity < 0:
        raise ValueError('dome_light_intensity must be finite and nonnegative')
    return {'usd': usd, 'matrix': transform, 'inverse': np.linalg.inv(transform),
            'dome_light_intensity': dome_intensity}


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

    render_size = 512
    observation_size = 256
    agent_radius_m = 0.1

    def __init__(self, scene: dict, *, headless: bool):
        from isaacsim import SimulationApp
        self.app = SimulationApp({'headless': headless, 'multi_gpu': False,
                                  'create_new_stage': False,
                                  'enable_crashreporter': False,
                                  'width': 320, 'height': 240,
                                  'samples_per_pixel_per_frame': 1})
        try:
            import omni.usd
            from isaacsim.core.api import World
            from isaacsim.core.utils.stage import add_reference_to_stage
            from isaacsim.sensors.experimental.rtx import CameraSensor
            import isaacsim.core.experimental.utils.app as app_utils
            import omni.replicator.core as rep
            import carb.settings
            from pxr import Usd, UsdGeom, UsdLux, UsdPhysics

            self.scene = scene
            self.world = World(stage_units_in_meters=1.0)
            add_reference_to_stage(str(scene['usd']), '/World/Environment')
            if scene.get('dome_light_intensity', 0.0) > 0:
                dome = UsdLux.DomeLight.Define(self.world.stage, '/World/RVLN_DomeLight')
                dome.CreateIntensityAttr(scene['dome_light_intensity'])
            self.world.reset()
            camera = UsdGeom.Camera.Define(self.world.stage, '/World/RVLN_Camera')
            camera.GetHorizontalApertureAttr().Set(20.0)
            camera.GetVerticalApertureAttr().Set(20.0)
            camera.GetFocalLengthAttr().Set(10.0)
            carb.settings.get_settings().set('/rtx/post/dlss/execMode', 2)
            size = self.render_size
            self.camera = CameraSensor('/World/RVLN_Camera', resolution=(size, size),
                                       annotators=['rgb', 'distance_to_image_plane'])
            self.camera_prim = self.camera.authoring_object
            app_utils.play(commit=True)
            rep.orchestrator.step(rt_subframes=2, pause_timeline=False)
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
        import omni.replicator.core as rep
        import omni.usd
        from PIL import Image

        c, s = math.cos(self.yaw / 2), math.sin(self.yaw / 2)
        orientation = np.array([c + s, c + s, s - c, s - c]) * 0.5
        self.camera_prim.set_world_poses(
            positions=np.asarray([self.position + [0.0, 0.0, 1.25]]),
            orientations=np.asarray([orientation]))
        # A navigation action can teleport the camera between captures. Discard
        # temporal RTX buffers before reading RGB and depth for the new pose.
        omni.usd.get_context().reset_renderer_accumulation()
        rep.orchestrator.step(rt_subframes=4, pause_timeline=False,
                              wait_for_render=True)
        deadline = time.monotonic() + 5.0
        attempts = 0
        size = self.render_size
        while attempts < 240 and time.monotonic() < deadline:
            self.world.step(render=True)
            attempts += 1
            data, _ = self.camera.get_data('rgb')
            rgba = (data.numpy() if data is not None and hasattr(data, 'numpy')
                    else np.asarray(data))
            rgb_valid = rgba.shape in ((size, size, 3), (size, size, 4))
            if rgb_valid:
                rgb = np.array(rgba[:, :, :3], dtype='uint8', copy=True)
            depth_data, _ = self.camera.get_data('distance_to_image_plane')
            depth = (depth_data.numpy() if depth_data is not None and
                     hasattr(depth_data, 'numpy') else np.asarray(depth_data))
            if depth.shape == (size, size, 1):
                depth = depth[:, :, 0]
            if rgb_valid and depth.shape == (size, size) and \
                    np.any(np.isfinite(depth) & (depth > 0)):
                break
        else:
            raise RuntimeError('Isaac camera produced no RGB/depth frame after '
                               f'{attempts} steps: rgb={rgba.shape}, depth={depth.shape}')
        buffer = io.BytesIO()
        Image.fromarray(rgb).resize((self.observation_size,) * 2,
                                    Image.Resampling.BILINEAR).save(
                                        buffer, format='JPEG', quality=90)
        depth = np.asarray(Image.fromarray(np.asarray(depth, dtype='float32')).resize(
            (self.observation_size,) * 2, Image.Resampling.NEAREST))
        return Observation(buffer.getvalue(),
                           np.asarray(depth, dtype='<f4').tobytes(),
                           self.observation_size, self.observation_size)

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
            for height in (0.15, 0.75, 1.4):
                for offset in (-self.agent_radius_m, 0.0, self.agent_radius_m):
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
