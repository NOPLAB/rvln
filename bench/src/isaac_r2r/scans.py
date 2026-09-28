"""Import licensed MP3D GLB scans as static, collidable Isaac USD assets."""
from __future__ import annotations

import asyncio
import json
import math
from pathlib import Path

from bench.artifacts import sha256


def validate_scan(source: Path, output: Path) -> tuple[Path, Path]:
    """Reject missing or invalid GLB inputs before starting Kit."""
    source = source.expanduser().resolve()
    output = output.expanduser().resolve()
    if source == output or output.exists() or output.with_suffix('.json').exists():
        raise ValueError('output USD must be a new file distinct from the source')
    if source.suffix.lower() != '.glb' or not source.is_file():
        raise ValueError(f'missing MP3D GLB: {source}')
    if output.suffix.lower() not in ('.usd', '.usda', '.usdc'):
        raise ValueError('output must have a USD extension')
    with source.open('rb') as stream:
        header = stream.read(12)
    if (len(header) != 12 or header[:4] != b'glTF'
            or int.from_bytes(header[4:8], 'little') != 2
            or int.from_bytes(header[8:12], 'little') != source.stat().st_size):
        raise ValueError(f'invalid GLB v2 header or length: {source}')
    return source, output


def add_static_collisions(stage) -> dict:
    """Keep imported visuals and author triangle-mesh collision on every mesh."""
    from pxr import UsdGeom, UsdPhysics

    up_axis = UsdGeom.GetStageUpAxis(stage)
    if up_axis not in (UsdGeom.Tokens.y, UsdGeom.Tokens.z):
        raise ValueError(f'unsupported imported up axis: {up_axis}')
    metres = UsdGeom.GetStageMetersPerUnit(stage)
    if not math.isclose(metres, 1.0, rel_tol=0, abs_tol=1e-6):
        raise ValueError(f'converted scan must use metres, got {metres}')
    meshes = [prim for prim in stage.Traverse() if prim.IsA(UsdGeom.Mesh)]
    if not meshes:
        raise ValueError('converted scan has no mesh prims')
    if up_axis == UsdGeom.Tokens.y:
        roots = {mesh.GetPath().GetPrefixes()[0] for mesh in meshes}
        for path in roots:
            prim = stage.GetPrimAtPath(path)
            xform = UsdGeom.Xformable(prim)
            if not xform:
                raise ValueError(f'cannot rotate imported root: {path}')
            original = xform.GetOrderedXformOps()
            reset = xform.GetResetXformStack()
            axis_rotation = xform.AddRotateXOp(
                precision=UsdGeom.XformOp.PrecisionDouble,
                opSuffix='isaac_y_up_to_z_up')
            axis_rotation.Set(90.0)
            xform.SetXformOpOrder([axis_rotation, *original], resetXformStack=reset)
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    for prim in meshes:
        UsdPhysics.CollisionAPI.Apply(prim)
        collision = UsdPhysics.MeshCollisionAPI.Apply(prim)
        collision.CreateApproximationAttr().Set('none')
    return {'meshes': len(meshes), 'source_up_axis': str(up_axis)}


def convert_scan(source: Path, output: Path) -> dict:
    """Run Isaac's glTF importer, then add static triangle-mesh collision."""
    source, output = validate_scan(source, output)
    from isaacsim import SimulationApp

    output.parent.mkdir(parents=True, exist_ok=True)
    app = SimulationApp({'headless': True, 'multi_gpu': False,
                         'enable_crashreporter': False})
    try:
        from isaacsim.core.utils.extensions import enable_extension
        from pxr import Usd

        enable_extension('omni.kit.asset_converter')
        import omni.kit.asset_converter

        context = omni.kit.asset_converter.AssetConverterContext()
        context.use_meter_as_world_unit = True
        context.ignore_material = False

        async def import_glb():
            task = omni.kit.asset_converter.get_instance().create_converter_task(
                str(source), str(output), lambda *_: None, context)
            if not await task.wait_until_finished():
                raise RuntimeError(f'Isaac asset conversion failed: '
                                   f'{task.get_status()}: {task.get_error_message()}')

        future = asyncio.ensure_future(import_glb())
        while not future.done() and app.is_running():
            app.update()
        if not future.done():
            raise RuntimeError('Isaac stopped before GLB conversion finished')
        future.result()
        stage = Usd.Stage.Open(str(output))
        if stage is None:
            raise RuntimeError(f'Isaac did not write a USD stage: {output}')
        geometry = add_static_collisions(stage)
        stage.GetRootLayer().Save()
    except BaseException:
        output.unlink(missing_ok=True)
        raise
    finally:
        app.close()
    metadata = {
        'schema': 1,
        'source_glb': str(source),
        'source_sha256': sha256(source),
        'usd': str(output),
        **geometry,
    }
    output.with_suffix('.json').write_text(json.dumps(metadata, indent=2) + '\n',
                                           encoding='utf-8')
    return metadata
