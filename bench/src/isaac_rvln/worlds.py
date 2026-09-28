"""Convert the three owned Gazebo pilot worlds to collidable Isaac USD scenes."""
from __future__ import annotations

import hashlib
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path


def parse_boxes(path: Path) -> list[dict]:
    """Extract static collision boxes; reject unsupported geometry explicitly."""
    root = ET.parse(path).getroot()
    world = root.find('world')
    if world is None:
        raise ValueError('SDF has no world')
    boxes = []
    for model in world.findall('model'):
        name = model.attrib['name']
        if model.findtext('static') != 'true':
            raise ValueError(f'{name}: dynamic SDF models are unsupported')
        pose = [float(value) for value in model.findtext('pose', '').split()]
        if len(pose) != 6 or not all(math.isfinite(value) for value in pose):
            raise ValueError(f'{name}: invalid pose')
        links = model.findall('link')
        if len(links) != 1:
            raise ValueError(f'{name}: expected one link')
        collision = links[0].find('collision/geometry/box/size')
        visual = links[0].find('visual/geometry/box/size')
        if collision is None or visual is None:
            raise ValueError(f'{name}: only visual and collision boxes are supported')
        size = [float(value) for value in collision.text.split()]
        visual_size = [float(value) for value in visual.text.split()]
        if len(size) != 3 or size != visual_size or any(value <= 0 for value in size):
            raise ValueError(f'{name}: invalid or mismatched box sizes')
        color_text = links[0].findtext('visual/material/diffuse', '0.7 0.7 0.7 1')
        color = [float(value) for value in color_text.split()]
        if len(color) != 4 or any(value < 0 or value > 1 for value in color):
            raise ValueError(f'{name}: invalid color')
        boxes.append({'name': name, 'pose': pose, 'size': size, 'rgba': color})
    if not boxes:
        raise ValueError('SDF has no static collision boxes')
    if len({item['name'] for item in boxes}) != len(boxes):
        raise ValueError('duplicate model names')
    return boxes


def write_usd(boxes: list[dict], output: Path) -> None:
    """Write a Z-up, metre-scale USD stage with PhysX-ready static colliders."""
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics

    stage = Usd.Stage.CreateNew(str(output))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    world = UsdGeom.Xform.Define(stage, '/World')
    stage.SetDefaultPrim(world.GetPrim())
    UsdPhysics.Scene.Define(stage, Sdf.Path('/World/Physics'))

    def cube(name, position, size, rgb):
        xform = UsdGeom.Xform.Define(stage, f'/World/{name}')
        xform.AddTranslateOp().Set(Gf.Vec3d(*position[:3]))
        xform.AddRotateXYZOp().Set(Gf.Vec3f(*(math.degrees(value)
                                              for value in position[3:])))
        shape = UsdGeom.Cube.Define(stage, f'/World/{name}/Body')
        shape.CreateSizeAttr(1.0)
        shape.AddScaleOp().Set(Gf.Vec3f(*size))
        shape.CreateDisplayColorAttr([Gf.Vec3f(*rgb[:3])])
        UsdPhysics.CollisionAPI.Apply(shape.GetPrim())

    cube('Ground', [0, 0, -0.025, 0, 0, 0], [100, 100, 0.05], [0.35] * 3)
    light = UsdLux.DistantLight.Define(stage, '/World/Sun')
    light.CreateIntensityAttr(500)
    for item in boxes:
        cube(item['name'], item['pose'], item['size'], item['rgba'])
    stage.GetRootLayer().Save()


def convert_world(source: Path, output: Path) -> dict:
    boxes = parse_boxes(source)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_usd(boxes, output)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    metadata = {'schema': 1, 'source_world': str(source.resolve()),
                'source_sha256': digest, 'box_models': [item['name'] for item in boxes],
                'usd': str(output.resolve())}
    output.with_suffix('.json').write_text(json.dumps(metadata, indent=2) + '\n',
                                             encoding='utf-8')
    return metadata
