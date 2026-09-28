"""Prepare the pinned Raspicat description for Isaac's URDF importer."""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path


WHEEL_RADIUS_M = 0.0762
WHEEL_SEPARATION_M = 0.27918


def _child(parent, tag: str, **attributes):
    return ET.SubElement(parent, tag, {key: str(value) for key, value in attributes.items()})


def _box(link, kind: str, size: str) -> None:
    _child(_child(_child(link, kind), 'geometry'), 'box', size=size)


def _inertial(link, mass: float, diagonal: tuple[float, float, float]) -> None:
    inertial = _child(link, 'inertial')
    _child(inertial, 'origin', xyz='0 0 0', rpy='0 0 0')
    _child(inertial, 'mass', value=mass)
    _child(inertial, 'inertia', ixx=diagonal[0], iyy=diagonal[1],
           izz=diagonal[2], ixy=0, ixz=0, iyz=0)


def prepare_robot(description_root: Path, output: Path) -> dict:
    """Recreate the Gazebo D435 mount and physical links from pinned sources.

    The upstream expanded URDF is visual-only; collision and mass values come
    from its xacro files. This function never edits the upstream checkout.
    """
    source = description_root / 'urdf' / 'raspicat.urdf'
    if not source.is_file():
        raise ValueError(f'missing upstream Raspicat URDF: {source}')
    root = ET.parse(source).getroot()
    links = {link.attrib['name']: link for link in root.findall('link')}
    joints = {joint.attrib['name']: joint for joint in root.findall('joint')}
    required_links = {'base_link', 'left_wheel_link', 'right_wheel_link',
                      'caster_link', 'caster_wheel_link', 'camera_link'}
    if not required_links.issubset(links) or 'camera_joint' not in joints:
        raise ValueError('upstream Raspicat URDF has unexpected links or joints')

    uri = 'package://raspicat_description/'
    for mesh in root.iter('mesh'):
        filename = mesh.attrib.get('filename', '')
        if not filename.startswith(uri):
            raise ValueError(f'unexpected mesh URI: {filename}')
        mesh_path = (description_root / filename[len(uri):]).resolve()
        if not mesh_path.is_file():
            raise ValueError(f'missing Raspicat mesh: {mesh_path}')
        mesh.set('filename', mesh_path.as_posix())

    base = links['base_link']
    base_collision = _child(base, 'collision')
    _child(_child(base_collision, 'geometry'), 'box', size='0.48 0.28 0.15')
    _inertial(base, 5.778318, (0.0485945486, 0.07472816111, 0.08479504851))
    for side in ('left', 'right'):
        wheel = links[f'{side}_wheel_link']
        collision = _child(wheel, 'collision')
        _child(_child(collision, 'geometry'), 'cylinder',
               radius=WHEEL_RADIUS_M, length=0.0254)
        _inertial(wheel, 0.255393, (0.000382815198, 0.00038254522,
                                   0.00074855867))
    caster = links['caster_link']
    _box(caster, 'collision', '0.09 0.08 0.07')
    _inertial(caster, 0.454, (0.000688529016, 0.000802824209, 0.000410878112))
    caster_wheel = links['caster_wheel_link']
    collision = _child(caster_wheel, 'collision')
    _child(_child(collision, 'geometry'), 'cylinder', radius=0.05, length=0.025)
    _inertial(caster_wheel, 0.245, (0.000382815198, 0.00038254522,
                                   0.00074855867))

    mount = _child(root, 'link', name='camera_mount_link')
    _box(mount, 'visual', '0.05 0.055 0.004')
    _box(mount, 'collision', '0.05 0.055 0.004')
    _inertial(mount, 0.04, (0.0001, 0.0001, 0.0001))
    joint = _child(root, 'joint', name='camera_mount_joint', type='fixed')
    _child(joint, 'origin', xyz='0.09 0 0.1288', rpy='0 0 0')
    _child(joint, 'parent', link='base_link')
    _child(joint, 'child', link='camera_mount_link')
    camera_joint = joints['camera_joint']
    camera_joint.find('parent').set('link', 'camera_mount_link')
    camera_joint.find('origin').set('xyz', '0.01 0 0.0145')
    camera = links['camera_link']
    _box(camera, 'visual', '0.09 0.025 0.025')
    _box(camera, 'collision', '0.09 0.025 0.025')
    _inertial(camera, 0.08, (0.0001, 0.0001, 0.0001))

    output.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(root, space='  ')
    output.write_text(ET.tostring(root, encoding='unicode') + '\n', encoding='utf-8')
    return {'schema': 1, 'source': str(source.resolve()),
            'robot_urdf': str(output.resolve()),
            'wheel_radius_m': WHEEL_RADIUS_M,
            'wheel_separation_m': WHEEL_SEPARATION_M}
