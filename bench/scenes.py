"""Generate the three versioned Gazebo Classic pilot scenes."""
from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path


# name, x, y, z, size_x, size_y, size_z, RGB
SCENES = {
    'corridor': [
        ('left_wall', 1.5, 0.8, 0.35, 4.0, 0.1, 0.7, '0.7 0.7 0.7 1'),
        ('right_wall', 1.5, -0.8, 0.35, 4.0, 0.1, 0.7, '0.7 0.7 0.7 1'),
        ('red_marker', 2.8, 0.0, 0.35, 0.25, 0.25, 0.7, '0.9 0.05 0.05 1'),
    ],
    'junction': [
        ('end_wall', 2.0, 0.0, 0.35, 0.1, 1.6, 0.7, '0.7 0.7 0.7 1'),
        ('left_marker', 1.35, 1.35, 0.35, 0.25, 0.25, 0.7, '0.9 0.05 0.05 1'),
        ('right_marker', 1.35, -1.35, 0.35, 0.25, 0.25, 0.7, '0.05 0.1 0.9 1'),
    ],
    'weave': [
        ('left_block', 1.0, 0.25, 0.3, 0.4, 0.55, 0.6, '0.65 0.65 0.65 1'),
        ('right_block', 1.9, -0.25, 0.3, 0.4, 0.55, 0.6, '0.65 0.65 0.65 1'),
        ('red_marker', 3.2, 0.0, 0.35, 0.25, 0.25, 0.7, '0.9 0.05 0.05 1'),
    ],
}


def _box(world: ET.Element, item: tuple) -> None:
    name, x, y, z, sx, sy, sz, rgba = item
    model = ET.SubElement(world, 'model', name=name)
    ET.SubElement(model, 'static').text = 'true'
    ET.SubElement(model, 'pose').text = f'{x} {y} {z} 0 0 0'
    link = ET.SubElement(model, 'link', name='body')
    for kind in ('visual', 'collision'):
        element = ET.SubElement(link, kind, name='box')
        ET.SubElement(ET.SubElement(ET.SubElement(element, 'geometry'), 'box'),
                      'size').text = f'{sx} {sy} {sz}'
        if kind == 'visual':
            material = ET.SubElement(element, 'material')
            ET.SubElement(material, 'ambient').text = rgba
            ET.SubElement(material, 'diffuse').text = rgba


def generate(scene: str) -> str:
    if scene not in SCENES:
        raise ValueError(f'unknown scene {scene!r}')
    sdf = ET.Element('sdf', version='1.6')
    world = ET.SubElement(sdf, 'world', name='default')
    ET.SubElement(ET.SubElement(world, 'scene'), 'shadows').text = 'false'
    for model_name in ('ground_plane', 'sun'):
        ET.SubElement(ET.SubElement(world, 'include'), 'uri').text = f'model://{model_name}'
    plugin = ET.SubElement(world, 'plugin', name='gazebo_ros_state',
                           filename='libgazebo_ros_state.so')
    ET.SubElement(plugin, 'update_rate').text = '20'
    for item in SCENES[scene]:
        _box(world, item)
    ET.indent(sdf, space='  ')
    return '<?xml version="1.0"?>\n' + ET.tostring(sdf, encoding='unicode') + '\n'


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', type=Path, default=Path('bench/worlds'))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for scene in SCENES:
        target = args.out_dir / f'{scene}.world'
        target.write_text(generate(scene), encoding='utf-8')
        print(target)


if __name__ == '__main__':
    main()
