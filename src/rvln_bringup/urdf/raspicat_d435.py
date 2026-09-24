"""Expand the upstream raspicat URDF with a D435 on the front handrail."""

import os
import sys

import xacro
from ament_index_python.packages import get_package_share_directory


def add_element(document, parent, tag, attributes=None):
    element = document.createElement(tag)
    for key, value in (attributes or {}).items():
        element.setAttribute(key, value)
    parent.appendChild(element)
    return element


def add_box(document, link, tag, size, material=None):
    geometry = add_element(document, add_element(document, link, tag), 'geometry')
    add_element(document, geometry, 'box', {'size': size})
    if material:
        add_element(document, geometry.parentNode, 'material', {'name': material})


def add_benchmark_contact_sensors(document, robot):
    """Expose every robot collision body to the closed-loop benchmark."""
    collisions = (
        ('base_footprint', 'base',
         'base_footprint_fixed_joint_lump__base_link_collision'),
        ('base_footprint', 'mount',
         'base_footprint_fixed_joint_lump__camera_mount_link_collision_1'),
        ('base_footprint', 'camera',
         'base_footprint_fixed_joint_lump__camera_link_collision_2'),
        ('caster_link', 'caster', 'caster_link_collision'),
        ('caster_wheel_link', 'caster_wheel', 'caster_wheel_link_collision'),
        ('left_wheel_link', 'left_wheel', 'left_wheel_link_collision'),
        ('right_wheel_link', 'right_wheel', 'right_wheel_link_collision'),
    )
    for link_name, sensor_name, collision_name in collisions:
        gazebo = add_element(document, robot, 'gazebo', {'reference': link_name})
        sensor = add_element(document, gazebo, 'sensor', {
            'name': f'bench_{sensor_name}_contact', 'type': 'contact',
        })
        add_element(document, sensor, 'always_on').appendChild(document.createTextNode('true'))
        add_element(document, sensor, 'update_rate').appendChild(document.createTextNode('50'))
        contact = add_element(document, sensor, 'contact')
        add_element(document, contact, 'collision').appendChild(
            document.createTextNode(collision_name))
        plugin = add_element(document, sensor, 'plugin', {
            'name': f'bench_{sensor_name}_bumper',
            'filename': 'libgazebo_ros_bumper.so',
        })
        ros = add_element(document, plugin, 'ros')
        add_element(document, ros, 'remapping').appendChild(
            document.createTextNode(
                f'bumper_states:=/bench/contacts/{sensor_name}'))


def generate_urdf():
    upstream = os.path.join(
        get_package_share_directory('raspicat_description'), 'urdf', 'raspicat.urdf.xacro',
    )
    document = xacro.process_file(upstream)
    robot = document.documentElement

    # The aluminum crossbar spans the robot front at z=0.1268 m in base_link.
    # A 4 mm plate rests on it; the 25 mm camera body rests on that plate.
    mount = add_element(document, robot, 'link', {'name': 'camera_mount_link'})
    add_box(document, mount, 'visual', '0.05 0.055 0.004', 'camera_mount_gray')
    add_box(document, mount, 'collision', '0.05 0.055 0.004')
    joint = add_element(document, robot, 'joint', {
        'name': 'camera_mount_joint', 'type': 'fixed',
    })
    add_element(document, joint, 'origin', {'xyz': '0.09 0 0.1288', 'rpy': '0 0 0'})
    add_element(document, joint, 'parent', {'link': 'base_link'})
    add_element(document, joint, 'child', {'link': 'camera_mount_link'})

    camera_joint = next(j for j in robot.getElementsByTagName('joint')
                        if j.getAttribute('name') == 'camera_joint')
    camera_joint.getElementsByTagName('parent')[0].setAttribute('link', 'camera_mount_link')
    camera_joint.getElementsByTagName('origin')[0].setAttribute('xyz', '0.01 0 0.0145')

    camera = next(link for link in robot.getElementsByTagName('link')
                  if link.getAttribute('name') == 'camera_link')
    # camera_link's local X runs along the D435 width; local Z points forward.
    add_box(document, camera, 'visual', '0.09 0.025 0.025', 'camera_body_black')
    add_box(document, camera, 'collision', '0.09 0.025 0.025')
    for x in ('0.032', '-0.012'):
        visual = add_element(document, camera, 'visual')
        add_element(document, visual, 'origin', {'xyz': f'{x} 0 0.013', 'rpy': '0 0 0'})
        geometry = add_element(document, visual, 'geometry')
        add_element(document, geometry, 'cylinder', {'radius': '0.006', 'length': '0.001'})
        add_element(document, visual, 'material', {'name': 'camera_lens_blue'})

    # Keep software rendering usable on the local simulator while preserving
    # the upstream D435 color and depth topic contracts.
    for gazebo in robot.getElementsByTagName('gazebo'):
        for sensor in gazebo.getElementsByTagName('sensor'):
            if sensor.getAttribute('type') not in ('camera', 'depth'):
                continue
            image = sensor.getElementsByTagName('image')[0]
            image.getElementsByTagName('width')[0].firstChild.data = '640'
            image.getElementsByTagName('height')[0].firstChild.data = '480'
            sensor.getElementsByTagName('update_rate')[0].firstChild.data = '10'

    for name, rgba in (
        ('camera_mount_gray', '0.65 0.68 0.70 1'),
        ('camera_body_black', '0.08 0.09 0.10 1'),
        ('camera_lens_blue', '0.10 0.24 0.34 1'),
    ):
        material = add_element(document, robot, 'material', {'name': name})
        add_element(document, material, 'color', {'rgba': rgba})

    if os.environ.get('RVLN_BENCH_CONTACTS') == '1':
        add_benchmark_contact_sensors(document, robot)

    return document.toxml()


if __name__ == '__main__':
    sys.stdout.write(generate_urdf())
