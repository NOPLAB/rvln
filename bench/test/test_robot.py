"""Ensure the pinned visual model is prepared with physical collision links."""
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

from isaac_rvln.robot import WHEEL_RADIUS_M, prepare_robot


class RobotTest(unittest.TestCase):
    def test_prepare_pinned_description(self):
        source = Path(__file__).resolve().parents[2] / 'src' / 'raspicat_description'
        if not source.is_dir():
            self.skipTest('optional pinned upstream description is unavailable')
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'raspicat.urdf'
            metadata = prepare_robot(source, output)
            root = ET.parse(output).getroot()
            links = {link.attrib['name']: link for link in root.findall('link')}
            self.assertEqual(metadata['wheel_radius_m'], WHEEL_RADIUS_M)
            for name in ('base_link', 'left_wheel_link', 'right_wheel_link',
                         'caster_link', 'caster_wheel_link', 'camera_link'):
                self.assertIsNotNone(links[name].find('collision'))
                self.assertIsNotNone(links[name].find('inertial'))
            for mesh in root.iter('mesh'):
                filename = mesh.attrib['filename']
                self.assertFalse(filename.startswith('file:'))
                self.assertTrue(Path(filename).is_file(), filename)
            camera = next(j for j in root.findall('joint')
                          if j.attrib['name'] == 'camera_joint')
            self.assertEqual(camera.find('parent').attrib['link'], 'camera_mount_link')


if __name__ == '__main__':
    unittest.main()
