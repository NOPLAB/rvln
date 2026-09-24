from setuptools import find_packages, setup

package_name = 'rvln_core'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=[
        'setuptools',
        'numpy',
        'opencv-python',
        'Pillow',
    ],
    zip_safe=True,
    maintainer='nop',
    maintainer_email='noplab90@gmail.com',
    description='ROS-free OmniVLA-edge inference core shared by edge and remote.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={'console_scripts': []},
)
