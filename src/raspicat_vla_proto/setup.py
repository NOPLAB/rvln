from setuptools import setup

package_name = 'raspicat_vla_proto'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools', 'grpcio>=1.50', 'protobuf>=4.21'],
    zip_safe=True,
    maintainer='nop',
    maintainer_email='noplab90@gmail.com',
    description='Mobile EdgeActionService stubs and fp16 helpers.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={'console_scripts': []},
)
