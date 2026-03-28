from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'multi_explorer'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
            glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'),
            glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='user@example.com',
    description='Multi-robot exploration package',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'virtual_simulator   = multi_explorer.virtual_simulator:main',
            'exploration_planner = multi_explorer.exploration_planner:main',
            'map_merger          = multi_explorer.map_merger:main',
            'robot_state_machine = multi_explorer.robot_state_machine:main',
            'map_visualizer      = multi_explorer.map_visualizer:main',
        ],
    },
)
