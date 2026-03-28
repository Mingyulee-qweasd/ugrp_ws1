"""
변경사항:
- entry_points 를 새 디렉토리 구조(nodes/, simulation/)에 맞게 수정
- 기존 exploration_planner, map_merger, robot_state_machine, map_visualizer 는
  더 이상 직접 노드로 실행하지 않고 core/, perception/ 의 라이브러리로 사용
- 실제 ROS2 노드는 nodes/ 안의 파일들이 담당
"""
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
            # nodes/ (ROS2 진입점)
            'explorer_node      = multi_explorer.nodes.explorer_node:main',
            'coordinator_node   = multi_explorer.nodes.coordinator_node:main',
            'map_merger_node    = multi_explorer.nodes.map_merger_node:main',
            'task_manager_node  = multi_explorer.nodes.task_manager_node:main',
            'visualizer_node    = multi_explorer.nodes.visualizer_node:main',
            'robot_agent_node   = multi_explorer.nodes.robot_agent_node:main',
            # simulation
            'virtual_simulator  = multi_explorer.simulation.virtual_simulator:main',
        ],
    },
)
