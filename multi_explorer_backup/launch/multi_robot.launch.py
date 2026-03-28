from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution
import os

def generate_launch_description():
    # 터틀봇 URDF 읽기
    urdf_path = '/opt/ros/humble/share/turtlebot3_description/urdf/turtlebot3_burger.urdf'
    with open(urdf_path, 'r') as f:
        robot_desc = f.read()

    robots = [
        {'name': 'tb3_0', 'x': '-1.5', 'y': '0.0'},
        {'name': 'tb3_1', 'x':  '1.5', 'y': '0.0'},
    ]

    ld = LaunchDescription()

    for robot in robots:
        name = robot['name']

        # robot_state_publisher
        ld.add_action(Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            namespace=name,
            parameters=[{
                'robot_description': robot_desc,
                'use_sim_time': True,
            }],
            remappings=[('/tf', 'tf'), ('/tf_static', 'tf_static')],
        ))

        # Gazebo에 로봇 spawn
        ld.add_action(Node(
            package='gazebo_ros',
            executable='spawn_entity.py',
            arguments=[
                '-entity', name,
                '-robot_namespace', name,
                '-x', robot['x'],
                '-y', robot['y'],
                '-z', '0.01',
                '-topic', f'/{name}/robot_description',
            ],
            output='screen',
        ))

    return ld
