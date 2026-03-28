"""
launch/gazebo_decentralized.launch.py

Gazebo 시뮬레이션 + Decentralized 탐색:
- Gazebo world (turtlebot3_house)
- TurtleBot3 × 3 spawn (tb3_0, tb3_1, tb3_2)
- robot_state_publisher × 3
- slam_toolbox × 3
- nav2 × 3
- robot_agent_node × 3
- coordinator_node × 1
- visualizer_node × 1
"""
import os
import xml.etree.ElementTree as ET

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    IncludeLaunchDescription, TimerAction, GroupAction,
    RegisterEventHandler, DeclareLaunchArgument,
)
from launch.event_handlers import OnShutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node, PushRosNamespace

PKG = 'multi_explorer'
ROBOT_NS = ['tb3_0', 'tb3_1', 'tb3_2']
NUM_ROBOTS = 3

# 로봇 초기 위치 (house world 내)
SPAWN_POSES = [
    (-1.0, 0.0),     # tb3_0
    (1.0, 0.0),      # tb3_1
    (0.0, 1.5),      # tb3_2
]

DWB_CRITICS = ['RotateToGoal', 'Oscillation', 'BaseObstacle',
               'GoalAlign', 'PathAlign', 'PathDist', 'GoalDist']


def dwb_params(ns):
    return {
        'FollowPath.plugin': 'dwb_core::DWBLocalPlanner',
        'FollowPath.critics': DWB_CRITICS,
        'FollowPath.min_vel_x': 0.0,
        'FollowPath.max_vel_x': 0.26,
        'FollowPath.max_vel_y': 0.0,
        'FollowPath.max_vel_theta': 1.0,
        'FollowPath.min_speed_xy': 0.0,
        'FollowPath.max_speed_xy': 0.26,
        'FollowPath.min_speed_theta': 0.0,
        'FollowPath.acc_lim_x': 2.5,
        'FollowPath.acc_lim_y': 0.0,
        'FollowPath.acc_lim_theta': 3.2,
        'FollowPath.decel_lim_x': -2.5,
        'FollowPath.decel_lim_y': 0.0,
        'FollowPath.decel_lim_theta': -3.2,
        'FollowPath.vx_samples': 20,
        'FollowPath.vy_samples': 5,
        'FollowPath.vtheta_samples': 20,
        'FollowPath.sim_time': 1.7,
        'FollowPath.linear_granularity': 0.05,
        'FollowPath.angular_granularity': 0.025,
        'FollowPath.transform_tolerance': 0.2,
        'FollowPath.xy_goal_tolerance': 0.25,
        'FollowPath.trans_stopped_velocity': 0.25,
        'FollowPath.short_circuit_trajectory_evaluation': True,
        'FollowPath.stateful': True,
        'FollowPath.BaseObstacle.scale': 0.02,
        'FollowPath.PathAlign.scale': 32.0,
        'FollowPath.PathAlign.forward_point_distance': 0.1,
        'FollowPath.GoalAlign.scale': 24.0,
        'FollowPath.GoalAlign.forward_point_distance': 0.1,
        'FollowPath.PathDist.scale': 32.0,
        'FollowPath.GoalDist.scale': 24.0,
        'FollowPath.RotateToGoal.scale': 32.0,
        'FollowPath.RotateToGoal.slowing_factor': 5.0,
        'FollowPath.RotateToGoal.lookahead_time': -1.0,
    }


def nav2_nodes(ns):
    config_dir = os.path.join(get_package_share_directory(PKG), 'config')
    yaml = os.path.join(config_dir, f'nav2_{ns}.yaml')
    nodes = []

    nodes.append(Node(
        package='nav2_controller',
        executable='controller_server',
        name='controller_server',
        namespace=ns, output='screen',
        parameters=[yaml, dwb_params(ns)],
        remappings=[('tf', '/tf'), ('tf_static', '/tf_static'),
                    ('cmd_vel', f'/{ns}/cmd_vel')]))

    for pkg, exe, name in [
        ('nav2_smoother',          'smoother_server',    'smoother_server'),
        ('nav2_planner',           'planner_server',     'planner_server'),
        ('nav2_behaviors',         'behavior_server',    'behavior_server'),
        ('nav2_bt_navigator',      'bt_navigator',       'bt_navigator'),
        ('nav2_waypoint_follower', 'waypoint_follower',  'waypoint_follower'),
        ('nav2_velocity_smoother', 'velocity_smoother',  'velocity_smoother'),
    ]:
        nodes.append(Node(
            package=pkg, executable=exe, name=name,
            namespace=ns, output='screen',
            parameters=[yaml],
            remappings=[('tf', '/tf'), ('tf_static', '/tf_static')]))

    nodes.append(Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_navigation',
        namespace=ns, output='screen',
        parameters=[{
            'use_sim_time': True,
            'autostart': True,
            'node_names': [
                'controller_server', 'smoother_server',
                'planner_server', 'behavior_server',
                'bt_navigator', 'waypoint_follower',
                'velocity_smoother',
            ],
        }]))
    return nodes


def generate_launch_description():
    TURTLEBOT3_MODEL = os.environ.get('TURTLEBOT3_MODEL', 'burger')

    tb3_gazebo_dir = get_package_share_directory('turtlebot3_gazebo')
    pkg_gazebo_ros = get_package_share_directory('gazebo_ros')
    multi_explorer_dir = get_package_share_directory(PKG)
    config_dir = os.path.join(multi_explorer_dir, 'config')
    launch_file_dir = os.path.join(tb3_gazebo_dir, 'launch')

    model_folder = 'turtlebot3_' + TURTLEBOT3_MODEL
    urdf_path = os.path.join(tb3_gazebo_dir, 'models', model_folder, 'model.sdf')
    save_dir = os.path.join(tb3_gazebo_dir, 'models', model_folder)

    # ── Gazebo world ──
    world = os.path.join(tb3_gazebo_dir, 'worlds', 'turtlebot3_house.world')

    ld = LaunchDescription()

    # Gazebo server
    ld.add_action(IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_gazebo_ros, 'launch', 'gzserver.launch.py')),
        launch_arguments={'world': world}.items()))

    # Gazebo client (GUI)
    ld.add_action(IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_gazebo_ros, 'launch', 'gzclient.launch.py'))))

    # ── 로봇 spawn × 3 ──
    sdf_paths = []
    for i, ns in enumerate(ROBOT_NS):
        # SDF에서 frame prefix 수정
        tree = ET.parse(urdf_path)
        root = tree.getroot()
        for tag in root.iter('odometry_frame'):
            tag.text = f'{ns}/odom'
        for tag in root.iter('robot_base_frame'):
            tag.text = f'{ns}/base_footprint'
        for tag in root.iter('frame_name'):
            tag.text = f'{ns}/base_scan'

        sdf_save = os.path.join(save_dir, f'tmp_{ns}.sdf')
        sdf_paths.append(sdf_save)
        modified = '<?xml version="1.0" ?>\n' + ET.tostring(
            tree.getroot(), encoding='unicode')
        with open(sdf_save, 'w') as f:
            f.write(modified)

        # robot_state_publisher
        rsp = IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(launch_file_dir, 'robot_state_publisher.launch.py')),
            launch_arguments={
                'use_sim_time': 'true',
                'frame_prefix': ns,
            }.items())

        # spawn
        spawn = IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(launch_file_dir, 'multi_spawn_turtlebot3.launch.py')),
            launch_arguments={
                'x_pose': str(SPAWN_POSES[i][0]),
                'y_pose': str(SPAWN_POSES[i][1]),
                'robot_name': f'{TURTLEBOT3_MODEL}_{ns}',
                'namespace': ns,
                'sdf_path': sdf_save,
            }.items())

        ld.add_action(GroupAction([
            PushRosNamespace(ns),
            rsp,
            spawn,
        ]))

    # 종료 시 임시 SDF 삭제
    ld.add_action(RegisterEventHandler(OnShutdown(
        on_shutdown=lambda event, context: [
            os.remove(p) for p in sdf_paths if os.path.exists(p)])))

    # ── SLAM Toolbox × 3 (5초 후) ──
    slam_pkg = get_package_share_directory('slam_toolbox')
    for ns in ROBOT_NS:
        ld.add_action(TimerAction(period=5.0, actions=[
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    os.path.join(slam_pkg, 'launch', 'online_async_launch.py')),
                launch_arguments={
                    'use_sim_time': 'true',
                    'slam_params_file': os.path.join(config_dir, f'slam_{ns}.yaml'),
                }.items())]))

    # ── multirobot_map_merge — 중앙 모니터링용 (7초 후) ──
    map_merge_config = os.path.join(config_dir, 'map_merge_params.yaml')
    ld.add_action(TimerAction(period=7.0, actions=[Node(
        package='multirobot_map_merge',
        executable='map_merge',
        name='map_merge',
        namespace='/',
        output='screen',
        parameters=[
            map_merge_config,
            {'use_sim_time': True},
            {'known_init_poses': True},
        ],
        remappings=[('/tf', 'tf'), ('/tf_static', 'tf_static')],
    )]))

    # ── 로봇별 map_merge 제거 — robot_agent_node가 자체 map_merger.py로 합성 ──

    # ── Nav2 × 3 (8초 후) ──
    for ns in ROBOT_NS:
        for node in nav2_nodes(ns):
            ld.add_action(TimerAction(period=8.0, actions=[node]))

    # ── Decentralized 탐색 노드 (12초 후) ──
    for i, ns in enumerate(ROBOT_NS):
        ld.add_action(TimerAction(period=12.0, actions=[Node(
            package=PKG, executable='robot_agent_node',
            name=f'robot_agent_{i}', output='screen',
            parameters=[{
                'use_sim_time': True,
                'robot_id': i,
                'robot_ns': ns,
                'num_robots': NUM_ROBOTS,
            }])]))

    ld.add_action(TimerAction(period=12.0, actions=[Node(
        package=PKG, executable='coordinator_node',
        name='coordinator_node', output='screen',
        parameters=[{'use_sim_time': True}])]))

    ld.add_action(TimerAction(period=12.0, actions=[Node(
        package=PKG, executable='visualizer_node',
        name='visualizer_node', output='screen',
        parameters=[{'use_sim_time': True}])]))

    return ld
