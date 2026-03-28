from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
import os

PKG    = 'multi_explorer'
NS     = ['tb3_0', 'tb3_1', 'tb3_2']
CONFIG = os.path.join(get_package_share_directory(PKG), 'config')

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
    yaml = os.path.join(CONFIG, f'nav2_{ns}.yaml')
    nodes = []

    # controller_server: yaml(costmap용) + dict(critics용)
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
    ld = LaunchDescription()

    ld.add_action(Node(
        package=PKG, executable='virtual_simulator',
        name='virtual_simulator', output='screen',
        parameters=[{'use_sim_time': False}]))

    slam_pkg = get_package_share_directory('slam_toolbox')
    for ns in NS:
        ld.add_action(TimerAction(period=2.0, actions=[
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    os.path.join(slam_pkg, 'launch', 'online_async_launch.py')),
                launch_arguments={
                    'use_sim_time': 'true',
                    'slam_params_file': os.path.join(CONFIG, f'slam_{ns}.yaml'),
                }.items())]))

    for ns in NS:
        for node in nav2_nodes(ns):
            ld.add_action(TimerAction(period=4.0, actions=[node]))

    for exe, name in [
        ('map_merger',         'map_merger'),
        ('robot_state_machine','robot_state_machine'),
        ('exploration_planner','exploration_planner'),
        ('map_visualizer',     'map_visualizer'),
    ]:
        ld.add_action(TimerAction(period=6.0, actions=[Node(
            package=PKG, executable=exe, name=name,
            output='screen', parameters=[{'use_sim_time': True}])]))

    return ld
