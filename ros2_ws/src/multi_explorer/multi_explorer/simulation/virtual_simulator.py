"""
simulation/virtual_simulator.py

Gazebo 없이 가상 환경에서 멀티로봇 시뮬레이션.
LiDAR, odom, TF, /clock 퍼블리시.

원본: 기존 virtual_simulator.py 그대로 (이미 노드+로직 일체형이라 유지)
오픈소스 대응: demo.m
"""
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import TransformStamped, Twist
from tf2_ros import TransformBroadcaster
from visualization_msgs.msg import Marker, MarkerArray
from rosgraph_msgs.msg import Clock
import numpy as np
import math

GRID_SIZE = 200
RESOLUTION = 0.1
ROBOT_SPEED = 0.26
LIDAR_RANGE = 4.0
LIDAR_RAYS = 360
UPDATE_HZ = 20.0
NUM_ROBOTS = 3
ROBOT_NS = ['tb3_0', 'tb3_1', 'tb3_2']
ROBOT_STARTS = [(0.0, 0.0), (0.5, 0.0), (0.25, 0.4)]


class VirtualRobot:
    def __init__(self, x, y, robot_id):
        self.x = x
        self.y = y
        self.theta = 0.0
        self.id = robot_id
        self.vx = 0.0
        self.vtheta = 0.0


class VirtualSimulator(Node):
    def __init__(self):
        super().__init__('virtual_simulator')
        self.map_grid = self._create_map()
        self.robots = [VirtualRobot(x, y, i)
                       for i, (x, y) in enumerate(ROBOT_STARTS[:NUM_ROBOTS])]
        self.tf_broadcaster = TransformBroadcaster(self)
        self._sim_time = 0.0

        self.clock_pub = self.create_publisher(Clock, '/clock', 10)
        self.scan_pubs = [self.create_publisher(LaserScan, f'/{ns}/scan', 10)
                          for ns in ROBOT_NS[:NUM_ROBOTS]]
        self.odom_pubs = [self.create_publisher(Odometry, f'/{ns}/odom', 10)
                          for ns in ROBOT_NS[:NUM_ROBOTS]]
        self.marker_pub = self.create_publisher(MarkerArray, '/robot_markers', 10)

        self.cmd_subs = [
            self.create_subscription(Twist, f'/{ns}/cmd_vel',
                                     lambda msg, i=i: self._cmd_cb(msg, i), 10)
            for i, ns in enumerate(ROBOT_NS[:NUM_ROBOTS])]

        self.create_timer(1.0 / UPDATE_HZ, self.update)
        self.get_logger().info(f'Virtual Simulator 시작 — {NUM_ROBOTS}대')

    def _cmd_cb(self, msg, robot_id):
        self.robots[robot_id].vx = msg.linear.x
        self.robots[robot_id].vtheta = msg.angular.z

    def _create_map(self):
        g = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)
        g[0, :] = 1; g[-1, :] = 1; g[:, 0] = 1; g[:, -1] = 1

        def wall(r, c, h, w):
            g[r:min(r + h, GRID_SIZE), c:min(c + w, GRID_SIZE)] = 1

        wall(20, 20, 70, 3); wall(20, 20, 3, 60)
        wall(20, 75, 3, 25); wall(75, 20, 25, 3)
        wall(20, 120, 70, 3); wall(20, 178, 3, 60)
        wall(20, 120, 3, 25); wall(75, 155, 25, 3)
        wall(125, 20, 55, 3); wall(125, 20, 3, 55)
        wall(125, 70, 3, 55); wall(175, 20, 55, 3)
        wall(125, 130, 55, 3); wall(125, 178, 3, 55)
        wall(125, 130, 3, 25); wall(175, 130, 55, 3)
        wall(85, 85, 30, 3); wall(85, 118, 30, 3)
        wall(88, 88, 3, 24); wall(118, 88, 3, 24)
        for pos in [(45, 95), (45, 105), (95, 45), (95, 155),
                    (155, 45), (155, 155), (105, 95), (105, 105)]:
            wall(pos[0], pos[1], 5, 5)
        wall(40, 50, 3, 20); wall(40, 130, 3, 20)
        wall(160, 50, 3, 20); wall(160, 130, 3, 20)
        wall(50, 40, 20, 3); wall(50, 157, 20, 3)
        wall(130, 40, 20, 3); wall(130, 157, 20, 3)
        return g

    def _w2g(self, wx, wy):
        cx = int(wx / RESOLUTION) + GRID_SIZE // 2
        cy = int(wy / RESOLUTION) + GRID_SIZE // 2
        return np.clip(cy, 0, GRID_SIZE - 1), np.clip(cx, 0, GRID_SIZE - 1)

    def update(self):
        dt = 1.0 / UPDATE_HZ
        self._sim_time += dt
        clock_msg = Clock()
        clock_msg.clock.sec = int(self._sim_time)
        clock_msg.clock.nanosec = int((self._sim_time % 1.0) * 1e9)
        self.clock_pub.publish(clock_msg)

        for robot in self.robots:
            robot.theta += robot.vtheta * dt
            nx = robot.x + robot.vx * math.cos(robot.theta) * dt
            ny = robot.y + robot.vx * math.sin(robot.theta) * dt
            r, c = self._w2g(nx, ny)
            if self.map_grid[r, c] == 0:
                robot.x = nx
                robot.y = ny
            self._lidar(robot)
            self._tf(robot)
            self._odom(robot)
        self._markers()

    def _lidar(self, robot):
        ranges = []
        angle_min = -math.pi
        angle_inc = 2 * math.pi / LIDAR_RAYS
        for i in range(LIDAR_RAYS):
            angle = robot.theta + angle_min + i * angle_inc
            hit = LIDAR_RANGE
            for d in np.arange(0.1, LIDAR_RANGE, RESOLUTION):
                rx = robot.x + d * math.cos(angle)
                ry = robot.y + d * math.sin(angle)
                r, c = self._w2g(rx, ry)
                if self.map_grid[r, c] == 1:
                    hit = d
                    break
            ranges.append(hit)
        scan = LaserScan()
        scan.header.stamp = self._ros_time()
        scan.header.frame_id = f'{ROBOT_NS[robot.id]}/base_scan'
        scan.angle_min = angle_min
        scan.angle_max = math.pi
        scan.angle_increment = angle_inc
        scan.range_min = 0.1
        scan.range_max = LIDAR_RANGE
        scan.ranges = ranges
        self.scan_pubs[robot.id].publish(scan)

    def _tf(self, robot):
        ns = ROBOT_NS[robot.id]
        now = self._ros_time()
        transforms = []

        t2 = TransformStamped()
        t2.header.stamp = now
        t2.header.frame_id = f'{ns}/odom'
        t2.child_frame_id = f'{ns}/base_footprint'
        t2.transform.translation.x = robot.x
        t2.transform.translation.y = robot.y
        t2.transform.rotation.z = math.sin(robot.theta / 2)
        t2.transform.rotation.w = math.cos(robot.theta / 2)
        transforms.append(t2)

        t3 = TransformStamped()
        t3.header.stamp = now
        t3.header.frame_id = f'{ns}/base_footprint'
        t3.child_frame_id = f'{ns}/base_scan'
        t3.transform.rotation.w = 1.0
        transforms.append(t3)

        self.tf_broadcaster.sendTransform(transforms)

    def _odom(self, robot):
        ns = ROBOT_NS[robot.id]
        msg = Odometry()
        msg.header.stamp = self._ros_time()
        msg.header.frame_id = f'{ns}/odom'
        msg.child_frame_id = f'{ns}/base_footprint'
        msg.pose.pose.position.x = robot.x
        msg.pose.pose.position.y = robot.y
        msg.pose.pose.orientation.z = math.sin(robot.theta / 2)
        msg.pose.pose.orientation.w = math.cos(robot.theta / 2)
        msg.twist.twist.linear.x = robot.vx
        msg.twist.twist.angular.z = robot.vtheta
        self.odom_pubs[robot.id].publish(msg)

    def _markers(self):
        colors = [(1, 0, 0), (0, 0, 1), (0, 1, 0)]
        arr = MarkerArray()
        for robot in self.robots:
            m = Marker()
            m.header.frame_id = 'map'
            m.header.stamp = self._ros_time()
            m.ns = 'robots'; m.id = robot.id
            m.type = Marker.CYLINDER
            m.action = Marker.ADD
            m.pose.position.x = robot.x
            m.pose.position.y = robot.y
            m.pose.position.z = 0.1
            m.scale.x = m.scale.y = 0.3
            m.scale.z = 0.2
            c = colors[robot.id % len(colors)]
            m.color.r = float(c[0]); m.color.g = float(c[1])
            m.color.b = float(c[2]); m.color.a = 1.0
            arr.markers.append(m)
        self.marker_pub.publish(arr)

    def _ros_time(self):
        from builtin_interfaces.msg import Time
        t = Time()
        t.sec = int(self._sim_time)
        t.nanosec = int((self._sim_time % 1.0) * 1e9)
        return t


def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(VirtualSimulator())
    rclpy.shutdown()
