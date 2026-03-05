#!/usr/bin/env python3
"""
Month 2: K-means 공간 분할 + Frontier 비용함수 + A* 경로 계획
Phase 1: 단일 로봇 테스트
"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from nav_msgs.msg import OccupancyGrid
from nav2_msgs.action import NavigateToPose
from visualization_msgs.msg import Marker, MarkerArray

import numpy as np
import cv2
import math
import heapq
from sklearn.cluster import KMeans

# ── 파라미터 ──────────────────────────────────────────────────
NUM_ROBOTS        = 1
ROBOT_NAMESPACES  = ['']    # '' = 네임스페이스 없음 (단일 로봇)
DELTA_PENALTY     = 5.0
SOBEL_THRESHOLD   = 20
OBSTACLE_INFLATE  = 5
MIN_CLUSTER_SIZE  = 2
MAP_UPDATE_PERIOD = 3.0
# ──────────────────────────────────────────────────────────────


def astar(grid, start, goal):
    h, w = grid.shape
    def heuristic(a, b):
        return math.hypot(a[0]-b[0], a[1]-b[1])
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}
    while open_set:
        _, current = heapq.heappop(open_set)
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1),
                        (-1,-1),(-1,1),(1,-1),(1,1)]:
            nr, nc = current[0]+dr, current[1]+dc
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if grid[nr, nc] == 1:
                continue
            tentative_g = g_score[current] + math.hypot(dr, dc)
            neighbor = (nr, nc)
            if tentative_g < g_score.get(neighbor, float('inf')):
                came_from[neighbor] = current
                g_score[neighbor]   = tentative_g
                f = tentative_g + heuristic(neighbor, goal)
                heapq.heappush(open_set, (f, neighbor))
    return None


class RobotController:
    def __init__(self, node, namespace, robot_id):
        self.node    = node
        self.ns      = namespace
        self.id      = robot_id
        self.navigating   = False
        self.robot_x      = 0.0
        self.robot_y      = 0.0
        self.path_history = []
        self.visited      = []

        nav_action = '/navigate_to_pose' if not namespace \
                     else f'/{namespace}/navigate_to_pose'
        self._nav_client = ActionClient(node, NavigateToPose, nav_action)

        from tf2_ros import Buffer, TransformListener
        self.tf_buffer   = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, node)

    def get_pose(self):
        map_frame  = 'map'         if not self.ns else f'{self.ns}/map'
        base_frame = 'base_footprint' if not self.ns else f'{self.ns}/base_footprint'
        try:
            t = self.tf_buffer.lookup_transform(
                map_frame, base_frame,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.5))
            self.robot_x = t.transform.translation.x
            self.robot_y = t.transform.translation.y
            return True
        except Exception:
            base_frame2 = 'base_link' if not self.ns else f'{self.ns}/base_link'
            try:
                t = self.tf_buffer.lookup_transform(
                    map_frame, base_frame2,
                    rclpy.time.Time(),
                    timeout=rclpy.duration.Duration(seconds=0.5))
                self.robot_x = t.transform.translation.x
                self.robot_y = t.transform.translation.y
                return True
            except Exception as e:
                self.node.get_logger().warn(f'[{self.ns}] TF 실패: {e}')
                return False

    def navigate_to(self, point, result_cb):
        if not self._nav_client.wait_for_server(timeout_sec=3.0):
            self.node.get_logger().warn(f'[{self.ns}] Nav2 없음')
            self.navigating = False
            return
        goal = NavigateToPose.Goal()
        map_frame = 'map' if not self.ns else f'{self.ns}/map'
        goal.pose.header.frame_id    = map_frame
        goal.pose.header.stamp       = self.node.get_clock().now().to_msg()
        goal.pose.pose.position.x    = float(point[0])
        goal.pose.pose.position.y    = float(point[1])
        goal.pose.pose.orientation.w = 1.0
        self.navigating = True
        self.current_goal = point
        self.node.get_logger().info(
            f'[{self.ns or "tb3"}] 이동: ({point[0]:.2f}, {point[1]:.2f})')
        future = self._nav_client.send_goal_async(
            goal, feedback_callback=self._feedback_cb)
        future.add_done_callback(
            lambda f: self._response_cb(f, result_cb))

    def _feedback_cb(self, feedback):
        dist = feedback.feedback.distance_remaining
        self.node.get_logger().info(
            f'  [{self.ns or "tb3"}] 남은 거리: {dist:.2f}m',
            throttle_duration_sec=2.0)

    def _response_cb(self, future, result_cb):
        handle = future.result()
        if not handle.accepted:
            self.node.get_logger().warn(f'[{self.ns}] 목표 거부')
            self.navigating = False
            return
        handle.get_result_async().add_done_callback(result_cb)


class ExplorationPlanner(Node):

    def __init__(self):
        super().__init__('exploration_planner')

        self.robots = [
            RobotController(self, ns, i)
            for i, ns in enumerate(ROBOT_NAMESPACES)
        ]

        self.map_data  = None
        self.map_info  = None
        self.last_time = self.get_clock().now()

        map_topic = '/map' if not ROBOT_NAMESPACES[0] \
                    else f'/{ROBOT_NAMESPACES[0]}/map'
        self.map_sub = self.create_subscription(
            OccupancyGrid, map_topic, self.map_callback, 10)

        self.marker_pub = self.create_publisher(
            MarkerArray, '/exploration_markers', 10)
        
        self.path_pub = self.create_publisher(
            MarkerArray, '/astar_paths', 10)

        self.get_logger().info(
            f'Exploration Planner 시작 — 로봇 {NUM_ROBOTS}대')

    def map_callback(self, msg: OccupancyGrid):
        if all(r.navigating for r in self.robots):
            return
        now = self.get_clock().now()
        if (now - self.last_time).nanoseconds / 1e9 < MAP_UPDATE_PERIOD:
            return
        self.last_time = now

        self.map_info = msg.info
        h, w = msg.info.height, msg.info.width
        raw  = np.array(msg.data, dtype=np.int8).reshape((h, w))
        self.map_data = raw

        for r in self.robots:
            r.get_pose()

        frontiers = self.detect_frontiers(raw)
        if not frontiers:
            self.get_logger().info('Frontier 없음 — 탐색 완료')
            self.clear_markers()   # ← 탐색 완료 시 마커 전부 삭제
            return

        res = msg.info.resolution
        ox  = msg.info.origin.position.x
        oy  = msg.info.origin.position.y
        world_f = [(col*res+ox, row*res+oy) for (row, col) in frontiers]

        zones = self.kmeans_partition(raw)

        # 필터링된 frontier만 추출
        valid_f = self.get_valid_frontiers(world_f)

        if not valid_f:
            self.get_logger().info('유효 Frontier 없음 — 탐색 완료')
            self.clear_markers()   # ← 마커 삭제
            return

        # 유효한 frontier만 마커로 표시
        self.publish_markers(valid_f, zones, msg.info)
        self.assign_and_navigate(world_f, zones, msg.info)
        
    def detect_frontiers(self, raw):
        map_tri = np.full(raw.shape, 128, dtype=np.uint8)
        map_tri[raw >= 0]  = 255
        map_tri[raw > 50]  = 0
        sx  = cv2.Sobel(map_tri, cv2.CV_64F, 1, 0, ksize=3)
        sy  = cv2.Sobel(map_tri, cv2.CV_64F, 0, 1, ksize=3)
        mag = np.sqrt(sx**2 + sy**2)
        _, edge = cv2.threshold(
            mag.astype(np.uint8), SOBEL_THRESHOLD, 255, cv2.THRESH_BINARY)
        occ = (raw > 50).astype(np.uint8)
        k   = cv2.getStructuringElement(
            cv2.MORPH_RECT, (OBSTACLE_INFLATE, OBSTACLE_INFLATE))
        edge[cv2.dilate(occ, k) == 1] = 0
        edge[raw == -1] = 0
        edge[raw > 50]  = 0
        num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(
            edge, connectivity=8)
        centers = []
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= MIN_CLUSTER_SIZE:
                centers.append((int(centroids[i][1]), int(centroids[i][0])))
        self.get_logger().info(f'Frontier 수: {len(centers)}')
        return centers

    def kmeans_partition(self, raw):
        unknown_pts = np.column_stack(np.where(raw == -1))
        if len(unknown_pts) < NUM_ROBOTS:
            return [[] for _ in range(NUM_ROBOTS)]
        if NUM_ROBOTS == 1:
            return [list(map(tuple, unknown_pts))]
        if len(unknown_pts) > 2000:
            idx = np.random.choice(len(unknown_pts), 2000, replace=False)
            unknown_pts = unknown_pts[idx]
        kmeans = KMeans(n_clusters=NUM_ROBOTS, n_init=5, random_state=42)
        labels = kmeans.fit_predict(unknown_pts)
        zones  = [[] for _ in range(NUM_ROBOTS)]
        for pt, label in zip(unknown_pts, labels):
            zones[label].append(tuple(pt))
        self.get_logger().info(f'K-means 영역: {[len(z) for z in zones]}')
        return zones

    def assign_and_navigate(self, world_frontiers, zones, info):
        res = info.resolution
        ox  = info.origin.position.x
        oy  = info.origin.position.y

        def world_to_pixel(wx, wy):
            return int((wy-oy)/res), int((wx-ox)/res)

        for robot in self.robots:
            if robot.navigating:
                continue
            best_f    = None
            best_cost = float('inf')
            zone_set  = set(zones[robot.id]) if robot.id < len(zones) else set()

            for (fx, fy) in world_frontiers:
                if any(math.hypot(fx-v[0], fy-v[1]) < 0.4
                       for v in robot.visited):
                    continue
                dist    = math.hypot(fx-robot.robot_x, fy-robot.robot_y)
                in_zone = world_to_pixel(fx, fy) in zone_set
                cost    = dist + (0.0 if in_zone else DELTA_PENALTY)
                if cost < best_cost:
                    best_cost = cost
                    best_f    = (fx, fy)

            if best_f:
                robot.path_history.append((robot.robot_x, robot.robot_y))

                if self.map_data is not None:
                    self.visualize_astar_path(robot, best_f, info)

                robot.navigate_to(
                    best_f,
                    lambda f, r=robot: self._result_cb(f, r))

    def _result_cb(self, future, robot):
        status = future.result().status
        self.get_logger().info(
            f'[{robot.ns or "tb3"}] 완료 status={status}')
        # visited에 추가 (성공/실패 모두)
        if hasattr(robot, 'current_goal') and robot.current_goal:
            robot.visited.append(robot.current_goal)
            robot.current_goal = None
        robot.navigating = False

    def publish_markers(self, world_frontiers, zones, info):
        colors = [
            (1.0, 0.3, 0.3),
            (0.3, 0.3, 1.0),
            (0.3, 1.0, 0.3),
        ]
        arr = MarkerArray()
        d = Marker(); d.action = Marker.DELETEALL
        arr.markers.append(d)
        self.marker_pub.publish(arr)

        arr = MarkerArray()
        map_frame = 'map' if not ROBOT_NAMESPACES[0] \
                    else f'{ROBOT_NAMESPACES[0]}/map'

        for i, (fx, fy) in enumerate(world_frontiers):
            m = Marker()
            m.header.frame_id = map_frame
            m.header.stamp    = self.get_clock().now().to_msg()
            m.ns = 'frontiers'; m.id = i
            m.type = Marker.SPHERE; m.action = Marker.ADD
            m.pose.position.x = fx
            m.pose.position.y = fy
            m.pose.position.z = 0.1
            m.scale.x = m.scale.y = m.scale.z = 0.15
            m.color.r = 1.0; m.color.g = 0.8
            m.color.a = 1.0
            m.lifetime.sec = 5
            arr.markers.append(m)

        for robot in self.robots:
            c = colors[robot.id % len(colors)]
            for j, (px, py) in enumerate(robot.path_history[-30:]):
                m = Marker()
                m.header.frame_id = map_frame
                m.header.stamp    = self.get_clock().now().to_msg()
                m.ns = f'path_{robot.id}'; m.id = j
                m.type = Marker.SPHERE; m.action = Marker.ADD
                m.pose.position.x = px
                m.pose.position.y = py
                m.pose.position.z = 0.05
                m.scale.x = m.scale.y = m.scale.z = 0.08
                m.color.r = c[0]; m.color.g = c[1]
                m.color.b = c[2]; m.color.a = 0.7
                m.lifetime.sec = 10
                arr.markers.append(m)

        self.marker_pub.publish(arr)
        
    def get_valid_frontiers(self, world_frontiers):
        """모든 로봇의 visited를 합산해서 필터링"""
        all_visited = []
        for r in self.robots:
            all_visited.extend(r.visited)

        valid = []
        for (fx, fy) in world_frontiers:
            if not any(math.hypot(fx-v[0], fy-v[1]) < 0.4
                       for v in all_visited):
                valid.append((fx, fy))
        return valid

    def clear_markers(self):
        """RViz2 마커 전체 삭제"""
        arr = MarkerArray()
        d = Marker()
        d.action = Marker.DELETEALL
        arr.markers.append(d)
        self.marker_pub.publish(arr)
        self.get_logger().info('마커 삭제 완료')
    
    def visualize_astar_path(self, robot, goal_world, info):
        """A*로 경로 계산 후 RViz2에 LINE_STRIP으로 표시"""
        res = info.resolution
        ox  = info.origin.position.x
        oy  = info.origin.position.y

        # 월드 → 픽셀 변환
        def w2p(wx, wy):
            return (int((wy - oy) / res), int((wx - ox) / res))

        # 픽셀 → 월드 변환
        def p2w(row, col):
            return (col * res + ox, row * res + oy)

        # OG Map → A* grid (0=free, 1=obstacle)
        grid = np.zeros_like(self.map_data, dtype=np.uint8)
        grid[self.map_data > 50]  = 1   # occupied

        start = w2p(robot.robot_x, robot.robot_y)
        goal  = w2p(goal_world[0], goal_world[1])

        # 범위 체크
        h, w = grid.shape
        if not (0 <= start[0] < h and 0 <= start[1] < w and
                0 <= goal[0]  < h and 0 <= goal[1]  < w):
            self.get_logger().warn('A* 범위 초과 — 시각화 스킵')
            return

        path = astar(grid, start, goal)

        if path is None:
            self.get_logger().warn(f'[{robot.ns or "tb3"}] A* 경로 없음')
            return

        self.get_logger().info(
            f'[{robot.ns or "tb3"}] A* 경로 길이: {len(path)} 셀')

        # ── LINE_STRIP 마커 생성 ──────────────────────────
        colors = [
            (1.0, 0.3, 0.3),   # 로봇0: 빨강
            (0.3, 0.3, 1.0),   # 로봇1: 파랑
            (0.3, 1.0, 0.3),   # 로봇2: 초록
        ]
        c = colors[robot.id % len(colors)]

        arr = MarkerArray()

        # 경로 선
        line = Marker()
        map_frame       = 'map' if not ROBOT_NAMESPACES[0] \
                          else f'{ROBOT_NAMESPACES[0]}/map'
        line.header.frame_id = map_frame
        line.header.stamp    = self.get_clock().now().to_msg()
        line.ns              = f'astar_line_{robot.id}'
        line.id              = 0
        line.type            = Marker.LINE_STRIP
        line.action          = Marker.ADD
        line.scale.x         = 0.03   # 선 두께
        line.color.r         = c[0]
        line.color.g         = c[1]
        line.color.b         = c[2]
        line.color.a         = 0.9
        line.lifetime.sec    = 8

        from geometry_msgs.msg import Point
        for (row, col) in path[::3]:   # 3셀마다 하나씩 (성능)
            wx, wy = p2w(row, col)
            pt = Point()
            pt.x = wx; pt.y = wy; pt.z = 0.05
            line.points.append(pt)
        arr.markers.append(line)

        # 시작점 마커 (구체)
        start_m = Marker()
        start_m.header.frame_id = map_frame
        start_m.header.stamp    = self.get_clock().now().to_msg()
        start_m.ns              = f'astar_start_{robot.id}'
        start_m.id              = 1
        start_m.type            = Marker.SPHERE
        start_m.action          = Marker.ADD
        start_m.pose.position.x = robot.robot_x
        start_m.pose.position.y = robot.robot_y
        start_m.pose.position.z = 0.1
        start_m.scale.x = start_m.scale.y = start_m.scale.z = 0.2
        start_m.color.r = c[0]; start_m.color.g = c[1]
        start_m.color.b = c[2]; start_m.color.a = 1.0
        start_m.lifetime.sec = 8
        arr.markers.append(start_m)

        # 목표점 마커 (구체)
        goal_m = Marker()
        goal_m.header.frame_id = map_frame
        goal_m.header.stamp    = self.get_clock().now().to_msg()
        goal_m.ns              = f'astar_goal_{robot.id}'
        goal_m.id              = 2
        goal_m.type            = Marker.SPHERE
        goal_m.action          = Marker.ADD
        goal_m.pose.position.x = goal_world[0]
        goal_m.pose.position.y = goal_world[1]
        goal_m.pose.position.z = 0.1
        goal_m.scale.x = goal_m.scale.y = goal_m.scale.z = 0.25
        goal_m.color.r = 1.0; goal_m.color.g = 1.0
        goal_m.color.b = 0.0; goal_m.color.a = 1.0   # 노란색
        goal_m.lifetime.sec = 8
        arr.markers.append(goal_m)

        self.path_pub.publish(arr)


def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(ExplorationPlanner())
    rclpy.shutdown()

