"""
nodes/explorer_node.py

ROS2 노드: frontier 탐색 계획 + Nav2 이동 명령
순수 로직은 core/exploration_planner.py, perception/frontier_detector.py 에 위임.

원본: 기존 exploration_planner.py 에서 ROS 통신부 분리
오픈소스 대응: robots.m, robotsMain.m
"""
#!/usr/bin/env python3
import rclpy
import rclpy.time
import rclpy.duration
from rclpy.node import Node
from rclpy.action import ActionClient
from nav_msgs.msg import OccupancyGrid, Path
from geometry_msgs.msg import PoseStamped, Point
from nav2_msgs.action import NavigateToPose
from std_msgs.msg import Bool, String
from visualization_msgs.msg import Marker, MarkerArray
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy
from tf2_ros import Buffer, TransformListener
import numpy as np
import math
import json

from multi_explorer.core.exploration_planner import ExplorationPlanner
from multi_explorer.perception.frontier_detector import FrontierDetector
from multi_explorer.planning.astar_planner import astar
from multi_explorer.planning.path_utils import grid_path_to_world

NUM_ROBOTS = 3
ROBOT_NAMESPACES = ['tb3_0', 'tb3_1', 'tb3_2']
ROBOT_STARTS = [(0.0, 0.0), (0.5, 0.0), (0.25, 0.4)]
MAP_UPDATE_PERIOD = 3.0


class RobotHandle:
    """로봇 1대의 상태/Nav2 통신을 관리하는 헬퍼."""

    def __init__(self, node, namespace, robot_id):
        self.node = node
        self.ns = namespace
        self.id = robot_id
        self.navigating = False
        self.robot_x = ROBOT_STARTS[robot_id][0]
        self.robot_y = ROBOT_STARTS[robot_id][1]
        self.visited = []
        self.astar_history = []
        self.current_goal = None
        self.zone_cells = set()
        self._goal_handle = None
        self.failed_goals = {}         # {key: expire_time} TTL 블랙리스트
        self._nav_failed = False
        self.BLACKLIST_TTL = 30.0      # 30초 후 자동 해제
        self.trajectory = []           # 실제 이동 궤적 (TF 샘플링)

        self._nav_client = ActionClient(
            node, NavigateToPose, f'/{namespace}/navigate_to_pose')
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, node)

    def get_pose(self):
        try:
            t = self.tf_buffer.lookup_transform(
                'map', f'{self.ns}/base_footprint',
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.3))
            self.robot_x = t.transform.translation.x
            self.robot_y = t.transform.translation.y
            # 궤적 기록: 이전 위치와 0.05m 이상 차이날 때만
            if (not self.trajectory or
                    math.hypot(self.robot_x - self.trajectory[-1][0],
                               self.robot_y - self.trajectory[-1][1]) > 0.05):
                self.trajectory.append((self.robot_x, self.robot_y))
            return True
        except Exception:
            return False

    def navigate_to(self, point, astar_pts=None):
        if not self._nav_client.wait_for_server(timeout_sec=1.0):
            self.node.get_logger().warn(f'[{self.ns}] Nav2 서버 없음')
            return
        self.navigating = True
        self.current_goal = point
        if astar_pts:
            self.astar_history.extend(astar_pts)

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.header.stamp = self.node.get_clock().now().to_msg()
        goal_msg.pose.pose.position.x = float(point[0])
        goal_msg.pose.pose.position.y = float(point[1])
        goal_msg.pose.pose.orientation.w = 1.0

        self.node.get_logger().info(
            f'[{self.ns}] 이동: ({point[0]:.2f}, {point[1]:.2f})')
        future = self._nav_client.send_goal_async(goal_msg)
        future.add_done_callback(self._goal_response_cb)

    def cancel_goal(self):
        if self._goal_handle is not None:
            self._goal_handle.cancel_goal_async()
            self._goal_handle = None
        self.navigating = False
        self.current_goal = None

    def _goal_response_cb(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.node.get_logger().warn(f'[{self.ns}] 목표 거절됨')
            self.navigating = False
            return
        self._goal_handle = goal_handle
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._result_cb)

    def _result_cb(self, future):
        self._goal_handle = None
        self.get_pose()
        if self.current_goal:
            dist = math.hypot(self.robot_x - self.current_goal[0],
                              self.robot_y - self.current_goal[1])
            if dist < 1.0:
                self.visited.append(self.current_goal)
                self._nav_failed = False
                self.node.get_logger().info(f'[{self.ns}] 도착')
            else:
                # TTL 블랙리스트: 30초 후 자동 해제
                key = (round(self.current_goal[0], 1), round(self.current_goal[1], 1))
                import time
                self.failed_goals[key] = time.time() + self.BLACKLIST_TTL
                self._nav_failed = True
                self.node.get_logger().warn(
                    f'[{self.ns}] 실패 → 블랙리스트 {key} ({self.BLACKLIST_TTL}s)')
            self.current_goal = None
        self.navigating = False
        self.node.publish_path(self.id)

    def get_active_blacklist(self) -> set:
        """만료되지 않은 블랙리스트 키만 반환."""
        import time
        now = time.time()
        expired = [k for k, t in self.failed_goals.items() if now >= t]
        for k in expired:
            del self.failed_goals[k]
        return set(self.failed_goals.keys())

    def to_dict(self) -> dict:
        """core/exploration_planner 에 넘길 dict 형태."""
        return {
            'id': self.id,
            'x': self.robot_x,
            'y': self.robot_y,
            'navigating': self.navigating,
            'visited': list(self.visited),
            'zone_cells': self.zone_cells,
            'failed_goals': self.get_active_blacklist(),
        }


class ExplorerNode(Node):
    def __init__(self):
        super().__init__('explorer_node')

        self.robots = [RobotHandle(self, ns, i)
                       for i, ns in enumerate(ROBOT_NAMESPACES)]
        self.planner = ExplorationPlanner(num_robots=NUM_ROBOTS)
        self.frontier_detector = FrontierDetector()

        self.map_data = None
        self.map_info = None
        self.system_state = 'INIT_SCAN'
        self.kmeans_initialized = False  # 첫 K-means 완료 여부
        self.last_kmeans_time = 0.0      # 마지막 K-means 실행 시간

        # 공유맵 구독
        map_qos = QoSProfile(
            depth=1,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            reliability=QoSReliabilityPolicy.RELIABLE)
        self.map_sub = self.create_subscription(
            OccupancyGrid, '/map', self._map_cb, map_qos)

        # 상태 구독
        self.state_sub = self.create_subscription(
            String, '/system_state', self._state_cb, 10)

        # merge 이벤트 → K-means 재할당
        self.merge_sub = self.create_subscription(
            Bool, '/merge_event', self._merge_event_cb, 10)

        # 퍼블리셔
        self.done_pub = self.create_publisher(Bool, '/exploration_done', 10)
        self.marker_pub = self.create_publisher(
            MarkerArray, '/exploration_markers', 10)
        self.astar_pub = self.create_publisher(MarkerArray, '/astar_paths', 10)
        self.cluster_info_pub = self.create_publisher(
            String, '/cluster_info', 10)
        self.path_pubs = [
            self.create_publisher(Path, f'/visited_path/{ns}', 10)
            for ns in ROBOT_NAMESPACES]

        self.create_timer(MAP_UPDATE_PERIOD, self._plan_tick)
        self.create_timer(1.0, self._sample_trajectory)  # 1초마다 궤적 샘플링
        self.get_logger().info('Explorer Node 시작')

    def _sample_trajectory(self):
        """1초마다 각 로봇의 TF 위치를 궤적에 기록 + path publish."""
        for r in self.robots:
            r.get_pose()
            self.publish_path(r.id)

    # ── 콜백 ──────────────────────────────────────────────────

    def _state_cb(self, msg: String):
        self.system_state = msg.data

    def _map_cb(self, msg: OccupancyGrid):
        h, w = msg.info.height, msg.info.width
        self.map_info = msg.info
        self.map_data = np.array(msg.data, dtype=np.int8).reshape((h, w))

    def _merge_event_cb(self, msg: Bool):
        if msg.data and self.map_data is not None:
            import time as _time
            now = _time.time()

            # 첫 K-means 또는 마지막 실행 후 30초 경과 시에만 재실행
            KMEANS_COOLDOWN = 30.0
            if self.kmeans_initialized and (now - self.last_kmeans_time) < KMEANS_COOLDOWN:
                self.get_logger().info('Merge 이벤트 — K-means 쿨다운 중, 맵만 업데이트')
                return

            self.get_logger().info('Merge 이벤트 → K-means 영역 재할당 (auction)')
            # 로봇 위치 갱신
            for r in self.robots:
                r.get_pose()
            robot_positions = [(r.robot_x, r.robot_y) for r in self.robots]
            res = self.map_info.resolution
            ox = self.map_info.origin.position.x
            oy = self.map_info.origin.position.y
            zones = self.planner.kmeans_partition(
                self.map_data, res, ox, oy, robot_positions)
            for r in self.robots:
                r.zone_cells = set(zones[r.id])

            self.kmeans_initialized = True
            self.last_kmeans_time = now

            # 클러스터 정보를 coordinator_node 에 전달 (랑데부 포인트 계산용)
            cluster_msg = String()
            cluster_msg.data = json.dumps({
                'centers': [list(c) for c in self.planner.cluster_centers],
                'sizes': [len(z) for z in zones],
            })
            self.cluster_info_pub.publish(cluster_msg)

    # ── 계획 루프 ─────────────────────────────────────────────

    def _plan_tick(self):
        if self.system_state != 'EXPLORE':
            return
        if self.map_data is None:
            return
        if all(r.navigating for r in self.robots):
            return

        for r in self.robots:
            r.get_pose()

        res = self.map_info.resolution
        ox = self.map_info.origin.position.x
        oy = self.map_info.origin.position.y
        h, w = self.map_data.shape

        # ── 논문 5번: 이동 중인 로봇의 목표가 이미 탐사됐으면 취소 ──
        for r in self.robots:
            if r.navigating and r.current_goal is not None:
                gx, gy = r.current_goal
                gc = int((gx - ox) / res)
                gr = int((gy - oy) / res)
                if 0 <= gr < h and 0 <= gc < w:
                    # 목표 주변이 이미 전부 free면 → 이미 탐사됨
                    if self._is_already_explored(gr, gc, self.map_data, h, w):
                        self.get_logger().info(
                            f'[{r.ns}] 목표 ({gx:.2f},{gy:.2f}) 이미 탐사됨 → 재계획')
                        r.cancel_goal()

        # frontier 검출
        grid_frontiers = self.frontier_detector.detect(self.map_data)
        if not grid_frontiers:
            self.get_logger().info('Frontier 없음 — 탐색 완료')
            self.done_pub.publish(Bool(data=True))
            return

        world_f = self.frontier_detector.to_world(grid_frontiers, res, ox, oy)

        # 맵 범위 내 필터링 (margin 3셀)
        margin = 3
        world_f = [
            (wx, wy) for (wx, wy) in world_f
            if margin <= int((wy - oy) / res) < h - margin
            and margin <= int((wx - ox) / res) < w - margin
        ]
        if not world_f:
            self.get_logger().info('범위 내 Frontier 없음 — 탐색 완료')
            self.done_pub.publish(Bool(data=True))
            return

        # 모든 로봇의 활성 블랙리스트 통합
        all_failed = set()
        for r in self.robots:
            all_failed |= r.get_active_blacklist()

        # 이동 중인 로봇의 current_goal도 제외 (중복 할당 방지)
        in_progress = set()
        for r in self.robots:
            if r.navigating and r.current_goal is not None:
                in_progress.add(
                    (round(r.current_goal[0], 1), round(r.current_goal[1], 1)))

        # 블랙리스트 + 진행 중 목표 필터링
        world_f_filtered = [
            (wx, wy) for (wx, wy) in world_f
            if (round(wx, 1), round(wy, 1)) not in all_failed
            and (round(wx, 1), round(wy, 1)) not in in_progress
        ]
        if not world_f_filtered:
            self.get_logger().info('블랙리스트 초기화 — 모든 frontier 재시도')
            for r in self.robots:
                r.failed_goals.clear()
            world_f_filtered = [
                (wx, wy) for (wx, wy) in world_f
                if (round(wx, 1), round(wy, 1)) not in in_progress
            ]
            if not world_f_filtered:
                world_f_filtered = world_f

        # 목표 할당
        map_info_dict = {'resolution': res, 'origin_x': ox, 'origin_y': oy}
        robot_dicts = [r.to_dict() for r in self.robots]
        assignments = self.planner.assign_targets(
            world_f_filtered, robot_dicts, map_info_dict)

        # Nav2 이동 + A* 검증 (경로 못 찾으면 보내지 않음)
        import time as _time
        sent = 0
        for robot_id, target in assignments.items():
            robot = self.robots[robot_id]
            # A* grid: occupied만 장애물 (unknown은 통과 허용 — frontier는 unknown 경계에 있으니까)
            grid = np.zeros_like(self.map_data, dtype=np.uint8)
            grid[self.map_data > 50] = 1   # occupied만 장애물

            start = (int((robot.robot_y - oy) / res),
                     int((robot.robot_x - ox) / res))
            goal = (int((target[1] - oy) / res),
                    int((target[0] - ox) / res))

            # 범위 체크
            if not (0 <= start[0] < h and 0 <= start[1] < w and
                    0 <= goal[0] < h and 0 <= goal[1] < w):
                key = (round(target[0], 1), round(target[1], 1))
                robot.failed_goals[key] = _time.time() + robot.BLACKLIST_TTL
                continue

            # start/goal이 장애물 위면 가장 가까운 free 셀로 보정
            if grid[start[0], start[1]] == 1:
                start = self._find_nearest_free(grid, start)
            if grid[goal[0], goal[1]] == 1:
                goal = self._find_nearest_free(grid, goal)

            if start is None or goal is None:
                key = (round(target[0], 1), round(target[1], 1))
                robot.failed_goals[key] = _time.time() + robot.BLACKLIST_TTL
                continue

            path = astar(grid, start, goal)
            if path:
                self._publish_astar(path, robot_id)
                astar_pts = grid_path_to_world(path, res, ox, oy)
                robot.navigate_to(target, astar_pts)
                sent += 1
            else:
                key = (round(target[0], 1), round(target[1], 1))
                robot.failed_goals[key] = _time.time() + robot.BLACKLIST_TTL
                self.get_logger().warn(
                    f'[{robot.ns}] A* 실패 → 블랙리스트 {key} ({robot.BLACKLIST_TTL}s)')

        self._publish_markers(world_f_filtered)
        self.get_logger().info(
            f'Frontier {len(world_f_filtered)}개, 할당 {len(assignments)}건, 전송 {sent}건')

    # ── 시각화 ────────────────────────────────────────────────

    @staticmethod
    def _find_nearest_free(grid, pos, radius=10):
        """grid에서 pos 주변의 가장 가까운 free(0) 셀을 찾아 반환."""
        h, w = grid.shape
        best, best_d = None, float('inf')
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                nr, nc = pos[0] + dr, pos[1] + dc
                if 0 <= nr < h and 0 <= nc < w and grid[nr, nc] == 0:
                    d = abs(dr) + abs(dc)
                    if d < best_d:
                        best_d = d
                        best = (nr, nc)
        return best

    @staticmethod
    def _is_already_explored(row, col, map_data, h, w, radius=5):
        """목표 주변에 unknown(-1)이 없으면 이미 탐사된 것으로 판단."""
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                nr, nc = row + dr, col + dc
                if 0 <= nr < h and 0 <= nc < w:
                    if map_data[nr, nc] == -1:
                        return False  # unknown이 있으면 아직 탐사 안 됨
        return True  # 주변 전부 free/occupied → 이미 탐사됨

    def publish_path(self, robot_id):
        robot = self.robots[robot_id]
        msg = Path()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        # trajectory (실제 이동 궤적) 사용, 없으면 visited fallback
        pts = robot.trajectory if robot.trajectory \
            else [ROBOT_STARTS[robot_id]] + robot.visited
        for (px, py) in pts:
            ps = PoseStamped()
            ps.header = msg.header
            ps.pose.position.x = float(px)
            ps.pose.position.y = float(py)
            ps.pose.orientation.w = 1.0
            msg.poses.append(ps)
        self.path_pubs[robot_id].publish(msg)

    def _publish_astar(self, path, robot_id):
        colors = [(1.0, 0.2, 0.2), (0.2, 0.2, 1.0), (0.2, 0.8, 0.2)]
        res = self.map_info.resolution
        ox = self.map_info.origin.position.x
        oy = self.map_info.origin.position.y
        arr = MarkerArray()
        m = Marker()
        m.header.frame_id = 'map'
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = f'astar_{robot_id}'; m.id = robot_id
        m.type = Marker.LINE_STRIP; m.action = Marker.ADD
        m.scale.x = 0.08
        c = colors[robot_id % len(colors)]
        m.color.r = c[0]; m.color.g = c[1]
        m.color.b = c[2]; m.color.a = 0.9
        m.lifetime.sec = 15
        for (r, col) in path:
            p = Point()
            p.x = col * res + ox
            p.y = r * res + oy
            m.points.append(p)
        arr.markers.append(m)
        self.astar_pub.publish(arr)

    def _publish_markers(self, world_frontiers):
        arr = MarkerArray()
        d = Marker(); d.action = Marker.DELETEALL
        arr.markers.append(d)
        self.marker_pub.publish(arr)
        arr = MarkerArray()
        for i, (fx, fy) in enumerate(world_frontiers):
            m = Marker()
            m.header.frame_id = 'map'
            m.header.stamp = self.get_clock().now().to_msg()
            m.ns = 'frontiers'; m.id = i
            m.type = Marker.SPHERE; m.action = Marker.ADD
            m.pose.position.x = fx
            m.pose.position.y = fy
            m.pose.position.z = 0.1
            m.scale.x = m.scale.y = m.scale.z = 0.2
            m.color.r = 1.0; m.color.g = 0.6
            m.color.b = 0.0; m.color.a = 1.0
            m.lifetime.sec = 6
            arr.markers.append(m)
        self.marker_pub.publish(arr)


def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(ExplorerNode())
    rclpy.shutdown()
