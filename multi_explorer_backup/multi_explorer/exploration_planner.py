#!/usr/bin/env python3
"""
exploration_planner.py

변경사항:
  - /map 단일 구독 (map_merger 가 만든 공유맵)
  - /system_state 구독 → EXPLORE 상태일 때만 동작
  - /exploration_done 퍼블리시 → frontier 없을 때
  - K-means 영역 할당은 merge_event 직후 수행
"""
import rclpy
#추가한 부분
import rclpy.time
#추가한 부분
import rclpy.duration
from rclpy.node import Node
from rclpy.action import ActionClient
from nav_msgs.msg import OccupancyGrid, Path
from geometry_msgs.msg import PoseStamped, Point
from nav2_msgs.action import NavigateToPose
from std_msgs.msg import Bool, String
from visualization_msgs.msg import Marker, MarkerArray
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy
import numpy as np
import cv2, math, heapq
from sklearn.cluster import KMeans

NUM_ROBOTS        = 3
ROBOT_NAMESPACES  = ['tb3_0', 'tb3_1', 'tb3_2']
ROBOT_STARTS      = [(0.0, 0.0), (0.5, 0.0), (0.25, 0.4)]
DELTA_PENALTY     = 3.0
SOBEL_THRESHOLD   = 20
OBSTACLE_INFLATE  = 5
MIN_CLUSTER_SIZE  = 2
MAP_UPDATE_PERIOD = 3.0
VISITED_RADIUS    = 0.8
#추가한 부분
# RENDEZVOUS_PERIOD, RENDEZVOUS_POS, rendezvous_cmd 퍼블리셔 제거
#추가한 부분
# → 집결 명령은 map_merger 가 /rendezvous_command 로 담당

def astar(grid, start, goal):
    h, w = grid.shape
    def heuristic(a, b): return math.hypot(a[0]-b[0], a[1]-b[1])
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
            if not (0 <= nr < h and 0 <= nc < w): continue
            if grid[nr, nc] == 1: continue
            tentative_g = g_score[current] + math.hypot(dr, dc)
            neighbor = (nr, nc)
            if tentative_g < g_score.get(neighbor, float('inf')):
                came_from[neighbor] = current
                g_score[neighbor]   = tentative_g
                heapq.heappush(open_set,
                    (tentative_g + heuristic(neighbor, goal), neighbor))
    return None


class RobotController:
    def __init__(self, node, namespace, robot_id):
        self.node          = node
        self.ns            = namespace
        self.id            = robot_id
        self.navigating    = False
        self.robot_x       = ROBOT_STARTS[robot_id][0]
        self.robot_y       = ROBOT_STARTS[robot_id][1]
        self.visited       = []
        self.astar_history = []
        self.current_goal  = None
        self.zone_cells    = set()   # K-means 할당 영역
        #추가한 부분
        self._goal_handle  = None    # Nav2 목표 핸들 저장 (취소 요청용)

        self._nav_client = ActionClient(
            node, NavigateToPose, f'/{namespace}/navigate_to_pose')

        from tf2_ros import Buffer, TransformListener
        self.tf_buffer   = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, node)

    def get_pose(self):
        try:
            t = self.tf_buffer.lookup_transform(
                'map', f'{self.ns}/base_footprint',
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.3))
            self.robot_x = t.transform.translation.x
            self.robot_y = t.transform.translation.y
            return True
        except Exception:
            return False

    def navigate_to(self, point, astar_pts=None):
        if not self._nav_client.wait_for_server(timeout_sec=1.0):
            self.node.get_logger().warn(f'[{self.ns}] Nav2 서버 없음')
            return
        self.navigating   = True
        self.current_goal = point
        if astar_pts:
            self.astar_history.extend(astar_pts)

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.header.stamp    = self.node.get_clock().now().to_msg()
        goal_msg.pose.pose.position.x    = float(point[0])
        goal_msg.pose.pose.position.y    = float(point[1])
        goal_msg.pose.pose.orientation.w = 1.0

        self.node.get_logger().info(
            f'[{self.ns}] 이동: ({point[0]:.2f}, {point[1]:.2f})')
        future = self._nav_client.send_goal_async(goal_msg)
        future.add_done_callback(self._goal_response_cb)

    def cancel_goal(self):
        #추가한 부분
        """Nav2 에 실제 취소 요청 전송 후 상태 초기화"""
        #추가한 부분
        if self._goal_handle is not None:
            #추가한 부분
            self._goal_handle.cancel_goal_async()
            #추가한 부분
            self._goal_handle = None
        self.navigating   = False
        self.current_goal = None

    def _goal_response_cb(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.node.get_logger().warn(f'[{self.ns}] 목표 거절됨')
            self.navigating = False
            return
        #추가한 부분
        self._goal_handle = goal_handle   # 취소 요청에 사용하기 위해 저장
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._result_cb)

    def _result_cb(self, future):
        #추가한 부분
        self._goal_handle = None          # 완료됐으므로 핸들 초기화
        self.get_pose()
        if self.current_goal:
            dist = math.hypot(self.robot_x - self.current_goal[0],
                              self.robot_y - self.current_goal[1])
            if dist < 1.0:
                self.visited.append(self.current_goal)
                self.node.get_logger().info(f'[{self.ns}] 도착')
            else:
                self.node.get_logger().warn(f'[{self.ns}] 미달 dist={dist:.2f}m')
            self.current_goal = None
        self.navigating = False
        self.node.publish_path(self.id)


class ExplorationPlanner(Node):
    def __init__(self):
        super().__init__('exploration_planner')

        self.robots     = [RobotController(self, ns, i)
                           for i, ns in enumerate(ROBOT_NAMESPACES)]
        self.map_data   = None
        self.map_info   = None
        self.last_plan_t = self.get_clock().now()
        #추가한 부분
        # last_rv_t 제거 → 집결 주기는 map_merger 가 관리
        self.system_state = 'INIT_SCAN'
        self.zones        = [[] for _ in range(NUM_ROBOTS)]

        # 공유맵 구독 (transient_local)
        map_qos = QoSProfile(
            depth=1,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            reliability=QoSReliabilityPolicy.RELIABLE)
        self.map_sub = self.create_subscription(
            OccupancyGrid, '/map', self._map_cb, map_qos)

        # 상태머신 상태 구독
        self.state_sub = self.create_subscription(
            String, '/system_state', self._state_cb, 10)

        # merge 이벤트 구독 → K-means 재할당 트리거
        self.merge_sub = self.create_subscription(
            Bool, '/merge_event', self._merge_event_cb, 10)

        # 퍼블리셔
        self.done_pub    = self.create_publisher(Bool, '/exploration_done', 10)
        self.marker_pub  = self.create_publisher(MarkerArray, '/exploration_markers', 10)
        self.astar_pub   = self.create_publisher(MarkerArray, '/astar_paths', 10)
        self.path_pubs   = [
            self.create_publisher(Path, f'/visited_path/{ns}', 10)
            for ns in ROBOT_NAMESPACES]

        self.create_timer(MAP_UPDATE_PERIOD, self._plan_tick)
        self.get_logger().info('Exploration Planner 시작')

    # ── 콜백 ──────────────────────────────────────────────────

    def _state_cb(self, msg: String):
        self.system_state = msg.data
        self.get_logger().info(f'상태 수신: {self.system_state}')

    def _map_cb(self, msg: OccupancyGrid):
        h, w = msg.info.height, msg.info.width
        self.map_info = msg.info
        self.map_data = np.array(msg.data, dtype=np.int8).reshape((h, w))

    def _merge_event_cb(self, msg: Bool):
        """merge 완료 → K-means 영역 재할당 + 진행 중 목표 취소"""
        if msg.data and self.map_data is not None:
            self.get_logger().info('Merge 이벤트 → K-means 영역 재할당')
            self.zones = self._kmeans_partition(self.map_data)
            for r in self.robots:
                r.zone_cells = set(self.zones[r.id])
            #추가한 부분
            # cancel_goal() 이 Nav2 에 실제 취소 요청을 보내도록 수정됨
            for r in self.robots:
                r.cancel_goal()

    # ── 계획 루프 ──────────────────────────────────────────────

    def _plan_tick(self):
        if self.system_state != 'EXPLORE':
            return
        if self.map_data is None:
            return
        if all(r.navigating for r in self.robots):
            return

        for r in self.robots:
            r.get_pose()

        frontiers = self._detect_frontiers(self.map_data)
        if not frontiers:
            self.get_logger().info('Frontier 없음 — 탐색 완료')
            self.done_pub.publish(Bool(data=True))
            return

        res = self.map_info.resolution
        ox  = self.map_info.origin.position.x
        oy  = self.map_info.origin.position.y
        world_f = [(col*res+ox, row*res+oy) for (row, col) in frontiers]

        self._assign_and_navigate(world_f)
        self._publish_markers(world_f)

    # ── Frontier 검출 ─────────────────────────────────────────

    def _detect_frontiers(self, raw):
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
        num_labels, _, stats, centroids = \
            cv2.connectedComponentsWithStats(edge, connectivity=8)
        centers = []
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= MIN_CLUSTER_SIZE:
                centers.append((int(centroids[i][1]), int(centroids[i][0])))
        self.get_logger().info(f'Frontier 수: {len(centers)}')
        return centers

    # ── K-means 영역 분할 ────────────────────────────────────

    def _kmeans_partition(self, raw):
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

    # ── 목표 할당 ─────────────────────────────────────────────

    def _assign_and_navigate(self, world_frontiers):
        res = self.map_info.resolution
        ox  = self.map_info.origin.position.x
        oy  = self.map_info.origin.position.y

        def w2p(wx, wy):
            return (int((wy-oy)/res), int((wx-ox)/res))

        assigned = set()

        for robot in self.robots:
            if robot.navigating:
                continue

            zone_set  = robot.zone_cells
            best_f, best_cost = None, float('inf')

            # 1단계: zone 우선 + visited 블랙리스트
            for (fx, fy) in world_frontiers:
                key = (round(fx,1), round(fy,1))
                if key in assigned: continue
                if any(math.hypot(fx-v[0], fy-v[1]) < VISITED_RADIUS
                       for v in robot.visited):
                    continue
                dist = math.hypot(fx-robot.robot_x, fy-robot.robot_y)
                cost = dist + (0.0 if w2p(fx,fy) in zone_set else DELTA_PENALTY)
                if cost < best_cost:
                    best_cost = cost; best_f = (fx, fy)

            # 2단계: visited 블랙리스트 무시
            if best_f is None:
                for (fx, fy) in world_frontiers:
                    key = (round(fx,1), round(fy,1))
                    if key in assigned: continue
                    dist = math.hypot(fx-robot.robot_x, fy-robot.robot_y)
                    if dist < best_cost:
                        best_cost = dist; best_f = (fx, fy)

            # 3단계: assigned 무시
            if best_f is None:
                for (fx, fy) in world_frontiers:
                    dist = math.hypot(fx-robot.robot_x, fy-robot.robot_y)
                    if dist < best_cost:
                        best_cost = dist; best_f = (fx, fy)

            if best_f:
                assigned.add((round(best_f[0],1), round(best_f[1],1)))
                astar_pts = None
                if self.map_data is not None:
                    grid = np.zeros_like(self.map_data, dtype=np.uint8)
                    grid[self.map_data > 50] = 1
                    start = w2p(robot.robot_x, robot.robot_y)
                    goal  = w2p(best_f[0], best_f[1])
                    path  = astar(grid, start, goal)
                    if path:
                        self._publish_astar(path, robot.id, self.map_info)
                        astar_pts = [(c*res+ox, r*res+oy) for (r,c) in path]
                robot.navigate_to(best_f, astar_pts)

    # ── 시각화 퍼블리셔 ───────────────────────────────────────

    def publish_path(self, robot_id):
        robot = self.robots[robot_id]
        msg = Path()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        pts = robot.astar_history if robot.astar_history \
              else [ROBOT_STARTS[robot_id]] + robot.visited
        for (px, py) in pts:
            ps = PoseStamped()
            ps.header = msg.header
            ps.pose.position.x = float(px)
            ps.pose.position.y = float(py)
            ps.pose.orientation.w = 1.0
            msg.poses.append(ps)
        self.path_pubs[robot_id].publish(msg)

    def _publish_astar(self, path, robot_id, info):
        colors = [(1.0,0.2,0.2),(0.2,0.2,1.0),(0.2,0.8,0.2)]
        res = info.resolution
        ox  = info.origin.position.x
        oy  = info.origin.position.y
        arr = MarkerArray()
        m   = Marker()
        m.header.frame_id = 'map'
        m.header.stamp    = self.get_clock().now().to_msg()
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
            p.y = r   * res + oy
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
            m.header.stamp    = self.get_clock().now().to_msg()
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
    rclpy.spin(ExplorationPlanner())
    rclpy.shutdown()
