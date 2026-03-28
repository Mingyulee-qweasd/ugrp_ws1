"""
nodes/robot_agent_node.py

Decentralized 로봇 에이전트 노드.
각 로봇 1대당 1개 실행. 독립적으로 탐색하며,
proximity/랑데뷰 시 만난 로봇들과 맵 교환 + K-means 재분배.

맵 합성: 자체 map_merger.py 사용 (origin 기반 정확한 합성)
Nav2 costmap: slam_toolbox raw map 기반
Frontier 탐색: local_map 기반
K-means 영역 분배: merged_map 기반
"""
#!/usr/bin/env python3
import rclpy
import rclpy.time
import rclpy.duration
import time as _time
import math
import json
import numpy as np
from rclpy.node import Node
from rclpy.action import ActionClient
from nav_msgs.msg import OccupancyGrid, Path
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
from std_msgs.msg import String, Bool
from visualization_msgs.msg import Marker, MarkerArray
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy
from tf2_ros import Buffer, TransformListener
from action_msgs.msg import GoalStatus

from multi_explorer.core.exploration_planner import ExplorationPlanner
from multi_explorer.perception.frontier_detector import FrontierDetector
from multi_explorer.perception.map_merger import MapMerger
from multi_explorer.planning.astar_planner import astar
from multi_explorer.planning.path_utils import grid_path_to_world

COMM_RANGE = 3.0
MERGE_COOLDOWN = 15.0
PLAN_PERIOD = 3.0
BLACKLIST_TTL = 30.0
RENDEZVOUS_TIMEOUT = 30.0
MERGE_PERIOD = 2.0

# 각 로봇의 spawn 위치 (launch 파일과 동일)
SPAWN_POSES = {
    0: (-1.0, 0.0),
    1: (1.0, 0.0),
    2: (0.0, 1.5),
}


class RobotAgentNode(Node):
    def __init__(self):
        super().__init__('robot_agent_node')

        self.declare_parameter('robot_id', 0)
        self.declare_parameter('robot_ns', 'tb3_0')
        self.declare_parameter('num_robots', 3)

        self.robot_id = self.get_parameter('robot_id').value
        self.ns = self.get_parameter('robot_ns').value
        self.num_robots = self.get_parameter('num_robots').value
        self.all_ns = [f'tb3_{i}' for i in range(self.num_robots)]

        # ── 상태 ──
        self.state = 'INIT_SCAN'
        self.init_scan_start = self.get_clock().now()

        # 자기 로컬맵 (slam_toolbox)
        self.local_map = None
        self.local_info = None

        # merged 맵 (자체 map_merger로 합성)
        self.merged_map = None
        self.merged_info = None

        # 다른 로봇으로부터 수신한 맵 저장소
        self.received_maps = {}  # {robot_id: {'data': np.ndarray, 'info': dict}}

        # 다른 로봇 위치
        self.other_positions = {}
        self.my_x, self.my_y = 0.0, 0.0

        # 탐색 로직
        self.planner = ExplorationPlanner(num_robots=self.num_robots)
        self.frontier_detector = FrontierDetector()
        self.merger = MapMerger()
        self.zone_cells = set()
        self.navigating = False
        self.current_goal = None
        self._goal_handle = None
        self.failed_goals = {}
        self.visited = []
        self.trajectory = []
        self.astar_path = []
        self.current_frontiers = []

        # 맵 교환 관련
        self.last_merge_time = 0.0
        self.in_comm_robots = set()

        # TF
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # ── 구독 ──
        self.map_sub = self.create_subscription(
            OccupancyGrid, f'/{self.ns}/map', self._local_map_cb, 10)

        self.shared_map_subs = {}
        for i in range(self.num_robots):
            if i == self.robot_id:
                continue
            other_ns = self.all_ns[i]
            sub = self.create_subscription(
                OccupancyGrid, f'/{other_ns}/shared_map',
                lambda msg, rid=i: self._received_shared_map_cb(msg, rid), 10)
            self.shared_map_subs[i] = sub

        self.rv_sub = self.create_subscription(
            Bool, '/rendezvous_command', self._rv_command_cb, 10)
        self.kmeans_sub = self.create_subscription(
            String, '/kmeans_assignment', self._kmeans_assignment_cb, 10)

        # ── 퍼블리셔 ──
        self.shared_map_pub = self.create_publisher(
            OccupancyGrid, f'/{self.ns}/shared_map', 10)
        self.kmeans_pub = self.create_publisher(
            String, '/kmeans_assignment', 10)

        map_qos = QoSProfile(
            depth=1,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            reliability=QoSReliabilityPolicy.RELIABLE)
        self.merged_map_pub = self.create_publisher(
            OccupancyGrid, f'/{self.ns}/merged_map', map_qos)

        self.state_pub = self.create_publisher(String, f'/{self.ns}/agent_state', 10)
        self.cluster_pub = self.create_publisher(String, f'/{self.ns}/cluster_info', 10)
        self.path_pub = self.create_publisher(Path, f'/visited_path/{self.ns}', 10)
        self.frontier_pub = self.create_publisher(MarkerArray, f'/{self.ns}/frontiers', 10)
        self.astar_pub = self.create_publisher(Path, f'/{self.ns}/astar_path', 10)

        self.nav_client = ActionClient(self, NavigateToPose, f'/{self.ns}/navigate_to_pose')

        # ── 타이머 ──
        self.create_timer(0.5, self._tick)
        self.create_timer(PLAN_PERIOD, self._plan)
        self.create_timer(1.0, self._monitor_comm)
        self.create_timer(1.0, self._sample_trajectory)
        self.create_timer(5.0, self._publish_shared_map)
        self.create_timer(MERGE_PERIOD, self._do_merge)

        self.get_logger().info(f'Robot Agent [{self.ns}] (ID={self.robot_id}) 시작')

    # ══════════════════════════════════════════════════════════
    # 콜백
    # ══════════════════════════════════════════════════════════

    def _local_map_cb(self, msg: OccupancyGrid):
        h, w = msg.info.height, msg.info.width
        self.local_info = msg.info
        self.local_map = np.array(msg.data, dtype=np.int8).reshape((h, w))
        if self.merged_map is None:
            self.merged_map = self.local_map.copy()
            self.merged_info = msg.info

    def _received_shared_map_cb(self, msg: OccupancyGrid, sender_id: int):
        if sender_id not in self.in_comm_robots:
            return

        h, w = msg.info.height, msg.info.width
        data = np.array(msg.data, dtype=np.int8).reshape((h, w))
        # spawn offset 보정: 상대 로봇 좌표계를 자기 좌표계로 변환
        my_spawn = SPAWN_POSES[self.robot_id]
        sender_spawn = SPAWN_POSES[sender_id]
        offset_x = sender_spawn[0] - my_spawn[0]
        offset_y = sender_spawn[1] - my_spawn[1]
        info = {
            'height': h, 'width': w,
            'resolution': msg.info.resolution,
            'origin_x': msg.info.origin.position.x + offset_x,
            'origin_y': msg.info.origin.position.y + offset_y,
        }
        self.received_maps[sender_id] = {'data': data, 'info': info}
        self.get_logger().info(f'[{self.ns}] tb3_{sender_id}의 맵 수신 ({w}x{h})')

        now = _time.time()
        if (now - self.last_merge_time) < MERGE_COOLDOWN:
            return
        self.last_merge_time = now
        participants = sorted([self.robot_id] + list(self.in_comm_robots))
        if participants[0] == self.robot_id:
            self._run_kmeans_and_assign(participants)

    def _rv_command_cb(self, msg: Bool):
        if msg.data and self.state == 'EXPLORE':
            self.state = 'RENDEZVOUS'
            self.rv_start_time = self.get_clock().now()
            self.get_logger().info(f'[{self.ns}] 랑데뷰 명령 수신 → RENDEZVOUS')

    def _kmeans_assignment_cb(self, msg: String):
        try:
            data = json.loads(msg.data)
            leader = data['leader']
            if leader == self.robot_id:
                return
            participants = data['participants']
            if self.robot_id not in participants:
                return
            if leader not in self.in_comm_robots:
                return
            centers = [tuple(c) for c in data['centers']]
            zones_data = data.get('zones', {})
            my_idx = participants.index(self.robot_id)
            my_zone_key = str(self.robot_id)
            if my_zone_key in zones_data:
                self.zone_cells = set(tuple(cell) for cell in zones_data[my_zone_key])
            if my_idx < len(centers):
                self.planner.cluster_centers[self.robot_id] = centers[my_idx]
            self.get_logger().info(
                f'[{self.ns}] K-means 할당 수신 (leader=tb3_{leader}, 내 영역: {len(self.zone_cells)}셀)')
        except Exception as e:
            self.get_logger().warn(f'[{self.ns}] K-means 할당 파싱 실패: {e}')

    # ══════════════════════════════════════════════════════════
    # 자체 맵 합성 (map_merger.py)
    # ══════════════════════════════════════════════════════════

    def _do_merge(self):
        if self.local_map is None or self.local_info is None:
            return

        my_map_dict = {
            'data': self.local_map,
            'info': {
                'height': self.local_info.height,
                'width': self.local_info.width,
                'resolution': self.local_info.resolution,
                'origin_x': self.local_info.origin.position.x,
                'origin_y': self.local_info.origin.position.y,
            }
        }
        local_maps = [my_map_dict]
        for rid, rmap in self.received_maps.items():
            local_maps.append(rmap)

        ref_info = self._compute_ref_info(local_maps)
        merged = self.merger.merge(local_maps, ref_info)
        self.merged_map = merged

        # merged_info를 MapMetaData로 저장
        from nav_msgs.msg import MapMetaData
        mi = MapMetaData()
        mi.height = ref_info['height']
        mi.width = ref_info['width']
        mi.resolution = ref_info['resolution']
        mi.origin.position.x = ref_info['origin_x']
        mi.origin.position.y = ref_info['origin_y']
        self.merged_info = mi

        self._publish_merged_map()

    def _compute_ref_info(self, local_maps):
        min_x, min_y = float('inf'), float('inf')
        max_x, max_y = float('-inf'), float('-inf')
        res = None
        for lm in local_maps:
            if lm is None:
                continue
            info = lm['info']
            if res is None:
                res = info['resolution']
            ox, oy = info['origin_x'], info['origin_y']
            max_x = max(max_x, ox + info['width'] * info['resolution'])
            max_y = max(max_y, oy + info['height'] * info['resolution'])
            min_x = min(min_x, ox)
            min_y = min(min_y, oy)
        if res is None:
            res = 0.05
        width = int(np.ceil((max_x - min_x) / res))
        height = int(np.ceil((max_y - min_y) / res))
        return {
            'height': height, 'width': width, 'resolution': res,
            'origin_x': min_x, 'origin_y': min_y,
        }

    def _publish_merged_map(self):
        if self.merged_map is None or self.merged_info is None:
            return
        msg = OccupancyGrid()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        msg.info = self.merged_info
        msg.data = self.merged_map.flatten().tolist()
        self.merged_map_pub.publish(msg)

    # ══════════════════════════════════════════════════════════
    # 통신 범위 모니터링
    # ══════════════════════════════════════════════════════════

    def _monitor_comm(self):
        self._update_my_pose()
        self.in_comm_robots.clear()
        for i in range(self.num_robots):
            if i == self.robot_id:
                continue
            ns = self.all_ns[i]
            try:
                t = self.tf_buffer.lookup_transform(
                    'map', f'{ns}/base_footprint',
                    rclpy.time.Time(),
                    timeout=rclpy.duration.Duration(seconds=0.2))
                ox = t.transform.translation.x
                oy = t.transform.translation.y
                self.other_positions[i] = (ox, oy)
                dist = math.hypot(self.my_x - ox, self.my_y - oy)
                if dist <= COMM_RANGE:
                    self.in_comm_robots.add(i)
            except Exception:
                pass

    def _update_my_pose(self):
        try:
            t = self.tf_buffer.lookup_transform(
                'map', f'{self.ns}/base_footprint',
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.2))
            self.my_x = t.transform.translation.x
            self.my_y = t.transform.translation.y
        except Exception:
            pass

    # ══════════════════════════════════════════════════════════
    # 맵 broadcast
    # ══════════════════════════════════════════════════════════

    def _publish_shared_map(self):
        if self.local_map is None or self.local_info is None:
            return
        msg = OccupancyGrid()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        msg.info = self.local_info
        msg.data = self.local_map.flatten().tolist()
        self.shared_map_pub.publish(msg)

    def _run_kmeans_and_assign(self, participant_ids):
        if self.merged_map is None or self.merged_info is None:
            return
        res = self.merged_info.resolution
        ox = self.merged_info.origin.position.x
        oy = self.merged_info.origin.position.y

        robot_positions = []
        for rid in participant_ids:
            if rid == self.robot_id:
                robot_positions.append((self.my_x, self.my_y))
            elif rid in self.other_positions:
                robot_positions.append(self.other_positions[rid])
            else:
                robot_positions.append((0.0, 0.0))

        planner = ExplorationPlanner(num_robots=len(participant_ids))
        zones = planner.kmeans_partition(self.merged_map, res, ox, oy, robot_positions)

        my_idx = participant_ids.index(self.robot_id)
        self.zone_cells = set(zones[my_idx])
        self.planner.cluster_centers[self.robot_id] = planner.cluster_centers[my_idx]

        zones_by_robot = {}
        for idx, pid in enumerate(participant_ids):
            zone_list = [[int(c) for c in cell] for cell in zones[idx][:2000]]
            zones_by_robot[str(int(pid))] = zone_list

        centers_native = [[float(v) for v in c] for c in planner.cluster_centers]
        sizes_native = [int(len(z)) for z in zones]

        assignment_msg = String()
        assignment_msg.data = json.dumps({
            'leader': int(self.robot_id),
            'participants': [int(p) for p in participant_ids],
            'centers': centers_native,
            'sizes': sizes_native,
            'zones': zones_by_robot,
        })
        self.kmeans_pub.publish(assignment_msg)
        self.cluster_pub.publish(assignment_msg)
        self.get_logger().info(
            f'[{self.ns}] K-means 실행 + 할당 전파 (참여: {participant_ids}, 내 영역: {len(self.zone_cells)}셀)')

    # ══════════════════════════════════════════════════════════
    # FSM
    # ══════════════════════════════════════════════════════════

    def _tick(self):
        if self.state == 'INIT_SCAN':
            elapsed = (self.get_clock().now() - self.init_scan_start).nanoseconds / 1e9
            if elapsed > 8.0:
                self.state = 'EXPLORE'
                self.get_logger().info(f'[{self.ns}] INIT_SCAN → EXPLORE')
                msg = String()
                msg.data = 'EXPLORE'
                self.state_pub.publish(msg)
        elif self.state == 'RENDEZVOUS':
            rv_elapsed = (self.get_clock().now() - self.rv_start_time).nanoseconds / 1e9
            if rv_elapsed > RENDEZVOUS_TIMEOUT:
                self.state = 'EXPLORE'
                self.get_logger().warn(f'[{self.ns}] 랑데뷰 타임아웃 → EXPLORE 복귀')

    # ══════════════════════════════════════════════════════════
    # Frontier 계획 (local_map 기반)
    # ══════════════════════════════════════════════════════════

    def _plan(self):
        if self.state != 'EXPLORE':
            return
        if self.local_map is None or self.local_info is None:
            return
        if self.navigating:
            if self.current_goal is not None:
                self._check_goal_validity()
            return

        grid_frontiers = self.frontier_detector.detect(self.local_map)
        if not grid_frontiers:
            self.get_logger().info(f'[{self.ns}] Frontier 없음')
            return

        res = self.local_info.resolution
        ox = self.local_info.origin.position.x
        oy = self.local_info.origin.position.y
        h, w = self.local_map.shape

        world_f = self.frontier_detector.to_world(grid_frontiers, res, ox, oy)
        margin = 3
        world_f = [
            (wx, wy) for (wx, wy) in world_f
            if margin <= int((wy - oy) / res) < h - margin
            and margin <= int((wx - ox) / res) < w - margin
        ]
        if not world_f:
            return

        active_bl = self._get_active_blacklist()
        world_f_filtered = [
            (wx, wy) for (wx, wy) in world_f
            if (round(wx, 1), round(wy, 1)) not in active_bl
        ]
        if not world_f_filtered:
            self.failed_goals.clear()
            world_f_filtered = world_f

        self._publish_frontier_markers(world_f_filtered)
        self.current_frontiers = world_f_filtered

        robot_dict = {
            'id': self.robot_id, 'x': self.my_x, 'y': self.my_y,
            'navigating': False, 'visited': self.visited,
            'zone_cells': self.zone_cells, 'failed_goals': active_bl,
        }
        if self.merged_info is not None:
            map_info_dict = {
                'resolution': self.merged_info.resolution,
                'origin_x': self.merged_info.origin.position.x,
                'origin_y': self.merged_info.origin.position.y,
            }
        else:
            map_info_dict = {'resolution': res, 'origin_x': ox, 'origin_y': oy}

        assignments = self.planner.assign_targets(world_f_filtered, [robot_dict], map_info_dict)
        if self.robot_id not in assignments:
            return

        target = assignments[self.robot_id]

        grid = np.zeros_like(self.local_map, dtype=np.uint8)
        grid[self.local_map > 50] = 1
        start = (int((self.my_y - oy) / res), int((self.my_x - ox) / res))
        goal = (int((target[1] - oy) / res), int((target[0] - ox) / res))

        if not (0 <= start[0] < h and 0 <= start[1] < w and
                0 <= goal[0] < h and 0 <= goal[1] < w):
            return

        path = astar(grid, start, goal)
        if path:
            astar_world = grid_path_to_world(path, res, ox, oy)
            self.astar_path = astar_world
            self._publish_astar_path(astar_world)
            self._navigate_to(target)
        else:
            key = (round(target[0], 1), round(target[1], 1))
            self.failed_goals[key] = _time.time() + BLACKLIST_TTL

    def _check_goal_validity(self):
        if self.local_map is None or self.current_goal is None:
            return
        res = self.local_info.resolution
        ox = self.local_info.origin.position.x
        oy = self.local_info.origin.position.y
        h, w = self.local_map.shape
        gx, gy = self.current_goal
        gc = int((gx - ox) / res)
        gr = int((gy - oy) / res)
        if 0 <= gr < h and 0 <= gc < w:
            has_unknown = False
            for dr in range(-5, 6):
                for dc in range(-5, 6):
                    nr, nc = gr + dr, gc + dc
                    if 0 <= nr < h and 0 <= nc < w:
                        if self.local_map[nr, nc] == -1:
                            has_unknown = True
                            break
                if has_unknown:
                    break
            if not has_unknown:
                self._cancel_goal()
                self.get_logger().info(f'[{self.ns}] 목표 이미 탐사됨 → 재계획')

    # ══════════════════════════════════════════════════════════
    # Nav2 이동
    # ══════════════════════════════════════════════════════════

    def _navigate_to(self, target):
        if not self.nav_client.wait_for_server(timeout_sec=0.5):
            return
        goal = NavigateToPose.Goal()
        goal.pose.header.frame_id = 'map'
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.pose.position.x = float(target[0])
        goal.pose.pose.position.y = float(target[1])
        goal.pose.pose.orientation.w = 1.0
        future = self.nav_client.send_goal_async(goal)
        future.add_done_callback(self._goal_response_cb)
        self.current_goal = target
        self.navigating = True

    def _goal_response_cb(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.navigating = False
            self.current_goal = None
            return
        self._goal_handle = goal_handle
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._goal_result_cb)

    def _goal_result_cb(self, future):
        self._goal_handle = None
        self._update_my_pose()
        if self.current_goal:
            dist = math.hypot(self.my_x - self.current_goal[0], self.my_y - self.current_goal[1])
            if dist < 1.0:
                self.visited.append(self.current_goal)
            else:
                key = (round(self.current_goal[0], 1), round(self.current_goal[1], 1))
                self.failed_goals[key] = _time.time() + BLACKLIST_TTL
            self.current_goal = None
        self.navigating = False

    def _cancel_goal(self):
        if self._goal_handle is not None:
            self._goal_handle.cancel_goal_async()
        self._goal_handle = None
        self.navigating = False
        self.current_goal = None

    # ══════════════════════════════════════════════════════════
    # 유틸리티
    # ══════════════════════════════════════════════════════════

    def _get_active_blacklist(self) -> set:
        now = _time.time()
        expired = [k for k, t in self.failed_goals.items() if now >= t]
        for k in expired:
            del self.failed_goals[k]
        return set(self.failed_goals.keys())

    def _sample_trajectory(self):
        self._update_my_pose()
        if (not self.trajectory or
                math.hypot(self.my_x - self.trajectory[-1][0],
                           self.my_y - self.trajectory[-1][1]) > 0.05):
            self.trajectory.append((self.my_x, self.my_y))
        self._publish_path()

    def _publish_path(self):
        msg = Path()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        for (px, py) in self.trajectory:
            ps = PoseStamped()
            ps.header = msg.header
            ps.pose.position.x = float(px)
            ps.pose.position.y = float(py)
            ps.pose.orientation.w = 1.0
            msg.poses.append(ps)
        self.path_pub.publish(msg)

    def _publish_frontier_markers(self, frontiers):
        arr = MarkerArray()
        delete = Marker()
        delete.action = 3
        delete.header.frame_id = 'map'
        delete.ns = 'frontiers'
        arr.markers.append(delete)
        for idx, (fx, fy) in enumerate(frontiers):
            m = Marker()
            m.header.frame_id = 'map'
            m.header.stamp = self.get_clock().now().to_msg()
            m.ns = 'frontiers'
            m.id = idx
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x = fx
            m.pose.position.y = fy
            m.scale.x = m.scale.y = m.scale.z = 0.15
            m.color.r = 1.0
            m.color.g = 0.5
            m.color.a = 1.0
            arr.markers.append(m)
        self.frontier_pub.publish(arr)

    def _publish_astar_path(self, astar_world):
        msg = Path()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        for (px, py) in astar_world:
            ps = PoseStamped()
            ps.header = msg.header
            ps.pose.position.x = float(px)
            ps.pose.position.y = float(py)
            ps.pose.orientation.w = 1.0
            msg.poses.append(ps)
        self.astar_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = RobotAgentNode()
    rclpy.spin(node)
    rclpy.shutdown()
