#!/usr/bin/env python3
"""
map_merger.py

역할:
  - 각 로봇의 /tb3_N/map 을 구독
  - 로봇 간 거리가 MERGE_DISTANCE 이하면 merge 트리거
  - merge된 공유맵을 /map 으로 퍼블리시
  - exploration_planner 에게 /merge_event (std_msgs/Bool) 신호 전송
  - 주기 도달 시 /rendezvous_command 신호 전송
"""
import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Bool
from geometry_msgs.msg import PoseStamped
from tf2_ros import Buffer, TransformListener
import numpy as np
import rclpy.duration, rclpy.time
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy

NUM_ROBOTS      = 3
ROBOT_NS        = ['tb3_0', 'tb3_1', 'tb3_2']
MERGE_DISTANCE  = 2.0   # 이 거리 이하면 merge
RENDEZVOUS_PERIOD = 60.0  # 강제 집결 주기 (초)

class MapMerger(Node):
    def __init__(self):
        super().__init__('map_merger')

        self.local_maps   = [None] * NUM_ROBOTS   # 각 로봇의 최신 맵
        self.robot_pos    = [(0.0, 0.0)] * NUM_ROBOTS
        self.merged_map   = None
        self.last_merge_t = self.get_clock().now()
        #추가한 부분
        self.rv_sent      = False  # 같은 주기에 집결 명령 중복 전송 방지

        # TF
        self.tf_buffer   = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # 각 로봇 맵 구독
        self.map_subs = []
        for i, ns in enumerate(ROBOT_NS):
            sub = self.create_subscription(
                OccupancyGrid, f'/{ns}/map',
                lambda msg, idx=i: self._map_cb(msg, idx), 10)
            self.map_subs.append(sub)

        # 공유맵 퍼블리셔 (transient_local - 늦게 구독해도 받을 수 있게)
        map_qos = QoSProfile(
            depth=1,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            reliability=QoSReliabilityPolicy.RELIABLE)
        self.map_pub   = self.create_publisher(OccupancyGrid, '/map', map_qos)

        # merge 이벤트 신호
        self.event_pub = self.create_publisher(Bool, '/merge_event', 10)

        #추가한 부분
        # 집결 명령 퍼블리셔 (state_machine 이 구독)
        #추가한 부분
        self.rv_pub = self.create_publisher(Bool, '/rendezvous_command', 10)

        # 주기 체크 타이머
        self.create_timer(2.0, self._check_merge)
        self.get_logger().info('Map Merger 시작')

    def _map_cb(self, msg: OccupancyGrid, robot_idx: int):
        self.local_maps[robot_idx] = msg

    def _get_robot_positions(self):
        """TF에서 각 로봇 위치 조회"""
        for i, ns in enumerate(ROBOT_NS):
            try:
                t = self.tf_buffer.lookup_transform(
                    'map', f'{ns}/base_footprint',
                    rclpy.time.Time(),
                    timeout=rclpy.duration.Duration(seconds=0.3))
                self.robot_pos[i] = (
                    t.transform.translation.x,
                    t.transform.translation.y)
            except Exception:
                pass

    def _check_merge(self):
        self._get_robot_positions()

        # 강제 집결 주기 체크
        now   = self.get_clock().now()
        elapsed = (now - self.last_merge_t).nanoseconds / 1e9
        force_merge = elapsed >= RENDEZVOUS_PERIOD

        # 거리 기반 merge 조건: 모든 로봇 쌍이 MERGE_DISTANCE 이내
        proximity_merge = True
        for i in range(NUM_ROBOTS):
            for j in range(i+1, NUM_ROBOTS):
                dx = self.robot_pos[i][0] - self.robot_pos[j][0]
                dy = self.robot_pos[i][1] - self.robot_pos[j][1]
                if (dx**2 + dy**2)**0.5 > MERGE_DISTANCE:
                    proximity_merge = False
                    break

        #추가한 부분
        # 주기 도달 시 집결 명령 전송 (아직 명령 안 보낸 경우만)
        #추가한 부분
        if force_merge and not self.rv_sent:
            #추가한 부분
            self.rv_pub.publish(Bool(data=True))
            #추가한 부분
            self.rv_sent = True
            #추가한 부분
            self.get_logger().info('집결 명령 전송 (/rendezvous_command)')

        if (proximity_merge or force_merge) and any(m is not None for m in self.local_maps):
            self.get_logger().info(
                f'Map Merge 트리거 (force={force_merge}, proximity={proximity_merge})')
            self._do_merge()
            self.last_merge_t = now
            #추가한 부분
            self.rv_sent = False  # 다음 주기를 위해 플래그 초기화

    def _do_merge(self):
        """
        사용 가능한 모든 local map을 OR-merge
        unknown(-1) < free(0~50) < occupied(51~100) 우선순위
        """
        valid_maps = [m for m in self.local_maps if m is not None]
        if not valid_maps:
            return

        # 기준 맵: 첫 번째 유효 맵의 메타데이터 사용
        ref = valid_maps[0]
        h   = ref.info.height
        w   = ref.info.width
        res = ref.info.resolution
        ox  = ref.info.origin.position.x
        oy  = ref.info.origin.position.y

        merged = np.full((h, w), -1, dtype=np.int8)  # 초기: 전부 unknown

        for local_map in valid_maps:
            lh = local_map.info.height
            lw = local_map.info.width
            lres = local_map.info.resolution
            lox  = local_map.info.origin.position.x
            loy  = local_map.info.origin.position.y

            local_arr = np.array(local_map.data, dtype=np.int8).reshape((lh, lw))

            for r in range(lh):
                for c in range(lw):
                    val = local_arr[r, c]
                    if val == -1:
                        continue
                    # 월드 좌표 → 공유맵 그리드 좌표
                    wx = lox + c * lres
                    wy = loy + r * lres
                    mc = int((wx - ox) / res)
                    mr = int((wy - oy) / res)
                    if 0 <= mr < h and 0 <= mc < w:
                        cur = merged[mr, mc]
                        # occupied 우선, 그 다음 free, unknown은 덮어쓰기
                        if cur == -1:
                            merged[mr, mc] = val
                        elif val > 50:          # occupied
                            merged[mr, mc] = val
                        elif val <= 50 and cur <= 50:
                            merged[mr, mc] = min(cur, val)

        # 공유맵 퍼블리시
        msg = OccupancyGrid()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        msg.info            = ref.info
        msg.data            = merged.flatten().tolist()
        self.merged_map     = msg
        self.map_pub.publish(msg)

        # merge 완료 이벤트
        self.event_pub.publish(Bool(data=True))
        self.get_logger().info(f'공유맵 퍼블리시 완료 ({h}x{w})')


def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(MapMerger())
    rclpy.shutdown()
