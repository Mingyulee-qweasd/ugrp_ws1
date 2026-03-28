"""
nodes/map_merger_node.py

ROS2 노드: 각 로봇 맵 subscribe → merge → 글로벌맵 publish
순수 로직은 perception/map_merger.py, core/rendezvous_manager.py 에 위임.
"""
#!/usr/bin/env python3
import rclpy
import time as _time
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Bool
from tf2_ros import Buffer, TransformListener
import numpy as np
import rclpy.duration
import rclpy.time
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy

from multi_explorer.perception.map_merger import MapMerger
from multi_explorer.core.rendezvous_manager import RendezvousManager

NUM_ROBOTS = 3
ROBOT_NS = ['tb3_0', 'tb3_1', 'tb3_2']


class MapMergerNode(Node):
    def __init__(self):
        super().__init__('map_merger_node')

        self.merger = MapMerger()
        self.rv_manager = RendezvousManager(
            merge_distance=3.0, rendezvous_period=60.0)

        self.local_maps = [None] * NUM_ROBOTS
        self.robot_pos = [(0.0, 0.0)] * NUM_ROBOTS
        self.last_merge_t = self.get_clock().now()
        self.rv_sent = False
        self.first_merge_done = False
        self.start_time = _time.time()

        # TF
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # 각 로봇 맵 구독
        self.map_subs = []
        for i, ns in enumerate(ROBOT_NS):
            sub = self.create_subscription(
                OccupancyGrid, f'/{ns}/map',
                lambda msg, idx=i: self._map_cb(msg, idx), 10)
            self.map_subs.append(sub)

        # 공유맵 퍼블리셔
        map_qos = QoSProfile(
            depth=1,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            reliability=QoSReliabilityPolicy.RELIABLE)
        self.map_pub = self.create_publisher(OccupancyGrid, '/map', map_qos)

        # merge 이벤트 + 집결 명령
        self.event_pub = self.create_publisher(Bool, '/merge_event', 10)
        self.rv_pub = self.create_publisher(Bool, '/rendezvous_command', 10)

        self.create_timer(2.0, self._check_merge)
        self.get_logger().info('Map Merger Node 시작')

    def _map_cb(self, msg: OccupancyGrid, robot_idx: int):
        self.local_maps[robot_idx] = msg

    def _get_robot_positions(self):
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

        now = self.get_clock().now()
        elapsed = (now - self.last_merge_t).nanoseconds / 1e9

        proximity_merge = self.rv_manager.check_proximity(self.robot_pos)
        force_merge = self.rv_manager.check_period(elapsed)

        # 주기 도달 → 집결 명령만 전송
        if force_merge and not self.rv_sent:
            self.rv_pub.publish(Bool(data=True))
            self.rv_sent = True
            self.get_logger().info('집결 명령 전송 (merge 없이 명령만)')

        MAP_PUBLISH_COOLDOWN = 5.0
        MERGE_EVENT_COOLDOWN = 10.0

        if elapsed >= MAP_PUBLISH_COOLDOWN and any(m is not None for m in self.local_maps):
            self._do_merge(publish_event=False)
            self.last_merge_t = now

            # 첫 merge_event: 시작 후 12초 이후 (INIT_SCAN 8초 + 여유)
            if not self.first_merge_done and (_time.time() - self.start_time) > 12.0:
                self.first_merge_done = True
                self.event_pub.publish(Bool(data=True))
                self.get_logger().info('첫 merge → merge_event 발행')
                return

            # 이후: proximity + 쿨다운일 때만 merge_event
            if self.first_merge_done and proximity_merge and elapsed >= MERGE_EVENT_COOLDOWN:
                self.event_pub.publish(Bool(data=True))
                self.rv_sent = False
                self.get_logger().info('로봇 근접 → merge_event 발행 (K-means 트리거)')

    def _do_merge(self, publish_event=False):
        """local_maps → 순수 merger 호출 → 공유맵 퍼블리시."""
        valid = [m for m in self.local_maps if m is not None]
        if not valid:
            return

        ref = valid[0]
        ref_info = {
            'height': ref.info.height,
            'width': ref.info.width,
            'resolution': ref.info.resolution,
            'origin_x': ref.info.origin.position.x,
            'origin_y': ref.info.origin.position.y,
        }

        local_dicts = []
        for lm in self.local_maps:
            if lm is None:
                local_dicts.append(None)
                continue
            local_dicts.append({
                'data': np.array(lm.data, dtype=np.int8).reshape(
                    (lm.info.height, lm.info.width)),
                'info': {
                    'height': lm.info.height,
                    'width': lm.info.width,
                    'resolution': lm.info.resolution,
                    'origin_x': lm.info.origin.position.x,
                    'origin_y': lm.info.origin.position.y,
                }
            })

        merged = self.merger.merge(local_dicts, ref_info)

        msg = OccupancyGrid()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        msg.info = ref.info
        msg.data = merged.flatten().tolist()
        self.map_pub.publish(msg)

        if publish_event:
            self.event_pub.publish(Bool(data=True))

        self.get_logger().info(
            f'공유맵 퍼블리시 ({ref_info["height"]}x{ref_info["width"]})')


def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(MapMergerNode())
    rclpy.shutdown()
