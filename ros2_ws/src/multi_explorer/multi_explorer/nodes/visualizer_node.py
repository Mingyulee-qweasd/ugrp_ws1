"""
nodes/visualizer_node.py (decentralized 버전)

4패널: 각 로봇 merged_map (frontier + 경로 + A*) + 모니터링 맵
"""
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid, Path
from std_msgs.msg import String
from visualization_msgs.msg import MarkerArray
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy
import numpy as np
import os
import json
from datetime import datetime

from multi_explorer.perception.map_visualizer import MapRenderer
from multi_explorer.perception.map_merger import MapMerger

NUM_ROBOTS = 3
ROBOT_NS = ['tb3_0', 'tb3_1', 'tb3_2']
SAVE_DIR = '/ros2_ws/map_outputs'


class VisualizerNode(Node):
    def __init__(self):
        super().__init__('visualizer_node')

        self.renderer = MapRenderer(num_robots=NUM_ROBOTS)
        self.merger = MapMerger()

        self.robot_merged_maps = [None] * NUM_ROBOTS
        self.robot_merged_infos = [None] * NUM_ROBOTS
        self.local_maps = [None] * NUM_ROBOTS
        self.local_infos = [None] * NUM_ROBOTS

        self.robot_frontiers = [[] for _ in range(NUM_ROBOTS)]
        self.path_accum = [[] for _ in range(NUM_ROBOTS)]
        self.astar_paths = [[] for _ in range(NUM_ROBOTS)]
        self.robot_pos = [(0.0, 0.0)] * NUM_ROBOTS
        self.cluster_centers = None

        os.makedirs(SAVE_DIR, exist_ok=True)

        map_qos = QoSProfile(
            depth=1,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            reliability=QoSReliabilityPolicy.RELIABLE)

        for i, ns in enumerate(ROBOT_NS):
            self.create_subscription(
                OccupancyGrid, f'/{ns}/merged_map',
                lambda msg, idx=i: self._merged_map_cb(msg, idx), map_qos)
            self.create_subscription(
                OccupancyGrid, f'/{ns}/map',
                lambda msg, idx=i: self._local_map_cb(msg, idx), 10)
            self.create_subscription(
                MarkerArray, f'/{ns}/frontiers',
                lambda msg, idx=i: self._frontier_cb(msg, idx), 10)
            self.create_subscription(
                Path, f'/{ns}/astar_path',
                lambda msg, idx=i: self._astar_cb(msg, idx), 10)
            self.create_subscription(
                Path, f'/visited_path/{ns}',
                lambda msg, idx=i: self._path_cb(msg, idx), 10)
            self.create_subscription(
                String, f'/{ns}/cluster_info',
                self._cluster_info_cb, 10)

        self.create_timer(10.0, self._save_snapshot)
        self.get_logger().info(f'Visualizer Node (decentralized) 시작 — 저장: {SAVE_DIR}')

    def _merged_map_cb(self, msg, idx):
        h, w = msg.info.height, msg.info.width
        self.robot_merged_maps[idx] = np.array(msg.data, dtype=np.int8).reshape((h, w))
        self.robot_merged_infos[idx] = msg.info

    def _local_map_cb(self, msg, idx):
        h, w = msg.info.height, msg.info.width
        self.local_maps[idx] = np.array(msg.data, dtype=np.int8).reshape((h, w))
        self.local_infos[idx] = msg.info

    def _frontier_cb(self, msg, idx):
        pts = []
        for m in msg.markers:
            if m.action == 3:
                continue
            if m.ns == 'frontiers':
                pts.append((m.pose.position.x, m.pose.position.y))
        self.robot_frontiers[idx] = pts

    def _astar_cb(self, msg, idx):
        self.astar_paths[idx] = [(ps.pose.position.x, ps.pose.position.y)
                                  for ps in msg.poses]

    def _path_cb(self, msg, idx):
        pts = [(ps.pose.position.x, ps.pose.position.y) for ps in msg.poses]
        self.path_accum[idx] = pts
        if pts:
            self.robot_pos[idx] = pts[-1]

    def _cluster_info_cb(self, msg):
        try:
            data = json.loads(msg.data)
            self.cluster_centers = [tuple(c) for c in data['centers']]
        except Exception:
            pass

    def _save_snapshot(self):
        if not any(m is not None for m in self.robot_merged_maps):
            if not any(m is not None for m in self.local_maps):
                self.get_logger().warn('맵 없음 — 대기')
                return

        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        ms = datetime.now().microsecond // 1000
        filepath = f'{SAVE_DIR}/snapshot_{ts}_{ms:04d}.png'

        robot_map_dicts = []
        for i in range(NUM_ROBOTS):
            mdata = self.robot_merged_maps[i]
            minfo = self.robot_merged_infos[i]
            if mdata is None:
                mdata = self.local_maps[i]
                minfo = self.local_infos[i]
            if mdata is not None and minfo is not None:
                robot_map_dicts.append({
                    'data': mdata,
                    'resolution': minfo.resolution,
                    'origin_x': minfo.origin.position.x,
                    'origin_y': minfo.origin.position.y,
                })
            else:
                robot_map_dicts.append(None)

        global_map, global_res, global_ox, global_oy = self._build_monitoring_map()

        self.renderer.render_decentralized(
            filepath=filepath,
            robot_maps=robot_map_dicts,
            global_map=global_map,
            global_res=global_res,
            global_ox=global_ox,
            global_oy=global_oy,
            path_accum=self.path_accum,
            astar_paths=self.astar_paths,
            robot_positions=self.robot_pos,
            cluster_centers=self.cluster_centers,
            robot_frontiers=self.robot_frontiers,
        )
        self.get_logger().info(f'저장: {filepath}')

    def _build_monitoring_map(self):
        valid = [i for i in range(NUM_ROBOTS)
                 if self.local_maps[i] is not None and self.local_infos[i] is not None]
        if not valid:
            return None, 0.05, 0.0, 0.0

        # 가장 큰 맵을 ref로 사용
        best_idx = valid[0]
        best_size = 0
        for i in valid:
            sz = self.local_infos[i].height * self.local_infos[i].width
            if sz > best_size:
                best_size = sz
                best_idx = i

        ref_info_obj = self.local_infos[best_idx]
        ref_info = {
            'height': ref_info_obj.height, 'width': ref_info_obj.width,
            'resolution': ref_info_obj.resolution,
            'origin_x': ref_info_obj.origin.position.x,
            'origin_y': ref_info_obj.origin.position.y,
        }
        local_dicts = []
        for i in range(NUM_ROBOTS):
            if self.local_maps[i] is not None and self.local_infos[i] is not None:
                info = self.local_infos[i]
                local_dicts.append({
                    'data': self.local_maps[i],
                    'info': {
                        'height': info.height, 'width': info.width,
                        'resolution': info.resolution,
                        'origin_x': info.origin.position.x,
                        'origin_y': info.origin.position.y,
                    }
                })
            else:
                local_dicts.append(None)
        merged = self.merger.merge(local_dicts, ref_info)
        return merged, ref_info['resolution'], ref_info['origin_x'], ref_info['origin_y']


def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(VisualizerNode())
    rclpy.shutdown()
