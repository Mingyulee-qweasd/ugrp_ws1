#!/usr/bin/env python3
"""
Month 2 산출물: matplotlib으로
- OG Map (free/unknown/occupied)
- K-means 영역 분할 (로봇별 색깔)
- Frontier 포인트
- 로봇 이동 경로
를 PNG로 저장
"""

import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from visualization_msgs.msg import MarkerArray

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import datetime
import os

from sklearn.cluster import KMeans

SAVE_DIR   = '/ros2_ws/map_outputs'
NUM_ROBOTS = 1   # 현재 로봇 수 (2대/3대로 바꿀 때 여기만 수정)

# 로봇별 색깔 (matplotlib 색상)
ROBOT_COLORS = [
    (1.0, 0.3, 0.3),   # 빨강 — 로봇0
    (0.3, 0.3, 1.0),   # 파랑 — 로봇1
    (0.3, 0.8, 0.3),   # 초록 — 로봇2
]


class MapVisualizer(Node):

    def __init__(self):
        super().__init__('map_visualizer')

        self.map_data    = None
        self.map_info    = None
        self.frontiers   = []
        self.robot_paths_accum = [[] for _ in range(NUM_ROBOTS)]  # 전체 누적
        self.robot_paths_current = [[] for _ in range(NUM_ROBOTS)]  # 현재 A* 경로
        self.snapshot_count = 0

        os.makedirs(SAVE_DIR, exist_ok=True)

        # 맵 구독
        map_topic = '/map' if NUM_ROBOTS == 1 else '/tb3_0/map'
        self.map_sub = self.create_subscription(
            OccupancyGrid, map_topic, self.map_callback, 10)
            
        # frontier + 경로 마커 구독
        self.marker_sub = self.create_subscription(
            MarkerArray, '/exploration_markers',
            self.marker_callback, 10)

        # A* 경로 마커 구독
        self.path_sub = self.create_subscription(
            MarkerArray, '/astar_paths',
            self.path_callback, 10)

        # 10초마다 스냅샷 저장
        self.timer = self.create_timer(10.0, self.save_snapshot)

        self.get_logger().info(f'Map Visualizer 시작 — 저장: {SAVE_DIR}')

    def map_callback(self, msg: OccupancyGrid):
        self.map_info = msg.info
        h, w = msg.info.height, msg.info.width
        self.map_data = np.array(msg.data, dtype=np.int8).reshape((h, w))

    def marker_callback(self, msg: MarkerArray):
        """exploration_markers에서 frontier 위치 추출"""
        self.frontiers = [
            (m.pose.position.x, m.pose.position.y)
            for m in msg.markers
            if m.action != 3 and m.ns == 'frontiers'
        ]

    def path_callback(self, msg: MarkerArray):
        """astar_paths에서 로봇별 경로 포인트 추출 + 누적"""
        for m in msg.markers:
            if m.type == 4:  # LINE_STRIP
                try:
                    robot_id = int(m.ns.split('_')[-1])
                    if robot_id < NUM_ROBOTS:
                        new_pts = [(p.x, p.y) for p in m.points]
                        # 현재 A* 경로 업데이트
                        self.robot_paths_current[robot_id] = new_pts
                        # 누적 경로에 추가 (중복 제거)
                        if new_pts:
                            last = self.robot_paths_accum[robot_id]
                            if not last or last[-1] != new_pts[-1]:
                                self.robot_paths_accum[robot_id].extend(new_pts)
                except Exception:
                    pass

    def save_snapshot(self):
        if self.map_data is None:
            self.get_logger().warn('맵 없음 — 대기 중')
            return
        self.snapshot_count += 1
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        filepath  = f'{SAVE_DIR}/snapshot_{timestamp}_{self.snapshot_count:04d}.png'
        self._render(filepath)

    def _render(self, filepath):
        raw  = self.map_data
        info = self.map_info
        h, w = raw.shape
        res  = info.resolution
        ox   = info.origin.position.x
        oy   = info.origin.position.y

        # ── K-means 영역 계산 ─────────────────────────────────
        zones = self._compute_zones(raw)

        # ── 베이스 맵 이미지 ──────────────────────────────────
        # unknown=회색, free=흰색, occupied=검정
        base_rgb = np.full((h, w, 3), 0.5)          # 전체 회색(unknown)
        base_rgb[(raw >= 0) & (raw <= 50)] = 1.0     # free → 흰색
        base_rgb[raw > 50] = 0.0                     # occupied → 검정

        # ── K-means 영역 오버레이 ─────────────────────────────
        # free 영역에만 반투명 색깔 덮기
        overlay = base_rgb.copy()
        for k, zone_mask in enumerate(zones):
            c = ROBOT_COLORS[k % len(ROBOT_COLORS)]
            # zone_mask: bool 2D array
            free_zone = zone_mask & ((raw >= 0) & (raw <= 50))
            overlay[free_zone, 0] = c[0] * 0.4 + base_rgb[free_zone, 0] * 0.6
            overlay[free_zone, 1] = c[1] * 0.4 + base_rgb[free_zone, 1] * 0.6
            overlay[free_zone, 2] = c[2] * 0.4 + base_rgb[free_zone, 2] * 0.6

        # ── 플롯 ─────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(10, 10))
        extent = [ox, ox + w*res, oy, oy + h*res]
        ax.imshow(overlay, origin='lower', extent=extent, vmin=0, vmax=1)

        # Frontier 포인트
        if self.frontiers:
            fx = [f[0] for f in self.frontiers]
            fy = [f[1] for f in self.frontiers]
            ax.scatter(fx, fy, c='orange', s=60, zorder=5,
                       edgecolors='black', linewidths=0.5,
                       label=f'Frontier ({len(self.frontiers)}개)')

        # 로봇별 이동 경로
        for k in range(NUM_ROBOTS):
            # 누적 전체 경로 (얇게)
            accum = self.robot_paths_accum[k]
            if len(accum) >= 2:
                px = [p[0] for p in accum]
                py = [p[1] for p in accum]
                c  = ROBOT_COLORS[k % len(ROBOT_COLORS)]
                ax.plot(px, py, color=c, linewidth=1.0,
                        alpha=0.4, zorder=5,
                        label=f'Robot {k} 전체 경로')

            # 현재 A* 경로 (굵게)
            path = self.robot_paths_current[k]
            if len(path) >= 2:
                px = [p[0] for p in path]
                py = [p[1] for p in path]
                c  = ROBOT_COLORS[k % len(ROBOT_COLORS)]
                ax.plot(px, py, color=c, linewidth=2.5,
                        alpha=0.9, zorder=6,
                        label=f'Robot {k} 현재 경로')
                # 현재 위치 별표
                ax.scatter(px[-1], py[-1], color=c, s=150,
                           zorder=7, marker='*',
                           edgecolors='black', linewidths=0.5)
        
        
        # ── 범례 ─────────────────────────────────────────────
        legend_patches = [
            mpatches.Patch(color='white',  label='Free'),
            mpatches.Patch(color='gray',   label='Unknown'),
            mpatches.Patch(color='black',  label='Occupied'),
            mpatches.Patch(color='orange', label='Frontier'),
        ]
        for k in range(NUM_ROBOTS):
            c = ROBOT_COLORS[k % len(ROBOT_COLORS)]
            legend_patches.append(
                mpatches.Patch(color=c, alpha=0.6,
                               label=f'Robot {k} 영역'))

        ax.legend(handles=legend_patches, loc='upper right', fontsize=10)
        ax.set_title(
            f'K-means 영역 분할 + Frontier  '
            f'(snapshot #{self.snapshot_count})',
            fontsize=13)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.grid(True, alpha=0.2)

        plt.tight_layout()
        plt.savefig(filepath, dpi=150)
        plt.close()
        self.get_logger().info(f'저장: {filepath}')

    def _compute_zones(self, raw):
        """
        K-means로 free 영역을 NUM_ROBOTS개로 분할
        반환: [mask_0, mask_1, ...] — 각각 (h,w) bool 배열
        """
        h, w = raw.shape
        zones = [np.zeros((h, w), dtype=bool) for _ in range(NUM_ROBOTS)]

        if NUM_ROBOTS == 1:
            zones[0][(raw >= 0) & (raw <= 50)] = True
            return zones

        # free 셀 좌표 추출
        free_pts = np.column_stack(np.where((raw >= 0) & (raw <= 50)))
        if len(free_pts) < NUM_ROBOTS:
            return zones

        # 샘플링
        if len(free_pts) > 3000:
            idx = np.random.choice(len(free_pts), 3000, replace=False)
            sample = free_pts[idx]
        else:
            sample = free_pts

        kmeans = KMeans(n_clusters=NUM_ROBOTS, n_init=5, random_state=42)
        kmeans.fit(sample)

        # 전체 free 셀에 레이블 예측
        all_labels = kmeans.predict(free_pts)
        for i, (r, c) in enumerate(free_pts):
            zones[all_labels[i]][r, c] = True

        return zones


def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(MapVisualizer())
    rclpy.shutdown()
