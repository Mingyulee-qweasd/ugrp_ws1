#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid, Path
from visualization_msgs.msg import MarkerArray
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from sklearn.cluster import KMeans
import os, math
from datetime import datetime

NUM_ROBOTS  = 3
SAVE_DIR    = '/ros2_ws/map_outputs'
COLORS      = ['red', 'blue', 'green']
ZONE_COLORS = [(220,150,150), (150,150,220), (150,220,150)]
ROBOT_STARTS = [(-7.0, 0.0), (7.0, 0.0), (0.0, 7.0)]

class MapVisualizer(Node):
    def __init__(self):
        super().__init__('map_visualizer')
        self.map_data   = None
        self.map_info   = None
        self.frontiers  = []
        # 각 로봇의 실제 이동 경로 (visited 기준)
        self.path_accum = [[] for _ in range(NUM_ROBOTS)]
        # 현재 A* 예정 경로
        self.astar_paths = [[] for _ in range(NUM_ROBOTS)]
        # 로봇 현재 위치
        self.robot_pos  = [ROBOT_STARTS[i] for i in range(NUM_ROBOTS)]

        os.makedirs(SAVE_DIR, exist_ok=True)

        map_qos = QoSProfile(depth=1,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            reliability=QoSReliabilityPolicy.RELIABLE)

        self.map_sub = self.create_subscription(
            OccupancyGrid, '/map', self.map_callback, map_qos)
        self.frontier_sub = self.create_subscription(
            MarkerArray, '/exploration_markers', self.frontier_callback, 10)
        self.astar_sub = self.create_subscription(
            MarkerArray, '/astar_paths', self.astar_callback, 10)
        # 각 로봇의 visited_path 구독
        self.path_subs = [
            self.create_subscription(
                Path, f'/visited_path/{ns}',
                lambda msg, i=i: self.path_callback(msg, i), 10)
            for i, ns in enumerate(['tb3_0','tb3_1','tb3_2'][:NUM_ROBOTS])]

        self.create_timer(10.0, self.save_snapshot)
        self.get_logger().info(f'Map Visualizer 시작 — 저장: {SAVE_DIR}')

    def map_callback(self, msg):
        self.map_info = msg.info
        h, w = msg.info.height, msg.info.width
        self.map_data = np.array(msg.data, dtype=np.int8).reshape((h, w))

    def frontier_callback(self, msg):
        self.frontiers = []
        for m in msg.markers:
            if m.action == 3: continue
            if m.ns == 'frontiers':
                self.frontiers.append((m.pose.position.x, m.pose.position.y))

    def astar_callback(self, msg):
        for m in msg.markers:
            if m.action == 3: continue
            if m.id < NUM_ROBOTS:
                self.astar_paths[m.id] = [(p.x, p.y) for p in m.points]

    def path_callback(self, msg, robot_id):
        pts = [(ps.pose.position.x, ps.pose.position.y) for ps in msg.poses]
        self.path_accum[robot_id] = pts
        if pts:
            self.robot_pos[robot_id] = pts[-1]

    def save_snapshot(self):
        if self.map_data is None:
            self.get_logger().warn('맵 없음 — 대기 중')
            return
        ts  = datetime.now().strftime('%Y%m%d_%H%M%S')
        ms  = datetime.now().microsecond // 1000
        self._render(f'{SAVE_DIR}/snapshot_{ts}_{ms:04d}.png')

    def _render(self, filepath):
        raw  = self.map_data
        info = self.map_info
        h, w = raw.shape
        res  = info.resolution
        ox   = info.origin.position.x
        oy   = info.origin.position.y

        # ── OG맵 RGB ──────────────────────────────────────────
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        rgb[raw == -1]                = [128,128,128]
        rgb[(raw >= 0)&(raw <= 50)]   = [255,255,255]
        rgb[raw > 50]                 = [0,  0,  0  ]

        # ── K-means 보로노이 영역 오버레이 ─────────────────────
        free_mask = (raw >= 0) & (raw <= 50)
        free_pts  = np.column_stack(np.where(free_mask))
        if len(free_pts) >= NUM_ROBOTS and NUM_ROBOTS > 1:
            sample = free_pts
            if len(free_pts) > 3000:
                idx    = np.random.choice(len(free_pts), 3000, replace=False)
                sample = free_pts[idx]
            try:
                km      = KMeans(n_clusters=NUM_ROBOTS, n_init=5, random_state=42)
                km.fit(sample)
                centers = km.cluster_centers_
                diff    = free_pts[:,None,:] - centers[None,:,:]
                dists   = np.sum(diff**2, axis=2)
                labels  = np.argmin(dists, axis=1)
                alpha   = 0.40
                for pt, label in zip(free_pts, labels):
                    c = ZONE_COLORS[label % len(ZONE_COLORS)]
                    rgb[pt[0], pt[1]] = [
                        int(255*(1-alpha) + c[0]*alpha),
                        int(255*(1-alpha) + c[1]*alpha),
                        int(255*(1-alpha) + c[2]*alpha)]
            except:
                pass

        # ── 플롯 ──────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.imshow(rgb, origin='lower',
                  extent=[ox, ox+w*res, oy, oy+h*res])

        # frontier 표시
        if self.frontiers:
            fx = [f[0] for f in self.frontiers]
            fy = [f[1] for f in self.frontiers]
            ax.scatter(fx, fy, c='orange', s=250, zorder=7,
                       edgecolors='black', linewidths=1.5, marker='*')

        # 실제 이동 경로: 얇은 실선 (visited 포인트 연결)
        for i in range(NUM_ROBOTS):
            pts = self.path_accum[i]
            if len(pts) >= 2:
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                ax.plot(xs, ys,
                        color=COLORS[i % len(COLORS)],
                        linewidth=1.2, alpha=0.8, zorder=4)

        # A* 예정 경로: 굵은 실선
        for i in range(NUM_ROBOTS):
            pts = self.astar_paths[i]
            if len(pts) >= 2:
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                ax.plot(xs, ys,
                        color=COLORS[i % len(COLORS)],
                        linewidth=3.0, alpha=0.9, zorder=6)

        # 로봇 현재 위치: 원
        for i in range(NUM_ROBOTS):
            rx, ry = self.robot_pos[i]
            ax.scatter([rx], [ry],
                       c=COLORS[i % len(COLORS)],
                       s=200, zorder=8, marker='o',
                       edgecolors='white', linewidths=2.0)
            ax.annotate(f'R{i}', (rx, ry),
                        textcoords='offset points', xytext=(6,6),
                        fontsize=10, fontweight='bold',
                        color=COLORS[i % len(COLORS)])

        # 범례
        legend_elements = [
            mpatches.Patch(facecolor='white',  edgecolor='gray', label='Free'),
            mpatches.Patch(color='gray',  label='Unknown'),
            mpatches.Patch(color='black', label='Occupied'),
            Line2D([0],[0], marker='*', color='orange', markersize=12,
                   label=f'Frontier ({len(self.frontiers)})',
                   linestyle='None', markeredgecolor='black'),
        ]
        for i in range(NUM_ROBOTS):
            legend_elements += [
                Line2D([0],[0], color=COLORS[i%len(COLORS)],
                       linewidth=1.2, label=f'Robot {i} 경로'),
                Line2D([0],[0], color=COLORS[i%len(COLORS)],
                       linewidth=3.0, label=f'Robot {i} A*'),
            ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=8)
        ax.set_title(f'Multi-Robot Exploration ({NUM_ROBOTS} robots)', fontsize=14)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(filepath, dpi=150)
        plt.close()
        self.get_logger().info(f'저장: {filepath}')

def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(MapVisualizer())
    rclpy.shutdown()
