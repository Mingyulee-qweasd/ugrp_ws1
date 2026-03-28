"""
perception/map_visualizer.py

맵 시각화 순수 로직 (ROS 의존 없음)

수정: 독립 K-means 제거 → explorer_node에서 전달받은 cluster_centers로
      보로노이 영역을 그림 (스냅샷마다 색깔이 바뀌던 문제 해결)

오픈소스 대응: plotInline.m, shade.m
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

COLORS = ['red', 'blue', 'green']
ZONE_COLORS = [(220, 150, 150), (150, 150, 220), (150, 220, 150)]


class MapRenderer:
    """맵 + 로봇 경로 + frontier 를 이미지로 렌더링."""

    def __init__(self, num_robots=3):
        self.num_robots = num_robots

    def render(self, filepath: str, map_data: np.ndarray,
               resolution: float, origin_x: float, origin_y: float,
               frontiers: list = None,
               path_accum: list = None,
               astar_paths: list = None,
               robot_positions: list = None,
               cluster_centers: list = None):
        """
        스냅샷을 파일로 저장.

        Args:
            cluster_centers: [(wx, wy), ...] explorer_node에서 받은 K-means 중심
                             None이면 보로노이 오버레이 안 그림
        """
        h, w = map_data.shape
        ox, oy, res = origin_x, origin_y, resolution
        frontiers = frontiers or []
        path_accum = path_accum or [[] for _ in range(self.num_robots)]
        astar_paths = astar_paths or [[] for _ in range(self.num_robots)]
        robot_positions = robot_positions or [(0, 0)] * self.num_robots

        # OG맵 RGB
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        rgb[map_data == -1] = [128, 128, 128]
        rgb[(map_data >= 0) & (map_data <= 50)] = [255, 255, 255]
        rgb[map_data > 50] = [0, 0, 0]

        # 보로노이 오버레이 — cluster_centers가 있을 때만, unknown 영역만
        if cluster_centers and len(cluster_centers) == self.num_robots:
            unk_mask = (map_data == -1)
            unk_pts = np.column_stack(np.where(unk_mask))  # (N, 2) row, col

            if len(unk_pts) > 0:
                # world 좌표의 cluster_centers를 grid 좌표로 변환
                grid_centers = np.array([
                    [(cy - oy) / res, (cx - ox) / res]
                    for (cx, cy) in cluster_centers
                ])  # (num_robots, 2) row, col

                # 각 unknown 셀에서 가장 가까운 cluster center 찾기 (보로노이)
                diff = unk_pts[:, None, :] - grid_centers[None, :, :]
                labels = np.argmin(np.sum(diff ** 2, axis=2), axis=1)

                alpha = 0.40
                for pt, label in zip(unk_pts, labels):
                    c = ZONE_COLORS[label % len(ZONE_COLORS)]
                    rgb[pt[0], pt[1]] = [
                        int(128 * (1 - alpha) + c[0] * alpha),
                        int(128 * (1 - alpha) + c[1] * alpha),
                        int(128 * (1 - alpha) + c[2] * alpha)]

        # 플롯
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.imshow(rgb, origin='lower',
                  extent=[ox, ox + w * res, oy, oy + h * res])

        # frontier
        if frontiers:
            fx = [f[0] for f in frontiers]
            fy = [f[1] for f in frontiers]
            ax.scatter(fx, fy, c='orange', s=250, zorder=7,
                       edgecolors='black', linewidths=1.5, marker='*')

        # 이동 경로 (실선)
        for i in range(self.num_robots):
            pts = path_accum[i]
            if len(pts) >= 2:
                ax.plot([p[0] for p in pts], [p[1] for p in pts],
                        color=COLORS[i % len(COLORS)],
                        linewidth=1.2, alpha=0.8, zorder=4)

        # A* 예정 경로 (굵은 실선)
        for i in range(self.num_robots):
            pts = astar_paths[i]
            if len(pts) >= 2:
                ax.plot([p[0] for p in pts], [p[1] for p in pts],
                        color=COLORS[i % len(COLORS)],
                        linewidth=3.0, alpha=0.9, zorder=6)

        # 로봇 위치
        for i in range(self.num_robots):
            rx, ry = robot_positions[i]
            ax.scatter([rx], [ry], c=COLORS[i % len(COLORS)],
                       s=200, zorder=8, marker='o',
                       edgecolors='white', linewidths=2.0)
            ax.annotate(f'R{i}', (rx, ry),
                        textcoords='offset points', xytext=(6, 6),
                        fontsize=10, fontweight='bold',
                        color=COLORS[i % len(COLORS)])

        # 범례
        legend_elements = [
            mpatches.Patch(facecolor='white', edgecolor='gray', label='Free'),
            mpatches.Patch(color='gray', label='Unknown'),
            mpatches.Patch(color='black', label='Occupied'),
            Line2D([0], [0], marker='*', color='orange', markersize=12,
                   label=f'Frontier ({len(frontiers)})',
                   linestyle='None', markeredgecolor='black'),
        ]
        for i in range(self.num_robots):
            legend_elements += [
                Line2D([0], [0], color=COLORS[i], linewidth=1.2,
                       label=f'Robot {i} path'),
                Line2D([0], [0], color=COLORS[i], linewidth=3.0,
                       label=f'Robot {i} A*'),
            ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=8)
        ax.set_title(f'Multi-Robot Exploration ({self.num_robots} robots)', fontsize=14)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(filepath, dpi=150)
        plt.close()

    def render_multi(self, filepath: str, global_map: np.ndarray,
                     resolution: float, origin_x: float, origin_y: float,
                     local_maps: list = None,
                     frontiers: list = None,
                     path_accum: list = None,
                     astar_paths: list = None,
                     robot_positions: list = None,
                     cluster_centers: list = None):
        """
        4패널 스냅샷: 로컬맵 3장 (좌상, 우상, 좌하) + 전역맵 1장 (우하).

        Args:
            global_map: 전역 merged 맵
            local_maps: [{'data': ndarray, 'resolution': float,
                          'origin_x': float, 'origin_y': float} | None, ...]
        """
        frontiers = frontiers or []
        path_accum = path_accum or [[] for _ in range(self.num_robots)]
        astar_paths = astar_paths or [[] for _ in range(self.num_robots)]
        robot_positions = robot_positions or [(0, 0)] * self.num_robots
        local_maps = local_maps or [None] * self.num_robots

        fig, axes = plt.subplots(2, 2, figsize=(20, 20))

        # ── 로컬맵 3장 (좌상, 우상, 좌하) ──
        panel_positions = [(0, 0), (0, 1), (1, 0)]
        for i in range(self.num_robots):
            r, c = panel_positions[i]
            ax = axes[r][c]

            if local_maps[i] is not None:
                lm = local_maps[i]
                ldata = lm['data']
                lh, lw = ldata.shape
                lres = lm['resolution']
                lox = lm['origin_x']
                loy = lm['origin_y']

                rgb = np.zeros((lh, lw, 3), dtype=np.uint8)
                rgb[ldata == -1] = [128, 128, 128]
                rgb[(ldata >= 0) & (ldata <= 50)] = [255, 255, 255]
                rgb[ldata > 50] = [0, 0, 0]

                ax.imshow(rgb, origin='lower',
                          extent=[lox, lox + lw * lres, loy, loy + lh * lres])

                # 해당 로봇의 경로
                pts = path_accum[i]
                if len(pts) >= 2:
                    ax.plot([p[0] for p in pts], [p[1] for p in pts],
                            color=COLORS[i], linewidth=1.5, alpha=0.8)

                # 해당 로봇 위치
                rx, ry = robot_positions[i]
                ax.scatter([rx], [ry], c=COLORS[i], s=200, zorder=8,
                           marker='o', edgecolors='white', linewidths=2.0)
                ax.annotate(f'R{i}', (rx, ry), textcoords='offset points',
                            xytext=(6, 6), fontsize=10, fontweight='bold',
                            color=COLORS[i])
            else:
                ax.text(0.5, 0.5, 'No map yet', ha='center', va='center',
                        transform=ax.transAxes, fontsize=14, color='gray')

            ax.set_title(f'Robot {i} Local Map (tb3_{i})', fontsize=12,
                         color=COLORS[i], fontweight='bold')
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.grid(True, alpha=0.3)

        # ── 전역맵 (우하) ──
        ax = axes[1][1]
        h, w = global_map.shape
        ox, oy, res = origin_x, origin_y, resolution

        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        rgb[global_map == -1] = [128, 128, 128]
        rgb[(global_map >= 0) & (global_map <= 50)] = [255, 255, 255]
        rgb[global_map > 50] = [0, 0, 0]

        # 보로노이 오버레이 (unknown만)
        if cluster_centers and len(cluster_centers) == self.num_robots:
            unk_mask = (global_map == -1)
            unk_pts = np.column_stack(np.where(unk_mask))
            if len(unk_pts) > 0:
                grid_centers = np.array([
                    [(cy - oy) / res, (cx - ox) / res]
                    for (cx, cy) in cluster_centers
                ])
                diff = unk_pts[:, None, :] - grid_centers[None, :, :]
                labels = np.argmin(np.sum(diff ** 2, axis=2), axis=1)
                alpha = 0.40
                for pt, label in zip(unk_pts, labels):
                    c = ZONE_COLORS[label % len(ZONE_COLORS)]
                    rgb[pt[0], pt[1]] = [
                        int(128 * (1 - alpha) + c[0] * alpha),
                        int(128 * (1 - alpha) + c[1] * alpha),
                        int(128 * (1 - alpha) + c[2] * alpha)]

        ax.imshow(rgb, origin='lower',
                  extent=[ox, ox + w * res, oy, oy + h * res])

        # frontier
        if frontiers:
            ax.scatter([f[0] for f in frontiers], [f[1] for f in frontiers],
                       c='orange', s=250, zorder=7, edgecolors='black',
                       linewidths=1.5, marker='*')

        # 모든 로봇 경로 + A* + 위치
        for i in range(self.num_robots):
            pts = path_accum[i]
            if len(pts) >= 2:
                ax.plot([p[0] for p in pts], [p[1] for p in pts],
                        color=COLORS[i], linewidth=1.2, alpha=0.8, zorder=4)
            pts = astar_paths[i]
            if len(pts) >= 2:
                ax.plot([p[0] for p in pts], [p[1] for p in pts],
                        color=COLORS[i], linewidth=3.0, alpha=0.9, zorder=6)
            rx, ry = robot_positions[i]
            ax.scatter([rx], [ry], c=COLORS[i], s=200, zorder=8,
                       marker='o', edgecolors='white', linewidths=2.0)
            ax.annotate(f'R{i}', (rx, ry), textcoords='offset points',
                        xytext=(6, 6), fontsize=10, fontweight='bold',
                        color=COLORS[i])

        # 범례
        legend_elements = [
            mpatches.Patch(facecolor='white', edgecolor='gray', label='Free'),
            mpatches.Patch(color='gray', label='Unknown'),
            mpatches.Patch(color='black', label='Occupied'),
            Line2D([0], [0], marker='*', color='orange', markersize=12,
                   label=f'Frontier ({len(frontiers)})',
                   linestyle='None', markeredgecolor='black'),
        ]
        for i in range(self.num_robots):
            legend_elements += [
                Line2D([0], [0], color=COLORS[i], linewidth=1.2,
                       label=f'Robot {i} path'),
                Line2D([0], [0], color=COLORS[i], linewidth=3.0,
                       label=f'Robot {i} A*'),
            ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=7)
        ax.set_title('Global Merged Map', fontsize=12, fontweight='bold')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.grid(True, alpha=0.3)

        plt.suptitle(f'Multi-Robot Exploration ({self.num_robots} robots)',
                     fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(filepath, dpi=120)
        plt.close()
    def render_decentralized(self, filepath: str,
                             robot_maps: list,
                             global_map: np.ndarray = None,
                             global_res: float = 0.05,
                             global_ox: float = 0.0,
                             global_oy: float = 0.0,
                             path_accum: list = None,
                             astar_paths: list = None,
                             robot_positions: list = None,
                             cluster_centers: list = None,
                             robot_frontiers: list = None):
        """
        Decentralized 4패널:
        - 좌상/우상/좌하: 각 로봇의 merged_map + frontier + 경로 + A*
        - 우하: PC 모니터링용 전체 합성맵
        """
        path_accum = path_accum or [[] for _ in range(self.num_robots)]
        astar_paths = astar_paths or [[] for _ in range(self.num_robots)]
        robot_positions = robot_positions or [(0, 0)] * self.num_robots
        robot_frontiers = robot_frontiers or [[] for _ in range(self.num_robots)]

        fig, axes = plt.subplots(2, 2, figsize=(20, 20))
        panel_positions = [(0, 0), (0, 1), (1, 0)]

        # ── 각 로봇의 merged_map ──
        for i in range(self.num_robots):
            r, c = panel_positions[i]
            ax = axes[r][c]

            if robot_maps[i] is not None:
                rm = robot_maps[i]
                rdata = rm['data']
                rh, rw = rdata.shape
                rres = rm['resolution']
                rox = rm['origin_x']
                roy = rm['origin_y']

                rgb = self._map_to_rgb(rdata)

                # K-means 영역 오버레이
                if cluster_centers and len(cluster_centers) >= 2:
                    unk_mask = (rdata == -1)
                    unk_pts = np.column_stack(np.where(unk_mask))
                    if len(unk_pts) > 0:
                        grid_centers = np.array([
                            [(cy - roy) / rres, (cx - rox) / rres]
                            for (cx, cy) in cluster_centers
                        ])
                        diff = unk_pts[:, None, :] - grid_centers[None, :, :]
                        labels = np.argmin(np.sum(diff ** 2, axis=2), axis=1)
                        alpha = 0.35
                        for pt, label in zip(unk_pts, labels):
                            zc = ZONE_COLORS[label % len(ZONE_COLORS)]
                            rgb[pt[0], pt[1]] = [
                                int(128 * (1 - alpha) + zc[0] * alpha),
                                int(128 * (1 - alpha) + zc[1] * alpha),
                                int(128 * (1 - alpha) + zc[2] * alpha)]

                ax.imshow(rgb, origin='lower',
                          extent=[rox, rox + rw * rres, roy, roy + rh * rres])

                # frontier 표시
                fpts = robot_frontiers[i]
                if fpts:
                    ax.scatter([f[0] for f in fpts], [f[1] for f in fpts],
                               c='orange', s=150, zorder=7, edgecolors='black',
                               linewidths=1.0, marker='*')

                # 이동 경로 (얇은 선)
                pts = path_accum[i]
                if len(pts) >= 2:
                    ax.plot([p[0] for p in pts], [p[1] for p in pts],
                            color=COLORS[i], linewidth=1.2, alpha=0.8, zorder=4)

                # A* 경로 (굵은 선)
                apts = astar_paths[i]
                if len(apts) >= 2:
                    ax.plot([p[0] for p in apts], [p[1] for p in apts],
                            color=COLORS[i], linewidth=3.0, alpha=0.9, zorder=6)

                # 로봇 위치
                rx, ry = robot_positions[i]
                ax.scatter([rx], [ry], c=COLORS[i], s=200, zorder=8,
                           marker='o', edgecolors='white', linewidths=2.0)
                ax.annotate(f'R{i}', (rx, ry), textcoords='offset points',
                            xytext=(6, 6), fontsize=10, fontweight='bold',
                            color=COLORS[i])
            else:
                ax.text(0.5, 0.5, 'No map yet', ha='center', va='center',
                        transform=ax.transAxes, fontsize=14, color='gray')

            n_f = len(robot_frontiers[i])
            ax.set_title(f'Robot {i} Merged Map (tb3_{i}) — Frontier: {n_f}',
                         fontsize=12, color=COLORS[i], fontweight='bold')
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.grid(True, alpha=0.3)

        # ── 모니터링용 전체 합성맵 (우하) ──
        ax = axes[1][1]
        if global_map is not None:
            h, w = global_map.shape
            rgb = self._map_to_rgb(global_map)
            ax.imshow(rgb, origin='lower',
                      extent=[global_ox, global_ox + w * global_res,
                              global_oy, global_oy + h * global_res])

            # 모든 로봇 경로 + A* + 위치
            for i in range(self.num_robots):
                pts = path_accum[i]
                if len(pts) >= 2:
                    ax.plot([p[0] for p in pts], [p[1] for p in pts],
                            color=COLORS[i], linewidth=1.2, alpha=0.8, zorder=4)
                apts = astar_paths[i]
                if len(apts) >= 2:
                    ax.plot([p[0] for p in apts], [p[1] for p in apts],
                            color=COLORS[i], linewidth=3.0, alpha=0.9, zorder=6)
                rx, ry = robot_positions[i]
                ax.scatter([rx], [ry], c=COLORS[i], s=200, zorder=8,
                           marker='o', edgecolors='white', linewidths=2.0)
                ax.annotate(f'R{i}', (rx, ry), textcoords='offset points',
                            xytext=(6, 6), fontsize=10, fontweight='bold',
                            color=COLORS[i])
        else:
            ax.text(0.5, 0.5, 'No monitoring map', ha='center', va='center',
                    transform=ax.transAxes, fontsize=14, color='gray')

        ax.set_title('Monitoring Map (PC only — robots do not see this)',
                     fontsize=11, fontweight='bold', color='gray')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.grid(True, alpha=0.3)

        plt.suptitle(f'Decentralized Multi-Robot Exploration ({self.num_robots} robots)',
                     fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(filepath, dpi=120)
        plt.close()

    @staticmethod
    def _map_to_rgb(map_data):
        """맵 데이터를 RGB 이미지로 변환."""
        h, w = map_data.shape
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        rgb[map_data == -1] = [128, 128, 128]
        rgb[(map_data >= 0) & (map_data <= 50)] = [255, 255, 255]
        rgb[map_data > 50] = [0, 0, 0]
        return rgb
