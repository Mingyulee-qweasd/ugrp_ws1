"""
core/exploration_planner.py

순수 탐색 계획 로직 (ROS 의존 없음)
- K-means 영역 분할 + auction 기반 클러스터 할당
- frontier 기반 목표 할당 (논문 수식 (6), (7) 구현)

논문: "Coordinated Multi-Agent Exploration, Rendezvous, & Task Allocation
       in Unknown Environments with Limited Connectivity"

  F_kp = ||f_p - x_k|| + Δ·||f_p - c_k||    (f_p ∉ ζ_k)
  F_kp = ||f_p - x_k||                        (f_p ∈ ζ_k)
  f_{k,p*} = argmin_p F_{k,p}

오픈소스 대응: robotsMain.m, frontierPoint.m
"""
import numpy as np
import math
from sklearn.cluster import KMeans

DELTA = 8.0          # Δ: 영역 밖 패널티 상수 (논문 수식 (6))
VISITED_RADIUS = 0.8


class ExplorationPlanner:
    """
    frontier 기반 탐색 목표를 선정하는 순수 로직 클래스.
    맵 데이터와 로봇 위치를 받아 각 로봇의 목표를 반환.
    """

    def __init__(self, num_robots=3, delta=DELTA):
        self.num_robots = num_robots
        self.delta = delta
        # 각 로봇에 할당된 클러스터 중심 (월드 좌표)
        self.cluster_centers = [(0.0, 0.0)] * num_robots
        # 각 로봇에 할당된 클러스터 셀 집합
        self.cluster_zones = [set() for _ in range(num_robots)]

    def kmeans_partition(self, map_data: np.ndarray,
                         resolution: float = 0.05,
                         origin_x: float = 0.0,
                         origin_y: float = 0.0,
                         robot_positions: list = None) -> list:
        """
        unknown(-1) 영역을 K-means 로 num_robots 개 영역으로 분할.
        논문: auction 방식으로 각 로봇에 클러스터 할당
              (각 로봇이 가장 가까운 클러스터에 bid, tie는 index로 해소)

        Args:
            map_data: OccupancyGrid 2D 배열 (h, w), 값: -1/0~100
            resolution, origin_x, origin_y: 맵 메타 (중심 좌표 변환용)
            robot_positions: [(x, y), ...] 각 로봇 월드 좌표 (auction 용)

        Returns:
            zones: [[(row, col), ...], ...] 로봇별 담당 셀 리스트
        """
        unknown_pts = np.column_stack(np.where(map_data == -1))
        if len(unknown_pts) < self.num_robots:
            return [[] for _ in range(self.num_robots)]
        if self.num_robots == 1:
            return [list(map(tuple, unknown_pts))]
        if len(unknown_pts) > 2000:
            idx = np.random.choice(len(unknown_pts), 2000, replace=False)
            unknown_pts = unknown_pts[idx]

        kmeans = KMeans(n_clusters=self.num_robots, n_init=5, random_state=42)
        labels = kmeans.fit_predict(unknown_pts)

        # 클러스터 중심을 월드 좌표로 변환 (row, col → x, y)
        centers_world = []
        for center in kmeans.cluster_centers_:
            cx = center[1] * resolution + origin_x  # col → x
            cy = center[0] * resolution + origin_y  # row → y
            centers_world.append((cx, cy))

        # 클러스터별 셀 그룹
        cluster_cells = [[] for _ in range(self.num_robots)]
        for pt, label in zip(unknown_pts, labels):
            cluster_cells[label].append(tuple(pt))

        # ── auction 기반 클러스터 ↔ 로봇 할당 ────────────────
        if robot_positions and len(robot_positions) == self.num_robots:
            zones = self._auction_assign(
                centers_world, cluster_cells, robot_positions)
        else:
            # robot_positions 없으면 label 순서 그대로
            zones = cluster_cells

        return zones

    def _auction_assign(self, centers_world, cluster_cells,
                        robot_positions) -> list:
        """
        논문: 각 로봇이 클러스터 중심까지 거리로 bid,
              가장 가까운 로봇에 할당. tie는 robot index로 해소.
        """
        n_clusters = len(centers_world)
        n_robots = len(robot_positions)

        # 거리 행렬: cost[robot][cluster]
        cost = np.zeros((n_robots, n_clusters))
        for r in range(n_robots):
            for c in range(n_clusters):
                cost[r, c] = math.hypot(
                    robot_positions[r][0] - centers_world[c][0],
                    robot_positions[r][1] - centers_world[c][1])

        # 탐욕적 할당: 가장 cost 낮은 (robot, cluster) 쌍부터 매칭
        assigned_robots = set()
        assigned_clusters = set()
        robot_to_cluster = {}

        # (cost, robot_idx, cluster_idx) 정렬
        pairs = []
        for r in range(n_robots):
            for c in range(n_clusters):
                pairs.append((cost[r, c], r, c))
        pairs.sort()  # cost 오름차순, tie 시 robot index 작은 것 우선

        for _, r, c in pairs:
            if r in assigned_robots or c in assigned_clusters:
                continue
            robot_to_cluster[r] = c
            assigned_robots.add(r)
            assigned_clusters.add(c)
            if len(assigned_robots) == n_robots:
                break

        # 할당 결과로 zones 재배치 + 중심 저장
        zones = [[] for _ in range(n_robots)]
        for robot_id in range(n_robots):
            cluster_id = robot_to_cluster.get(robot_id, robot_id % n_clusters)
            zones[robot_id] = cluster_cells[cluster_id]
            self.cluster_centers[robot_id] = centers_world[cluster_id]
            self.cluster_zones[robot_id] = set(cluster_cells[cluster_id])

        return zones

    def assign_targets(
        self,
        world_frontiers: list,
        robots: list,
        map_info: dict,
    ) -> dict:
        """
        논문 수식 (6), (7) 기반 frontier 할당.

        F_kp = ||f_p - x_k|| + Δ·||f_p - c_k||    (f_p ∉ ζ_k)
        F_kp = ||f_p - x_k||                        (f_p ∈ ζ_k)
        f_{k,p*} = argmin_p F_{k,p}

        Args:
            world_frontiers: [(wx, wy), ...] 월드 좌표 frontier 리스트
            robots: [{'id': int, 'x': float, 'y': float,
                      'navigating': bool, 'visited': [(x,y),...],
                      'zone_cells': set, 'failed_goals': set}, ...]
            map_info: {'resolution': float, 'origin_x': float, 'origin_y': float}

        Returns:
            assignments: {robot_id: (target_x, target_y)}
        """
        res = map_info['resolution']
        ox = map_info['origin_x']
        oy = map_info['origin_y']

        def w2p(wx, wy):
            return (int((wy - oy) / res), int((wx - ox) / res))

        assigned_keys = set()
        assignments = {}

        for robot in robots:
            if robot['navigating']:
                continue

            robot_id = robot['id']
            zone_set = robot.get('zone_cells', set())
            failed = robot.get('failed_goals', set())
            c_k = self.cluster_centers[robot_id]  # 클러스터 중심
            best_f, best_cost = None, float('inf')

            # 1단계: 논문 수식 (6) + visited/failed 블랙리스트
            for (fx, fy) in world_frontiers:
                key = (round(fx, 1), round(fy, 1))
                if key in assigned_keys or key in failed:
                    continue
                if any(math.hypot(fx - v[0], fy - v[1]) < VISITED_RADIUS
                       for v in robot['visited']):
                    continue

                # 수식 (6): F_kp
                dist_to_robot = math.hypot(fx - robot['x'], fy - robot['y'])
                if w2p(fx, fy) in zone_set:
                    cost = dist_to_robot
                else:
                    dist_to_center = math.hypot(fx - c_k[0], fy - c_k[1])
                    cost = dist_to_robot + self.delta * dist_to_center

                if cost < best_cost:
                    best_cost = cost
                    best_f = (fx, fy)

            # 2단계: visited 블랙리스트 무시 (수식 (6) 유지)
            if best_f is None:
                for (fx, fy) in world_frontiers:
                    key = (round(fx, 1), round(fy, 1))
                    if key in assigned_keys or key in failed:
                        continue
                    dist_to_robot = math.hypot(fx - robot['x'], fy - robot['y'])
                    if w2p(fx, fy) in zone_set:
                        cost = dist_to_robot
                    else:
                        dist_to_center = math.hypot(fx - c_k[0], fy - c_k[1])
                        cost = dist_to_robot + self.delta * dist_to_center
                    if cost < best_cost:
                        best_cost = cost
                        best_f = (fx, fy)

            # 3단계: 모든 제약 무시 (최근접)
            if best_f is None:
                for (fx, fy) in world_frontiers:
                    dist = math.hypot(fx - robot['x'], fy - robot['y'])
                    if dist < best_cost:
                        best_cost = dist
                        best_f = (fx, fy)

            if best_f:
                assigned_keys.add((round(best_f[0], 1), round(best_f[1], 1)))
                assignments[robot['id']] = best_f

        return assignments
