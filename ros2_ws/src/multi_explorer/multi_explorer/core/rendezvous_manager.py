"""
core/rendezvous_manager.py

랑데부 조건 판단 + 집결 위치 선정 (순수 Python, ROS 의존 없음)

논문: "Coordinated Multi-Agent Exploration, Rendezvous, & Task Allocation
       in Unknown Environments with Limited Connectivity"

랑데부 포인트:
  x_r = argmin_r  Σ(|ζ_j| · ||x_r - c_j||) / Σ|ζ_j|
  → 클러스터 중심까지 거리를 파티션 크기로 가중 평균한 값이 최소인 지점

경로 정책:
  R_φ = min_φ  φ·(p_m + γ·n_unk) + (1-φ)·p_k
  → γ → 0: unknown 통과 허용, γ → 1: 안전 경로만

오픈소스 대응: rzDecide.m
"""
import math
import numpy as np


class RendezvousManager:
    """로봇 간 집결 조건 판단 + 집결 위치 선정."""

    def __init__(self, merge_distance=2.0, rendezvous_period=60.0,
                 gamma=0.5):
        self.merge_distance = merge_distance
        self.rendezvous_period = rendezvous_period
        self.gamma = gamma  # risk coefficient: 0=unknown 무시, 1=unknown 위험

    def check_proximity(self, robot_positions: list) -> bool:
        """아무 로봇 쌍이라도 merge_distance 이내이면 True."""
        n = len(robot_positions)
        for i in range(n):
            for j in range(i + 1, n):
                dx = robot_positions[i][0] - robot_positions[j][0]
                dy = robot_positions[i][1] - robot_positions[j][1]
                if math.hypot(dx, dy) <= self.merge_distance:
                    return True
        return False

    def check_period(self, elapsed_sec: float) -> bool:
        """주기 기반 강제 집결 조건 확인."""
        return elapsed_sec >= self.rendezvous_period

    # ── 랑데부 포인트 선정 (논문 수식) ─────────────────────────

    def compute_rendezvous_point(
        self,
        cluster_centers: list,
        cluster_sizes: list,
        map_data: np.ndarray = None,
        resolution: float = 0.05,
        origin_x: float = 0.0,
        origin_y: float = 0.0,
    ) -> tuple:
        """
        논문 수식: 클러스터 중심의 가중 평균으로 랑데부 포인트 계산.

        x_r = argmin_r  Σ(|ζ_j| · ||x_r - c_j||) / Σ|ζ_j|

        이 수식의 최적해는 가중 중심(weighted centroid)이므로:
        x_r = Σ(|ζ_j| · c_j) / Σ|ζ_j|

        Args:
            cluster_centers: [(cx, cy), ...] 각 로봇의 클러스터 중심 (월드 좌표)
            cluster_sizes: [|ζ_0|, |ζ_1|, ...] 각 클러스터의 셀 개수
            map_data: (h, w) int8 배열 (보정용, optional)
            resolution, origin_x, origin_y: 맵 메타 (보정용)

        Returns:
            (rx, ry) 랑데부 포인트 월드 좌표
        """
        if not cluster_centers or not cluster_sizes:
            return (0.0, 0.0)

        total_weight = sum(cluster_sizes)
        if total_weight == 0:
            return (0.0, 0.0)

        # 가중 중심 계산
        wx = sum(s * c[0] for s, c in zip(cluster_sizes, cluster_centers))
        wy = sum(s * c[1] for s, c in zip(cluster_sizes, cluster_centers))
        rx = wx / total_weight
        ry = wy / total_weight

        # 맵이 있으면 occupied/unknown 셀이 아닌 free 셀로 보정
        if map_data is not None:
            rx, ry = self._snap_to_free(
                rx, ry, map_data, resolution, origin_x, origin_y)

        return (rx, ry)

    def _snap_to_free(self, wx, wy, map_data, resolution, origin_x, origin_y,
                      search_radius=20) -> tuple:
        """
        논문: 선정된 point가 occupied or unknown인 경우
        Moore's Neighborhood 사용 → 주변 셀에서 free 셀을 찾아 보정

        Args:
            wx, wy: 원래 월드 좌표
            map_data: (h, w) int8 배열
            search_radius: 탐색 반경 (그리드 셀 수)

        Returns:
            보정된 (wx, wy) 월드 좌표
        """
        h, w = map_data.shape
        col = int((wx - origin_x) / resolution)
        row = int((wy - origin_y) / resolution)

        # 이미 free 셀이면 그대로
        if 0 <= row < h and 0 <= col < w:
            if 0 <= map_data[row, col] <= 50:
                return (wx, wy)

        # Moore's Neighborhood: 나선형으로 free 셀 탐색
        best_dist = float('inf')
        best_pos = (wx, wy)

        for dr in range(-search_radius, search_radius + 1):
            for dc in range(-search_radius, search_radius + 1):
                nr, nc = row + dr, col + dc
                if not (0 <= nr < h and 0 <= nc < w):
                    continue
                if 0 <= map_data[nr, nc] <= 50:  # free 셀
                    dist = math.hypot(dr, dc)
                    if dist < best_dist:
                        best_dist = dist
                        best_pos = (
                            nc * resolution + origin_x,
                            nr * resolution + origin_y)

        return best_pos

    # ── 경로 정책 (빠른 길 vs 안전한 길) ──────────────────────

    def select_route_policy(self, p_m: float, n_unk: float,
                            p_k: float) -> dict:
        """
        논문 수식: R_φ = min_φ  φ·(p_m + γ·n_unk) + (1-φ)·p_k

        Args:
            p_m: 최단 경로 길이 (unknown 구간 포함)
            n_unk: 최단 경로 중 unknown 구간 길이
            p_k: 안전 경로 길이 (known & free 셀만)

        Returns:
            {'use_shortest': bool, 'cost': float}
            use_shortest=True → p_m 경로 선택 (φ=1)
            use_shortest=False → p_k 경로 선택 (φ=0)
        """
        # φ=1: 최단 경로 (unknown 포함)
        cost_shortest = p_m + self.gamma * n_unk
        # φ=0: 안전 경로 (free만)
        cost_safe = p_k

        if cost_shortest <= cost_safe:
            return {'use_shortest': True, 'cost': cost_shortest}
        else:
            return {'use_shortest': False, 'cost': cost_safe}
