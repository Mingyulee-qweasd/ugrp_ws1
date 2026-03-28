"""
planning/goal_selector.py

frontier 중 최적 goal 선택 (순수 Python, ROS 의존 없음)
정보이득 + 거리 기반 스코어링

오픈소스 대응: frontierPoint.m 일부
"""
import math
import numpy as np


def score_frontier(frontier_pos: tuple, robot_pos: tuple,
                   map_data: np.ndarray, resolution: float,
                   origin_x: float, origin_y: float,
                   info_gain_radius: int = 10) -> float:
    """
    frontier 의 점수를 계산 (높을수록 좋음).

    Args:
        frontier_pos: (wx, wy) 월드 좌표
        robot_pos: (rx, ry) 로봇 월드 좌표
        map_data: (h, w) int8 배열
        resolution, origin_x, origin_y: 맵 메타
        info_gain_radius: 정보이득 계산 반경 (그리드 셀 수)

    Returns:
        score: float (높을수록 우선)
    """
    fx, fy = frontier_pos
    rx, ry = robot_pos

    dist = math.hypot(fx - rx, fy - ry)

    col = int((fx - origin_x) / resolution)
    row = int((fy - origin_y) / resolution)
    h, w = map_data.shape

    r_min = max(0, row - info_gain_radius)
    r_max = min(h, row + info_gain_radius + 1)
    c_min = max(0, col - info_gain_radius)
    c_max = min(w, col + info_gain_radius + 1)

    patch = map_data[r_min:r_max, c_min:c_max]
    unknown_count = np.sum(patch == -1)

    score = unknown_count / (dist + 1.0)
    return score
