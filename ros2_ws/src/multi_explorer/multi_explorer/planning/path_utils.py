"""
planning/path_utils.py

경로 보간, 좌표 변환 유틸 (순수 Python, ROS 의존 없음)
오픈소스 대응: interparc.m
"""
import math


def grid_path_to_world(path: list, resolution: float,
                       origin_x: float, origin_y: float) -> list:
    """A* 그리드 경로 → 월드 좌표 리스트."""
    return [(col * resolution + origin_x, row * resolution + origin_y)
            for (row, col) in path]


def path_length(points: list) -> float:
    """[(x,y), ...] 경로의 총 길이."""
    total = 0.0
    for i in range(1, len(points)):
        total += math.hypot(points[i][0] - points[i - 1][0],
                            points[i][1] - points[i - 1][1])
    return total


def interpolate_path(points: list, spacing: float = 0.1) -> list:
    """경로를 일정 간격으로 보간."""
    if len(points) < 2:
        return points
    result = [points[0]]
    for i in range(1, len(points)):
        dx = points[i][0] - points[i - 1][0]
        dy = points[i][1] - points[i - 1][1]
        seg_len = math.hypot(dx, dy)
        if seg_len < 1e-6:
            continue
        n_steps = int(seg_len / spacing)
        for s in range(1, n_steps + 1):
            t = s / max(n_steps, 1)
            result.append((points[i - 1][0] + dx * t,
                           points[i - 1][1] + dy * t))
    if result[-1] != points[-1]:
        result.append(points[-1])
    return result
