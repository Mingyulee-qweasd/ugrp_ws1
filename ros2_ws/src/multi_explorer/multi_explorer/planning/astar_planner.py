"""
planning/astar_planner.py

그리드 기반 A* 경로 계획 (순수 Python, ROS 의존 없음)
nav2 의 A* 를 메인으로 사용하되, 시각화/보조 목적으로 유지.

원본: 기존 exploration_planner.py 의 astar() 분리
오픈소스 대응: mapAStarGrid.m
"""
import math
import heapq


def astar(grid, start: tuple, goal: tuple) -> list:
    """
    그리드 기반 A* 경로 탐색.

    Args:
        grid: (h, w) uint8 배열, 0=free, 1=occupied
        start: (row, col)
        goal: (row, col)

    Returns:
        [(row, col), ...] 경로 or None
    """
    h, w = grid.shape

    def heuristic(a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}

    while open_set:
        _, current = heapq.heappop(open_set)
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]

        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1),
                       (-1, -1), (-1, 1), (1, -1), (1, 1)]:
            nr, nc = current[0] + dr, current[1] + dc
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if grid[nr, nc] == 1:
                continue
            tentative_g = g_score[current] + math.hypot(dr, dc)
            neighbor = (nr, nc)
            if tentative_g < g_score.get(neighbor, float('inf')):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                heapq.heappush(
                    open_set,
                    (tentative_g + heuristic(neighbor, goal), neighbor))

    return None
