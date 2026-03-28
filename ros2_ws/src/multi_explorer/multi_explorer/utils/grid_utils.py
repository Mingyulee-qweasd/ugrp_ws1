"""
utils/grid_utils.py

그리드 관련 범용 유틸 (순수 Python)
오픈소스 대응: neighbourND.m, combinator.m, cumsumall.m
"""


def get_neighbors_4(row: int, col: int, height: int, width: int) -> list:
    """4방향 이웃."""
    neighbors = []
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = row + dr, col + dc
        if 0 <= nr < height and 0 <= nc < width:
            neighbors.append((nr, nc))
    return neighbors


def get_neighbors_8(row: int, col: int, height: int, width: int) -> list:
    """8방향 이웃."""
    neighbors = []
    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            if dr == 0 and dc == 0:
                continue
            nr, nc = row + dr, col + dc
            if 0 <= nr < height and 0 <= nc < width:
                neighbors.append((nr, nc))
    return neighbors


def flood_fill(grid, start: tuple, target_val: int) -> set:
    """
    BFS flood fill — target_val 과 같은 값을 가진 연결 영역 반환.

    Args:
        grid: 2D numpy 배열
        start: (row, col)
        target_val: 찾을 값

    Returns:
        set of (row, col)
    """
    h, w = grid.shape
    visited = set()
    queue = [start]
    visited.add(start)
    while queue:
        r, c = queue.pop(0)
        for nr, nc in get_neighbors_4(r, c, h, w):
            if (nr, nc) not in visited and grid[nr, nc] == target_val:
                visited.add((nr, nc))
                queue.append((nr, nc))
    return visited
