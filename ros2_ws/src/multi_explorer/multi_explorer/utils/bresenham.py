"""
utils/bresenham.py

Bresenham 라인 알고리즘 (ray-casting 용)
오픈소스 대응: bresenham.m
"""


def bresenham(x0: int, y0: int, x1: int, y1: int) -> list:
    """
    두 그리드 셀 사이의 Bresenham 라인.

    Returns:
        [(x, y), ...] 경로상의 그리드 좌표
    """
    points = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    while True:
        points.append((x0, y0))
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy

    return points
