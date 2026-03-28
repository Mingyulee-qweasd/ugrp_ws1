"""
perception/occupancy_map.py

OccupancyGrid 데이터 래핑 + 좌표 변환 유틸 (ROS 의존 없음)
오픈소스 대응: setMap.m, findGrid.m, findSquarePoints.m
"""
import numpy as np


class OccupancyMap:
    """OccupancyGrid 를 감싸는 유틸 클래스."""

    def __init__(self, data: np.ndarray, resolution: float,
                 origin_x: float, origin_y: float):
        self.data = data  # (h, w) int8
        self.resolution = resolution
        self.origin_x = origin_x
        self.origin_y = origin_y
        self.height, self.width = data.shape

    def world_to_grid(self, wx: float, wy: float) -> tuple:
        """월드 좌표 → 그리드 (row, col)."""
        col = int((wx - self.origin_x) / self.resolution)
        row = int((wy - self.origin_y) / self.resolution)
        return (row, col)

    def grid_to_world(self, row: int, col: int) -> tuple:
        """그리드 (row, col) → 월드 좌표 (x, y)."""
        wx = col * self.resolution + self.origin_x
        wy = row * self.resolution + self.origin_y
        return (wx, wy)

    def is_free(self, row: int, col: int) -> bool:
        if not self._in_bounds(row, col):
            return False
        return 0 <= self.data[row, col] <= 50

    def is_occupied(self, row: int, col: int) -> bool:
        if not self._in_bounds(row, col):
            return True
        return self.data[row, col] > 50

    def is_unknown(self, row: int, col: int) -> bool:
        if not self._in_bounds(row, col):
            return True
        return self.data[row, col] == -1

    def _in_bounds(self, row: int, col: int) -> bool:
        return 0 <= row < self.height and 0 <= col < self.width

    def get_info_dict(self) -> dict:
        """map_merger 등에서 사용하는 info dict 반환."""
        return {
            'height': self.height,
            'width': self.width,
            'resolution': self.resolution,
            'origin_x': self.origin_x,
            'origin_y': self.origin_y,
        }
