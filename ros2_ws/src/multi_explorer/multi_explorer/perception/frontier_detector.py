"""
perception/frontier_detector.py

OccupancyGrid 에서 frontier 셀 추출 (Sobel edge 방식)
순수 Python (ROS 의존 없음)

수정사항:
- 문제5: MIN_CLUSTER_SIZE 올림 (2→5) — 작은 unknown 패치 제거
- frontier 중심이 free 셀 위에 있는지 검증
- free 셀과 인접한 frontier만 유효 처리

오픈소스 대응: frontierPoint.m
"""
import numpy as np
import cv2

SOBEL_THRESHOLD = 20
OBSTACLE_INFLATE = 5    # 벽 근처 frontier 제거 (원래값 복원)
MIN_CLUSTER_SIZE = 3    # 작은 노이즈는 거르되 유효 frontier는 유지


class FrontierDetector:
    """Sobel edge 기반 frontier 검출기."""

    def __init__(self, sobel_threshold=SOBEL_THRESHOLD,
                 obstacle_inflate=OBSTACLE_INFLATE,
                 min_cluster_size=MIN_CLUSTER_SIZE):
        self.sobel_threshold = sobel_threshold
        self.obstacle_inflate = obstacle_inflate
        self.min_cluster_size = min_cluster_size

    def detect(self, map_data: np.ndarray) -> list:
        """
        OccupancyGrid 데이터에서 frontier 중심점 추출.

        Args:
            map_data: (h, w) int8 배열, -1=unknown, 0~50=free, 51~100=occupied

        Returns:
            [(row, col), ...] frontier 중심 그리드 좌표 리스트
        """
        h, w = map_data.shape

        # 3값 맵: unknown=128, free=255, occupied=0
        map_tri = np.full(map_data.shape, 128, dtype=np.uint8)
        map_tri[map_data >= 0] = 255
        map_tri[map_data > 50] = 0

        # Sobel edge
        sx = cv2.Sobel(map_tri, cv2.CV_64F, 1, 0, ksize=3)
        sy = cv2.Sobel(map_tri, cv2.CV_64F, 0, 1, ksize=3)
        mag = np.sqrt(sx ** 2 + sy ** 2)
        _, edge = cv2.threshold(
            mag.astype(np.uint8), self.sobel_threshold, 255, cv2.THRESH_BINARY)

        # occupied 팽창 → frontier 에서 제외
        occ = (map_data > 50).astype(np.uint8)
        k = cv2.getStructuringElement(
            cv2.MORPH_RECT, (self.obstacle_inflate, self.obstacle_inflate))
        edge[cv2.dilate(occ, k) == 1] = 0
        edge[map_data == -1] = 0
        edge[map_data > 50] = 0

        # connected components → 클러스터 중심
        num_labels, labels, stats, centroids = \
            cv2.connectedComponentsWithStats(edge, connectivity=8)

        centers = []
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] < self.min_cluster_size:
                continue

            cr = int(centroids[i][1])
            cc = int(centroids[i][0])

            # 중심이 맵 범위 내인지
            if not (0 <= cr < h and 0 <= cc < w):
                continue

            # 중심이 free 셀 위에 있지 않으면 가장 가까운 free 셀로 보정
            if not (0 <= map_data[cr, cc] <= 50):
                cr, cc = self._snap_to_free(cr, cc, map_data, h, w)
                if cr is None:
                    continue

            # free 셀과 인접한 unknown이 있는 진짜 frontier인지 검증
            if not self._has_adjacent_unknown(cr, cc, map_data, h, w):
                continue

            centers.append((cr, cc))

        return centers

    def _snap_to_free(self, row, col, map_data, h, w, radius=5):
        """가장 가까운 free 셀을 찾아 반환."""
        import math
        best_r, best_c, best_d = None, None, float('inf')
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                nr, nc = row + dr, col + dc
                if 0 <= nr < h and 0 <= nc < w:
                    if 0 <= map_data[nr, nc] <= 50:
                        d = math.hypot(dr, dc)
                        if d < best_d:
                            best_d = d
                            best_r, best_c = nr, nc
        return best_r, best_c

    def _has_adjacent_unknown(self, row, col, map_data, h, w, radius=3):
        """주변에 unknown(-1) 셀이 있는지 확인."""
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                nr, nc = row + dr, col + dc
                if 0 <= nr < h and 0 <= nc < w:
                    if map_data[nr, nc] == -1:
                        return True
        return False

    def to_world(self, grid_frontiers: list, resolution: float,
                 origin_x: float, origin_y: float) -> list:
        """그리드 좌표 → 월드 좌표 변환."""
        return [(col * resolution + origin_x, row * resolution + origin_y)
                for (row, col) in grid_frontiers]
