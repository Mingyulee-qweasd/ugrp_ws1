"""
perception/map_merger.py

다중 로봇 맵 합성 순수 로직 (ROS 의존 없음)
unknown(-1) < free(0~50) < occupied(51~100) 우선순위

후처리:
1. 고립 occupied 제거: 주변에 occupied 이웃이 적은 occupied 셀 → free
2. 고립 unknown 제거: 주변이 대부분 free인 unknown 셀 → free
"""
import numpy as np
from scipy import ndimage


class MapMerger:
    """여러 local map 을 하나의 공유맵으로 합성."""

    def merge(self, local_maps: list, ref_info: dict) -> np.ndarray:
        h = ref_info['height']
        w = ref_info['width']
        res = ref_info['resolution']
        ox = ref_info['origin_x']
        oy = ref_info['origin_y']

        merged = np.full((h, w), -1, dtype=np.int8)

        for lm in local_maps:
            if lm is None:
                continue
            ldata = lm['data']
            linfo = lm['info']
            lh = linfo['height']
            lw = linfo['width']
            lres = linfo['resolution']
            lox = linfo['origin_x']
            loy = linfo['origin_y']

            # vectorized 좌표 변환
            cols_l = np.arange(lw)
            rows_l = np.arange(lh)
            cc, rr = np.meshgrid(cols_l, rows_l)

            wx = lox + cc * lres
            wy = loy + rr * lres

            mc = np.round((wx - ox) / res).astype(int)
            mr = np.round((wy - oy) / res).astype(int)

            valid = (mr >= 0) & (mr < h) & (mc >= 0) & (mc < w)
            known = ldata != -1
            mask = valid & known

            src_vals = ldata[mask]
            dst_r = mr[mask]
            dst_c = mc[mask]

            # vectorized 합성: occupied(>50) 우선, free는 최솟값
            for i in range(len(src_vals)):
                val = src_vals[i]
                r, c = dst_r[i], dst_c[i]
                cur = merged[r, c]
                if cur == -1:
                    merged[r, c] = val
                elif val > 50:
                    merged[r, c] = val
                elif val <= 50 and cur <= 50:
                    merged[r, c] = min(cur, val)

        # 후처리
        merged = self._remove_isolated_occupied(merged)
        merged = self._fill_isolated_unknown(merged)

        return merged

    @staticmethod
    def _remove_isolated_occupied(merged: np.ndarray) -> np.ndarray:
        """
        고립된 occupied 노이즈 제거.
        occupied 셀(>50) 주변 8셀 중 occupied가 2개 미만이면 free(0)로 변환.
        벽은 보통 연속된 occupied 셀이라 유지되고, LiDAR 노이즈 점은 제거됨.
        """
        h, w = merged.shape
        occupied_mask = (merged > 50).astype(np.float32)

        # 3x3 커널로 주변 occupied 셀 수 계산 (자기 제외)
        kernel = np.array([[1, 1, 1],
                           [1, 0, 1],
                           [1, 1, 1]], dtype=np.float32)
        neighbor_count = ndimage.convolve(occupied_mask, kernel, mode='constant', cval=0.0)

        # occupied인데 주변 occupied 이웃이 2개 미만 → 노이즈로 판단
        noise_mask = (merged > 50) & (neighbor_count < 2)
        result = merged.copy()
        result[noise_mask] = 0  # free로 변환

        return result

    @staticmethod
    def _fill_isolated_unknown(merged: np.ndarray) -> np.ndarray:
        """
        주변 8셀 중 6셀 이상이 free(0~50)인데 자신만 unknown(-1)인 셀을 free(0)로 채움.
        """
        h, w = merged.shape
        free_mask = ((merged >= 0) & (merged <= 50)).astype(np.float32)

        kernel = np.array([[1, 1, 1],
                           [1, 0, 1],
                           [1, 1, 1]], dtype=np.float32)
        free_neighbor_count = ndimage.convolve(free_mask, kernel, mode='constant', cval=0.0)

        fill_mask = (merged == -1) & (free_neighbor_count >= 6)
        result = merged.copy()
        result[fill_mask] = 0

        return result
