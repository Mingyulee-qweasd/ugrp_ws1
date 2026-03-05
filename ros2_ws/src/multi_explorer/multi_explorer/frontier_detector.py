#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker, MarkerArray
from nav2_msgs.action import NavigateToPose
import numpy as np
import cv2
import math

SOBEL_THRESHOLD   = 20
OBSTACLE_INFLATE  = 5
MIN_CLUSTER_SIZE  = 2
MAP_UPDATE_PERIOD = 3.0
MIN_FRONTIER_DIST = 0.5   # 현재 위치에서 이 거리 미만 frontier는 무시


class FrontierDetector(Node):

    def __init__(self):
        super().__init__('frontier_detector')
        self.map_sub = self.create_subscription(
            OccupancyGrid, '/map', self.map_callback, 10)
        self.marker_pub = self.create_publisher(
            MarkerArray, '/frontier_markers', 10)
        self._nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        self.navigating      = False
        self.last_map_time   = self.get_clock().now()
        self.current_goal    = None
        self.visited_goals   = []   # 방문했거나 실패한 목표 누적
        self.robot_x         = 0.0
        self.robot_y         = 0.0

        # TF로 로봇 현재 위치 추적
        from tf2_ros import Buffer, TransformListener
        self.tf_buffer   = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.get_logger().info('Frontier Detector 시작')

    def get_robot_pose(self):
        """TF에서 로봇 현재 위치 가져오기"""
        try:
            t = self.tf_buffer.lookup_transform(
                'map', 'base_footprint',
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.5))
            self.robot_x = t.transform.translation.x
            self.robot_y = t.transform.translation.y
            return True
        except Exception:
            try:
                t = self.tf_buffer.lookup_transform(
                    'map', 'base_link',
                    rclpy.time.Time(),
                    timeout=rclpy.duration.Duration(seconds=0.5))
                self.robot_x = t.transform.translation.x
                self.robot_y = t.transform.translation.y
                return True
            except Exception as e:
                self.get_logger().warn(f'TF 실패: {e}')
                return False

    def map_callback(self, msg: OccupancyGrid):
        if self.navigating:
            return
        now = self.get_clock().now()
        if (now - self.last_map_time).nanoseconds / 1e9 < MAP_UPDATE_PERIOD:
            return
        self.last_map_time = now

        # 로봇 현재 위치 업데이트
        self.get_robot_pose()
        self.get_logger().info(
            f'로봇 위치: ({self.robot_x:.2f}, {self.robot_y:.2f})')

        h, w = msg.info.height, msg.info.width
        raw  = np.array(msg.data, dtype=np.int8).reshape((h, w))
        frontiers = self.detect_frontiers(raw)

        if len(frontiers) == 0:
            self.get_logger().info('Frontier 없음 — 탐색 완료')
            return

        res = msg.info.resolution
        ox  = msg.info.origin.position.x
        oy  = msg.info.origin.position.y
        world = [(col * res + ox, row * res + oy) for (row, col) in frontiers]

        # 방문했던 곳 + 너무 가까운 곳 제거
        world = self.filter_goals(world)
        if not world:
            self.get_logger().warn('유효 frontier 없음 — 방문 기록 초기화')
            self.visited_goals = []
            return

        self.publish_markers(world)
        best = self.select_best(world)
        self.get_logger().info(
            f'선택된 frontier: ({best[0]:.2f}, {best[1]:.2f}), '
            f'거리: {math.hypot(best[0]-self.robot_x, best[1]-self.robot_y):.2f}m')
        self.navigate_to(best)

    def detect_frontiers(self, raw):
        map_tri = np.full(raw.shape, 128, dtype=np.uint8)
        map_tri[raw >= 0]  = 255
        map_tri[raw > 50]  = 0

        sx  = cv2.Sobel(map_tri, cv2.CV_64F, 1, 0, ksize=3)
        sy  = cv2.Sobel(map_tri, cv2.CV_64F, 0, 1, ksize=3)
        mag = np.sqrt(sx**2 + sy**2)
        _, edge = cv2.threshold(
            mag.astype(np.uint8), SOBEL_THRESHOLD, 255, cv2.THRESH_BINARY)

        occ_mask    = (raw > 50).astype(np.uint8)
        kernel      = cv2.getStructuringElement(
            cv2.MORPH_RECT, (OBSTACLE_INFLATE, OBSTACLE_INFLATE))
        occ_dilated = cv2.dilate(occ_mask, kernel)
        edge[occ_dilated == 1] = 0
        edge[raw == -1] = 0
        edge[raw > 50]  = 0

        num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(
            edge, connectivity=8)
        centers = []
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= MIN_CLUSTER_SIZE:
                centers.append((int(centroids[i][1]), int(centroids[i][0])))
        self.get_logger().info(f'Frontier 클러스터 수: {len(centers)}')
        return centers

    def filter_goals(self, frontiers, blacklist_radius=0.4):
        result = []
        for f in frontiers:
            # 로봇과 너무 가까우면 제외
            dist_to_robot = math.hypot(
                f[0] - self.robot_x, f[1] - self.robot_y)
            if dist_to_robot < MIN_FRONTIER_DIST:
                continue
            # 이미 방문한 곳이면 제외
            if any(math.hypot(f[0]-v[0], f[1]-v[1]) < blacklist_radius
                   for v in self.visited_goals):
                continue
            result.append(f)
        return result

    def select_best(self, frontiers):
        """로봇에서 가장 먼 frontier 선택 (탐색 극대화)"""
        return max(frontiers,
                   key=lambda p: math.hypot(
                       p[0] - self.robot_x, p[1] - self.robot_y))

    def navigate_to(self, point):
        if not self._nav_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().warn('Nav2 서버 없음')
            return
        goal = NavigateToPose.Goal()
        goal.pose.header.frame_id    = 'map'
        goal.pose.header.stamp       = self.get_clock().now().to_msg()
        goal.pose.pose.position.x    = float(point[0])
        goal.pose.pose.position.y    = float(point[1])
        goal.pose.pose.orientation.w = 1.0
        self.navigating   = True
        self.current_goal = point
        self.get_logger().info(f'이동 시작: ({point[0]:.2f}, {point[1]:.2f})')
        future = self._nav_client.send_goal_async(
            goal, feedback_callback=self.feedback_cb)
        future.add_done_callback(self.goal_response_cb)

    def feedback_cb(self, feedback):
        """이동 중 현재 거리 로깅"""
        dist = feedback.feedback.distance_remaining
        self.get_logger().info(f'  남은 거리: {dist:.2f}m', throttle_duration_sec=2.0)

    def goal_response_cb(self, future):
        handle = future.result()
        if not handle.accepted:
            self.get_logger().warn('목표 거부됨')
            self.navigating = False
            if self.current_goal:
                self.visited_goals.append(self.current_goal)
            return
        handle.get_result_async().add_done_callback(self.result_cb)

    def result_cb(self, future):
        status = future.result().status
        if status == 4:
            self.get_logger().info('도착 완료 — 재탐색 시작')
        else:
            self.get_logger().warn(f'Nav2 상태 {status} — 방문 목록 추가')
        # 성공/실패 모두 방문 기록에 추가 → 같은 곳 재방문 방지
        if self.current_goal:
            self.visited_goals.append(self.current_goal)
        self.navigating = False

    def publish_markers(self, frontiers):
        arr = MarkerArray()
        d = Marker()
        d.action = Marker.DELETEALL
        arr.markers.append(d)
        self.marker_pub.publish(arr)

        arr = MarkerArray()
        for i, (wx, wy) in enumerate(frontiers):
            m = Marker()
            m.header.frame_id    = 'map'
            m.header.stamp       = self.get_clock().now().to_msg()
            m.ns                 = 'frontiers'
            m.id                 = i
            m.type               = Marker.SPHERE
            m.action             = Marker.ADD
            m.pose.position.x    = wx
            m.pose.position.y    = wy
            m.pose.position.z    = 0.15
            m.scale.x = m.scale.y = m.scale.z = 0.2
            m.color.r = 1.0
            m.color.g = 0.5
            m.color.a = 1.0
            m.lifetime.sec       = 5
            arr.markers.append(m)
        self.marker_pub.publish(arr)


def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(FrontierDetector())
    rclpy.shutdown()
