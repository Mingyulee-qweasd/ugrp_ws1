#!/usr/bin/env python3
"""
robot_state_machine.py

전체 탐색 흐름 제어:
  INIT_SCAN  → 로봇들이 제자리 회전하며 초기 스캔
  INIT_MERGE → 초기 맵 merge + K-means 영역 할당
  EXPLORE    → frontier 탐색 (exploration_planner 가 담당)
  RENDEZVOUS → 집결지 이동 (merge_event or 주기 도달)
  DONE       → frontier 없음, 탐색 완료
"""
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
from geometry_msgs.msg import Twist
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
import math, time

NUM_ROBOTS       = 3
ROBOT_NS         = ['tb3_0', 'tb3_1', 'tb3_2']
INIT_SCAN_SEC    = 8.0        # 초기 제자리 회전 시간
RENDEZVOUS_POS   = (0.0, 0.0) # 집결 위치 (추후 동적으로 변경 가능)
RENDEZVOUS_TOL   = 0.4        # 집결 도착 판정 반경

class StateMachine(Node):
    def __init__(self):
        super().__init__('robot_state_machine')

        self.state         = 'INIT_SCAN'
        self.init_scan_start = self.get_clock().now()
        self.robots_at_rv  = [False] * NUM_ROBOTS   # 집결지 도착 여부
        self.explore_done  = False
        #추가한 부분
        self.rendezvous_sent = False  # 목표 중복 전송 방지 플래그

        # 초기 스캔용 cmd_vel 퍼블리셔
        self.cmd_pubs = [
            self.create_publisher(Twist, f'/{ns}/cmd_vel', 10)
            for ns in ROBOT_NS]

        # Nav2 액션 클라이언트 (집결 이동용)
        self.nav_clients = [
            ActionClient(self, NavigateToPose, f'/{ns}/navigate_to_pose')
            for ns in ROBOT_NS]

        # merge_event 구독 (map_merger 에서 옴)
        self.merge_sub = self.create_subscription(
            Bool, '/merge_event', self._merge_cb, 10)

        # 탐색 완료 신호 구독 (exploration_planner 에서 옴)
        self.done_sub = self.create_subscription(
            Bool, '/exploration_done', self._done_cb, 10)

        #추가한 부분
        # 집결 명령 구독 (map_merger 에서 옴)
        #추가한 부분
        self.rv_sub = self.create_subscription(
            #추가한 부분
            Bool, '/rendezvous_command', self._rv_command_cb, 10)

        # 상태 퍼블리셔 (exploration_planner 가 구독)
        self.state_pub = self.create_publisher(String, '/system_state', 10)

        self.create_timer(0.5, self._tick)
        self.get_logger().info('State Machine 시작 — INIT_SCAN')

    # ── 콜백 ──────────────────────────────────────────────────

    # _merge_cb 중복 정의 제거 후 INIT_MERGE / RENDEZVOUS 통합 처리
    def _merge_cb(self, msg: Bool):
        if not msg.data:
            return
        if self.state == 'INIT_MERGE':
            self.get_logger().info('초기 Merge 완료 → EXPLORE')
            self.state = 'EXPLORE'
            self._publish_state()
        elif self.state == 'RENDEZVOUS':
            self.get_logger().info('Rendezvous Merge 완료 → EXPLORE')
            self.state = 'EXPLORE'
            self.robots_at_rv = [False] * NUM_ROBOTS
            self.explore_done = False
            #추가한 부분
            self.rendezvous_sent = False
            self._publish_state()

    def _done_cb(self, msg: Bool):
        if msg.data:
            self.explore_done = True

    #추가한 부분
    def _rv_command_cb(self, msg: Bool):
        #추가한 부분
        """map_merger 의 주기 신호를 받아 EXPLORE → RENDEZVOUS 전환"""
        #추가한 부분
        if msg.data and self.state == 'EXPLORE':
            #추가한 부분
            self.get_logger().info('집결 명령 수신 → RENDEZVOUS')
            #추가한 부분
            self.state = 'RENDEZVOUS'
            #추가한 부분
            self.rendezvous_sent = False
            #추가한 부분
            self._publish_state()

    # ── 메인 틱 ───────────────────────────────────────────────

    def _tick(self):
        if self.state == 'INIT_SCAN':
            self._do_init_scan()
        elif self.state == 'INIT_MERGE':
            self._do_init_merge()
        elif self.state == 'EXPLORE':
            self._do_explore()
        elif self.state == 'RENDEZVOUS':
            self._do_rendezvous()
        elif self.state == 'DONE':
            self._do_done()

    def _do_init_scan(self):
        """로봇들을 제자리 회전시켜 초기 스캔"""
        elapsed = (self.get_clock().now() - self.init_scan_start
                   ).nanoseconds / 1e9
        if elapsed < INIT_SCAN_SEC:
            twist = Twist()
            twist.angular.z = 0.5   # 천천히 회전
            for pub in self.cmd_pubs:
                pub.publish(twist)
        else:
            # 회전 정지
            for pub in self.cmd_pubs:
                pub.publish(Twist())
            self.get_logger().info(f'초기 스캔 완료 ({INIT_SCAN_SEC}s) → INIT_MERGE')
            self.state = 'INIT_MERGE'
            self._publish_state()

    def _do_init_merge(self):
        """map_merger 가 merge_event 를 줄 때까지 대기
        (map_merger 의 proximity 조건이 시작 위치에서 바로 충족됨)"""
        pass  # map_merger 의 /merge_event 를 _merge_cb 에서 처리

    def _do_explore(self):
        if self.explore_done:
            self.get_logger().info('탐색 완료 → DONE')
            self.state = 'DONE'
            self._publish_state()

    def _do_rendezvous(self):
        #추가한 부분
        """목표 전송 후 모든 로봇 도착 대기, 도착 완료 시 merge_event 콜백이 EXPLORE 로 전환"""
        #추가한 부분
        if not self.rendezvous_sent:
            #추가한 부분
            self._send_rendezvous_goals()
            #추가한 부분
            self.rendezvous_sent = True
        #추가한 부분
        if all(self.robots_at_rv):
            #추가한 부분
            self.get_logger().info('모든 로봇 집결 완료 — merge 대기 중')
            # map_merger 가 proximity 조건을 감지 → merge → /merge_event
            # → _merge_cb 에서 EXPLORE 로 전환

    def _do_done(self):
        #추가한 부분
        """탐색 완료 후 집결지로 이동 (중복 전송 방지)"""
        #추가한 부분
        if not self.rendezvous_sent:
            #추가한 부분
            self._send_rendezvous_goals()
            #추가한 부분
            self.rendezvous_sent = True

    def _send_rendezvous_goals(self):
        """모든 로봇에게 집결지 목표 전송"""
        rx, ry = RENDEZVOUS_POS
        for i, client in enumerate(self.nav_clients):
            if self.robots_at_rv[i]:
                continue
            if not client.wait_for_server(timeout_sec=0.5):
                continue
            goal = NavigateToPose.Goal()
            goal.pose.header.frame_id = 'map'
            goal.pose.header.stamp    = self.get_clock().now().to_msg()
            goal.pose.pose.position.x    = float(rx)
            goal.pose.pose.position.y    = float(ry)
            goal.pose.pose.orientation.w = 1.0
            future = client.send_goal_async(goal)
            future.add_done_callback(
                lambda f, idx=i: self._rv_goal_cb(f, idx))

    def _rv_goal_cb(self, future, robot_idx):
        goal_handle = future.result()
        if not goal_handle.accepted:
            return
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(
            lambda f, idx=robot_idx: self._rv_result_cb(f, idx))

    def _rv_result_cb(self, future, robot_idx):
        self.robots_at_rv[robot_idx] = True
        self.get_logger().info(f'[tb3_{robot_idx}] 집결지 도착')

    def _publish_state(self):
        msg = String()
        msg.data = self.state
        self.state_pub.publish(msg)
        self.get_logger().info(f'상태 전환: {self.state}')


def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(StateMachine())
    rclpy.shutdown()
