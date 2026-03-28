"""
nodes/coordinator_node.py (decentralized 버전)

역할 축소: 랑데뷰 타이밍만 주기적으로 알림.
목표 설정, 맵 합성 등은 각 robot_agent_node가 독립 처리.
"""
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
from geometry_msgs.msg import Twist

NUM_ROBOTS = 3
ROBOT_NS = ['tb3_0', 'tb3_1', 'tb3_2']
RENDEZVOUS_PERIOD = 60.0
INIT_SCAN_SEC = 8.0


class CoordinatorNode(Node):
    def __init__(self):
        super().__init__('coordinator_node')

        self.init_scan_start = self.get_clock().now()
        self.init_done = False
        self.last_rv_time = None

        self.cmd_pubs = [
            self.create_publisher(Twist, f'/{ns}/cmd_vel', 10)
            for ns in ROBOT_NS]

        self.rv_pub = self.create_publisher(Bool, '/rendezvous_command', 10)

        self.create_timer(0.5, self._tick)
        self.get_logger().info('Coordinator Node (decentralized) 시작')

    def _tick(self):
        now = self.get_clock().now()

        if not self.init_done:
            elapsed = (now - self.init_scan_start).nanoseconds / 1e9
            if elapsed < INIT_SCAN_SEC:
                twist = Twist()
                twist.angular.z = 0.5
                for pub in self.cmd_pubs:
                    pub.publish(twist)
            else:
                for pub in self.cmd_pubs:
                    pub.publish(Twist())
                self.init_done = True
                self.last_rv_time = now
                self.get_logger().info('초기 스캔 완료')
            return

        if self.last_rv_time is not None:
            rv_elapsed = (now - self.last_rv_time).nanoseconds / 1e9
            if rv_elapsed >= RENDEZVOUS_PERIOD:
                self.rv_pub.publish(Bool(data=True))
                self.last_rv_time = now
                self.get_logger().info(
                    f'랑데뷰 명령 전송 ({RENDEZVOUS_PERIOD}초 주기)')


def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(CoordinatorNode())
    rclpy.shutdown()
