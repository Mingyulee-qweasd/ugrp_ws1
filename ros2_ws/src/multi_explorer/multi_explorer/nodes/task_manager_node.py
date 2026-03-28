"""
nodes/task_manager_node.py

ROS2 노드: task 발견/할당 결과 publish
순수 로직은 core/task_allocator.py 에 위임.

오픈소스 대응: TaskAllocation_DS.m 호출부
현재는 스켈레톤. 추후 구현 예정.
"""
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

from multi_explorer.core.task_allocator import TaskAllocator


class TaskManagerNode(Node):
    def __init__(self):
        super().__init__('task_manager_node')
        self.allocator = TaskAllocator(num_robots=3)
        self.get_logger().info('Task Manager Node 시작 (스켈레톤)')


def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(TaskManagerNode())
    rclpy.shutdown()
