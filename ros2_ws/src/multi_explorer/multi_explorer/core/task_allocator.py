"""
core/task_allocator.py

Task 발견/분배 로직 (순수 Python, ROS 의존 없음)
오픈소스 대응: TaskAllocation_DS.m, allocateTasks.m, generateTask.m

현재는 스켈레톤. 추후 오픈소스 기반으로 구현 예정.
"""


class TaskAllocator:
    """발견된 task를 로봇들에게 분배하는 로직."""

    def __init__(self, num_robots=3):
        self.num_robots = num_robots
        self.tasks = []  # 발견된 task 리스트

    def add_task(self, task_pos: tuple, priority: float = 1.0):
        """새로운 task 등록."""
        self.tasks.append({'pos': task_pos, 'priority': priority, 'assigned': None})

    def allocate(self, robot_positions: list) -> dict:
        """
        로봇 위치 기반으로 task 할당.

        Args:
            robot_positions: [(x, y), ...] 각 로봇 위치

        Returns:
            {robot_id: task_index} 할당 결과
        """
        # TODO: 오픈소스 TaskAllocation_DS.m 기반 구현
        return {}
