"""
core/robot_state_machine.py

순수 FSM 로직 (ROS 의존 없음)
상태: INIT_SCAN → INIT_MERGE → EXPLORE → RENDEZVOUS → DONE

원본: 기존 robot_state_machine.py 에서 ROS 통신 제거
오픈소스 대응: mainFSM_run.m
"""


class RobotFSM:
    """
    상태 전이 규칙만 관리하는 순수 클래스.
    ROS 토픽/액션은 nodes/coordinator_node.py 가 담당.
    """

    STATES = ('INIT_SCAN', 'INIT_MERGE', 'EXPLORE', 'RENDEZVOUS', 'DONE')

    def __init__(self, num_robots=3, init_scan_sec=8.0, rendezvous_tol=0.4):
        self.state = 'INIT_SCAN'
        self.num_robots = num_robots
        self.init_scan_sec = init_scan_sec
        self.rendezvous_tol = rendezvous_tol

        self.robots_at_rv = [False] * num_robots
        self.explore_done = False
        self.rendezvous_sent = False

    # ── 상태 전이 메서드 ──────────────────────────────────────

    def tick_init_scan(self, elapsed_sec: float) -> bool:
        """초기 스캔 시간이 지났으면 INIT_MERGE 로 전이. 전이 여부 반환."""
        if self.state != 'INIT_SCAN':
            return False
        if elapsed_sec >= self.init_scan_sec:
            self.state = 'INIT_MERGE'
            return True
        return False

    def on_merge_complete(self) -> str:
        """merge_event 수신 시 호출. 새 상태 반환."""
        if self.state == 'INIT_MERGE':
            self.state = 'EXPLORE'
        elif self.state == 'RENDEZVOUS':
            self.state = 'EXPLORE'
            self.robots_at_rv = [False] * self.num_robots
            self.explore_done = False
            self.rendezvous_sent = False
        return self.state

    def on_rendezvous_command(self) -> bool:
        """집결 명령 수신 시 호출. 전이 여부 반환."""
        if self.state == 'EXPLORE':
            self.state = 'RENDEZVOUS'
            self.rendezvous_sent = False
            return True
        return False

    def on_exploration_done(self):
        """frontier 없음 신호 수신."""
        self.explore_done = True

    def tick_explore(self) -> bool:
        """EXPLORE 상태에서 탐색 완료 여부 확인. DONE 전이 여부 반환."""
        if self.state != 'EXPLORE':
            return False
        if self.explore_done:
            self.state = 'DONE'
            return True
        return False

    def tick_rendezvous(self) -> dict:
        """
        RENDEZVOUS 상태 틱.
        반환: {'need_send_goals': bool, 'all_arrived': bool}
        """
        result = {'need_send_goals': False, 'all_arrived': False}
        if self.state != 'RENDEZVOUS':
            return result
        if not self.rendezvous_sent:
            result['need_send_goals'] = True
            self.rendezvous_sent = True
        if all(self.robots_at_rv):
            result['all_arrived'] = True
        return result

    def tick_done(self) -> dict:
        """DONE 상태 틱. 집결지 이동 필요 여부 반환."""
        result = {'need_send_goals': False}
        if self.state != 'DONE':
            return result
        if not self.rendezvous_sent:
            result['need_send_goals'] = True
            self.rendezvous_sent = True
        return result

    def mark_robot_arrived(self, robot_idx: int):
        """로봇이 집결지에 도착했음을 표시."""
        if 0 <= robot_idx < self.num_robots:
            self.robots_at_rv[robot_idx] = True
