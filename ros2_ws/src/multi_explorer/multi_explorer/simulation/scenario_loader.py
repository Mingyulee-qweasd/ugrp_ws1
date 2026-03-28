"""
simulation/scenario_loader.py

맵/로봇 초기위치 설정 파일 로딩 (순수 Python)
오픈소스 대응: setMap.m 일부

현재는 스켈레톤. 추후 YAML 기반 시나리오 로딩 구현 예정.
"""


class ScenarioLoader:
    """시뮬레이션 시나리오(맵, 로봇 초기 위치 등) 로딩."""

    def __init__(self, config_path: str = None):
        self.config_path = config_path
        self.robot_starts = []
        self.map_file = None

    def load(self) -> dict:
        """
        설정 파일을 읽어 시나리오 정보 반환.

        Returns:
            {'robot_starts': [(x,y), ...], 'map_file': str, ...}
        """
        # TODO: YAML 파일 파싱 구현
        return {
            'robot_starts': [(0.0, 0.0), (0.5, 0.0), (0.25, 0.4)],
            'map_file': None,
            'num_robots': 3,
        }
