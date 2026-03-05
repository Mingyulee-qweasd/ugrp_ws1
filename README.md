# 🤖 UGRP Multi-Robot Exploration Simulation

ROS2 Humble + TurtleBot3 기반 멀티로봇 자율탐색 시뮬레이션 프로젝트입니다.

---

## 📁 프로젝트 구조

```
ugrp_ws/
├── docker/
│   ├── Dockerfile          # ROS2 환경 이미지 정의
│   └── entrypoint.sh       # 컨테이너 진입점 스크립트
├── docker-compose.yml      # 컨테이너 실행 설정
└── ros2_ws/
    └── src/
        └── multi_explorer/ # 멀티로봇 탐색 패키지
```

---

## ✅ 사전 요구사항

### 필수 설치
- **Ubuntu 22.04** (권장) 또는 20.04
- **Docker** → [설치 가이드](https://docs.docker.com/engine/install/ubuntu/)
- **Docker Compose** → [설치 가이드](https://docs.docker.com/compose/install/)

### Docker 설치 빠른 방법
```bash
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER
newgrp docker
```

---

## 🚀 처음 세팅하는 방법 (최초 1회)

### 1. 저장소 클론
```bash
git clone https://github.com/<팀장_계정>/ugrp_ws.git
cd ugrp_ws
```

### 2. 자동 세팅 스크립트 실행
```bash
chmod +x setup.sh
./setup.sh
```
> ⏳ Docker 이미지 빌드는 **10~20분** 소요됩니다. 최초 1회만 필요해요.

### 3. (또는) 수동 세팅
```bash
# X11 GUI 권한 허용
xhost +local:docker

# Docker 이미지 빌드
docker build -t tb3_humble:latest ./docker/

# ROS2 워크스페이스 빌드
docker compose run --rm ros2_humble bash -c \
  "cd /ros2_ws && colcon build --symlink-install"
```

---

## ▶️ 시뮬레이션 실행 방법

### 컨테이너 시작 및 접속
```bash
# 백그라운드로 컨테이너 시작
docker compose up -d

# 컨테이너 내부 접속
docker compose exec ros2_humble bash
```

### 컨테이너 안에서 시뮬레이션 실행
```bash
# (컨테이너 내부)
# Gazebo 시뮬레이션 실행 예시
ros2 launch turtlebot3_gazebo turtlebot3_world.launch.py

# 멀티로봇 탐색 실행 예시 (별도 터미널로 접속 후)
docker compose exec ros2_humble bash
ros2 launch multi_explorer <launch_file>.launch.py
```

### 여러 터미널로 접속하는 방법
```bash
# 새 터미널마다 아래 명령어로 같은 컨테이너에 접속
docker compose exec ros2_humble bash
```

### 컨테이너 종료
```bash
docker compose down
```

---

## 🖥️ GUI (RViz2 / Gazebo) 안 열릴 때

```bash
# 실행 전 반드시 아래 명령어 실행
xhost +local:docker

# DISPLAY 변수 확인
echo $DISPLAY   # :1 또는 :0 이 나와야 함
```

---

## 🔄 코드 수정 후 재빌드

`ros2_ws/src/` 안의 코드를 수정했다면:
```bash
docker compose exec ros2_humble bash -c \
  "cd /ros2_ws && colcon build --symlink-install"
```

---

## ⚙️ 환경 정보

| 항목 | 버전/설정 |
|------|-----------|
| ROS2 | Humble |
| OS Base | Ubuntu 22.04 |
| 시뮬레이터 | Gazebo (Classic) |
| 로봇 모델 | TurtleBot3 Burger |
| DDS | CycloneDDS |
| SLAM | slam-toolbox, Cartographer |
| Navigation | Nav2 |

---

## ❓ 자주 발생하는 문제

**Q. `docker: permission denied` 오류**
```bash
sudo usermod -aG docker $USER
newgrp docker
```

**Q. Gazebo 화면이 안 뜸**
```bash
xhost +local:docker
# docker-compose.yml의 DISPLAY 값 확인
echo $DISPLAY
```

**Q. colcon build 실패**
```bash
# 의존성 재설치 후 재빌드
docker compose exec ros2_humble bash -c \
  "cd /ros2_ws && rosdep install --from-paths src --ignore-src -r -y && colcon build"
```

---

## 📬 문의
문제 발생 시 팀 단톡방 또는 GitHub Issues에 남겨주세요.
