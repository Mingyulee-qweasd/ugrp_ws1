#!/bin/bash
# =============================================================
# UGRP Multi-Robot Exploration - 팀원 초기 세팅 스크립트
# =============================================================
set -e

echo "======================================"
echo " UGRP 시뮬레이션 환경 세팅 시작"
echo "======================================"

# 1. Docker 설치 확인
if ! command -v docker &> /dev/null; then
    echo "[ERROR] Docker가 설치되지 않았습니다."
    echo "  → https://docs.docker.com/engine/install/ubuntu/ 참고"
    exit 1
fi

if ! command -v docker compose &> /dev/null; then
    echo "[ERROR] Docker Compose가 설치되지 않았습니다."
    echo "  → https://docs.docker.com/compose/install/ 참고"
    exit 1
fi

echo "[OK] Docker 확인 완료"

# 2. DISPLAY 환경변수 확인 및 설정
if [ -z "$DISPLAY" ]; then
    echo "[WARN] DISPLAY 변수가 없습니다. :0 으로 설정합니다."
    export DISPLAY=:0
fi
echo "[OK] DISPLAY=$DISPLAY"

# 3. X11 디스플레이 권한 허용 (GUI용)
echo "[INFO] X11 디스플레이 권한 설정 중..."
xhost +local:docker 2>/dev/null || echo "[WARN] xhost 설정 실패 (GUI 없는 환경일 수 있음)"

# 4. GPU 환경 감지 안내
echo ""
echo "======================================"
echo " GPU 환경 확인"
echo "======================================"
if command -v nvidia-smi &> /dev/null; then
    echo "[INFO] NVIDIA GPU 감지됨 → docker-compose.yml의 소프트웨어 렌더링 주석 유지하세요"
elif lspci | grep -i "intel" | grep -i "graphics" &> /dev/null; then
    echo "[INFO] Intel GPU 감지됨"
    echo "       → Gazebo가 안 열리면 docker-compose.yml에서 아래 두 줄 주석 해제하세요:"
    echo "         LIBGL_ALWAYS_SOFTWARE=1"
    echo "         GALLIUM_DRIVER=llvmpipe"
else
    echo "[INFO] GPU 정보를 자동 감지하지 못했습니다. README를 참고하세요."
fi
echo ""

# 5. Docker 이미지 빌드
echo "[INFO] Docker 이미지 빌드 중... (최초 1회, 10~20분 소요)"
docker compose build

echo "[OK] Docker 이미지 빌드 완료"

# 6. ROS2 워크스페이스 빌드
echo "[INFO] ROS2 워크스페이스 빌드 중..."
docker compose run --rm ros2_humble bash -c "\
    cd /ros2_ws && \
    source /opt/ros/humble/setup.bash && \
    colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release"

echo "[OK] ROS2 워크스페이스 빌드 완료"

echo ""
echo "======================================"
echo " 세팅 완료!"
echo ""
echo " 실행 방법:"
echo "   docker compose up -d"
echo "   docker compose exec ros2_humble bash"
echo "======================================"
