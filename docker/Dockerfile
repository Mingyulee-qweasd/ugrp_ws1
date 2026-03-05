# =============================================================
# UGRP Multi-Robot Exploration - 노트북 개발용 Docker 이미지
# Base: Ubuntu 22.04 (Jammy) + ROS2 Humble
# =============================================================
FROM osrf/ros:humble-desktop-full

# --- 빌드 인자 ---
ARG DEBIAN_FRONTEND=noninteractive
ARG ROS_DISTRO=humble

# --- 시스템 패키지 업데이트 + 기본 도구/빌드도구/X11/Mesa/네트워크 ---
RUN apt-get update && apt-get upgrade -y && apt-get install -y \
    # 기본 개발 도구
    git wget curl vim nano htop tmux \
    # Python 도구
    python3-pip python3-dev \
    # ROS2 빌드 도구
    python3-colcon-common-extensions \
    python3-rosdep \
    python3-vcstool \
    # X11 디스플레이 (RViz2, Gazebo GUI)
    libgl1-mesa-glx libgl1-mesa-dri \
    mesa-utils x11-apps \
    # 네트워크 진단
    iputils-ping net-tools iproute2 \
    && rm -rf /var/lib/apt/lists/*

# --- TurtleBot3 & 내비게이션/SLAM/Gazebo 등 ROS2 패키지 ---
# 주의: ros-humble-multirobot-map-merge 는 apt 바이너리 없음 → 아래에서 소스 빌드
RUN apt-get update && apt-get install -y \
    # TurtleBot3 전체
    ros-${ROS_DISTRO}-turtlebot3 \
    ros-${ROS_DISTRO}-turtlebot3-msgs \
    ros-${ROS_DISTRO}-turtlebot3-simulations \
    # SLAM
    ros-${ROS_DISTRO}-slam-toolbox \
    ros-${ROS_DISTRO}-cartographer \
    ros-${ROS_DISTRO}-cartographer-ros \
    # Navigation2
    ros-${ROS_DISTRO}-navigation2 \
    ros-${ROS_DISTRO}-nav2-bringup \
    # DDS (멀티로봇 통신 안정화)
    ros-${ROS_DISTRO}-rmw-cyclonedds-cpp \
    # Gazebo (시뮬레이션)
    ros-${ROS_DISTRO}-gazebo-ros-pkgs \
    ros-${ROS_DISTRO}-gazebo-ros2-control \
    # 기타 유틸
    ros-${ROS_DISTRO}-teleop-twist-keyboard \
    ros-${ROS_DISTRO}-rqt* \
    && rm -rf /var/lib/apt/lists/*

# --- Python 알고리즘 의존성 ---
RUN pip3 install --no-cache-dir \
    numpy \
    scipy \
    scikit-learn \
    opencv-python-headless \
    matplotlib \
    pandas \
    transforms3d

# --- 환경 변수 ---
ENV RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
ENV TURTLEBOT3_MODEL=burger
ENV GAZEBO_MODEL_PATH=/opt/ros/humble/share/turtlebot3_gazebo/models

# --- rosdep 초기화 ---
RUN rosdep init 2>/dev/null || true && rosdep update

# --- multirobot_map_merge: apt 바이너리 없으므로 소스 직접 빌드 ---
WORKDIR /opt/map_merge_ws/src
RUN git clone -b main https://github.com/robo-friends/m-explore-ros2.git

WORKDIR /opt/map_merge_ws
RUN bash -c "\
    source /opt/ros/humble/setup.bash && \
    rosdep install --from-paths src --ignore-src -r -y && \
    colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release"

# map_merge 워크스페이스를 ROS2 환경에 영구 등록
RUN echo "source /opt/map_merge_ws/install/setup.bash" \
    >> /opt/ros/humble/setup.bash

# --- .bashrc에 ROS2 환경 영구 등록 ---
# docker compose exec로 접속할 때도 자동 소싱됨
RUN echo "source /opt/ros/humble/setup.bash" >> /root/.bashrc && \
    echo "source /opt/map_merge_ws/install/setup.bash" >> /root/.bashrc && \
    echo "if [ -f /ros2_ws/install/setup.bash ]; then source /ros2_ws/install/setup.bash; fi" >> /root/.bashrc && \
    echo "export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp" >> /root/.bashrc && \
    echo "export TURTLEBOT3_MODEL=burger" >> /root/.bashrc && \
    echo "export GAZEBO_MODEL_PATH=/opt/ros/humble/share/turtlebot3_gazebo/models" >> /root/.bashrc

# --- ROS2 워크스페이스 생성 ---
RUN mkdir -p /ros2_ws/src
WORKDIR /ros2_ws

# --- entrypoint 스크립트 복사 ---
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
