#!/bin/bash
set -e

# ROS2 Humble 기본 환경 소싱
source /opt/ros/humble/setup.bash

# map_merge 워크스페이스 소싱 (빌드 완료 시)
if [ -f /opt/map_merge_ws/install/setup.bash ]; then
  source /opt/map_merge_ws/install/setup.bash
fi

# 개발용 워크스페이스 소싱 (colcon build 이후)
if [ -f /ros2_ws/install/setup.bash ]; then
  source /ros2_ws/install/setup.bash
fi

# 환경 변수
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export TURTLEBOT3_MODEL=burger
export GAZEBO_MODEL_PATH=/opt/ros/humble/share/turtlebot3_gazebo/models

exec "$@"
