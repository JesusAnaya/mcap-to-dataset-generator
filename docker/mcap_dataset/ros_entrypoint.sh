#!/usr/bin/env bash
set -e

# setup ros environment
source /opt/ros/humble/setup.bash
source /opt/carla-ros-bridge/install/local_setup.bash

exec "$@"