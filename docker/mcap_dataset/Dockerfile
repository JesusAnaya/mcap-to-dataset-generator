FROM carla_ros_bridge:latest as bridge
FROM ros:humble-ros-base-jammy

ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN apt-get update && apt-get install -y \
    ros-humble-cv-bridge \
    ros-humble-ros-base \
    ros-humble-ros2bag \
    ros-humble-rosbag2-transport \
    ros-humble-rosbag2-storage-mcap \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip3 install --upgrade pip

COPY --from=bridge /opt/carla-ros-bridge/install /opt/carla-ros-bridge/install

WORKDIR /home/code
COPY ./docker/mcap_dataset/requirements.txt /home/code/requirements.txt
RUN pip3 install -r requirements.txt

# Copy the source code
COPY ./src /home/code

COPY ./docker/mcap_dataset/ros_entrypoint.sh /entrypoint.sh

RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]

CMD ["bash"]
