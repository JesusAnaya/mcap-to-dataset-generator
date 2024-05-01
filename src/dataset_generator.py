from datetime import datetime
from cv_bridge import CvBridge
from rclpy.serialization import deserialize_message
from carla_msgs.msg import CarlaEgoVehicleStatus, CarlaEgoVehicleControl
from sensor_msgs.msg import Image
from rosidl_runtime_py.utilities import get_message
from datetime import datetime, timedelta
from tqdm import tqdm
import rosbag2_py
import cv2
import os
import argparse
import pandas as pd
import numpy as np

# Assuming the import statements for ROS message types remain the same

parser = argparse.ArgumentParser(description='Process MCAP file and generate dataset')
parser.add_argument('--mcap_file_path', type=str, help='Path to the MCAP file')
parser.add_argument('--root_path', type=str, help='Root path to save the dataset')
bridge = CvBridge()


def deserialize_image_message(message: Image) -> np.ndarray:
    # Convert ROS Image message to JPEG RGB image by cv_bridge
    cv_image = bridge.imgmsg_to_cv2(message, desired_encoding="passthrough")

    # Corp the last 188 pixels from the bottom to up and resize the image to 200x66
    # cv_image = cv_image[:, :]
    cv_image = cv2.resize(cv_image, (224, 224))

    return cv_image


def deserialize_vehicle_status_message(message: CarlaEgoVehicleStatus) -> dict:
    # Implement based on your ROS vehicle status message format
    # The structure of the message is defined in carla_msgs.msg.CarlaEgoVehicleStatus
    # The steering angle is stored in message.control.steer where control is an instance
    # of CarlaEgoVehicleControl
    steering = np.clip(message.control.steer, -1.0, 1.0)
    throttle = np.clip(message.control.throttle, 0.0, 1.0)

    return dict(steering=steering, throttle=throttle)


def save_image(image, path):
    cv2.imwrite(path, image)


class MCAPProcessor(object):
    def __init__(self, mcap_file_path, root_path):
        timestamp = datetime.now().strftime('%Y-%m-%dT%H-%M-%S')

        self.mcap_file_path = mcap_file_path
        self.root_path = root_path
        self.output_dir = os.path.join(root_path, "images")
        self.csv_file_path = os.path.join(root_path, f"data_{timestamp}.csv")
        self.df = pd.DataFrame(
            columns=['timestamp', 'center_image', 'left_image', 'right_image', 'steering', 'throttle']
        )
        os.makedirs(self.output_dir, exist_ok=True)

    def read_messages(self):
        reader = rosbag2_py.SequentialReader()
        storage_options = rosbag2_py.StorageOptions(uri=self.mcap_file_path, storage_id="mcap")
        converter_options = rosbag2_py.ConverterOptions(
            input_serialization_format="cdr", output_serialization_format="cdr"
        )
        reader.open(storage_options, converter_options)

        topic_types = reader.get_all_topics_and_types()

        def typename(topic_name):
            for topic_type in topic_types:
                if topic_type.name == topic_name:
                    return topic_type.type
            raise ValueError(f"Topic {topic_name} not in bag")

        while reader.has_next():
            topic, data, timestamp = reader.read_next()
            msg_type = get_message(typename(topic))
            msg = deserialize_message(data, msg_type)
            yield topic, msg, timestamp
        del reader

    def process_mcap(self):
        checkpoint_interval = 0.05  # 50 milliseconds
        next_checkpoint_time = None
        messages_in_interval = []

        for topic, msg, timestamp_ns in tqdm(self.read_messages()):
            timestamp = datetime.fromtimestamp(timestamp_ns / 1e9)  # Convert nanoseconds to datetime

            if next_checkpoint_time is None or timestamp >= next_checkpoint_time:
                if messages_in_interval:
                    self.process_messages(messages_in_interval)
                    messages_in_interval = []

                # Set next checkpoint time
                if next_checkpoint_time is None:
                    next_checkpoint_time = timestamp
                while timestamp >= next_checkpoint_time:
                    next_checkpoint_time += timedelta(seconds=checkpoint_interval)

            # Collect all messages that are within the interval leading up to the next checkpoint
            if timestamp < next_checkpoint_time:
                messages_in_interval.append((topic, msg, timestamp_ns))

        # Process any remaining messages
        if messages_in_interval:
            self.process_messages(messages_in_interval)

        self.df.to_csv(self.csv_file_path, index=False)

    def process_messages(self, messages):
        images = {}
        steering = None
        throttle = None
        timestamp_str = ""
        # Find the message for each topic that's closest to the median timestamp of collected messages
        median_timestamp = np.median([timestamp_ns for _, _, timestamp_ns in messages])
        closest_messages = {}

        for topic, msg, timestamp_ns in messages:
            if topic not in closest_messages or abs(timestamp_ns - median_timestamp) < abs(closest_messages[topic][1] - median_timestamp):
                closest_messages[topic] = (msg, timestamp_ns)

        for topic, (msg, timestamp_ns) in closest_messages.items():
            timestamp = datetime.fromtimestamp(timestamp_ns / 1e9)
            timestamp_str = timestamp.strftime('%Y-%m-%dT%H-%M-%S-%f')

            if 'image' in topic:
                # Extract identifier (center, left, right) from the topic string
                identifier = topic.split('/')[3]  # Assuming the format "/carla/ego_vehicle/rgb_{identifier}/image"
                image_name = f"{timestamp_str}_{identifier}.jpg"
                images[identifier] = os.path.join("images", image_name)
                image_path = os.path.join(self.output_dir, image_name)
                image = deserialize_image_message(msg)
                save_image(image, image_path)
            elif 'vehicle_status' in topic:
                data_dict = deserialize_vehicle_status_message(msg)
                steering = data_dict['steering']
                throttle = data_dict['throttle']

        if images and steering is not None:
            center_image = images.get('rgb_center', '')
            left_image = images.get('rgb_left', '')
            right_image = images.get('rgb_right', '')

            if not center_image or not left_image or not right_image or steering is None or throttle is None:
                return

            new_row = {
                'timestamp': timestamp_str,
                'center_image': center_image,
                'left_image': left_image,
                'right_image': right_image,
                'steering': steering,
                'throttle': throttle
            }
            new_df = pd.DataFrame([new_row])
            self.df = pd.concat([self.df.astype(new_df.dtypes), new_df.astype(self.df.dtypes)], ignore_index=True)


def main(mcap_file_path, root_path):
    processor = MCAPProcessor(mcap_file_path, root_path)
    processor.process_mcap()


if __name__ == "__main__":
    args = parser.parse_args()
    main(args.mcap_file_path, args.root_path)
