from datetime import datetime
from tqdm import tqdm
import tensorflow as tf
import pandas as pd
import numpy as np
import cv2
import os
import argparse

parser = argparse.ArgumentParser(description='Process Dataset folder based on CSV file and images to generate TFRecord')
parser.add_argument('--dataset_path', type=str, help='Path to the dataset')
parser.add_argument('--dest_path', type=str, help='Path to save the TFRecord dataset')


class TFRecordCreator(object):
    def __init__(self, root_path, tfrecord_path):
        now_str = datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
        self.root_path = root_path
        self.csv_file_path = os.path.join(root_path, 'data.csv')
        self.tfrecord_file_path = os.path.join(tfrecord_path, f'dataset_{now_str}.tfrecord')

    @staticmethod
    def _bytes_feature(value):
        """Returns a bytes_list from a string / byte."""
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


    @staticmethod
    def _float_feature(value):
        """Returns a float_list from a float / double."""
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

    def image_to_byte_array(self, image_path):
        """Load an image and convert it to a byte array."""
        image = cv2.imread(image_path)
        # return bytes array of a JPG image
        img_encode = cv2.imencode('.jpg', image)[1]
        data_encode = np.array(img_encode)
        return data_encode.tobytes()

    def process_row(self, row, writer):
        """Process a single row of CSV file and return a tf.train.Example."""

        # Process center image
        center_steering_value = round(float(row['steering']), 4)
        center_image_path = os.path.join(self.root_path, row['center_image'])
        center_image_bytes = self.image_to_byte_array(center_image_path)
        tf_example = self.create_tf_example(center_image_bytes, center_steering_value)
        writer.write(tf_example.SerializeToString())

        # Process left image
        left_steering_value = round(center_steering_value + 0.1, 4)
        left_image_path = os.path.join(self.root_path, row['left_image'])
        left_image_bytes = self.image_to_byte_array(left_image_path)
        tf_example = self.create_tf_example(left_image_bytes, left_steering_value)
        writer.write(tf_example.SerializeToString())

        # Process right image
        right_steering_value = round(center_steering_value - 0.1, 4)
        right_image_path = os.path.join(self.root_path, row['right_image'])
        right_image_bytes = self.image_to_byte_array(right_image_path)
        tf_example = self.create_tf_example(right_image_bytes, right_steering_value)
        writer.write(tf_example.SerializeToString())

    def create_tf_example(self, image_bytes, steering_value):
        # Create a tf.train.Example
        feature = {
            'steering': self._float_feature(steering_value),
            'image': self._bytes_feature(image_bytes),
        }

        return tf.train.Example(features=tf.train.Features(feature=feature))

    def convert_to_tfrecord(self):
        """Read CSV and convert all rows to TFRecord format."""
        print(f"Reading CSV file from: {self.csv_file_path}")
        df = pd.read_csv(self.csv_file_path)

        print(f"Creating TFRecord file at: {self.tfrecord_file_path}")
        with tf.io.TFRecordWriter(self.tfrecord_file_path) as writer:
            for _, row in tqdm(df.iterrows()):
                self.process_row(row, writer)

        print(f"TFRecord file created at: {self.tfrecord_file_path}")


def main():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    args = parser.parse_args()
    creator = TFRecordCreator(
        root_path=args.dataset_path,
        tfrecord_path=args.dest_path
    )
    creator.convert_to_tfrecord()


if __name__ == "__main__":
    main()
