import unittest
import pickle
import time
import os
import warnings
import logging

import cv2

from critique.pose.estimator import PoseEstimator

TEST_DATA_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 
    'data'
)

class TestEstimator(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.estimator = PoseEstimator()

    def test_pose_estimation(self):
        # Load previous detections
        prev_det_path = os.path.join(TEST_DATA_PATH, 'tpose-pose.pkl')
        with open(prev_det_path, 'rb') as f:
            prev_det = pickle.load(f)

        image_path = os.path.join(TEST_DATA_PATH, 'tpose.jpg')
        image = cv2.imread(image_path)

        start_time = time.time()

        poses = self.estimator.estimate(image)

        est_time = time.time() - start_time

        with open(os.path.join(TEST_DATA_PATH, 'tpose-pose.pkl'), 'wb') as f:
            pickle.dump(poses, f)

        logging.info(f"Pose estimate took {est_time}s")

        if any([p1 != p2 for p1, p2 in zip(poses, prev_det)]):
            logging.warn("Poses are different than expected!")

        cv2.imwrite(os.path.join(TEST_DATA_PATH, 'tmp.jpg'), image)