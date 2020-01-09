import unittest
import pickle
import time
import os
import warnings
import logging

import cv2

from critique.pose.estimator import PoseEstimator
from tests.utils import load_test_poses, load_test_image

class TestEstimator(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.estimator = PoseEstimator()

    def test_pose_estimation(self):
        prev_det = load_test_poses('tpose')
        image = load_test_image('tpose')

        start_time = time.time()

        poses = self.estimator.estimate(image)

        est_time = time.time() - start_time

        logging.info(f"Pose estimate took {est_time}s")

        if any([p1 != p2 for p1, p2 in zip(poses, prev_det)]):
            logging.warn("Poses are different than expected!")