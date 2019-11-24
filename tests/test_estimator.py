import unittest
import pickle
import os

import cv2

from critique.estimator import PoseEstimator

TEST_DATA_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 
    'data'
)

class TestEstimator(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.estimator = PoseEstimator()

    def test_pose_estimation(self):
        gt_humans_path = os.path.join(TEST_DATA_PATH, 'tpose-humans.pkl')
        image_path = os.path.join(TEST_DATA_PATH, 'tpose.jpg')

        with open(gt_humans_path, 'rb') as f:
            gt_humans = pickle.load(f)
        image = cv2.imread(image_path)
        pred_humans = self.estimator.inference(image)

        self.assertEqual(len(pred_humans), len(gt_humans), msg="Predictions are different length.")