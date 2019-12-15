import unittest
import pickle
import time

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2

import warnings  
with warnings.catch_warnings():  
    import tensorflow as tf
    warnings.filterwarnings("ignore", category=FutureWarning)
    from src.critique.estimator import PoseEstimator

TEST_DATA_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 
    'data'
)

class TestEstimator(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        tf_config = tf.ConfigProto()
        tf_config.intra_op_parallelism_threads = 8
        tf_config.inter_op_parallelism_threads = 8
        cls.estimator = PoseEstimator()

    def test_pose_estimation(self):
        # gt_humans_path = os.path.join(TEST_DATA_PATH, 'tpose-humans.pkl')
        image_path = os.path.join(TEST_DATA_PATH, 'tpose.jpg')

        # with open(gt_humans_path, 'rb') as f:
        #     gt_humans = pickle.load(f)
        image = cv2.imread(image_path)

        start = time.time()
        num_iters = 30
        for i in range(num_iters):
            pred_humans = self.estimator.inference(image)
        elapsed = time.time() - start
        
        print(f"{num_iters} inferences took {elapsed} seconds total, {elapsed / num_iters} seconds on average.")


        # self.assertEqual(len(pred_humans), len(gt_humans), msg="Predictions are different length.")