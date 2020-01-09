import unittest
import pickle
import os
import logging

import cv2

from critique.measure import PoseHeuristics
from tests.utils import load_test_poses

TEST_DATA_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 
    'data'
)

class TestPoseHeuristics(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        pass

    def test_measurements(self):
        pose = load_test_poses('tpose')
        heuristics = PoseHeuristics(pose)

        logging.info(heuristics)