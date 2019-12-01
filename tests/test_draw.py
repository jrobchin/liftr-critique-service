import unittest
import pickle

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2

import warnings  
from src.estimator.estimator import draw_humans

TEST_DATA_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 
    'data'
)

class TestDraw(unittest.TestCase):
    def test_draw_humans(self):
        pass