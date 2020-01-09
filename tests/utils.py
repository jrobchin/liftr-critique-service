import os
import pickle

import cv2


TEST_DATA_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 
    'data'
)
TEST_POSE_PATH = os.path.join(
    TEST_DATA_PATH, 
    'pose'
)
TEST_IMAGE_PATH = os.path.join(
    TEST_DATA_PATH, 
    'image'
)


def load_test_poses(name, first=False):
    """
    Returns a 2D list of poses if first is False.

    If first is True, return the first pose of the first frame, useful for single
    image pose detections.
    """
    pose_path = os.path.join(TEST_POSE_PATH, f'{name}.pkl')
    with open(pose_path, 'rb') as f:
        poses = pickle.load(f)
    
    if first:
        return poses[0][0]
    return poses


def load_test_image(name):
    image_path = os.path.join(TEST_IMAGE_PATH, f'{name}.jpg')
    image = cv2.imread(image_path)

    return image