import os
import multiprocessing
multiprocessing.set_start_method('spawn')

import cv2

from critique.celery import celery_app
from critique.pose.estimator import PoseEstimator
from critique.io import url_to_image

estimator = PoseEstimator()

@celery_app.task()
def critique_image(path):

    if path[:4] == 'http': 
        img = url_to_image(path)
    else:
        img = cv2.imread(path)

    poses = estimator.estimate(img)

    return [pose.keypoints.tolist() for pose in poses]