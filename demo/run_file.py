import os
import time
import shutil
import logging
import argparse
import subprocess
import pickle
import io

import cv2
import numpy as np
import tensorflow as tf

from src.critique.settings import BASE_DIR, MODEL_PATH
from src.critique.estimator import PoseEstimator, draw_humans

WIDTH = 432
HEIGHT = 368
GRAPH_PATH = MODEL_PATH
CAMERA = 0
RESIZE_OUT_RATIO = 4.0

logger = logging.getLogger('TfPoseEstimator-WebCam')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

fps_time = 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run pose estimation on a video.')
    parser.add_argument('-v', '--video', help='Path to video.')
    parser.add_argument('-i', '--image', help='Path to image.')
    parser.add_argument('-s', '--skip', help='Number of frames to skip between detections. 1 means skip none, 2 means detect every other frame.', default=1, type=int)
    parser.add_argument('-f', '--frames', help='Number of frames to process.', default=-1, type=int)
    parser.add_argument('output', help='For videos this should be a folder path, for images an file path.')

    args = parser.parse_args()

    if args.video is None and args.image is None:
        raise ValueError("--video or --image must be given.")
    elif args.video is not None and args.image is not None:
        raise ValueError("--video and --image cannot be given together.")

    logger.debug('initialization %s' % (GRAPH_PATH))

    tf_config = tf.ConfigProto()

    e = PoseEstimator()

    if args.video:
        logger.debug('cam read+')
        cam = cv2.VideoCapture(args.video)
        ret_val, image = cam.read()
        logger.info('cam image=%dx%d' % (image.shape[1], image.shape[0]))

        tmp_folder = os.path.join(os.path.join(args.output, 'tmp'))

        try:
            os.mkdir(tmp_folder)
        except OSError:
            pass

        frame_num = 0
        while True:
            frame_num += 1

            ret_val, image = cam.read()

            if not ret_val:
                break

            if args.frames != -1 and frame_num > args.frames:
                break
                
            if frame_num % args.skip != 0:
                continue

            logger.debug('image process+')
            humans = e.inference(image, resize_to_default=True, upsample_size=RESIZE_OUT_RATIO)

            logger.debug('postprocess+')
            image = draw_humans(image, humans, imgcopy=False)

            logger.debug('show+')

            cv2.imwrite(os.path.join(os.path.join(args.output, 'tmp'), f'{frame_num:05}.jpg'), image)

            fps_time = time.time()
            logger.debug('finished+')

        cam.release()

        subprocess.run("ffmpeg -y -framerate {} -pattern_type glob -i {} {}".format(10, os.path.join(tmp_folder, '*.jpg'), os.path.join(args.output, 'output.mp4')).split())

        shutil.rmtree(tmp_folder)
    
    elif args.image:
        img = cv2.imread(args.image)
        
        humans = e.inference(img, resize_to_default=True, upsample_size=RESIZE_OUT_RATIO)
        res_image = draw_humans(img, humans, imgcopy=False)

        cv2.imwrite(args.output, res_image)