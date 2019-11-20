import os
import time
import shutil
import logging
import argparse
import subprocess

import cv2
import numpy as np

from critique.settings import BASE_DIR
from critique.estimator import PoseEstimator

WIDTH = 432
HEIGHT = 368
GRAPH_PATH = os.path.join(BASE_DIR, 'estimator/models/mobilenet_v2_large-graph_opt.pb')
CAMERA = 0
RESIZE_OUT_RATIO = 4.0
FRAME_SKIP = 5
FRAMES = 100

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
    parser.add_argument('video')
    parser.add_argument('output_dir')

    args = parser.parse_args()

    logger.debug('initialization %s' % (GRAPH_PATH))

    e = PoseEstimator(GRAPH_PATH, target_size=(WIDTH, HEIGHT))
    
    logger.debug('cam read+')
    cam = cv2.VideoCapture(args.video)
    ret_val, image = cam.read()
    logger.info('cam image=%dx%d' % (image.shape[1], image.shape[0]))

    tmp_folder = os.path.join(os.path.join(args.output_dir, 'tmp'))

    try:
        os.mkdir(tmp_folder)
    except OSError:
        pass

    frame_num = 0
    while True:
        frame_num += 1

        if frame_num > FRAMES*FRAME_SKIP:
            break
            
        if frame_num % FRAME_SKIP != 0:
            continue

        ret_val, image = cam.read()

        logger.debug('image process+')
        humans = e.inference(image, resize_to_default=True, upsample_size=RESIZE_OUT_RATIO)

        logger.debug('postprocess+')
        image = PoseEstimator.draw_humans(image, humans, imgcopy=False)

        logger.debug('show+')
        cv2.putText(image,
                    f"FRAME: {frame_num}",
                    (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 2)

        cv2.imwrite(os.path.join(os.path.join(args.output_dir, 'tmp'), f'{frame_num:03}.jpg'), image)

        fps_time = time.time()
        logger.debug('finished+')

    cam.release()
    # ffmpeg -framerate 6 -pattern_type glob -i 'tmp*.jpg' output.mp4
    subprocess.run("ffmpeg -framerate 6 -pattern_type glob -i {} {}".format(os.path.join(tmp_folder, '*.jpg'), os.path.join(args.output_dir, 'output.mp4')).split())

    shutil.rmtree(tmp_folder)