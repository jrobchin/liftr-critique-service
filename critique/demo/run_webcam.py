import argparse
import logging
import time

import cv2
import numpy as np

from critique.estimator import PoseEstimator

HEIGHT = 320
WIDTH = 320
GRAPH_PATH = '/src/estimator/models/mobilenet_v2_large-graph_opt.pb'
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
    logger.debug('initialization %s' % (GRAPH_PATH))

    e = PoseEstimator(GRAPH_PATH, target_size=(WIDTH, HEIGHT))
    
    logger.debug('cam read+')
    cam = cv2.VideoCapture(CAMERA)
    ret_val, image = cam.read()
    logger.info('cam image=%dx%d' % (image.shape[1], image.shape[0]))

    while True:
        ret_val, image = cam.read()

        logger.debug('image process+')
        humans = e.inference(image, resize_to_default=True, upsample_size=RESIZE_OUT_RATIO)

        logger.debug('postprocess+')
        image = PoseEstimator.draw_humans(image, humans, imgcopy=False)

        logger.debug('show+')
        cv2.putText(image,
                    "FPS: %f" % (1.0 / (time.time() - fps_time)),
                    (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 2)
        # cv2.imshow('tf-pose-estimation result', image)
        cv2.imwrite('/src/demo/images/tmp.jpg', image)
        fps_time = time.time()
        if cv2.waitKey(1) == 27:
            break
        logger.debug('finished+')

    cv2.destroyAllWindows()
