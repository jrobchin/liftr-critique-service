import os

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'estimator/models/mobilenet_v2_large-graph_opt.pb')

WIDTH = 432
HEIGHT = 368
CAMERA = 0
RESIZE_OUT_RATIO = 4.0