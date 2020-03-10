import os

from dotenv import load_dotenv
load_dotenv()

def get_env_bool(name):
    bool_map = {
        'false': False,
        'true': True,
        '0': False,
        '1': True
    }

    val = os.getenv(name)
    if val is None:
        return False

    try:
        return bool_map[val.lower()]
    except KeyError:
        raise ValueError(f"{name} environment variable should evaluate to True or False")

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
LOG_DIR = os.path.join(BASE_DIR, 'logs')
RES_DIR = os.path.join(os.path.dirname(BASE_DIR), 'res')

CHECKPOINT_PATH = os.path.join(BASE_DIR, 'pose/checkpoint_iter_370000.pth')
DEFAULT_CAMERA = os.getenv('DEFAULT_CAMERA')

BACKEND_DOMAIN = os.getenv('BACKEND_DOMAIN')

DEBUG_BUTTONS = get_env_bool('DEBUG_BUTTONS')
DEBUG_POSE = get_env_bool('DEBUG_POSE')
DISABLE_NET = get_env_bool('DISABLE_NET')
DEGREES = get_env_bool('DEGREES')
MV_HISTORY = 10
MV_HOLD_THRESH = 15

S3_BUCKET_NAME = os.getenv('S3_BUCKET_NAME')
S3_BUCKET_DOMAIN = os.getenv('S3_BUCKET_DOMAIN')
AWS_DEFAULT_REGION = os.getenv('AWS_DEFAULT_REGION')
AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')

OWM_API_KEY = os.getenv('OWM_API_KEY')
OWM_CITY_ID = os.getenv('OWM_CITY_ID')
