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
CHECKPOINT_PATH = os.path.join(BASE_DIR, 'pose/checkpoint_iter_370000.pth')

BACKEND_DOMAIN = os.getenv('BACKEND_DOMAIN')
DISABLE_NET = get_env_bool('DISABLE_NET')
