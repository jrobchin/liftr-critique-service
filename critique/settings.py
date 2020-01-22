import os

from dotenv import load_dotenv
load_dotenv()

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
LOG_DIR = os.path.join(BASE_DIR, 'logs')
CHECKPOINT_PATH = os.path.join(BASE_DIR, 'pose/checkpoint_iter_370000.pth')

BACKEND_DOMAIN = os.getenv('BACKEND_DOMAIN')
DISABLE_NET = os.getenv('DISABLE_NET')
