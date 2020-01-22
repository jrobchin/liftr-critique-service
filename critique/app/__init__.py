import os

KV_FILES = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'ui')

def get_kv_file(name):
    return os.path.join(KV_FILES, f'{name}.kv')