import os
import json
import logging
logging.basicConfig(level=logging.INFO)

import socketio

from critique import settings

class __SessionService__:
    def __init__(self):
        self.client: socketio.Client = socketio.Client()
        self.s_key: str = None
        self._events = {}

    def _set_s_key(self, key):
        logging.info(f"Setting session key {key}")
        self.s_key = key

    def connect(self, hostname=settings.BACKEND_DOMAIN, callback=None):
        if self.client.connected:
            self.client.disconnect()

        self.client.connect(f"http://{hostname}")
        if not self.client.connected:
            raise ConnectionError("Could not connect to backend...")
        logging.info("Connected to backend")

        def _cb(d):
            d = json.loads(d)
            self._set_s_key(d['data']['s_key'])
            callback()

        self.client.emit('register_machine', {
            'client_type': 'machine'
        }, callback=_cb)
        logging.info("Registered with backend")
    
    def bind(self, event, func):
        self.client.on(event, func)
        

session_service = __SessionService__()