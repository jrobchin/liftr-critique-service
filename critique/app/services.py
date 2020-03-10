import json
import logging
logging.basicConfig(level=logging.INFO)

import socketio # pylint: disable=wrong-import-position

from critique import settings # pylint: disable=wrong-import-position

class __SessionService__:
    def __init__(self):
        self.client: socketio.Client = socketio.Client()
        self.s_key: str = None
        self._events = {}

    def _set_s_key(self, key):
        logging.info(f"Setting session key {key}")
        self.s_key = key

    def connect(self, hostname=settings.BACKEND_DOMAIN, on_success=None, on_error=None):
        logging.info("Connecting to server...")
        if self.client.connected:
            self.client.disconnect()

        # TODO: Clean this up
        try:
            self.client.connect(f"http://{hostname}", transports=['websocket'])
        except socketio.exceptions.ConnectionError:
            if on_error:
                on_error()
                return
        if not self.client.connected:
            if on_error:
                on_error()
                return
        logging.info("Connected to backend")

        def _cb(d):
            d = json.loads(d)
            self._set_s_key(d['data']['s_key'])
            if on_success:
                on_success()

        self.client.emit('register_machine', {
            'client_type': 'machine'
        }, callback=_cb)
        logging.info("Registered with backend")

    def bind(self, event, func):
        self.client.on(event, func)

    def emit(self, event, data, callback=None):
        self.client.emit(event, data, callback=callback)

session_service = __SessionService__()
