import datetime

from kivy.app import App
from kivy.clock import Clock
from kivy.uix import screenmanager
from kivy.core.window import Window
import requests

from critique import settings
from critique.app.services import session_service
from critique.app.widgets import SessionKeyLabel # pylint: disable=unused-import


class MenuScreen(screenmanager.Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._keyboard = Window.request_keyboard(self._keyboard_closed, self)
        self._keyboard.bind(on_key_down=self._on_keyboard_down)

        session_service.bind('start_session', lambda _: self._transition_to_display())

        Clock.schedule_once(self.connect, 0)

        self._update_hud()
        Clock.schedule_interval(self._update_hud, 60)

    def connect(self, *args): # pylint: disable=unused-argument
        app = App.get_running_app()

        def _on_success():
            self.ids.session_key_label.text = f"{session_service.s_key}"
            app.state['connected'] = True

        def _on_error():
            self.ids.session_key_label.text = "Error"
            app.state['connected'] = False

        session_service.connect(on_success=_on_success, on_error=_on_error)

    def _keyboard_closed(self):
        self._keyboard.unbind(on_key_down=self._on_keyboard_down)
        self._keyboard = None

    def _on_keyboard_down(self, keyboard, keycode, text, modifiers):
        if keycode[1] == 'r':
            self.connect()
        return True

    def _transition_to_display(self):
        self.manager.get_screen('display').ids.display.start()
        self.manager.transition.direction = 'left'
        self.manager.current = 'display'

    def _transition_to_about(self):
        pass

    def _update_hud(self, dt=None): # pylint: disable=unused-argument
        # Update time
        now = datetime.datetime.now()
        self.ids.time_label.text = f"{now.strftime('%I:%M%p')}"

        # Update weather
        try:
            res = requests.get("https://api.openweathermap.org/data/2.5/weather?""units=metric&" + \
                            f"id={settings.OWM_CITY_ID}&appid={settings.OWM_API_KEY}")
            if res.status_code == 200:
                data = res.json()
                self.ids.weather_label.text = \
                    f"{data['weather'][0]['main']} {data['main']['temp']:.0f}°C"
            else:
                self.ids.weather_label.text = ""
        except requests.exceptions.ConnectionError:
            self.ids.weather_label.text = ""
