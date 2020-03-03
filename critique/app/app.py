# pylint: disable=wrong-import-position
import os

import kivy
kivy.require('1.11.1')
from kivy.config import Config
Config.set('graphics', 'borderless', '1')
from kivy.app import App
from kivy.uix import screenmanager
from kivy.lang.builder import Builder
from kivy.core.window import Window

from critique.app.screens import MenuScreen, DisplayScreen, AboutScreen

Window.maximize()

Builder.load_file(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), 'app.kv')
)

screen_manager = screenmanager.ScreenManager()
screen_manager.add_widget(MenuScreen(name='menu'))
screen_manager.add_widget(DisplayScreen(name='display'))
screen_manager.add_widget(AboutScreen(name='about'))

class LiftrApp(App):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.state = {
            'connected': False
        }

    def build(self):
        return screen_manager
    
    def get_screen(self):
        return self.root.current_screen
