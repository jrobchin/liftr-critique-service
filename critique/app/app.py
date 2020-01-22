import kivy
kivy.require('1.11.1')
from kivy.app import App
from kivy.clock import Clock
from kivy.uix import label, button, widget, boxlayout, screenmanager, image
from kivy.lang.builder import Builder
from kivy import properties
from kivy.core.window import Window
from kivy.graphics.texture import Texture
import cv2
import socketio

from critique import settings
from critique.io import VideoReader
from critique.app import get_kv_file
from critique.app.services import session_service
from critique.measure import PoseHeuristics
from critique.pose.estimator import PoseEstimator

from kivy.config import Config

# Window.maximize()

Builder.load_file(get_kv_file('app'))

class DisplayWidget(widget.Widget):
    camera = properties.NumericProperty(1)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.cap = None
        self.estimator = PoseEstimator()

        self.display = image.Image()
        self.display.allow_stretch = True
        self.add_widget(self.display)

        self.bind(
            camera=self.on_camera
        )
        self.update_loop = None
        
    def start(self):
        Clock.schedule_once(self._on_start, 0)

    def stop(self):
        if self.update_loop is not None:
            Clock.unschedule(self.update_loop)

    def _on_start(self, *args):
        self.error_label = self.parent.parent.ids.error_label
        self.on_camera()
        self.update_loop = Clock.schedule_interval(self.update, 0)

    def on_camera(self, *args):
        if self.cap:
            self.cap.release()
        try:
            camera = int(self.camera)
        except ValueError:
            camera = self.camera
        
        self.cap = cv2.VideoCapture(camera)

    def update(self, dt):
        if self.cap:
            ret, frame = self.cap.read()

            if ret: 
                self.error_label.opacity = 0
                self.display.opacity = 1
            else:
                self.error_label.opacity = 1
                self.display.opacity = 0
                return

            poses = self.estimator.estimate(frame)
            for pose in poses:
                pose.draw(frame)
                pose = pose.get_kpt_group()
                ph = PoseHeuristics(pose)
                ph.draw(frame)

            # convert it to texture
            buf1 = cv2.flip(frame, 0)
            buf = buf1.tostring()
            image_texture = Texture.create(
                size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            image_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')

            # display image from the texture
            self.display.texture = image_texture
            self.display.size = self.size
            self.display.pos = self.pos

class SessionKeyLabel(label.Label):
    s_key = properties.StringProperty()

    def on_s_key(self, value):
        print(value)

class MenuScreen(screenmanager.Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        Clock.schedule_once(self._on_start, 0)
        
    def _on_start(self, *args):
        def _cb():
            self.ids.session_key_label.text = f"Session Key: {session_service.s_key}"

        session_service.connect(callback=_cb)

class DisplayScreen(screenmanager.Screen):
    pass

class AboutScreen(screenmanager.Screen):
    pass

screen_manager = screenmanager.ScreenManager()
screen_manager.add_widget(MenuScreen(name='menu'))
screen_manager.add_widget(DisplayScreen(name='display'))
screen_manager.add_widget(AboutScreen(name='about'))

class LiftrApp(App):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self):
        return screen_manager