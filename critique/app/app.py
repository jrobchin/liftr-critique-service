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

        self._display = image.Image()
        self._display.allow_stretch = True
        self.add_widget(self._display)

        self._exercise = None

        self.bind(
            camera=self._on_camera
        )
        self._update_loop = None
        
    def start(self):
        Clock.schedule_once(self._on_start, 0)

    def stop(self):
        if self._update_loop is not None:
            Clock.unschedule(self._update_loop)

    def _on_start(self, *args):
        screen = App.get_running_app().get_screen()
        self.error_label = screen.ids.error_label
        self._on_camera()
        self._update_loop = Clock.schedule_interval(self.update, 0)
    
    def _on_camera(self, *args):
        if self.cap:
            self.cap.release()
        try:
            camera = int(self.camera)
        except ValueError:
            camera = self.camera
        
        self.cap = cv2.VideoCapture(camera)

    def _select_exercise(self, exercise):
        screen = App.get_running_app().get_screen()

        # Notify workout selection
        screen.ids.workout_label.opacity = 0
        screen.ids.workout_select_info_box.opacity = 1
        screen.ids.workout_select_info_label.opacity = 1
        exercise_label = screen.ids.workout_select_info_label
        exercise_label.text = f"Workout Selected: {exercise}"

        def _change_workout(dt):
            # Hide pop up and set workout label
            screen.ids.workout_label.text = f"Workout: {exercise}"
            screen.ids.workout_label.opacity = 1
            screen.ids.workout_select_info_box.opacity = 0
            screen.ids.workout_select_info_label.opacity = 0
        
        Clock.schedule_once(_change_workout, 3)
    
    def _start_session(self):
        pass

    def update(self, dt):
        if self.cap:
            ret, frame = self.cap.read()

            if ret: 
                self.error_label.opacity = 0
                self._display.opacity = 1
            else:
                self.error_label.opacity = 1
                self._display.opacity = 0
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
            self._display.texture = image_texture
            self._display.size = self.size
            self._display.pos = self.pos

class SessionKeyLabel(label.Label):
    s_key = properties.StringProperty()

    def on_s_key(self, value):
        print(value)

class MenuScreen(screenmanager.Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        session_service.bind('start_session', lambda _: self._transition_to_display())

        Clock.schedule_once(self.connect, 0)
        
    def connect(self, *args):
        app = App.get_running_app()
        
        def _on_success():
            self.ids.session_key_label.text = f"Session Key: {session_service.s_key}"
            app.state['connected'] = True
        
        def _on_error():
            self.ids.session_key_label.text = f"Unable to connect to server..."
            app.state['connected'] = False
        
        session_service.connect(on_success=_on_success, on_error=_on_error)

    def _transition_to_display(self):
        self.manager.get_screen('display').ids.display.start()
        self.manager.transition.direction = 'left' 
        self.manager.current = 'display'

    def _transition_to_about(self):
        pass

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

        self.state = {
            'connected': False
        }

    def build(self):
        return screen_manager
    
    def get_screen(self):
        return self.root.current_screen