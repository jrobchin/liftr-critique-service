import os
import uuid
import logging
import datetime

import kivy
kivy.require('1.11.1')
from kivy.config import Config
Config.set('graphics', 'borderless', '1')
from kivy.app import App
from kivy.clock import Clock
from kivy.uix import label, button, widget, boxlayout, screenmanager, image
from kivy.lang.builder import Builder
from kivy import properties
from kivy.core.window import Window
from kivy.graphics.texture import Texture
import cv2
import socketio
import boto3
from botocore.exceptions import ClientError
import requests

from critique import settings
from critique.io import VideoReader
from critique.app import get_kv_file
from critique.app.services import session_service
from critique.measure import PoseHeuristics
from critique.pose.estimator import PoseEstimator
from critique.app.exercises import Critique, EXERCISES

Window.maximize()

Builder.load_file(get_kv_file('app'))

class DisplayWidget(widget.Widget):
    camera = properties.StringProperty('1')

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.reset()
 
        self._display = image.Image()
        self._display.allow_stretch = True
        self.add_widget(self._display)

        session_service.bind('select_exercise', lambda d: self._select_exercise(d['exercise']))
        session_service.bind('start_exercise', lambda d: self._start_exercise(d['reps']))

        self.bind(
            camera=self._on_camera
        )
        
    def start(self):
        Clock.schedule_once(self._on_start, 0)

    def stop(self):
        if self._update_loop is not None:
            Clock.unschedule(self._update_loop)
        
    def reset(self):
        self._cap = None
        self._curr_frame = None
        self._estimator = PoseEstimator()
        self._heuristics = PoseHeuristics()
        self._exercise = None
        self._critiques_given = set()
        self._update_loop = None
        self._started = False
        self._reps = 0

    def _on_start(self, *args):
        screen = App.get_running_app().get_screen()
        self.error_label = screen.ids.error_label
        self._on_camera()
        self._update_loop = Clock.schedule_interval(self.update, 0)
    
    def _on_camera(self, *args):
        if self._cap:
            self._cap.release()

        try:
            camera = int(self.camera)
            self._cap = cv2.VideoCapture(camera)
        except ValueError:

            if isinstance(self.camera, str):
                # File path given, get path to file in res folder
                camera = os.path.join(settings.RES_DIR, self.camera)
                self._cap = cv2.VideoCapture(camera)
        
    def _select_exercise(self, exercise):
        selected_exercise = EXERCISES.get(exercise)

        screen = App.get_running_app().get_screen()

        if selected_exercise is None:
            # Notify error
            screen.ids.workout_label.opacity = 0
            screen.ids.workout_select_info_box.opacity = 1
            screen.ids.workout_select_info_label.opacity = 1
            screen.ids.workout_select_info_label.text = f"Selected workout not installed..."
            
            def _cb(dt):
                # Hide pop up and set workout label
                screen.ids.workout_select_info_box.opacity = 0
                screen.ids.workout_select_info_label.opacity = 0
            
            Clock.schedule_once(_cb, 1.5)

            return
        
        self._exercise = selected_exercise()

        # Notify workout selection
        screen.ids.workout_label.opacity = 0
        screen.ids.workout_select_info_box.opacity = 1
        screen.ids.workout_select_info_label.opacity = 1
        exercise_label = screen.ids.workout_select_info_label
        exercise_label.text = f"Workout Selected: {exercise}"

        def _cb(dt):
            # Hide pop up and set workout label
            screen.ids.workout_label.text = f"Workout: {exercise}"
            screen.ids.workout_label.opacity = 1
            screen.ids.workout_select_info_box.opacity = 0
            screen.ids.workout_select_info_label.opacity = 0
        
        Clock.schedule_once(_cb, 1.5)
    
    def _start_exercise(self, reps):
        screen = App.get_running_app().get_screen()
        
        def _count_down(n, callback):
            def _animation(n):
                screen.ids.count_down_label.text = str(n)

                screen.ids.count_down_box.opacity = 1
                screen.ids.count_down_label.opacity = 1

                n -= 1

                if n < 0:
                    _after()
                else:
                    Clock.schedule_once(lambda _: _animation(n), 1)

            def _after():
                screen.ids.count_down_box.opacity = 0
                screen.ids.count_down_label.opacity = 0
                callback()
            
            Clock.schedule_once(lambda _: _animation(n), 0)

        def _cb():
            self._started = True
            screen.ids.started_value_label.text = "True"

        _count_down(3, _cb)

    def _send_critique(self, critique:Critique, frame=None):
        if frame is None:
            frame = self._curr_frame

        # Upload screenshot
        tmp_fpath = 'tmp/tmp.jpg'
        cv2.imwrite(tmp_fpath, frame)
        s3 = boto3.client('s3')
        try:
            obj_name = uuid.uuid4().hex + ".jpg"
            response = s3.upload_file(tmp_fpath, settings.S3_BUCKET_NAME, obj_name, ExtraArgs={'ACL':'public-read'})
        except ClientError:
            logging.error("Error uploading to S3.")
        
        # Send critique to app
        session_service.emit('make_critique', {
            "exercise": self._exercise.name,
            "caption": critique.msg,
            "image": f"{settings.S3_BUCKET_DOMAIN}/{obj_name}"
        })

        screen = App.get_running_app().get_screen()
        screen.ids.critique_count_value_label.text = str(len(self._critiques_given))
    
    def _test_critique(self):
        frame = self._curr_frame

        # Upload screenshot
        tmp_fpath = 'tmp/tmp.jpg'
        cv2.imwrite(tmp_fpath, frame)
        s3 = boto3.client('s3')
        try:
            obj_name = uuid.uuid4().hex + ".jpg"
            response = s3.upload_file(tmp_fpath, settings.S3_BUCKET_NAME, obj_name, ExtraArgs={'ACL':'public-read'})
        except ClientError:
            logging.error("Error uploading to S3.")
        
        # Send critique to app
        session_service.emit('make_critique', {
            "exercise": self._exercise.name,
            "caption": 'test_critique',
            "image": f"{settings.S3_BUCKET_DOMAIN}/{obj_name}"
        })

        screen = App.get_running_app().get_screen()
        screen.ids.critique_count_value_label.text = str(len(self._critiques_given))
    
    def _update_rep_counter(self, reps):
        session_service.emit('update_reps', { "reps": reps })

    def update(self, dt):
        if self._cap:
            if self._curr_frame is not None:
                prev_frame = self._curr_frame.copy()
            ret, frame = self._cap.read()
            frame = cv2.flip(frame, 1)

            if ret: 
                self._curr_frame = frame.copy()
                self.error_label.opacity = 0
                self._display.opacity = 1
            else:
                self.error_label.opacity = 1
                self._display.opacity = 0
                return

            state = ''
            try:
                pose = self._estimator.estimate(frame)[0]
                pose.draw(frame)
                pose = pose.get_kpt_group()
                self._heuristics.update(pose)
                if self._started:
                    state, critiques = self._exercise.update(pose, self._heuristics)
                    for critique in critiques:
                        if critique.name not in self._critiques_given:
                            self._critiques_given.add(critique.name)
                            self._send_critique(critique, prev_frame)
                self._heuristics.draw(frame)
            except IndexError:
                pass

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

            # set state label text and update reps
            if self._started:
                screen = App.get_running_app().get_screen()
                screen.ids.state_value_label.text = state
                screen.ids.rep_value_label.text = str(self._exercise.reps)
                if self._exercise.reps > self._reps:
                    self._reps = self._exercise.reps
                    self._update_rep_counter(self._exercise.reps)

class SessionKeyLabel(label.Label):
    s_key = properties.StringProperty()

    def on_s_key(self, value):
        print(value)

class MenuScreen(screenmanager.Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        session_service.bind('start_session', lambda _: self._transition_to_display())

        Clock.schedule_once(self.connect, 0)

        self._update_hud()
        Clock.schedule_interval(self._update_hud, 60)
        
    def connect(self, *args):
        app = App.get_running_app()
        
        def _on_success():
            self.ids.session_key_label.text = f"{session_service.s_key}"
            app.state['connected'] = True
        
        def _on_error():
            self.ids.session_key_label.text = "Error"
            app.state['connected'] = False
        
        session_service.connect(on_success=_on_success, on_error=_on_error)

    def _transition_to_display(self):
        self.manager.get_screen('display').ids.display.start()
        self.manager.transition.direction = 'left' 
        self.manager.current = 'display'

    def _transition_to_about(self):
        pass

    def _update_hud(self, dt=None):
        # Update time
        now = datetime.datetime.now()
        self.ids.time_label.text = f"{now.strftime('%I:%M%p')}"

        # Update weather
        res = requests.get("https://api.openweathermap.org/data/2.5/weather?""units=metric&" + \
                          f"id={settings.OWM_CITY_ID}&appid={settings.OWM_API_KEY}")
        if res.status_code == 200:
            data = res.json()
            self.ids.weather_label.text = f"{data['weather'][0]['main']} {data['main']['temp']:.0f}°C"
        else:
            self.ids.weather_label.text = ""



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
