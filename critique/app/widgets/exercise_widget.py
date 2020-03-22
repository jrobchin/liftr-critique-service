import os
import uuid
import logging
from typing import Tuple

from kivy.core.window import Window
from kivy.app import App
from kivy.clock import Clock
from kivy.uix import widget, image
from kivy import properties
from kivy.graphics.texture import Texture # pylint: disable=no-name-in-module
from kivy.graphics import Color, Rectangle, Ellipse, Line
import cv2
import boto3
from botocore.exceptions import ClientError

from critique import settings
from critique.app.services import session_service
from critique.measure import PoseHeuristics, Pose, KEYPOINTS
from critique.pose.estimator import PoseEstimator
from critique.app.exercises import Critique, Exercise, EXERCISES, ExerciseState
from critique.app.threads import CallbackThread


class ExerciseWidget(widget.Widget):
    camera = properties.StringProperty('1') # pylint: disable=c-extension-no-member

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._keyboard = Window.request_keyboard(self._keyboard_closed, self)
        self._keyboard.bind(on_key_down=self._on_keyboard_down)

        # TODO: convert to properties
        self._cap = None
        self._curr_frame = None
        self._estimator: PoseEstimator = None
        self._heuristics: PoseHeuristics = None
        self._exercise: Exercise = None
        self._critiques_given: set = None
        self._update_loop = None
        self._started: bool = None
        self._reps: int = None
        self._state: ExerciseState = None

        self.reset()

        self._error_label = None
        self._display = image.Image()
        self._display.allow_stretch = True
        self.add_widget(self._display)

        session_service.bind('select_exercise', lambda d: self._select_exercise(d['exercise']))
        session_service.bind('start_exercise', lambda d: self._start_exercise(d['reps']))

        self.bind(
            camera=self._on_camera
        )

    def _keyboard_closed(self):
        self._keyboard.unbind(on_key_down=self._on_keyboard_down)
        self._keyboard = None

    def _on_keyboard_down(self, keyboard, keycode, text, modifiers):
        if keycode[1] == 'd':
            settings.DEBUG_POSE = not settings.DEBUG_POSE
        elif keycode[1] == 'b':
            root = App.get_running_app().root
            root.transition.direction = 'right'
            root.current = 'menu'
        return True

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
        self._error_label = screen.ids.error_label
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
        self._started = False

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
        self._reps = 0

        # Notify workout selection
        # screen.ids.workout_label.opacity = 0
        screen.ids.workout_select_info_box.opacity = 1
        screen.ids.workout_select_info_label.opacity = 1
        exercise_label = screen.ids.workout_select_info_label
        exercise_label.text = f"Workout Selected: {self._exercise.name}"

        def _cb(dt):
            # Hide pop up and set workout label
            screen.ids.workout_label.text = f"{self._exercise.name}"
            screen.ids.workout_label.opacity = 1
            screen.ids.workout_select_info_box.opacity = 0
            screen.ids.workout_select_info_label.opacity = 0

        Clock.schedule_once(_cb, 1.5)

    def _start_exercise(self, reps=None):
        if self._exercise is None:
            return

        self._state = self._exercise.start_state

        screen = App.get_running_app().get_screen()
        if screen.name != 'display':
            return
            
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
            # screen.ids.started_value_label.text = "Y"

        _count_down(3, _cb)

    def _send_critique(self, critique: Critique, frame=None):
        if frame is None:
            frame = self._curr_frame

        obj_name = uuid.uuid4().hex + ".jpg"

        def _upload_image():
            # Upload screenshot
            tmp_fpath = 'tmp/tmp.jpg'
            cv2.imwrite(tmp_fpath, frame)
            s3 = boto3.client('s3')
            try:
                s3.upload_file(
                    tmp_fpath,
                    settings.S3_BUCKET_NAME,
                    obj_name,
                    ExtraArgs={'ACL':'public-read'}
                )
            except ClientError:
                logging.error("Error uploading to S3.")
        
        def _cb():
            # Send critique to app
            session_service.emit('make_critique', {
                "exercise": self._exercise.name,
                "caption": critique.msg,
                "image": f"{settings.S3_BUCKET_DOMAIN}/{obj_name}"
            })

            screen = App.get_running_app().get_screen()
            screen.ids.critique_count_value_label.text = str(len(self._critiques_given))
        
        CallbackThread(
            name='critique_image_upload',
            target=_upload_image,
            callback=(_cb,),
        ).start()

    def _test_critique(self):
        self._send_critique(Critique('test_critique', [], 'testing critiques', None))

    def _update_rep_counter(self, reps):
        session_service.emit('update_reps', {"reps": reps})

    def _calc_tex_size(self, parent, texture):
        parent_aspect = parent.size[0] / parent.size[1]
        tex_aspect = texture.size[0] / texture.size[1]
        if parent_aspect > tex_aspect:
            return texture.size[0] * parent.size[1] // texture.size[1], parent.size[1]
        else:
            return parent.size[0], texture.size[1] * parent.size[0] // texture.size[0]

    def _draw_image(self, texture:Texture, size:Tuple[int, int], pos:Tuple[int, int]):
        self._display.canvas.add(Rectangle(texture=texture, pos=pos, size=size))

    def _draw_pose(self, pose: Pose, progress, texture_size:Tuple[int, int], texture_pos:Tuple[int, int], image_size:Tuple[int, int]):

        def _transform_kpt_pos(kpt_pos, scale_factors, pos_offset):
            return kpt_pos[0] * scale_factors[0] + pos_offset[0], \
                   (image_size[1] - kpt_pos[1]) * scale_factors[1] + pos_offset[1]

        scale_factors = (texture_size[0] / image_size[0], texture_size[1] / image_size[1])
        marker_size = 6

        _canvas = self._display.canvas

        # Draw pose
        # if pose is not None:
        #     for kpt_id in range(KEYPOINTS.NUM_KPTS):
        #         kpt_pos = pose.keypoints[kpt_id].tolist()
        #         if kpt_pos[0] == -1:
        #             continue
        #         kpt_pos_t = _transform_kpt_pos(kpt_pos, scale_factors, texture_pos)
        #         _canvas.add(Color(0, 0, 1, 0.5))
        #         _canvas.add(Ellipse(size=(marker_size, marker_size), pos=(kpt_pos_t[0] - marker_size//2, kpt_pos_t[1] - marker_size//2)))
        
        guide_size = 75
        for h_id, kpt_id, val in progress:
            fill_size = min(max(10, guide_size * val), guide_size)

            kpt_pos = pose.keypoints[kpt_id].tolist()
            if kpt_pos[0] == -1:
                continue
            kpt_pos_t = _transform_kpt_pos(kpt_pos, scale_factors, texture_pos)
            _canvas.add(Color(1-val, val, 0))
            _canvas.add(Ellipse(size=(fill_size, fill_size), pos=(kpt_pos_t[0] - fill_size//2, kpt_pos_t[1] - fill_size//2)))
            _canvas.add(Color(1, 1, 1))
            _canvas.add(
                Line(circle=(*kpt_pos_t, guide_size//2), width=2.5)
            )


    def update(self, dt):
        if self._cap:
            if self._curr_frame is not None:
                prev_frame = self._curr_frame.copy()
            ret, frame = self._cap.read()
            frame = cv2.flip(frame, 1)

            if ret:
                self._curr_frame = frame.copy()
                self._error_label.opacity = 0
                self._display.opacity = 1
            else:
                self._error_label.opacity = 1
                self._display.opacity = 0
                return
            
            self._display.canvas.clear()

            pose: Pose = None
            progress = []
            # TODO: extract all non-estimate steps from the try except
            try:
                pose = self._estimator.estimate(frame)[0]
                self._heuristics.update(pose)
                if settings.DEBUG_POSE:
                    self._heuristics.draw(frame)
                    pose.draw(frame)
                if self._started:
                    self._state, critiques, progress = self._exercise.update(pose, self._heuristics)
                    for critique in critiques:
                        if critique.name not in self._critiques_given:
                            self._critiques_given.add(critique.name)
                            self._send_critique(critique, prev_frame)
            except IndexError:
                pass

            # convert it to texture
            buf1 = cv2.flip(frame, 0)
            buf = buf1.tostring()
            image_texture = Texture.create(
                size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            image_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')

            # calculate target size and position for drawing the image texture
            texture_size = self._calc_tex_size(self, image_texture)
            texture_pos = (self.center_x-texture_size[0]//2, self.pos[1])

            # draw webcam feed
            self._draw_image(image_texture, texture_size, texture_pos)

            # display image from the texture
            self._draw_pose(pose, progress, texture_size, texture_pos, image_texture.size)
            self._display.size = self.size

            # set state label text and update reps
            if self._started:
                screen = App.get_running_app().get_screen()
                screen.ids.state_value_label.text = self._state.str
                screen.ids.rep_value_label.text = str(self._exercise.reps)
                if self._exercise.reps > self._reps:
                    self._reps = self._exercise.reps
                    self._update_rep_counter(self._exercise.reps)