import kivy
kivy.require('1.11.1')
from kivy.app import App
from kivy.clock import Clock
from kivy.uix import label, button, widget, boxlayout, screenmanager
from kivy.lang.builder import Builder
import cv2

from critique.gui import get_kv_file
from critique.io import VideoReader
from critique.gui.camera import FrameDisplayWidget
from critique.pose.estimator import PoseEstimator
from critique.measure import PoseHeuristics


Builder.load_file(get_kv_file('app'))


class DisplayWidget(boxlayout.BoxLayout):
    def __init__(self, video, **kwargs):
        super().__init__(orientation='vertical', **kwargs)

        self.estimator = PoseEstimator()

        try:
            video = int(video)
        except ValueError:
            pass

        self.frame_provider = VideoReader(video)

        self.frame_iter = iter(self.frame_provider)
        self.display = FrameDisplayWidget()
        self.add_widget(self.display)
    
        self.update_loop = Clock.schedule_interval(self.update, 0)

    def update(self, dt):
        try:
            frame = next(self.frame_iter)
            poses = self.estimator.estimate(frame)
            for pose in poses:
                pose.draw(frame)
                ph = PoseHeuristics(pose)
                ph.draw(frame)

            self.display.set_frame(frame)
        except StopIteration:
            Clock.unschedule(self.update_loop)


class MenuScreen(screenmanager.Screen):
    pass


class DisplayScreen(screenmanager.Screen):
    pass


screen_manager = screenmanager.ScreenManager()
screen_manager.add_widget(MenuScreen(name='menu'))
screen_manager.add_widget(DisplayScreen(name='display'))

class LiftrApp(App):
    def build(self):
        return screen_manager