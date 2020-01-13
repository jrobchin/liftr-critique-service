import cv2
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture


class CameraWidget(Image):
    def __init__(self, capture, fps, **kwargs):
        super().__init__(**kwargs)
        self.capture = capture
        Clock.schedule_interval(self.update, 1.0 / fps)

    def update(self, dt):
        ret, frame = self.capture.read()
        if ret:
            # convert it to texture
            buf1 = cv2.flip(frame, 0)
            buf = buf1.tostring()
            image_texture = Texture.create(
                size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            image_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            # display image from the texture
            self.texture = image_texture


class FrameDisplayWidget(Image):
    def __init__(self, fps:int=0, **kwargs):
        """
        Widget for displaying OpenCV frames.
        Args:
        fps (int) -- if given, lock display to `fps`
        """
        super().__init__(**kwargs)
        self.frame = None
        self.fps = fps

        Clock.schedule_interval(self.update, self.fps)

    def set_frame(self, frame):
        self.frame = frame

    def update(self, dt):
        if self.frame is not None:
            # convert it to texture
            buf1 = cv2.flip(self.frame, 0)
            buf = buf1.tostring()
            image_texture = Texture.create(
                size=(self.frame.shape[1], self.frame.shape[0]), colorfmt='bgr')
            image_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')

            # display image from the texture
            self.texture = image_texture