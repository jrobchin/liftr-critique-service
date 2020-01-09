import cv2
import numpy as np
import urllib

class ImageReader(object):
    def __init__(self, filename):
        self.filename = filename
        self.read = False
    
    def get_image(self):
        img = cv2.imread(self.filename, cv2.IMREAD_COLOR)
        return img

    def __iter__(self):
        return self

    def __next__(self):
        if self.read:
            raise StopIteration
        img = cv2.imread(self.filename, cv2.IMREAD_COLOR)
        self.read = True
        return img


class VideoReader(object):
    def __init__(self, file_name):
        self.file_name = file_name
        try:  # OpenCV needs int to read from webcam
            self.file_name = int(file_name)
        except ValueError:
            pass
        
        self.cap = cv2.VideoCapture(self.file_name)
        self.width, self.height, self.fps, self.frame_count = self.get_info()
    
    def get_info(self):
        """If a video capture is defined, return the `width`, `height`, `fps`, and `frame_count` in a tuple."""
        if self.cap:
            width  = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            frame_count = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
            return width, height, fps, frame_count

    def __iter__(self):
        self.cap = cv2.VideoCapture(self.file_name)
        if not self.cap.isOpened():
            raise IOError('Video {} cannot be opened'.format(self.file_name))
        return self

    def __next__(self):
        was_read, img = self.cap.read()
        if not was_read:
            raise StopIteration
        return img

def url_to_image(url):
	resp = urllib.request.urlopen(url)
	image = np.asarray(bytearray(resp.read()), dtype="uint8")
	image = cv2.imdecode(image, cv2.IMREAD_COLOR)
	return image