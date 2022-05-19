import cv2
import numpy as np

class Click:
    def __init__(self):
        self.points = [(-1, -1), (0, 0)]

    def click(self, event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points.append((x, y))

    def last_rectangle(self) -> tuple[int, int, int, int] | None:
        if len(self.points) < 2:
            return None
        x1, y1 = self.points[-2]
        x2, y2 = self.points[-1]
        return min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)


def open_image(file) -> np.ndarray:
    if isinstance(file, cv2.VideoCapture):
        return file.read()[1]
    elif isinstance(file, str):
        return cv2.imread(file, -1)
    elif isinstance(file, np.ndarray):
        return file
    else:
        raise ValueError('Unknown file type, neither a path nor a numpy array.')