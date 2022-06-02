from __future__ import annotations

import cv2
import numpy as np
from mediapipe.python.solutions.face_mesh import FaceMesh

import roi
from utils.utils import get_results_from_detector


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


def show(img, resize=None, destroy=True, name='0', delay=0):
    if isinstance(img, str): img = cv2.imread(img)
    elif isinstance(img, cv2.VideoCapture): img = img.read()[1]
    if resize is not None:
        if isinstance(resize, tuple):
            img = cv2.resize(img, resize)
        else:
            img = cv2.resize(img, (round(resize * img.shape[1]), round(resize * img.shape[0])))
    cv2.imshow(str(name), img)
    k = cv2.waitKey(delay)
    if destroy:
        cv2.destroyWindow(str(name))
    return k


def isimage(image: np.ndarray) -> bool:
    return (image is not None
            and isinstance(image, np.ndarray)
            and image.ndim == 3
            and image.shape[-1] == 3)


def islandmarks(landmarks: np.ndarray) -> bool:
    return (landmarks is not None
            and isinstance(landmarks, np.ndarray)
            and landmarks.dtype in [np.float32, np.float64, float]
            and landmarks.shape[-1] == 3)


def skin_mask(image: np.ndarray, hsv=(-5, 0, 0), th1=20) -> np.ndarray:
    image = open_image(image)
    HSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV_FULL)
    h, s, v = cv2.split(HSV)

    # Only pixels in this mask are considered skin.
    mask1 = abs(h.astype(np.int8) - hsv[0]).astype(np.uint8) < th1

    # only the pixels with a high saturation are considered skin.
    mask2 = s > hsv[1] - th1

    # only the pixels with a high value are considered skin.
    mask3 = v > hsv[2] - 2 * th1

    # Those pixels are the mouth, thus they are not considered skin.
    landmarks = get_face_landmarks(image)
    mask4 = np.logical_not(roi2mask(landmarks[list(roi.mouth_RoI)], image.shape[:2], bool))

    # Only the pixels that are in the face are considered skin.
    mask5 = roi2mask(landmarks[list(roi.face_RoI)], image.shape[:2], bool)

    return mask1 & mask2 & mask3 & mask4


def get_forhead_color(image: np.ndarray, face_detector) -> np.ndarray:
    image = open_image(image)
    landmarks = get_face_landmarks(image)

    # Get the average color of the forehead.
    forehead_point = landmarks[151, ..., :2].astype(int)
    forehead = image[forehead_point[1] - 10:forehead_point[1] + 10,
                     forehead_point[0] - 10:forehead_point[0] + 10]

    hsv = cv2.cvtColor(forehead, cv2.COLOR_BGR2HSV_FULL)
    hsv_mean = np.mean(hsv, axis=(0, 1))
    return hsv_mean


def get_face_landmarks(image):
    image = open_image(image)
    with FaceMesh() as fd:
        results = get_results_from_detector(fd, image)
        if results[0] is None:
            print('No face detected.')
            return None

    landmarks = np.array([[(l.x, l.y, l.z) for l in result.landmark] for result in results[0]])[0]
    landmarks[..., 0] *= image.shape[1]
    landmarks[..., 1] *= image.shape[0]

    return landmarks


def roi2mask(contour: np.ndarray, image_shape: tuple[int, int], dtype: type = np.uint8) -> np.ndarray:
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [contour[..., :2].astype(int)], 0, 255, -1)
    return mask.astype(dtype)
