import sys
import time
import logging

import cv2
import numpy as np

from roi import forehead_PoI, forehead_comb
from utils.images import get_results_from_detector

log = logging.getLogger(__name__)
log.addHandler(logging.StreamHandler(sys.stderr))
log.handlers[-1].setFormatter(logging.Formatter('\x1B[0;34m%(asctime)s - %(name)s.%(funcName)s\t- %(levelname)s - %(message)s\x1B[0m'))
log.setLevel(logging.DEBUG)


class Iterable:
    def __iter__(self): return self


class Camera(Iterable):
    def __init__(self, video_input, duration):
        self.capture = cv2.VideoCapture(video_input)
        self.time_duration = duration if isinstance(duration, float) else float('inf')
        self.frames_duration = duration if isinstance(duration, int) else float('inf')
        self.frame_count = 0
        self.until = None

    def __next__(self):
        success, image = self.capture.read()
        timestamp = time.time()
        self.until = self.until or timestamp + self.time_duration
        if self.frame_count >= self.frames_duration or timestamp > self.until:
            self.capture.release()
            raise StopIteration
        self.frame_count += 1
        return timestamp, image[..., ::-1]

    def __del__(self): self.capture.release()


def draw(image: np.ndarray, points, color=(255, 255, 255), thickness=2, copy=True):
    image = image.copy() if copy else image
    points = np.vstack(list(points))
    for x, y, *_ in points.astype(int).reshape(-1, 3):
        cv2.circle(image, (x, y), thickness, color, -1)
    return image


def detect(detector, image: np.ndarray, timestamp=None):
    if image is None:
        log.debug(f'Image {timestamp} is None.')
        return None
    results = get_results_from_detector(detector, image)
    if results[0] is None:
        log.debug(f'No detected {detector.__class__.__name__} mesh in {timestamp}.')
        return None
    landmarks = np.array([[(l.x, l.y, l.z) for l in result.landmark] for result in results[0]])[0]
    landmarks[..., 0] *= image.shape[1]
    landmarks[..., 1] *= image.shape[0]
    return landmarks


def roi(landmarks: np.ndarray, points=forehead_PoI, comb=None):
    comb = np.eye(len(points)) if comb is None else comb
    contour = landmarks.reshape(-1, 3)[list(points)]
    contour = np.einsum('ij,ik->kj', contour, np.array(comb).T)
    return contour


def ppg(image, contour, excludes=None):
    # Get mask from contour.
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [contour[..., :2].astype(int)], 0, 255, -1)

    # Remove things like the eyes and mouth.
    for exclude in excludes or []:
        cv2.drawContours(mask, [exclude[..., :2].astype(int)], 0, 0, -1)

    # Get mean of the mask.
    mean = np.mean(image[mask.astype(bool)], axis=0)
    if excludes:
        j = image.copy()
        j[np.logical_not(mask.astype(bool))] = 0
    return mean


def ppg_arround(image, landmarks):
    # TODO: test
    def gaussian(h, w, sigma=10):
        # TODO: comput as separate 1D filters.
        kernel = np.zeros((h, w))
        for i in range(h):
            for j in range(w):
                kernel[i, j] = np.exp(-((i - h // 2) ** 2 + (j - w // 2) ** 2) / (2 * sigma ** 2))
        return kernel

    h = w = 40
    kernel = gaussian(h, w, sigma=10)
    values = []

    for x, y in landmarks[..., :2]:
        for channel in np.split(image, image.shape[-1], axis=-1):
            value = kernel * channel[y - h // 2:y + h // 2,
                                     x - w // 2:x + w // 2].squeeze()
            values.append(value)
    return np.array(values).flatten()

