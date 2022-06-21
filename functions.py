import logging
import sys
import time

import cv2
import numpy as np

from utils.worker import KeepThemComing


log = logging.getLogger(__name__)
log.addHandler(logging.StreamHandler(sys.stderr))
log.handlers[-1].setFormatter(logging.Formatter('\x1B[0;34m%(asctime)s - %(name)s.%(funcName)s\t- %(levelname)s - %(message)s\x1B[0m'))
log.setLevel(logging.DEBUG)


class ImageReader:
    def __init__(self, video_input, duration=None):
        self.capture = cv2.VideoCapture(video_input)
        self.time_duration   = duration if isinstance(duration, float) else float('inf')
        self.frames_duration = duration if isinstance(duration,   int) else float('inf')
        self.until = None
        self.frame_count = 0
        self.__dict__['names'] = ('id', 'image')

    def __iter__(self): return self

    def __next__(self):
        success, image = self.capture.read()
        timestamp = time.monotonic()
        self.until = self.until or timestamp + self.time_duration
        if self.frame_count >= self.frames_duration or timestamp > self.until:
            self.capture.release()
            raise StopIteration
        self.frame_count += 1
        return timestamp, image[..., ::-1]

    __call__ = __next__


def get_results_from_detector(detector, image) -> list:
    """Unified way of getting results from different types of detectors (mainly FaceMesh and Hands)."""
    processed = detector.process(image)
    return [processed.__dict__[field] for field in processed._fields]


def landmarks(image, detector):
    if image is None:
        raise ValueError('Image {id} is None.')

    results = get_results_from_detector(detector, image)

    if results[0] is None:
        raise ValueError(f'No detected {detector.__class__.__name__} in {{id}}.')

    # MediPipe returns landmarks in an obscure format. Convert them to np.ndarray.
    landmarks = np.array([(l.x, l.y, l.z) for l in results[0][0].landmark])
    landmarks[..., :2] *= (image.shape[1], image.shape[0])  # From [0, 1] to image size.

    return landmarks


def contour(landmarks, points_indexes, combination=None):
    """
    Return contour of the ROI based on the landmarks.

    :param landmarks: landmarks of the face or hand.
    :param points_indexes: indexes of the points to use.
    :param combination: combination of the points to use. If None, use all points.
                        If it's a matrix, use the linear combination given by it.
    """
    contour = landmarks[list(points_indexes)]
    if combination is not None:
        contour = np.einsum('mn,nk->mk', np.array(combination), contour)
    return contour


def roi(image, landmarks, contour, excludes=(), blur=(5, 5)):
    # Get mask from contour.
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    contour = landmarks[list(contour)][..., :2]
    cv2.drawContours(mask, [np.round(contour).astype(int)], 0, 255, -1)

    # Remove excluded contours.
    for exclude in excludes:
        exclude = landmarks[list(exclude)][..., :2]
        cv2.drawContours(mask, [np.round(exclude).astype(int)], 0, 0, -1)

    # TODO: Remove pixels that are obviusly not skin.

    # Blur the mask.
    mask = cv2.GaussianBlur(mask, blur, 0)

    return mask


def ppg(image, roi):
    mean = np.average(image, weights=np.repeat(roi[..., np.newaxis], repeats=image.shape[-1], axis=-1), axis=(0, 1))
    return mean


def gaussian(h, w, sigma=10) -> np.ndarray:
    # TODO: compute as separate 1D filters.
    kernel = np.zeros((h, w))
    for i in range(h):
        for j in range(w):
            kernel[i, j] = np.exp(-((i - h // 2) ** 2 + (j - w // 2) ** 2) / (2 * sigma ** 2))
    return kernel / np.sum(kernel)


KERNEL = gaussian(40, 40, 15)


def ppg_around(image, landmarks, kernel=KERNEL):
    values = []
    kernel_ = kernel
    h, w = kernel.shape[:2]
    h2, w2 = h // 2, w // 2

    for x, y in np.round(landmarks[..., :2]).astype(int):
        x0, y0, x1, y1 = x - h2, y - w2, x + h2, y + w2
        kernel = kernel_

        # Make sure the kernel is inside the image.
        if y0 < 0: kernel = kernel[-y0:, :]
        if x0 < 0: kernel = kernel[:, -x0:]
        if y1 > image.shape[0]: kernel = kernel[:image.shape[0] - y1, :]
        if x1 > image.shape[1]: kernel = kernel[:, :image.shape[1] - x1]

        for channel in np.split(image, image.shape[-1], axis=-1):
            value = kernel * channel[y0:y1, x0:x1].squeeze()
            values.append(np.sum(value))
    return np.array(values).flatten()


def draw(image, landmarks, color=(255, 255, 255), radius=2, copy=True):
    image = image.copy() if copy else image
    for x, y in np.round(landmarks[..., :2]).astype(int):
        cv2.circle(image, (x, y), radius, color, cv2.FILLED)
    return image


def draw_contour(image, contour, color=(255, 255, 255), thickness=2, copy=True):
    image = image.copy() if copy else image
    cv2.drawContours(image, [np.round(contour[:, :2]).astype(int)], 0, color, thickness)
    return image


KERNEL = gaussian(40, 40, 2)
KERNEL /= KERNEL.max()
KERNEL = np.repeat(KERNEL[..., np.newaxis], repeats=3, axis=-1)


def draw_kernel(image, landmarks, color=(255, 255, 255), radius=2, copy=True, kernel=KERNEL):
    image = image.copy() if copy else image
    kernel_ = kernel
    h, w = kernel.shape[:2]
    h2, w2 = h // 2, w // 2

    for x, y in np.round(landmarks[..., :2]).astype(int):
        x0, y0, x1, y1 = x - h2, y - w2, x + h2, y + w2
        kernel = kernel_

        # Make sure the kernel is inside the image.
        if y0 < 0: kernel = kernel[-y0:, :]
        if x0 < 0: kernel = kernel[:, -x0:]
        if y1 > image.shape[0]: kernel = kernel[:image.shape[0] - y1, :]
        if x1 > image.shape[1]: kernel = kernel[:, :image.shape[1] - x1]

        image[y0:y1, x0:x1] = 255 - (255 - image[y0:y1, x0:x1]) * (1 - kernel)

    return image


class Save:
    def __init__(self, path):
        self.path = path
        self.values = []

    def __call__(self, ppg):
        self.values.append(ppg)

    def __del__(self):
        np.save(self.path, self.values)


class SaveCSV(Save):
    def __del__(self):
        np.savetxt(self.path, self.values, delimiter=',')


class Interpolate:
    def __init__(self):
        self.last_landmarks = None
        self.last_id = None
        self.ids = []

    def __call__(self, id, landmarks=None):
        # We'll have to interpolate this one later.
        if landmarks is None:
            self.ids.append(id)
            raise KeepThemComing(True)
        # We have new landmarks, but there are no ids to intepolate.
        elif not self.ids:
            # Save them as the most up-to-date ones.
            self.last_landmarks = landmarks
            self.last_id = id
            return landmarks
        # We have new landmarks and previous ids to intepolate.
        else:
            span = id - self.last_id
            # For each stored id, interpolate the landmarks.
            interpolateds = []
            for id_ in self.ids:
                alpha = (id_ - self.last_id) / span
                interpolated = self.last_landmarks * alpha + landmarks * (1 - alpha)
                interpolateds.append(interpolated)
            self.last_landmarks = landmarks
            self.last_id = id
            self.ids.clear()
            return interpolateds


def show(image, title='Image', delay=1):
    if len(image.shape) == 2:
        image = np.repeat(image[:, :, np.newaxis], 3, axis=2)
    cv2.imshow(title, image[..., ::-1, ::-1])
    cv2.waitKey(delay)


def show_roi(image, roi, title='Image', delay=1):
    if len(image.shape) == 2:
        image = np.repeat(image[:, :, np.newaxis], 3, axis=2)
    image[np.logical_not(roi.astype(bool))] = [0, 0, 0]
    cv2.imshow(title, image[..., ::-1, ::-1])
    cv2.waitKey(delay)


def print_ppg(ppg):
    print(ppg)


class SaveVideo:
    def __init__(self, path, shape=(480, 640), fps=30, fourcc='HFYU'):
        self.writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*fourcc), fps, shape[:2][::-1])

    def __call__(self, image): self.writer.write(image[..., ::-1])

    def __del__(self): self.writer.release()
