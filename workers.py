import logging
import sys
import threading
import time

import cv2
import numpy as np
from mediapipe.python.solutions.face_mesh import FaceMesh
from mediapipe.python.solutions.hands import Hands

from utils.queues import RequestQueue
from utils.images import isimage


log = logging.getLogger(__name__)
log.addHandler(logging.StreamHandler(sys.stderr))
log.handlers[-1].setFormatter(logging.Formatter('\x1B[0;34m%(asctime)s - %(name)s.%(funcName)s\t- %(levelname)s - %(message)s\x1B[0m'))
log.setLevel(logging.DEBUG)


def get_results_from_detector(detector: FaceMesh | Hands, image):
    """Unified way of getting results from different types of detectors."""
    processed = detector.process(image)
    return processed.__dict__[processed.__doc__[16:36]]


def add_images(video_input, images_queue: RequestQueue):
    """Add images to the queue."""
    cap = cv2.VideoCapture(video_input)
    success = images_queue.more
    while success and images_queue.more:
        success, image = cap.read()
        images_queue.append((time.time(), image))
    images_queue.off()
    cap.release()
    log.debug('End of add_images.')


def show_images(images_queue: RequestQueue, title='Images', delay=10):
    """Show images from the queue."""
    for timestamp, image, *_ in images_queue:
        cv2.imshow(title, image[:, ::-1])
        k = cv2.waitKey(delay)
        if k == 27:  # ESC.
            break
    cv2.destroyWindow(title)
    log.debug('End of show_images.')


def draw_images(images_queue: RequestQueue, drawn_images_queue: RequestQueue, copy=True):
    for timestamp, *rest in images_queue:
        image = next(filter(isimage, rest))
        image = image.copy() if copy else image
        for landmarks in rest:
            if (landmarks is None or not isinstance(landmarks, np.ndarray)
               or landmarks.dtype not in [np.float32, float]
               or not landmarks.shape[-1] == 3):
                continue
            for x, y, z in landmarks.reshape(-1, 3):
                x, y = round(x), round(y)
                cv2.circle(image, (x, y), 1, (255, 255, 255), -1)
        drawn_images_queue.append((timestamp, image, *rest))
    drawn_images_queue.off()
    log.debug('End of draw_images.')


def detect_mesh(images_queue: RequestQueue, mesh_queue: RequestQueue | list[RequestQueue],
                detector_type=FaceMesh):
    """Detect a face in the images."""
    with detector_type() as mesh_detector:
        for timestamp, image, *rest in images_queue:
            if image is None:
                log.debug(f'Image {timestamp} is None.')
                continue
            results = get_results_from_detector(mesh_detector, image)
            if results is None:
                log.debug(f'No detected mesh in {timestamp}.')
                continue

            landmarks = np.array([[(l.x, l.y, l.z) for l in result.landmark] for result in results])
            landmarks[..., 0] *= image.shape[1]
            landmarks[..., 1] *= image.shape[0]

            if isinstance(mesh_queue, RequestQueue):
                mesh_queue.append((timestamp, image, landmarks, *rest))
            else:
                # TODO: Left always first.
                mesh_queue[0].append((timestamp, image, landmarks[0], *rest))
                if len(landmarks) > 1:
                    mesh_queue[0].append((timestamp, image, landmarks[1], *rest))
    if isinstance(mesh_queue, RequestQueue):
        mesh_queue.off()
    else:
        for queue in mesh_queue:
            queue.off()
    log.debug('End of detect_mesh.')


roi_face_points = (103, 67, 109, 10, 338, 297, 332, 333, 299, 337, 151, 108, 69, 104)
roi_face_combination = 1.5 * np.eye(len(roi_face_points)) - .5 * np.eye(len(roi_face_points))[:, ::-1]

roi_hand_points = (0, 1, 5, 9, 13, 17)
# roi_hand_combination = 1.0 * np.eye(len(roi_hand_points))
roi_hand_combined = np.array([[.7,  0., 0., 0., 0., 0.],
                              [0.,  .8, 0., 0., 0., 0.],
                              [0., .05, 1., 0., 0., 0.],
                              [0., .05, 0., 1., 0., 0.],
                              [0., .05, 0., 0., 1., 0.],
                              [.3, .05, 0., 0., 0., 1.]])


def compute_roi(mesh_queue: RequestQueue, roi_queue: RequestQueue,
                roi_points=roi_face_points,
                roi_combination: np.ndarray = roi_face_combination):
    for timestamp, image, landmarks, *rest in mesh_queue:
        contour = landmarks.reshape(-1, 3)[list(roi_points)]
        contour = np.einsum('ij,ik->kj', contour, roi_combination)
        roi_queue.append((timestamp, image, contour, *rest))
    roi_queue.off()
    log.debug('End of compute_roi.')


def get_ppg(roi_queue: RequestQueue, ppg_queue: RequestQueue):
    for timestamp, image, contour, *rest in roi_queue:
        # Get mask from contour.
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [contour[..., :2].astype(int)], 0, 255, -1)
        # Get mean of the mask.
        mean = np.mean(image[mask.astype(bool)], axis=0)
        ppg_queue.append((timestamp, mean, *rest))
    ppg_queue.off()
    log.debug('End of get_ppg.')


def save_array(ppg_queue: RequestQueue, output_file='ppg.npy'):
    values = []
    for timestamp, ppg, *rest in ppg_queue:
        values.append(ppg)
    np.save(output_file, values)
    log.debug(f'End of save_ppg (saved in {output_file}).')


def save_video(images_queue: RequestQueue, output_file='video.mp4'):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, 30.0, (640, 480))

    for timestamp, image in images_queue:
        out.write(image)
    out.release()
    log.debug('End of record.')
