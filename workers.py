import sys
import threading
import time

import cv2
import numpy as np
from mediapipe.python.solutions.face_mesh import FaceMesh
from mediapipe.python.solutions.hands import Hands

from utils.queues import RequestQueue, RequestQueueLast, RequestQueueNext


def get_results_from_detector(detector: FaceMesh | Hands, image):
    """Unified way of getting results from different types of detectors."""
    processed = detector.process(image)
    return processed.__dict__[processed.__doc__[16:36]]


def add_images(images_queue: RequestQueue, video_input=0):
    """Add images to the queue."""
    cap = cv2.VideoCapture(video_input)
    success = images_queue.more
    while success and images_queue.more:
        success, image = cap.read()
        images_queue.append((image, time.time()))
    images_queue.off()
    cap.release()


def show_images(images_queue: RequestQueue, title='Images', delay=10):
    """Show images from the queue."""
    for image, timestamp, *_ in images_queue(RequestQueueLast):
        cv2.imshow(title, image)
        k = cv2.waitKey(delay)
        if k == 27:  # ESC.
            break
    cv2.destroyWindow(title)


def draw_images(images_queue: RequestQueue, drawn_images_queue: RequestQueue, copy=True):
    for image, timestamp, landmarks, *rest in images_queue(RequestQueueLast):
        image = image.copy() if copy else image
        for x, y, z in landmarks.reshape(-1, 3):
            x, y = round(x), round(y)
            cv2.circle(image, (x, y), 1, (255, 255, 255), -1)
        drawn_images_queue.append((image, timestamp, landmarks, *rest))


def detect_mesh(images_queue: RequestQueue, facemesh_queue: RequestQueue, detector_type=FaceMesh):
    """Detect a face in the images."""
    with detector_type() as mesh_detector:
        for image, timestamp, *rest in images_queue(RequestQueueLast):
            results = get_results_from_detector(mesh_detector, image)
            if results is None: continue

            landmarks = np.array([[(l.x, l.y, l.z) for l in result.landmark] for result in results])
            landmarks[..., 0] *= image.shape[1]
            landmarks[..., 1] *= image.shape[0]

            facemesh_queue.append((image, timestamp, landmarks, *rest))


roi_face_points = (103, 67, 109, 10, 338, 297, 332, 333, 299, 337, 151, 108, 69, 104)

roi_face_combination = 1.0 * np.eye(len(roi_face_points)) - .0 * np.eye(len(roi_face_points))[:, ::-1]


def compute_roi(mesh_queue: RequestQueue, roi_queue: RequestQueue,
                roi_points=roi_face_points,
                roi_combination: np.ndarray = roi_face_combination):
    # roi_combination = np.stack([roi_combination, roi_combination, roi_combination], axis=-1)
    for image, timestamp, landmarks, *rest in mesh_queue(RequestQueueLast):
        contour = landmarks.reshape(-1, 3)[list(roi_points)]
        contour = np.einsum('ij,ik->kj', contour, roi_combination)
        roi_queue.append((image, timestamp, contour, *rest))


def main(video_input=0):
    images_queue = RequestQueue()

    facemesh_queue = RequestQueue()
    handmesh_queue = RequestQueue()

    roi_queue = RequestQueue()

    drawn_face_queue = RequestQueue()
    drawn_hands_queue = RequestQueue()
    threads = [
        threading.Thread(target=add_images, args=(images_queue, video_input)),

        threading.Thread(target=detect_mesh, args=(images_queue, facemesh_queue)),
        #  threading.Thread(target=detect_mesh, args=(images_queue, handmesh_queue, Hands)),

        threading.Thread(target=compute_roi, args=(facemesh_queue, roi_queue)),

        #  threading.Thread(target=draw_images, args=(handmesh_queue, drawn_hands_queue)),
        threading.Thread(target=draw_images, args=(roi_queue, drawn_face_queue)),

        #  threading.Thread(target=show_images, args=(drawn_hands_queue, 'Hands')),
        threading.Thread(target=show_images, args=(drawn_face_queue, 'Face')),
    ]

    for thread in threads:
        thread.start()
    return images_queue


if __name__ == '__main__':
    main(*sys.argv[1:])
