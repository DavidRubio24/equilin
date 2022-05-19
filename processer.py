import sys
import time
from itertools import chain

import cv2
import numpy as np
from mediapipe.python.solutions.face_mesh import FaceMesh

from utils import ReQueue, ReQueueIterator


def add_images(images: ReQueue, video_input=0):
    """Add images to the queue."""
    cap = cv2.VideoCapture(video_input)
    success = images.more
    index = 0
    while success and images.more:
        success, image = cap.read()
        images.append((image, time.time()))
        index += 1
    images.more = False
    cap.release()


def show_images(images: ReQueue | ReQueueIterator, delay=10):
    """Show images from the queue."""
    for image, timestamp in images(ReQueue.LAST):
        cv2.imshow('Original images.', image)
        k = cv2.waitKey(delay)
        if k == 27:  # ESC.
            break


def show_images_drawn(images: ReQueue | ReQueueIterator, pointss: list = (), contours: list = (), delay=10):
    """Show images from the queue."""
    queue = images(ReQueue.LAST)
    for index, (image, timestamp) in enumerate(queue, start=queue.next_item):
        # All contours (-1), color green, thickness 3.
        cv2.drawContours(image, contours, -1, (0, 255, 0), 3)
        points = np.array(pointss, int).reshape(-1, 2)
        for x, y in points:
            # Radius 2, color white, -1 = filled.
            cv2.circle(image, (x, y), 2, (255, 255, 255), -1)
        cv2.imshow('Drawn images.', image)
        k = cv2.waitKey(delay)
        if k == 27:  # ESC.
            break


def detect_faces(images: ReQueue | ReQueueIterator, face_detector: FaceMesh):
    pass


def main(path='./'):
    pass


if __name__ == '__main__':
    main(*sys.argv[1:])
