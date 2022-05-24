import sys
import threading
import time

import cv2
import numpy as np
from mediapipe.python.solutions.face_mesh import FaceMesh
from mediapipe.python.solutions.hands import Hands

from utils.queues import RequestQueue, RequestQueueLast, RequestQueueNext


def add_images(images_queue: RequestQueue, video_input=0):
    """Add images to the queue."""
    cap = cv2.VideoCapture(video_input)
    success = images_queue.more
    while success and images_queue.more:
        success, image = cap.read()
        images_queue.append((image, time.time()))
    images_queue.off()
    cap.release()


def show_images(images_queue: RequestQueue, delay=10):
    """Show images from the queue."""
    for image, timestamp in images_queue(RequestQueueLast):
        cv2.imshow('Original images.', image)
        k = cv2.waitKey(delay)
        if k == 27:  # ESC.
            break
    cv2.destroyWindow('Original images.')


def detect_face_mesh(images_queue: RequestQueue,
                     mesh_detector: FaceMesh,
                     facemesh_queue: RequestQueue):
    for image, timestamp, *rest in images_queue(RequestQueueLast):
        start = time.time()
        results = mesh_detector.process(image).multi_face_landmarks
        if results is None: continue

        landmarks = np.array([(l.x, l.y, l.z) for l in results[0].landmark])
        landmarks[:, 0] *= image.shape[1]
        landmarks[:, 1] *= image.shape[0]

        facemesh_queue.append((image, timestamp, landmarks, *rest))
        end = time.time() - start

        print(f'Face: {1/end} fps', end='\r')


def detect_hand_mesh(images_queue: RequestQueue,
                     mesh_detector: Hands,
                     handmesh_queue: RequestQueue):
    times = [time.time()]
    for image, timestamp, *rest in images_queue(RequestQueueLast):
        results = mesh_detector.process(image).multi_hand_landmarks
        if results is None: continue

        landmarks = np.array([(l.x, l.y, l.z) for l in results[0].landmark])
        landmarks[:, 0] *= image.shape[1]
        landmarks[:, 1] *= image.shape[0]

        handmesh_queue.append((image, timestamp, landmarks, *rest))

        times.append(time.time())
        if len(times) > 60:
            print(f'Hand: {60/(times[-1] - times[-60])} fps', end='\r')


def show_images_drawn(facemesh_queue: RequestQueueNext):
    for image, timestamp, landmarks in facemesh_queue(RequestQueueLast):
        for x, y, z in landmarks:
            x, y = round(x), round(y)
            cv2.circle(image, (x, y), 1, (255, 255, 255), -1)
        cv2.imshow('Drawn images.', image)
        k = cv2.waitKey(10)
        if k == 27:
            facemesh_queue.off()
            break
    cv2.destroyWindow('Drawn images.')


def main(video_input=0):
    images_queue = RequestQueue()
    facemesh_queue = RequestQueue()
    handmesh_queue = RequestQueue()
    threads = [
        threading.Thread(target=add_images, args=(images_queue, video_input)),
        #  threading.Thread(target=show_images, args=(images_queue,)),
        threading.Thread(target=detect_face_mesh, args=(images_queue, FaceMesh(), facemesh_queue)),
        #  threading.Thread(target=detect_hand_mesh, args=(images_queue, Hands(), handmesh_queue)),
        #  threading.Thread(target=show_images_drawn, args=(facemesh_queue,)),
        #  threading.Thread(target=show_images_drawn, args=(facemesh_queue,)),
        #  threading.Thread(target=show_images_drawn, args=(handmesh_queue,)),
    ]

    for thread in threads:
        thread.start()
    return images_queue


if __name__ == '__main__':
    main(*sys.argv[1:])
