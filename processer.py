import sys
import time
from itertools import chain
from threading import Thread

import cv2
import numpy as np
from mediapipe.python.solutions.face_mesh import FaceMesh
from mediapipe.python.solutions.hands import Hands

from utils.requeue import ReQueue, ReQueueIterator


def add_images(images_queue: ReQueue, lists: list[list], constructor=dict, video_input=0):
    """Add images to the queue."""
    cap = cv2.VideoCapture(video_input)
    success = images_queue.more
    while success and images_queue.more:
        success, image = cap.read()
        timestamp = time.time()
        for linst_ in lists:
            linst_.append(constructor())
        images_queue.append((image, timestamp))
    images_queue.more = False
    cap.release()


def show_images(images_queue: ReQueue | ReQueueIterator, delay=10):
    """Show images from the queue."""
    for image, timestamp in images_queue(ReQueue.LAST):
        cv2.imshow('Original images.', image)
        k = cv2.waitKey(delay)
        if k == 27:  # ESC.
            break
    cv2.destroyWindow('Original images.')


def show_images_drawn(images_queue: ReQueue | ReQueueIterator, pointsss: list = (), contourss: list = (), delay=10):
    """
    Show images from the queue.

    @param images_queue: Queue of incoming images.
    @param pointsss: list of dict of list of points to draw.
    @param contourss: list of dict of contours to draw.
    @param delay: Miliseconds that each frame will be displayed.
    """
    queue = images_queue(ReQueue.LAST)  # Only show the last available image, don't try to show all.
    for index, (image, timestamp) in enumerate(queue, start=queue.next_item):
        for contour in contourss[index].values():
            # First contour (0), color green, thickness 3.
            cv2.drawContours(image, [contour], 0, (0, 255, 0), 3)

        pointss = pointsss[index].values()
        points = np.array(list(pointss), int).reshape(-1, 3)
        for x, y, z in points:
            # Radius 2, color white, -1 = filled.
            cv2.circle(image, (x, y), 1, (255, 255, 255), -1)
        cv2.imshow('Drawn images.', image)
        k = cv2.waitKey(delay)
        if k == 27:  # ESC.
            images_queue.off()
            break
    cv2.destroyWindow('Drawn images.')


def detect_face_mesh(images_queue: ReQueue | ReQueueIterator, mesh_detector: FaceMesh,
                     pointsss: list, meshes_detected: ReQueue):
    queue = images_queue(ReQueue.LAST)  # Only detect the last available image.
    with mesh_detector as mesh_detector:
        for index, (image, timestamp) in enumerate(queue, start=queue.next_item):
            results = mesh_detector.process(image).multi_face_landmarks
            if results is None: continue

            landmarks = np.array([(l.x, l.y, l.z) for l in results[0].landmark])
            landmarks[:, 0] *= image.shape[1]
            landmarks[:, 1] *= image.shape[0]

            pointsss[index]['face_mesh'] = landmarks
            meshes_detected.append(index)


def detect_hand_mesh(images_queue: ReQueue | ReQueueIterator, mesh_detector: Hands, pointsss: list = ()):
    queue = images_queue(ReQueue.LAST)  # Only detect the last available image.
    with mesh_detector as mesh_detector:
        for index, (image, timestamp) in enumerate(queue, start=queue.next_item):
            results = mesh_detector.process(image).multi_hand_landmarks
            if results is None: continue

            landmarks = np.array([(l.x, l.y, l.z) for l in results[0].landmark])
            landmarks[:, 0] *= image.shape[1]
            landmarks[:, 1] *= image.shape[0]

            pointsss[index]['hand_mesh'] = landmarks


def contour_from_mesh(meshes_detected: ReQueue, pointsss: list[dict], contourss: list[dict],
                     contour_shape: list[list[tuple[int, float]]], name='face_mesh'):
    """
    Compute the contour from th mesh.

    @param meshes_detected: Queue of indexes of meshes detected.
    @param pointsss: list of dict of meshes (a mesh is a list of 3D points).
    @param contourss: list of dict of list of contours to draw.
    @param contour_shape: we'll use a weighted sum of points for each vertex in the contour.
    @param name: name of the mesh.
    """
    for index in meshes_detected:
        if name in contourss[index] or name not in pointsss[index]:
            continue

        points = np.array(pointsss[index][name])
        contour = []
        for vertex_weighted_sum in contour_shape:
            sum_ = np.array([0, 0, 0])
            for point_index, weight in vertex_weighted_sum:
                sum_ += points[point_index] * weight
            contour.append(sum_)
        contourss[index][name] = contour


def main(video_input=0):
    images_queue = ReQueue()
    pointsss = []
    contourss = []
    meshes_detected = ReQueue()
    threads = []
    threads.append(Thread(target=add_images,  args=(images_queue, [pointsss, contourss])))
    threads.append(Thread(target=detect_face_mesh, args=(images_queue, FaceMesh(), pointsss, meshes_detected)))
    # threads.append(Thread(target=show_images, args=(images_queue,)))
    threads.append(Thread(target=show_images_drawn, args=(images_queue, pointsss, contourss)))

    for thread in threads:
        thread.start()

    return images_queue, pointsss, contourss, meshes_detected


if __name__ == '__main__':
    main(*sys.argv[1:])
