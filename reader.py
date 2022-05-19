import sys
import threading
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
from mediapipe.python.solutions.face_mesh import FaceMesh
from mediapipe.python.solutions.hands import Hands

from data import Data
from requeue import ReQueue, ReQueueIterator
import algorithms as algorithms

start = 50


def add_images(queue: ReQueue, values: list[Data], video_input=0):
    cap = cv2.VideoCapture(video_input)
    success = queue.more
    while success and queue.more:
        success, image = cap.read()
        timestamp = time.time()
        values.append(Data(timestamp))
        queue.append((image, timestamp))
    queue.more = False
    cap.release()


def show_images_cv2(queue: ReQueue | ReQueueIterator, window_name="Webcam"):
    for index, (image, timestamp) in queue(ReQueue.LAST):
        cv2.imshow(window_name, image)
        k = cv2.waitKey(1)
        if k == 27:  # Esc
            queue.off()
    cv2.destroyWindow(window_name)


def show_images_contour(queue: ReQueue | ReQueueIterator, values: list[Data], window_name="Webcam"):
    global start
    for index, (image, timestamp) in queue(ReQueue.LAST):
        if values[index].contour is None:
            time.sleep(0.02)
        if values[index].contour is not None:
            image = image.copy()
            cv2.drawContours(image, [values[index].contour.astype(int)], -1, (255, 0, 0), 2)
            for x, y, z in values[index].mesh:
                if x >= image.shape[1] or y >= image.shape[0]:
                    continue
                x, y = int(x), int(y)
                image[y, x] = 255
        if values[index].hand_contour is None:
            print('.')
            time.sleep(0.2)
        if values[index].hand_contour is not None:
            print(values[index].hand_contour)
            image = image.copy()
            for x, y, z in values[index].hand_contour:
                if x >= image.shape[1] or y >= image.shape[0]:
                    continue
                x, y = int(x), int(y)
                image[y, x] = 255
        cv2.imshow(window_name, image[:, ::-1])
        k = cv2.waitKey(1)
        if k == 27:  # Esc
            queue.off()
        elif k == ord('c'):  # CLean the graph
            start = index - 1
    cv2.destroyWindow(window_name)


def get_contours(queue: ReQueue | ReQueueIterator, values: list[Data],
                 face_mesh_detector: FaceMesh, algorithm=algorithms.algorithm_1):
    with face_mesh_detector:
        for index, (image, timestamp) in queue(ReQueue.NEXT):
            results = face_mesh_detector.process(image).multi_face_landmarks
            if results is None:
                continue
            landmarks = np.array([(l.x, l.y, l.z) for l in results[0].landmark])
            landmarks[:, 0] *= image.shape[1]
            landmarks[:, 1] *= image.shape[0]

            values[index].mesh = landmarks

            mesh, contour, mask, value = algorithm(image, [v.mesh for v in values[-10:] if v.mesh is not None])
            if contour is None or mask is None or value is None:
                print('.')
                continue
            values[index].ppg = value[..., ::-1]
            values[index].mesh = mesh
            values[index].contour = contour


def get_hand(queue: ReQueue | ReQueueIterator, values: list[Data],
             hand_mesh_detector: Hands, algorithm=algorithms.algorithm_1):
    with hand_mesh_detector:
        for index, (image, timestamp) in queue(ReQueue.NEXT):
            results = hand_mesh_detector.process(image).multi_hand_landmarks
            if results is None:
                continue
            landmarks = np.array([(l.x, l.y, l.z) for l in results[0].landmark])
            landmarks[:, 0] *= image.shape[1]
            landmarks[:, 1] *= image.shape[0]

            values[index].hand_mesh = landmarks


def get_hand_contour(queue: ReQueue | ReQueueIterator, values: list[Data]):
    PoI = [0, 1, 5, 9, 13, 17]
    for index, (image, timestamp) in queue(ReQueue.NEXT):
        if values[index].hand_mesh is None:
            time.sleep(0.02)
        if values[index].hand_mesh is None:
            continue
        values[index].hand_contour = values[index].hand_mesh[PoI]


def show_plots(queue: ReQueue | ReQueueIterator, values: list[Data], window_name="Webcam"):
    """It only works on main thread."""
    global start
    start = 50
    # plt.figure('PPG')
    figure, axes = plt.subplots(3, 2, figsize=(15, 8))   # 3 rows, 2 columns

    for index, (image, timestamp) in queue(ReQueue.LAST):
        if index <= start: continue
        ppg = np.array([value.ppg for value in values[start:index + 1] if value.ppg is not None])[-400:]
        if len(ppg) == 0: continue
        r, g, b = ppg[..., 0], ppg[..., 1], ppg[..., 2]
        for axis in axes.flat:
            axis.clear()
        axes = axes.reshape((axes.shape[0], -1))
        axes[0, 0].plot(r, color='r')
        axes[0, 0].set_title('ROJO')
        axes[1, 0].plot(g, color='g')
        axes[1, 0].set_title('VERDE')
        axes[2, 0].plot(b, color='b')
        axes[2, 0].set_title('AZUL')
        if axes.shape[1] > 1:
            axes[0, 1].plot(abs(np.fft.fft(r - np.mean(r), axis=0)[:len(r) // 2]), color='r')
            axes[0, 1].set_title('Frequencias ROJO')
            axes[1, 1].plot(abs(np.fft.fft(g - np.mean(g), axis=0)[:len(g) // 2]), color='g')
            axes[1, 1].set_title('Frequencias VERDE')
            axes[2, 1].plot(abs(np.fft.fft(b - np.mean(b), axis=0)[:len(b) // 2]), color='b')
            axes[2, 1].set_title('Frequencias AZUL')
        plt.pause(0.01)
    plt.close()


def main(video_input: int | str = 0, plots: bool = False, face=True, hand=False):
    queue = ReQueue()
    values = []
    video_input = int(video_input) if isinstance(video_input, str) and video_input.isdigit() else video_input
    t0 = threading.Thread(target=add_images, args=(queue, values, video_input))
    t0.start()

    if face:
        t2 = threading.Thread(target=get_contours, args=(queue, values, FaceMesh()))
        t2.start()

    if hand:
        t3 = threading.Thread(target=get_hand, args=(queue, values, Hands()))
        t4 = threading.Thread(target=get_hand_contour, args=(queue, values))
        t3.start()
        t4.start()

    if plots:
        t3 = threading.Thread(target=show_plots, args=(queue, values))
        t3.start()

    t1 = threading.Thread(target=show_images_contour, args=(queue, values))
    t1.start()
    return queue, values


if __name__ == '__main__':
    main(*sys.argv[1:])
