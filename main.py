import sys

import cv2
from mediapipe.python.solutions.face_mesh import FaceMesh

from functions import ImageReader, landmarks, contour, ppg, Save, show, draw, SaveVideo, ppg_around, roi
from roi import hand_PoI, hand_comb, forehead_PoI, forehead_comb, excludes, face_RoI
from utils.queues import RequestQueueLast, RequestQueueNext
from utils.worker import Worker, EmptyIterator, WORKERS


def main(video_input=0, output_file='ppg.npy', type_=RequestQueueLast, duration=300):
    image_q = Worker(ImageReader(video_input, duration).__next__, ['id', 'image'])(EmptyIterator())
    landmarks_q = Worker(landmarks, detector=FaceMesh())(image_q(type_))
    # contour_q = Worker(contour, points_indexes=forehead_PoI, combination=forehead_comb)(landmarks_q(type_))
    roi_q = Worker(roi, contour=face_RoI, excludes=excludes)(landmarks_q(type_))
    # draw_q = Worker(draw, ['image'])(roi_q(RequestQueueLast), )
    Worker(show, cleanup=cv2.destroyAllWindows)(roi_q(RequestQueueLast), {'roi': 'image'})
    # ppg_q = Worker(ppg_around, ['ppg'])(landmarks_q(type_))
    ppg_q = Worker(ppg)(roi_q(type_))
    Worker(Save(output_file), output=False)(ppg_q(type_))

    for worker in WORKERS:
        worker.start()
    WORKERS.clear()


def record(video_input=0, output_video='me00.avi', duration=300):
    image_q = Worker(ImageReader(video_input, duration).__next__, ['id', 'image'])(EmptyIterator())
    Worker(SaveVideo(output_video), output=False)(image_q(RequestQueueNext))

    for worker in WORKERS:
        worker.start()
    WORKERS.clear()


if __name__ == '__main__':
    main(*sys.argv[1:])
