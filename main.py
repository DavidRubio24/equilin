import sys

from mediapipe.python.solutions.face_mesh import FaceMesh

from functions import ImageReader, landmarks, contour, ppg, Save, show, draw, print_ppg
from roi import hand_PoI, hand_comb, forehead_PoI, forehead_comb
from utils.queues import RequestQueueLast, RequestQueueNext
from utils.worker import Worker, EmptyIterator, WORKERS


def main(video_input=0, output_file='ppg.npy'):
    image_q = Worker(ImageReader(video_input, 300).__next__, ['id', 'image'])(EmptyIterator())
    landmarks_q = Worker(landmarks, ['landmarks'], detector=FaceMesh())(image_q(RequestQueueLast))
    roi_q = Worker(contour, ['contour'], points_indexes=forehead_PoI, combination=forehead_comb)(landmarks_q(RequestQueueLast))
    ppg_q = Worker(ppg)(roi_q(RequestQueueNext))
    Worker(Save(output_file), output=False)(ppg_q(RequestQueueLast))


def second(video_input=0, output_file='ppg.npy'):
    image_q = Worker(ImageReader(video_input, 300).__next__, ['id', 'image'])(EmptyIterator())
    landmarks_q = Worker(landmarks, detector=FaceMesh())(image_q(RequestQueueLast))
    roi_q = Worker(contour, points_indexes=forehead_PoI, combination=forehead_comb)(landmarks_q(RequestQueueLast))
    draw_q = Worker(draw, ['image'])(roi_q(RequestQueueLast), {'contour': 'landmarks'})
    Worker(show)(draw_q(RequestQueueLast))
    ppg_q = Worker(ppg)(roi_q(RequestQueueLast))
    Worker(Save(output_file), output=False)(ppg_q(RequestQueueLast))

    for worker in WORKERS:
        worker.start()
    WORKERS.clear()


if __name__ == '__main__':
    main(*sys.argv[1:])
