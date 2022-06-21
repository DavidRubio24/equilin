import sys

from mediapipe.python.solutions.face_mesh import FaceMesh

from functions import ImageReader, landmarks, Save, SaveVideo, ppg_around, contour, draw_contour, show, draw_kernel
from roi import forehead_small_comb, forehead_small_PoI, left_under_eye_PoI, right_under_eye_PoI, nose_tip_PoI, \
    nose_PoI, face_RoI, upper_lip, lower_lip, chin, left_down_check, right_down_check, forehead_comb, forehead_PoI
from utils.queues import RequestQueueLast, RequestQueueNext
from utils.worker import Worker, WORKERS


def main(video_input=0, queue_type=RequestQueueNext, duration=600):
    iamges_q = Worker(ImageReader(video_input, duration))()
    landmarks_q = Worker(landmarks, detector=FaceMesh())(iamges_q(queue_type))
    drawn_images_q = Worker(draw_kernel, ['image'])(landmarks_q(queue_type))
    Worker(show, output=False)(drawn_images_q(queue_type))

    for worker in WORKERS:
        worker.start()
    WORKERS.clear()


def show_regions(video_input=0, queue_type=RequestQueueLast, duration=300):
    image_q = Worker(ImageReader(video_input, duration))()
    landmarks_q = Worker(landmarks, detector=FaceMesh())(image_q(queue_type))

    drawn_images_q = Worker(contour, points_indexes=forehead_small_PoI, combination=forehead_small_comb)(landmarks_q(queue_type))
    drawn_images_q = Worker(draw_contour, ['image'], color=(255, 0, 0))(drawn_images_q)
    drawn_images_q = Worker(contour, points_indexes=left_under_eye_PoI)(drawn_images_q)
    drawn_images_q = Worker(draw_contour, ['image'], color=(0, 255, 0))(drawn_images_q)
    drawn_images_q = Worker(contour, points_indexes=right_under_eye_PoI)(drawn_images_q)
    drawn_images_q = Worker(draw_contour, ['image'], color=(0, 0, 255))(drawn_images_q)
    drawn_images_q = Worker(contour, points_indexes=nose_PoI)(drawn_images_q)
    drawn_images_q = Worker(draw_contour, ['image'], color=(0, 255, 255))(drawn_images_q)
    drawn_images_q = Worker(contour, points_indexes=nose_tip_PoI)(drawn_images_q)
    drawn_images_q = Worker(draw_contour, ['image'], color=(255, 0, 255))(drawn_images_q)
    drawn_images_q = Worker(contour, points_indexes=face_RoI)(drawn_images_q)
    drawn_images_q = Worker(draw_contour, ['image'], color=(255, 255, 0))(drawn_images_q)
    drawn_images_q = Worker(contour, points_indexes=upper_lip)(drawn_images_q)
    drawn_images_q = Worker(draw_contour, ['image'], color=(100, 100, 100))(drawn_images_q)
    drawn_images_q = Worker(contour, points_indexes=lower_lip)(drawn_images_q)
    drawn_images_q = Worker(draw_contour, ['image'], color=(255, 80, 80))(drawn_images_q)
    drawn_images_q = Worker(contour, points_indexes=chin)(drawn_images_q)
    drawn_images_q = Worker(draw_contour, ['image'], color=(80, 255, 80))(drawn_images_q)
    drawn_images_q = Worker(contour, points_indexes=left_down_check)(drawn_images_q)
    drawn_images_q = Worker(draw_contour, ['image'], color=(80, 80, 255))(drawn_images_q)
    drawn_images_q = Worker(contour, points_indexes=right_down_check)(drawn_images_q)
    drawn_images_q = Worker(draw_contour, ['image'], color=(80, 80, 80))(drawn_images_q)
    drawn_images_q = Worker(contour, points_indexes=forehead_PoI, combination=forehead_comb)(drawn_images_q)
    drawn_images_q = Worker(draw_contour, ['image'], color=(0, 0, 0))(drawn_images_q)

    Worker(show, output=False)(drawn_images_q(queue_type))
    # Worker(Save(output_file), output=False)(drawn_images_q(queue_type))

    for worker in WORKERS:
        worker.start()
    WORKERS.clear()


def record(video_input=0, output_video='me00.avi', duration=300):
    image_q = Worker(ImageReader(video_input, duration))()
    Worker(SaveVideo(output_video), output=False)(image_q(RequestQueueNext))

    for worker in WORKERS:
        worker.start()
    WORKERS.clear()


if __name__ == '__main__':
    show_regions()
