import sys
import time

import cv2
import numpy as np
from mediapipe.python.solutions.face_mesh import FaceMesh
from mediapipe.python.solutions.hands import Hands

from functions import VideoStreamReader, landmarks, Save, SaveVideo, ppg_around, contour, draw_contour, show, \
    draw_kernel, ImageDirectoryReader, roi, ppg, SaveCSV, roi_int
from roi import forehead_small_comb, forehead_small_PoI, left_under_eye_PoI, right_under_eye_PoI, nose_tip_PoI, \
    nose_PoI, face_RoI, upper_lip, lower_lip, chin, left_down_check, right_down_check, forehead_comb, forehead_PoI, \
    forehead_left, forehead_right, left_eye_RoI, right_eye_RoI, right_eyebrow_PoI, left_eyebrow_PoI, mouth_RoI, \
    hand_PoI, hand_comb
from utils.queues import RequestQueueLast, RequestQueueNext
from utils.utils import bar
from utils.worker import Worker, WORKERS


def main(image_directory='data/u004_m_s1_m2_c', queue_type=RequestQueueNext):
    iamges_q = Worker(bar(ImageDirectoryReader(image_directory)).__next__, ('id', 'image'))()
    landmarks_q = Worker(landmarks, detector=FaceMesh())(iamges_q(queue_type))

    contour_q = Worker(contour, points_indexes=forehead_small_PoI, combination=forehead_small_comb)(landmarks_q(queue_type))
    contour_q = Worker(roi)(contour_q(queue_type))
    contour_q = Worker(ppg)(contour_q(queue_type))
    Worker(SaveCSV(image_directory + '_forehead_small_ppg.csv'), output=False)(contour_q(queue_type))
    # contour_q = Worker(contour, points_indexes=left_under_eye_PoI)(contour_q(queue_type))
    # contour_q = Worker(roi)(contour_q(queue_type))
    # contour_q = Worker(ppg)(contour_q(queue_type))
    # Worker(SaveCSV(image_directory + '_left_under_eye_ppg.csv'), output=False)(contour_q(queue_type))
    # contour_q = Worker(contour, points_indexes=right_under_eye_PoI)(contour_q(queue_type))
    # contour_q = Worker(roi)(contour_q(queue_type))
    # contour_q = Worker(ppg)(contour_q(queue_type))
    # Worker(SaveCSV(image_directory + '_right_under_eye_ppg.csv'), output=False)(contour_q(queue_type))
    # contour_q = Worker(contour, points_indexes=nose_PoI)(contour_q(queue_type))
    # contour_q = Worker(roi)(contour_q(queue_type))
    # contour_q = Worker(ppg)(contour_q(queue_type))
    # Worker(SaveCSV(image_directory + '_nose_ppg.csv'), output=False)(contour_q(queue_type))
    # contour_q = Worker(contour, points_indexes=nose_tip_PoI)(contour_q(queue_type))
    # contour_q = Worker(roi)(contour_q(queue_type))
    # contour_q = Worker(ppg)(contour_q(queue_type))
    # Worker(SaveCSV(image_directory + '_nose_tip_ppg.csv'), output=False)(contour_q(queue_type))
    # contour_q = Worker(contour, points_indexes=face_RoI)(contour_q(queue_type))
    # contour_q = Worker(roi)(contour_q(queue_type))
    # contour_q = Worker(ppg)(contour_q(queue_type))
    # Worker(SaveCSV(image_directory + '_face_ppg.csv'), output=False)(contour_q(queue_type))
    # contour_q = Worker(contour, points_indexes=upper_lip)(contour_q(queue_type))
    # contour_q = Worker(roi)(contour_q(queue_type))
    # contour_q = Worker(ppg)(contour_q(queue_type))
    # Worker(SaveCSV(image_directory + '_upper_lip_ppg.csv'), output=False)(contour_q(queue_type))
    # contour_q = Worker(contour, points_indexes=lower_lip)(contour_q(queue_type))
    # contour_q = Worker(roi)(contour_q(queue_type))
    # contour_q = Worker(ppg)(contour_q(queue_type))
    # Worker(SaveCSV(image_directory + '_lower_lip_ppg.csv'), output=False)(contour_q(queue_type))
    # contour_q = Worker(contour, points_indexes=chin)(contour_q(queue_type))
    # contour_q = Worker(roi)(contour_q(queue_type))
    # contour_q = Worker(ppg)(contour_q(queue_type))
    # Worker(SaveCSV(image_directory + '_chin_ppg.csv'), output=False)(contour_q(queue_type))
    # contour_q = Worker(contour, points_indexes=left_down_check)(contour_q(queue_type))
    # contour_q = Worker(roi)(contour_q(queue_type))
    # contour_q = Worker(ppg)(contour_q(queue_type))
    # Worker(SaveCSV(image_directory + '_left_down_check_ppg.csv'), output=False)(contour_q(queue_type))
    # contour_q = Worker(contour, points_indexes=right_down_check)(contour_q(queue_type))
    # contour_q = Worker(roi)(contour_q(queue_type))
    # contour_q = Worker(ppg)(contour_q(queue_type))
    # Worker(SaveCSV(image_directory + '_right_down_check_ppg.csv'), output=False)(contour_q(queue_type))
    # contour_q = Worker(contour, points_indexes=forehead_PoI, combination=forehead_comb)(contour_q(queue_type))
    # contour_q = Worker(roi)(contour_q(queue_type))
    # contour_q = Worker(ppg)(contour_q(queue_type))
    # Worker(SaveCSV(image_directory + '_forehead_ppg.csv'), output=False)(contour_q(queue_type))

    for worker in WORKERS:
        worker.start()
    WORKERS.clear()


def show_regions(video_input=0, queue_type=RequestQueueLast, duration=300):
    image_q = Worker(VideoStreamReader(video_input, duration))()
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
    image_q = Worker(VideoStreamReader(video_input, duration))()
    Worker(SaveVideo(output_video), output=False)(image_q(RequestQueueNext))

    for worker in WORKERS:
        worker.start()
    WORKERS.clear()


def sequential(image_directory='data/u004_m_s1_m2_c', x0=0, y0=0, x1=-1, y1=-1, values=None):
    values = values or {}
    detector = FaceMesh().__enter__()
    for id, image in bar(ImageDirectoryReader(image_directory, delay=0), append=image_directory):
        lms = landmarks(image[y0:y1, x0:x1], detector)[..., :2]
        lms[:, 0] += x0
        lms[:, 1] += y0
        lms_int = np.round(lms).astype(np.int)
        for points_indexes, combination, name in ([forehead_small_PoI, forehead_small_comb, 'forehead_small'],
                                                  [forehead_PoI, forehead_comb, 'forehead']):
            c = contour(lms, points_indexes, combination)
            r = roi(image, None, c, blur=False)
            p = ppg(image, r, min(lms_int[:, 0]), min(lms_int[:, 1]), max(lms_int[:, 0]), max(lms_int[:, 1]))
            value_roi = values.get(name, [])
            value_roi.append(p)
            values[name] = value_roi

        for points_indexes, name in ([left_under_eye_PoI,  'left_under_eye'],
                                     [right_under_eye_PoI, 'right_under_eye'],
                                     [nose_PoI,            'nose'],
                                     [nose_tip_PoI,        'nose_tip'],
                                     [face_RoI,            'face'],
                                     [upper_lip,           'upper_lip'],
                                     [lower_lip,           'lower_lip'],
                                     [chin,                'chin'],
                                     [left_down_check,     'left_down_check'],
                                     [right_down_check,    'right_down_check'],
                                     [forehead_left,       'forehead_left'],
                                     [forehead_right,      'forehead_right']):
            r = roi_int(image, lms_int, points_indexes=points_indexes, blur=False)
            p = ppg(image, r, min(lms_int[:, 0]), min(lms_int[:, 1]), max(lms_int[:, 0]), max(lms_int[:, 1]))
            value_roi = values.get(name, [])
            value_roi.append(p)
            values[name] = value_roi

        r = roi_int(image, lms_int, points_indexes=face_RoI, excludes=[left_eye_RoI, right_eye_RoI, right_eyebrow_PoI, left_eyebrow_PoI, mouth_RoI], blur=False)
        p = ppg(image, r, min(lms_int[:, 0]), min(lms_int[:, 1]), max(lms_int[:, 0]), max(lms_int[:, 1]))
        value_roi = values.get('face_skin', [])
        value_roi.append(p)
        values['face_skin'] = value_roi

    for name, value in values.items():
        np.savetxt(image_directory + '_' + name + '.csv', value, delimiter=',')
        np.save(image_directory + '_' + name + '.npy', value)

    detector.__exit__(None, None, None)


def sequential_hand(image_directory=r'data\pa01\m1\img\c'):

    detector = Hands().__enter__()
    value = []
    for id, image in bar(ImageDirectoryReader(image_directory, delay=0)):
        lms = landmarks(image[:-300, :-900], detector)

        c = contour(lms, hand_PoI, combination=hand_comb)
        r = roi(image, lms, c, blur=False)
        p = ppg(image, r, weighted=False)
        value.append(p)

    np.savetxt(image_directory + '_hand_rgb.csv', value, delimiter=',')
    np.save(   image_directory + '_hand_rgb.npy', value)

    detector.__exit__(None, None, None)


if __name__ == '__main__':
    show_regions()
