from __future__ import annotations

import logging
import sys
import time

import cv2
import numpy as np

from functions import Camera, draw, detect, roi, ppg, ppg_arround
from roi import forehead_PoI, forehead_comb
from utils.queues import RequestQueue
from utils.images import isimage, islandmarks
from utils.utils import get_image, get_landmarks
from utils.images import get_results_from_detector

log = logging.getLogger(__name__)
log.addHandler(logging.StreamHandler(sys.stderr))
log.handlers[-1].setFormatter(logging.Formatter('\x1B[0;34m%(asctime)s - %(name)s.%(funcName)s\t- %(levelname)s - %(message)s\x1B[0m'))
log.setLevel(logging.DEBUG)


def add_images(video_input, images_queue: RequestQueue, duration=None):
    for timestamp, image in Camera(video_input, duration):
        if not images_queue.more: break
        images_queue.append((timestamp, image))
    images_queue.off()
    log.debug('End of add_images_test.')


def show_images(images_queue: RequestQueue, title='Images', delay=10):
    """Show images from the queue."""
    for timestamp, image, *_ in images_queue:
        cv2.imshow(title, image[:, ::-1, ::-1])
        k = cv2.waitKey(delay)
        if k == 27:  # ESC.
            break
    cv2.destroyWindow(title)
    log.debug('End of show_images.')


def draw_images(images_queue: RequestQueue, drawn_queue: RequestQueue, copy=True):
    for timestamp, *rest in images_queue:
        image = next(filter(isimage, rest), None)
        if image is None: continue
        image = draw(image, filter(islandmarks, rest), copy=copy)
        drawn_queue.append((timestamp, image, *rest))
    drawn_queue.off()
    log.debug('End of draw_images.')


def detect_mesh(images_queue: RequestQueue, mesh_queue: RequestQueue | list[RequestQueue], detector):
    """Detect a face in the images."""
    with detector as mesh_detector:
        for timestamp, image, *rest in images_queue:
            landmarks = detect(mesh_detector, image, timestamp)
            if landmarks is None: continue
            mesh_queue.append((timestamp, image, landmarks, *rest))
    mesh_queue.off()
    log.debug('End of detect_mesh.')


def compute_roi_(mesh_queue: RequestQueue, roi_queue: RequestQueue, roi_points=forehead_PoI, roi_comb=forehead_comb):
    for timestamp, image, landmarks, *rest in mesh_queue:
        contour = landmarks.reshape(-1, 3)[list(roi_points)]
        contour = np.einsum('ij,ik->kj', contour, roi_comb)
        roi_queue.append((timestamp, image, contour, *rest))
    roi_queue.off()
    log.debug('End of compute_roi.')


def compute_roi(mesh_queue: RequestQueue, roi_queue: RequestQueue, roi_points, roi_comb=None):
    for timestamp, image, landmarks, *rest in mesh_queue:
        roi_queue.append((timestamp, image, roi(landmarks, roi_points, roi_comb), *rest))
    roi_queue.off()
    log.debug('End of compute_roi.')


def ppg_arround_landmarks(mesh_queue: RequestQueue, ppg_queue: RequestQueue):
    for timestamp, image, landmarks, *rest in mesh_queue:
        ppg_queue.append((timestamp, ppg_arround(image, landmarks), *rest))
    ppg_queue.off()
    log.debug('End of ppg_arround_landmarks.')


def get_ppg_(roi_queue: RequestQueue, ppg_queue: RequestQueue):
    for timestamp, image, contour, *rest in roi_queue:
        # Get mask from contour.
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [contour[..., :2].astype(int)], 0, 255, -1)
        # Get mean of the mask.
        mean = np.mean(image[mask.astype(bool)], axis=0)
        ppg_queue.append((timestamp, mean, *rest))
    ppg_queue.off()
    log.debug('End of get_ppg.')


def get_ppg(roi_queue: RequestQueue, ppg_queue: RequestQueue):
    for timestamp, image, contour, *rest in roi_queue:
        ppg_queue.append((timestamp, ppg(image, contour), *rest))
    ppg_queue.off()
    log.debug('End of get_ppg.')


def get_ppg_computing_rois(mesh_queue: RequestQueue, roi_queue: RequestQueue, roi_points, roi_comb=None, excludes=None):
    roi_comb = roi_comb or np.eye(len(roi_points))
    for timestamp, image, landmarks, *rest in mesh_queue:
        contour = roi(landmarks, roi_points, roi_comb)
        roi_queue.append((timestamp, ppg(image, contour, [roi(landmarks, exclude) for exclude in excludes]), *rest))
    roi_queue.off()
    log.debug('End of compute_roi.')


def save_array(ppg_queue: RequestQueue, output_file='ppg'):
    values = []
    for timestamp, ppg, *rest in ppg_queue:
        values.append(ppg)

    output_file_npy = output_file + '.npy' if not output_file.endswith('.npy') else output_file + '.npy'
    output_file_csv = output_file + '.csv' if not output_file.endswith('.csv') else output_file + '.csv'
    np.save(output_file_npy, values)
    np.savetxt(output_file_csv, values, delimiter=',\t', fmt='%g')

    log.debug(f'End of save_ppg (saved in {output_file}).')


def save_debug(ppg_queue: RequestQueue, output_file='ppg'):
    values = []
    timestamps = []
    for timestamp, ppg, *rest in ppg_queue:
        values.append(ppg)
        timestamps.append(timestamp)

    output_file_npy = output_file + '.npy' if not output_file.endswith('.npy') else output_file + '.npy'
    output_file_csv = output_file + '.csv' if not output_file.endswith('.csv') else output_file + '.csv'
    np.save(output_file_npy, values)
    np.savetxt(output_file_csv, values, delimiter=',')
    np.savetxt(output_file_csv[:-4] + '.timestamps' + '.csv', timestamps, delimiter=',')

    log.debug(f'End of save_ppg (saved in {output_file}).')


def save_video(images_queue: RequestQueue, output_file='video.mp4'):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, 30.0, (640, 480))

    for timestamp, image in images_queue:
        out.write(image)
    out.release()
    log.debug('End of record.')


def interpolate(images_queue: RequestQueue, mesh_queue: RequestQueue, interpolated_queue: RequestQueue):
    images_iterator = images_queue.__iter__()
    mesh_iterator = mesh_queue.__iter__()
    images_since_mesh = []
    last_landmarks = None
    last_timestamp = None

    while mesh_iterator.has_next() or mesh_iterator.more:
        # Get the next landmarks.
        timestamp_mesh, *rest = next(mesh_iterator, (None,))
        if timestamp_mesh is None: continue
        landmarks = next(filter(islandmarks, rest), None)
        if landmarks is None: continue

        # Get all the images since the last landmarks.
        while images_iterator.has_next() or images_iterator.more:
            timestamp_image, image, *rest = next(images_iterator, (None, None))
            if timestamp_image is None: continue
            # TODO: deal with images that are ahead of the mesh.
            images_since_mesh.append((timestamp_image, image))
            if timestamp_image >= timestamp_mesh:
                break

        # Interpolate the images.
        if last_landmarks is None:
            log.debug(f'No last landmarks. {len(images_since_mesh)} images.')
            # Worst interpolation ever: use the current landmarks.
            for timestamp_intermediate_image, image in images_since_mesh:
                # TODO: should't we copy the landmarks? Probably not.
                interpolated_queue.append((timestamp_intermediate_image, image, landmarks, *rest))
        elif last_timestamp is None:
            # How TF did we end up here? Anyway... interpolate using the frame number. Close enough.
            log.warning('No last timestamp to interpolate. Using frame number.')
            total = len(images_since_mesh) - 1
            for index, (timestamp_intermediate_image, image) in enumerate(images_since_mesh):
                # Interpolate between the current and the last landmarks.
                alpha = index / total
                interpolated_landmarks = (landmarks        * alpha
                                          + last_landmarks * (1 - alpha))
                interpolated_queue.append((timestamp_intermediate_image, image, interpolated_landmarks, *rest))
        else:
            # log.debug(f'Interpolating between {last_timestamp} and {timestamp_mesh}. {len(images_since_mesh)} images.')
            total = timestamp_mesh - last_timestamp
            for index, (timestamp_intermediate_image, image) in enumerate(images_since_mesh):
                # Interpolate between the current and the last landmarks.
                alpha = (timestamp_intermediate_image - last_timestamp) / total
                interpolated_landmarks = (landmarks        * alpha
                                          + last_landmarks * (1 - alpha))
                interpolated_queue.append((timestamp_intermediate_image, image, interpolated_landmarks, *rest))

        last_timestamp = timestamp_mesh
        last_landmarks = landmarks
        images_since_mesh = []

    # Get all the images after the last landmarks.
    while images_iterator.has_next() or images_iterator.more:
        timestamp_image, image, *rest = next(images_iterator, (None, None))
        if timestamp_image is None: continue
        # TODO: deal with images that are ahead of the mesh.
        images_since_mesh.append((timestamp_image, image))

    # The ramaining images don't get to be interpolated, just use the last landmarks.
    for timestamp_intermediate_image, image in images_since_mesh:
        log.debug(f'Adding the last images. Timestamp: {timestamp_intermediate_image}, last_landmarks is None {last_landmarks is None}.')
        interpolated_queue.append((timestamp_intermediate_image, image, last_landmarks))

    interpolated_queue.off()

    log.debug('End of interpolate.')
