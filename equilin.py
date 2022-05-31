import sys
import threading

from mediapipe.python.solutions.face_mesh import FaceMesh
from mediapipe.python.solutions.hands import Hands

from roi import hand_PoI, hand_comb, forehead_PoI, forehead_comb
from utils.queues import RequestQueue, RequestQueueNext, RequestQueueLast, join_queues
from workers import add_images, detect_mesh, compute_roi, draw_images, get_ppg, save_array, show_images, save_video

MODE = {int: RequestQueueLast,
        str: RequestQueueNext}


def ppg_face(video_input=0, output_file='ppg.npy', duration=10):
    images_queue   = RequestQueue()
    facemesh_queue = RequestQueue()
    roi_face_queue = RequestQueue()
    drawn_queue    = RequestQueue()
    ppg_queue      = RequestQueue()

    mode = MODE[type(video_input)]

    # We'll create a thread calling the first item with the second item as arguments.
    threads = [
        (add_images,  (video_input,          images_queue, duration)),
        (detect_mesh, (images_queue(mode),   facemesh_queue, FaceMesh)),
        (compute_roi, (facemesh_queue(mode), roi_face_queue)),
        (draw_images, (roi_face_queue,       drawn_queue)),
        (show_images, (drawn_queue,   'Face')),
        (get_ppg,     (roi_face_queue(mode), ppg_queue)),
        (save_array, (ppg_queue(mode),       output_file)),
    ]

    for thread in threads:
        threading.Thread(target=thread[0], args=thread[1], kwargs=thread[2] if len(thread) > 2 else {}).start()
    return images_queue


def ppg_hand(video_input=0, output_face_file='ppg_face.npy', output_hand_file='ppg_hand.npy', duration=30):
    images_queue   = RequestQueue()
    facemesh_queue = RequestQueue()
    handmesh_queue = RequestQueue()
    roi_face_queue = RequestQueue()
    roi_hand_queue = RequestQueue()
    joined_queue   = RequestQueue()
    drawn_queue    = RequestQueue()
    ppg_face_queue = RequestQueue()
    ppg_hand_queue = RequestQueue()

    mode = MODE[type(video_input)]

    # We'll create a thread calling the first item with the second item as arguments and the third as keyword arguments.
    threads = [
        (add_images,  (video_input,    images_queue, duration)),

        (detect_mesh, (images_queue(mode),   facemesh_queue, FaceMesh)),
        (detect_mesh, (images_queue(mode),   handmesh_queue, Hands)),

        (compute_roi, (facemesh_queue(mode), roi_face_queue, forehead_PoI, forehead_comb)),
        (compute_roi, (handmesh_queue(mode), roi_hand_queue,     hand_PoI,     hand_comb)),

        (join_queues, (joined_queue, roi_face_queue, roi_hand_queue)),
        (draw_images, (joined_queue, drawn_queue)),
        (show_images, (drawn_queue,    'Face')),

        (get_ppg,     (roi_face_queue(mode), ppg_face_queue)),
        (get_ppg,     (roi_hand_queue(mode), ppg_hand_queue)),
        (save_array,  (ppg_face_queue(mode), output_face_file)),
        (save_array,  (ppg_hand_queue(mode), output_hand_file)),
    ]

    for thread in threads:
        threading.Thread(target=thread[0], args=thread[1], kwargs=thread[2] if len(thread) > 2 else {}).start()
    return images_queue


def ppg_hand_test(video_input=0, output_face_file='ppg_face.npy', output_hand_file='ppg_hand.npy', duration=30):
    images_queue   = RequestQueue()
    handmesh_queue = RequestQueue()
    roi_hand_queue = RequestQueue()
    ppg_hand_queue = RequestQueue()

    mode = MODE[type(video_input)]

    # We'll create a thread calling the first item with the second item as arguments and the third as keyword arguments.
    threads = [
        (add_images,  (video_input,    images_queue, duration)),

        (detect_mesh, (images_queue(mode),   handmesh_queue, Hands)),

        (compute_roi, (handmesh_queue(mode), roi_hand_queue,     hand_PoI,     hand_comb)),

        (get_ppg,     (roi_hand_queue(mode), ppg_hand_queue)),
    ]

    for thread in threads:
        threading.Thread(target=thread[0], args=thread[1], kwargs=thread[2] if len(thread) > 2 else {}).start()
    return images_queue


def ppg_face_test(video_input=0, output_file='ppg_face.npy', duration=30):
    images_queue   = RequestQueue()
    facemesh_queue = RequestQueue()
    roi_face_queue = RequestQueue()
    ppg_face_queue = RequestQueue()

    mode = MODE[type(video_input)]

    # We'll create a thread calling the first item with the second item as arguments and the third as keyword arguments.
    threads = [
        (add_images,  (video_input,    images_queue, duration)),

        (detect_mesh, (images_queue(mode),   facemesh_queue)),

        (compute_roi, (facemesh_queue(mode), roi_face_queue, forehead_PoI, forehead_comb)),

        (get_ppg,     (roi_face_queue(mode), ppg_face_queue)),
    ]

    for thread in threads:
        threading.Thread(target=thread[0], args=thread[1], kwargs=thread[2] if len(thread) > 2 else {}).start()
    return images_queue


def ppg_face_n_hand_test(video_input=0, output_face_file='ppg_face.npy', output_hand_file='ppg_hand.npy', duration=30):
    images_queue   = RequestQueue()
    facemesh_queue = RequestQueue()
    handmesh_queue = RequestQueue()
    roi_face_queue = RequestQueue()
    roi_hand_queue = RequestQueue()
    ppg_face_queue = RequestQueue()
    ppg_hand_queue = RequestQueue()

    mode = MODE[type(video_input)]

    # We'll create a thread calling the first item with the second item as arguments and the third as keyword arguments.
    threads = [
        (add_images,  (video_input,    images_queue, duration)),

        (detect_mesh, (images_queue(mode),   facemesh_queue, FaceMesh)),
        (detect_mesh, (images_queue(mode),   handmesh_queue, Hands)),

        (compute_roi, (facemesh_queue(mode), roi_face_queue, forehead_PoI, forehead_comb)),
        (compute_roi, (handmesh_queue(mode), roi_hand_queue,     hand_PoI,     hand_comb)),


        (get_ppg,     (roi_face_queue(mode), ppg_face_queue)),
        (get_ppg,     (roi_hand_queue(mode), ppg_hand_queue)),
    ]

    for thread in threads:
        threading.Thread(target=thread[0], args=thread[1], kwargs=thread[2] if len(thread) > 2 else {}).start()
    return images_queue


def record(video_input=0, output_file='video.mp4', duration=10):
    images_queue = RequestQueue()

    threas = [
        (add_images,  (video_input,    images_queue, duration)),
        (save_video,  (images_queue,   output_file)),
        (show_images, (images_queue,   'Video')),
    ]

    for thread in threas:
        threading.Thread(target=thread[0], args=thread[1]).start()
    return images_queue


def main(video_input=0, *output_files, type_='hand'):
    if isinstance(video_input, str) and video_input.isdigit():
        video_input = int(video_input)

    if type_ == 'face':
        return ppg_face(video_input, *output_files[:1])
    elif type_ == 'hand':
        return ppg_hand(video_input, *output_files[:2])
    elif type_ == 'hand_test':
        return ppg_hand_test(video_input, *output_files[:2])
    elif type_ == 'record':
        return record(video_input, *output_files[:1])


if __name__ == '__main__':
    main(*sys.argv[1:])
