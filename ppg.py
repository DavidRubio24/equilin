from time import time

import cv2
import numpy as np

import utils as utils
import algorithms as algorithms


class PPG:
    def __init__(self, face_mesh_detector, algo=2,  show=False, save='video.avi'):
        self.face_meshes = []
        self.values = []
        self.rectangle_values = []
        self.times = []
        self.click = utils.Click()
        self.done = False
        self.face_mesh_detector = face_mesh_detector
        self.show = show
        self.algo = algo
        self.video_writer = None
        if isinstance(algo, int | str):
            self.algorithm = algorithms.__dict__.get(f'algorithm_{algo}', algorithms.algorithm_2)

        # Save video to file without compression
        if save:
            self.video_writer = cv2.VideoWriter(save, cv2.VideoWriter_fourcc(*'XVID'), 1.6666666666666666, (640, 480))

    def ppg(self, image):
        image = utils.open_image(image)
        if image is None:
            self.done = True
            return -1, -1, -1
        self.times.append(time())

        # MediaPipe input should be 8 bit RGB, not BGR or grayscale.
        img = image[:, :, ::-1] if len(image.shape) == 3 else np.stack([image] * 3, axis=-1)
        if img.dtype != np.uint8:
            img = img // (np.iinfo(img.dtype).max // (np.iinfo(np.uint8).max + 1))
            img = img.astype(np.uint8)

        results = self.face_mesh_detector.process(img).multi_face_landmarks  # Get entire face mesh.
        if results is None:
            return -1, -1, -1

        # Reformat results into a NumPy array.
        landmarks = np.array([(l.x, l.y, l.z) for l in results[0].landmark])

        landmarks[:, 0] *= image.shape[1]
        landmarks[:, 1] *= image.shape[0]

        self.face_meshes.append(landmarks)

        contour, mask, value = self.algorithm(img, self.face_meshes)

        self.values.append(list(value))

        # Simple rectange.
        x0, y0, x1, y1 = self.click.last_rectangle()
        r, g, b = np.mean(img[y0:y1, x0:x1], axis=(0, 1))
        self.rectangle_values.append((r, g, b))

        return value