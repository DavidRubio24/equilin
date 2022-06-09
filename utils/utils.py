from __future__ import annotations

import numpy as np

from utils.images import isimage, islandmarks


def get_image(images: list):
    return next(filter(isimage, images), None)


def get_landmarks(landmarks: list):
    return filter(islandmarks, landmarks)


def gaussian(h, w, sigma=10) -> np.ndarray:
    # TODO: compute as separate 1D filters.
    kernel = np.zeros((h, w))
    for i in range(h):
        for j in range(w):
            kernel[i, j] = np.exp(-((i - h // 2) ** 2 + (j - w // 2) ** 2) / (2 * sigma ** 2))
    return kernel / np.sum(kernel)
