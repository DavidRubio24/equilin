from __future__ import annotations

from utils.images import isimage, islandmarks


def get_image(images: list):
    return next(filter(isimage, images), None)


def get_landmarks(landmarks: list):
    return filter(islandmarks, landmarks)
