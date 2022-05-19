import cv2
import numpy as np
from scipy.signal import convolve2d
from scipy.ndimage import gaussian_filter


_default_tail = (100,) * 2 + (90,) * 8 + (70,) * 10 + (50,) * 10 + (20,) * 20 + (5,) * 20
_default_tail = (100, 80, 50, 30, 15, 5)


def algorithm_1(image, face_meshes: list, up=.5, down=.5):
    """Gets the contour of the forehead, based on the last face mesh."""
    if not face_meshes: return None, None, None
    contour_indices = [103,  67, 109,  10, 338, 297, 332,
                       333, 299, 337, 151, 108,  69, 104]
    contour = face_meshes[-1][contour_indices][:, [0, 1]]

    displace = contour[:7] - contour[7:][::-1]  # Upper points [:7] minus their respective [::-1] lower points [7:].

    contour[:7] += displace * up
    contour[7:] -= displace * down

    mask = np.zeros(image.shape[:2], np.uint8)
    cv2.drawContours(mask, [contour.astype(int)], contourIdx=0, color=1, thickness=-1)

    value = np.mean(image[mask.astype(bool)], axis=0)

    return face_meshes[-1], contour, mask, value


def algorithm_2(image, face_meshes: list, up=1., down=.6, tail=_default_tail):
    """Gets the contour of the forehead, based on many of the last face_meshes (weighted by tail)."""
    if not face_meshes: return None, None, None
    tail = np.array(tail[:len(face_meshes)], np.float64)
    tail /= np.sum(tail)
    face_mesh = np.zeros(face_meshes[-1].shape)
    for fm, weight in zip(face_meshes[::-1], tail):
        face_mesh += weight * fm

    return algorithm_1(image, [face_mesh], up, down)


def algorithm_3(image, face_meshes: list, up=.5, down=.5, tail=_default_tail, border=10):
    face_mesh, contour, mask, _ = algorithm_2(image, face_meshes, up, down, tail)
    if contour is None: return None, None, None

    kernel = np.ones((3, 3), dtype=np.int8)
    for _ in range(border):
        mask = convolve2d(mask, kernel, 'same')
    mask = gaussian_filter(mask, 5)
    mask = np.stack([mask] * 3, axis=-1) if len(image.shape) == 3 else mask
    value = np.average(image, axis=(0, 1), weights=mask)

    return face_mesh, contour, mask, value
