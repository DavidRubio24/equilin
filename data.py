import numpy as np


class Data:
    timestamp: float
    mesh: np.ndarray
    contour: np.ndarray
    ppg: list[float]
    hand_mesh: np.ndarray = None
    hand_contour = None

    def __init__(self, timestamp: float, mesh: np.ndarray = None, contour: np.ndarray = None, ppg: list[float] = None):
        self.timestamp = timestamp
        self.mesh = mesh
        self.contour = contour
        self.ppg = ppg

    def __repr__(self):
        return f"<Data {self.timestamp=}\n{self.contour=}\n{self.ppg=}\n{self.hand_contour=}>"
