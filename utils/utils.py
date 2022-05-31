from __future__ import annotations

from mediapipe.python.solutions.face_mesh import FaceMesh
from mediapipe.python.solutions.hands import Hands


def get_results_from_detector(detector: FaceMesh | Hands, image) -> list:
    """Unified way of getting results from different types of detectors."""
    processed = detector.process(image)
    return [processed.__dict__[field] for field in processed._fields]
