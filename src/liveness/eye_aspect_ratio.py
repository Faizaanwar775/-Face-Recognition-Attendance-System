"""
Eye Aspect Ratio (EAR) computation.

EAR is the classic Soukupova & Cech (2016) metric used to detect eye
blinks from facial landmarks:

    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)

where p1..p6 are the six 2D landmark points around a single eye. EAR
stays roughly constant while the eye is open and drops sharply toward
zero during a blink, then recovers -- which is exactly the multi-frame
signature this system looks for (see `liveness/blink_tracker.py`).
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

Point = Tuple[float, float]


def _euclidean(a: Point, b: Point) -> float:
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)))


def eye_aspect_ratio(eye_points: Sequence[Point]) -> float:
    """Compute EAR for a single eye given its 6 ordered landmark points.

    Point order follows the standard 68-point dlib convention:
    [outer_corner, top_left, top_right, inner_corner, bottom_right, bottom_left]
    """
    if len(eye_points) != 6:
        raise ValueError(f"Expected 6 eye landmark points, got {len(eye_points)}")

    p1, p2, p3, p4, p5, p6 = eye_points
    vertical_1 = _euclidean(p2, p6)
    vertical_2 = _euclidean(p3, p5)
    horizontal = _euclidean(p1, p4)

    if horizontal == 0:
        return 0.0
    return (vertical_1 + vertical_2) / (2.0 * horizontal)


def average_ear_from_landmarks(landmarks: dict) -> float | None:
    """Given a face_recognition-style landmarks dict, return the average
    EAR across both eyes, or None if eye landmarks are unavailable.

    `landmarks` is the dict returned per-face by
    `face_recognition.face_landmarks(image, model="large")`, expected to
    contain 'left_eye' and 'right_eye' keys with 6 points each.
    """
    left_eye = landmarks.get("left_eye")
    right_eye = landmarks.get("right_eye")
    if not left_eye or not right_eye or len(left_eye) != 6 or len(right_eye) != 6:
        return None

    left_ear = eye_aspect_ratio(left_eye)
    right_ear = eye_aspect_ratio(right_eye)
    return (left_ear + right_ear) / 2.0
