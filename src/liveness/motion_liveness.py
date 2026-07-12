"""
Secondary liveness signal: micro-movement + texture analysis.

Blink detection (`blink_tracker.py`) is the primary liveness mechanism
required by the assignment. This module adds a second, complementary
signal so a printed photo or a phone/monitor held up to the camera is
rejected quickly, without waiting for the full blink window:

  * Micro-movement: a real face held in front of a camera always shows
    small natural jitter (breathing, postural sway, hand tremor if
    hand-held). We track the bounding-box centroid across frames and
    require a small amount of variance -- a perfectly rigid image
    (photo taped to a wall, or a monitor on a stand) tends to produce
    suspiciously close-to-zero centroid variance.

  * Texture/frequency analysis: printed photos and screens generally
    have a flatter, more uniform local texture than real skin (paper
    grain / print dot pattern, or a screen's pixel grid and moire
    pattern) once you look at the variance of the Laplacian of the
    cropped face region. Real skin tends to show more irregular
    high-frequency detail (pores, subtle shading).

Both are HEURISTICS, not certainties -- documented explicitly:

KNOWN LIMITATIONS:
  * A photo that is gently hand-waved in front of the camera can
    produce enough centroid jitter to pass the movement check alone.
  * A very high resolution / high quality printed photo, or a very
    good screen, can produce a Laplacian variance close to real skin,
    especially at low camera resolution.
  * Combined with blink detection, these limitations are mitigated but
    not eliminated -- an attacker with a high quality video replay of a
    real blinking face remains the hardest case (see blink_tracker.py).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict

import cv2
import numpy as np

from src.config import settings
from src.storage.models import BoundingBox


@dataclass
class _MotionState:
    centroids: Deque[tuple[float, float]] = field(default_factory=lambda: deque(maxlen=20))
    texture_scores: Deque[float] = field(default_factory=lambda: deque(maxlen=20))


class MotionTextureLiveness:
    """Tracks centroid jitter and texture sharpness per track for a
    secondary, fast-rejecting liveness signal."""

    # Minimum centroid variance (pixels^2) expected from a genuinely
    # held/standing person over ~20 frames. Tuned conservatively low so
    # legitimate, mostly-still users aren't falsely rejected.
    MIN_CENTROID_VARIANCE = 0.15

    # Below this Laplacian variance, texture looks suspiciously flat
    # (consistent with a printed photo or screen at typical webcam
    # resolution/lighting). This is intentionally a soft signal, not a
    # hard gate, used only for the "obviously static" fast-path check.
    MIN_TEXTURE_VARIANCE = 15.0

    def __init__(self) -> None:
        self._states: Dict[int, _MotionState] = {}

    def update(self, track_id: int, bgr_face_crop: np.ndarray, bbox: BoundingBox) -> dict:
        state = self._states.setdefault(track_id, _MotionState())
        state.centroids.append(bbox.center())

        texture_score = self._laplacian_variance(bgr_face_crop)
        if texture_score is not None:
            state.texture_scores.append(texture_score)

        centroid_variance = self._centroid_variance(state.centroids)
        avg_texture = (
            float(np.mean(state.texture_scores)) if state.texture_scores else None
        )

        looks_static = (
            len(state.centroids) >= state.centroids.maxlen
            and centroid_variance < self.MIN_CENTROID_VARIANCE
        )
        looks_flat_texture = avg_texture is not None and avg_texture < self.MIN_TEXTURE_VARIANCE

        return {
            "centroid_variance": centroid_variance,
            "texture_variance": avg_texture,
            "suspected_static_spoof": looks_static and looks_flat_texture,
        }

    @staticmethod
    def _centroid_variance(centroids: Deque[tuple[float, float]]) -> float:
        if len(centroids) < 2:
            return float("inf")  # not enough data yet -> don't flag as static
        arr = np.asarray(centroids)
        return float(np.var(arr[:, 0]) + np.var(arr[:, 1]))

    @staticmethod
    def _laplacian_variance(bgr_face_crop: np.ndarray) -> float | None:
        if bgr_face_crop is None or bgr_face_crop.size == 0:
            return None
        gray = cv2.cvtColor(bgr_face_crop, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    def purge_stale(self, active_track_ids: set[int]) -> None:
        for track_id in list(self._states.keys()):
            if track_id not in active_track_ids:
                del self._states[track_id]
