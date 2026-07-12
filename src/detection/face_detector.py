"""
Face detection.

Wraps `face_recognition.face_locations` (which itself wraps dlib's HOG or
CNN face detector). Detection is deliberately kept separate from
embedding extraction so each stage can be reasoned about, tested, and
swapped independently (e.g. swapping in a different detector backend
later without touching matching/liveness code).
"""

from __future__ import annotations

from typing import List

import face_recognition
import numpy as np

from src.config import settings
from src.logging_events.logger import get_logger
from src.storage.models import BoundingBox, DetectedFace

logger = get_logger(__name__)


class FaceDetector:
    """Detects one or more faces in a BGR (OpenCV-style) image."""

    def __init__(self, model: str = settings.detection_model) -> None:
        if model not in ("hog", "cnn"):
            raise ValueError(f"Unsupported detection model '{model}'; expected 'hog' or 'cnn'")
        self._model = model

    def detect(self, bgr_image: np.ndarray, frame_id: int) -> List[DetectedFace]:
        """Return all detected faces in the given frame.

        Handles corrupted/empty frames gracefully by returning an empty
        list rather than raising, so a single bad frame never crashes
        the pipeline.
        """
        if bgr_image is None or bgr_image.size == 0:
            logger.warning("Received empty frame (frame_id=%s); skipping detection.", frame_id)
            return []

        try:
            rgb_image = bgr_image[:, :, ::-1]  # BGR -> RGB, no copy needed for detection
            raw_locations = face_recognition.face_locations(rgb_image, model=self._model)
        except Exception:  # noqa: BLE001 - a bad frame must never crash the app
            logger.exception("Face detection failed on frame_id=%s", frame_id)
            return []

        faces: List[DetectedFace] = []
        for top, right, bottom, left in raw_locations:
            try:
                bbox = BoundingBox(top=top, right=right, bottom=bottom, left=left)
            except ValueError as exc:
                logger.debug("Discarding invalid bounding box %s: %s", (top, right, bottom, left), exc)
                continue
            faces.append(DetectedFace(frame_id=frame_id, bbox=bbox))

        return faces
