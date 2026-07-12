"""
detection/detector.py

Face detection ONLY. Deliberately loads InsightFace with
allowed_modules=["detection"] rather than the full FaceAnalysis pack
(detection+recognition+landmark+age/gender) — even though InsightFace could
give us an embedding in the same call, we keep detection and embedding as
separate modules with separate model-loading, matching the required
capture/detection/embeddings/... separation of concerns. The cost is a
second forward pass in embeddings/extractor.py; the benefit is that each
module is independently understandable, testable, and swappable (e.g. you
could replace this detector with a Haar cascade or MediaPipe without
touching embeddings.py at all).
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from insightface.app import FaceAnalysis

from models.schemas import BoundingBox, DetectedFace

logger = logging.getLogger(__name__)


class DetectionError(RuntimeError):
    """Raised for unrecoverable detector failures (e.g. model failed to load)."""


class FaceDetector:
    """
    Thin wrapper around InsightFace's RetinaFace-based detector.

    ctx_id=-1 forces CPU (safe default for a Windows dev laptop without a
    configured CUDA/onnxruntime-gpu setup). Switch to ctx_id=0 if you have a
    working GPU onnxruntime install — that's a deployment/environment
    decision, not an architecture one, so it's a constructor arg here.
    """

    def __init__(self, ctx_id: int = -1, det_size: tuple[int, int] = (640, 640)) -> None:
        try:
            self._app = FaceAnalysis(
                name="buffalo_l",
                allowed_modules=["detection"],
            )
            self._app.prepare(ctx_id=ctx_id, det_size=det_size)
        except Exception as exc:  # model download / load failure, corrupt cache, etc.
            raise DetectionError(
                "Failed to initialize InsightFace detector. If this is the "
                "first run, it needs internet access once to download the "
                "'buffalo_l' model pack to ~/.insightface/models/."
            ) from exc
        logger.info("FaceDetector ready (buffalo_l, ctx_id=%d).", ctx_id)

    def detect(self, frame: np.ndarray, frame_id: int) -> list[DetectedFace]:
        """
        Runs detection on a single BGR frame (as returned by OpenCV).
        Returns a list of validated DetectedFace — empty list if no faces
        found. Never raises on a "no face" result; only raises DetectionError
        on a genuine failure (e.g. malformed frame array), which the caller
        (pipeline worker) can catch and skip without crashing the app.
        """
        if frame is None or frame.size == 0:
            logger.warning("Empty frame passed to detector; skipping.")
            return []

        try:
            raw_faces = self._app.get(frame)
        except Exception:
            logger.exception("Detector inference failed on frame %d; skipping frame.", frame_id)
            return []

        results: list[DetectedFace] = []
        for face in raw_faces:
            try:
                x1, y1, x2, y2 = face.bbox.astype(int)
                bbox = BoundingBox(
                    x=max(0, int(x1)),
                    y=max(0, int(y1)),
                    width=max(1, int(x2 - x1)),
                    height=max(1, int(y2 - y1)),
                )
                landmarks = (
                    [(float(px), float(py)) for px, py in face.kps]
                    if getattr(face, "kps", None) is not None
                    else None
                )
                det_conf = float(face.det_score)
                detected = DetectedFace(
                    frame_id=frame_id,
                    bbox=bbox,
                    detection_confidence=min(max(det_conf, 0.0), 1.0),
                    landmarks_5pt=landmarks,
                )
                results.append(detected)
            except Exception:
                # One malformed detection (e.g. a NaN bbox from a degenerate
                # frame) shouldn't drop every other valid face in the frame.
                logger.exception("Skipping one malformed detection in frame %d.", frame_id)
                continue

        return results

    def crop_face(self, frame: np.ndarray, detected: DetectedFace, margin: float = 0.0) -> Optional[np.ndarray]:
        """
        Utility for downstream modules (embeddings, liveness): crops the
        detected face region out of the frame, with an optional fractional
        margin. Returns None if the resulting crop would be degenerate.
        """
        h, w = frame.shape[:2]
        bx, by, bw, bh = detected.bbox.x, detected.bbox.y, detected.bbox.width, detected.bbox.height

        mx, my = int(bw * margin), int(bh * margin)
        x1, y1 = max(0, bx - mx), max(0, by - my)
        x2, y2 = min(w, bx + bw + mx), min(h, by + bh + my)

        if x2 <= x1 or y2 <= y1:
            return None
        return frame[y1:y2, x1:x2]
