"""
Facial embedding extraction.

This is the CPU-heaviest step in the pipeline (a forward pass through
dlib's ResNet embedding model), and is the piece of work this project
offloads to a `ProcessPoolExecutor` so it never blocks the video capture
or display loop.

IMPORTANT: `extract_embedding` is a free (module-level) function rather
than a method, specifically so it can be pickled and sent to worker
processes by `concurrent.futures.ProcessPoolExecutor`.
"""

from __future__ import annotations

from typing import List, Optional

import face_recognition
import numpy as np

from src.config import settings
from src.storage.models import BoundingBox, FaceEmbedding


def extract_embedding(
    rgb_image: np.ndarray,
    bbox_tuple: tuple[int, int, int, int],
    num_jitters: int = settings.num_jitters,
) -> Optional[List[float]]:
    """Extract a single 128-d embedding for the face at `bbox_tuple`.

    `bbox_tuple` is (top, right, bottom, left) as required by
    `face_recognition.face_encodings`. Runs in a worker process; must
    remain a pure, picklable function with no shared state.

    Returns None if extraction fails (e.g. the crop is degenerate) --
    callers must treat this as "no embedding available" rather than
    letting an exception propagate and kill a worker.
    """
    try:
        encodings = face_recognition.face_encodings(
            rgb_image,
            known_face_locations=[bbox_tuple],
            num_jitters=num_jitters,
            model=settings.embedding_model,
        )
    except Exception:  # noqa: BLE001 - defensive: never crash a worker process
        return None

    if not encodings:
        return None
    return encodings[0].tolist()


class EmbeddingExtractor:
    """Thin synchronous wrapper, useful for enrollment / single-shot use
    where offloading to a process pool would be unnecessary overhead."""

    def __init__(self, num_jitters: int = settings.num_jitters) -> None:
        self._num_jitters = num_jitters

    def extract(self, bgr_image: np.ndarray, bbox: BoundingBox) -> Optional[FaceEmbedding]:
        rgb_image = bgr_image[:, :, ::-1]
        vector = extract_embedding(rgb_image, bbox.as_tuple(), self._num_jitters)
        if vector is None:
            return None
        return FaceEmbedding(vector=vector)
