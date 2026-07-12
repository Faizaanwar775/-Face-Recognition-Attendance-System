"""
Matching: compares a freshly-extracted embedding against every enrolled
employee's stored embedding and returns the best match, subject to a
distance threshold.

Threshold justification (documented, not just hard-coded):
dlib's 128-d embedding space (used by `face_recognition`) is trained so
that faces of the same person are typically within ~0.6 Euclidean
distance, and different people are usually further apart. We default to
0.55 (`settings.match_distance_threshold`) -- slightly stricter than the
library's own 0.6 default -- to bias the system toward fewer false
accepts in an attendance/security context, accepting a small increase in
false rejects (a legitimate employee occasionally needing a second
attempt) as the safer trade-off.
"""

from __future__ import annotations

from typing import List

import numpy as np

from src.config import settings
from src.logging_events.logger import get_logger
from src.storage.models import EmployeeRecord, FaceEmbedding, MatchResult

logger = get_logger(__name__)


def _distance_to_confidence(distance: float, threshold: float) -> float:
    """Map a distance score to an intuitive 0..1 confidence value.

    Confidence is 1.0 at distance 0 (identical), 0.5 at the threshold,
    and approaches 0 as distance grows. This is a simple linear mapping
    used purely for display/logging -- the actual accept/reject decision
    is made on `distance` directly, not on this derived number.
    """
    if threshold <= 0:
        return 0.0
    confidence = 1.0 - (distance / (2 * threshold))
    return float(max(0.0, min(1.0, confidence)))


class FaceMatcher:
    """Compares live embeddings against the enrolled employee roster."""

    def __init__(self, distance_threshold: float = settings.match_distance_threshold) -> None:
        self._distance_threshold = distance_threshold

    def match(
        self, embedding: FaceEmbedding, enrolled: List[EmployeeRecord]
    ) -> MatchResult:
        if not enrolled:
            return MatchResult(distance=1.0, confidence=0.0, is_match=False)

        live_vector = np.asarray(embedding.vector, dtype=np.float64)
        best_distance = float("inf")
        best_record: EmployeeRecord | None = None

        for record in enrolled:
            enrolled_vector = np.asarray(record.embedding, dtype=np.float64)
            if enrolled_vector.shape != live_vector.shape:
                logger.warning(
                    "Skipping employee '%s': embedding dimension mismatch (%s vs %s)",
                    record.employee_id,
                    enrolled_vector.shape,
                    live_vector.shape,
                )
                continue
            distance = float(np.linalg.norm(live_vector - enrolled_vector))
            if distance < best_distance:
                best_distance = distance
                best_record = record

        if best_record is None:
            return MatchResult(distance=1.0, confidence=0.0, is_match=False)

        is_match = best_distance <= self._distance_threshold
        confidence = _distance_to_confidence(best_distance, self._distance_threshold)

        return MatchResult(
            employee_id=best_record.employee_id if is_match else None,
            full_name=best_record.full_name if is_match else None,
            distance=best_distance,
            confidence=confidence,
            is_match=is_match,
        )
