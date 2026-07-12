# enrollment/enroll.py
"""
enrollment/enroll.py

Onboarding flow: point the camera at a new employee, capture several
reference frames, extract one embedding per frame, and persist an
EmployeeRecord -- embeddings only, no image ever touches disk.

This module reuses Camera, FaceDetector, and EmbeddingExtractor exactly as
the live attendance pipeline does; enrollment and recognition are two
different *use cases* of the same underlying components, not two separate
implementations of face processing.
"""

from __future__ import annotations

import logging
import time

import cv2

from capture.camera import Camera
from detection.detector import DetectedFace, FaceDetector
from embeddings.extractor import EmbeddingExtractor
from models.schemas import EmployeeRecord, FaceEmbedding
from storage.database import Storage

logger = logging.getLogger(__name__)


class EnrollmentError(RuntimeError):
    """Raised when enrollment cannot be completed (e.g. no usable frames captured)."""


def _pick_primary_face(faces: list[DetectedFace]) -> DetectedFace | None:
    """
    Enrollment assumes one person in frame. If the detector sees more than
    one (someone walked behind the person enrolling), we pick the largest
    bounding box -- almost certainly the person standing closest to the
    camera, i.e. the one being enrolled -- and log a warning rather than
    silently enrolling the wrong face.
    """
    if not faces:
        return None
    if len(faces) > 1:
        logger.warning(
            "%d faces visible during enrollment capture; using the largest "
            "(closest) one. Ensure only the employee being enrolled is in frame.",
            len(faces),
        )
    return max(faces, key=lambda f: f.bbox.width * f.bbox.height)


def enroll_employee(
    employee_id: str,
    full_name: str,
    num_reference_frames: int = 5,
    seconds_between_captures: float = 0.6,
    show_preview: bool = True,
) -> EmployeeRecord:
    """
    Runs an interactive enrollment session and persists the resulting
    EmployeeRecord to storage. Raises EnrollmentError if fewer than
    `num_reference_frames` usable frames are captured after a reasonable
    number of attempts, rather than silently enrolling someone with a
    single low-quality reference.
    """
    detector = FaceDetector()
    extractor = EmbeddingExtractor()
    storage = Storage()

    embeddings: list[FaceEmbedding] = []
    max_attempts = num_reference_frames * 6  # generous budget for missed/blurry frames

    logger.info(
        "Starting enrollment for %s (%s). Look at the camera; slightly vary "
        "your head angle between captures for a more robust reference set.",
        employee_id, full_name,
    )

    with Camera() as cam:
        attempts = 0
        last_capture_time = 0.0

        for meta, frame in cam.frames():
            attempts += 1
            if attempts > max_attempts:
                break

            if show_preview:
                preview = frame.copy()
                cv2.putText(
                    preview,
                    f"Enrolling {full_name}: {len(embeddings)}/{num_reference_frames}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
                )
                cv2.imshow("Enrollment", preview)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            now = time.monotonic()
            if now - last_capture_time < seconds_between_captures:
                continue

            faces = detector.detect(frame, meta.frame_id)
            face = _pick_primary_face(faces)
            if face is None:
                continue

            embedding = extractor.extract(frame, face)
            if embedding is None:
                continue

            embeddings.append(embedding)
            last_capture_time = now
            logger.info("Captured reference %d/%d.", len(embeddings), num_reference_frames)

            if len(embeddings) >= num_reference_frames:
                break

    if show_preview:
        cv2.destroyAllWindows()

    if len(embeddings) < num_reference_frames:
        raise EnrollmentError(
            f"Only captured {len(embeddings)}/{num_reference_frames} usable "
            "reference frames. Try again with better lighting or a closer, "
            "more direct camera angle."
        )

    record = EmployeeRecord(employee_id=employee_id, full_name=full_name, embeddings=embeddings)
    storage.save_employee(record)
    storage.close()

    logger.info("Enrollment complete for %s (%s).", employee_id, full_name)
    return record