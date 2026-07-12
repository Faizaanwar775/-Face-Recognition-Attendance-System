"""
Employee enrollment flow.

Captures a short burst of live frames from the camera, extracts a face
embedding from each one that contains exactly one clear face, and stores
the AVERAGE embedding vector for the employee. At no point is a raw
frame or cropped face image written to disk -- only the final numeric
vector is persisted, via `AttendanceDatabase.upsert_employee`.
"""

from __future__ import annotations

import time
from typing import List

import cv2
import numpy as np

from src.capture.camera_stream import CameraStream
from src.config import settings
from src.detection.face_detector import FaceDetector
from src.embeddings.embedding_extractor import EmbeddingExtractor
from src.logging_events.logger import get_logger
from src.storage.database import AttendanceDatabase
from src.storage.models import EmployeeRecord, EnrollmentRequest
from src.utils.drawing import draw_face_box

logger = get_logger(__name__)


class EnrollmentError(RuntimeError):
    """Raised when enrollment cannot be completed."""


class EnrollmentService:
    """Runs the interactive enrollment flow for a new employee."""

    def __init__(self, database: AttendanceDatabase) -> None:
        self._db = database
        self._detector = FaceDetector()
        self._extractor = EmbeddingExtractor()

    def enroll_interactive(self, request: EnrollmentRequest, show_preview: bool = True) -> EmployeeRecord:
        """Run the live-camera enrollment flow. Blocks until enough good
        frames are captured (or the user aborts with 'q')."""
        camera = CameraStream().start()
        collected_vectors: List[List[float]] = []

        logger.info(
            "Starting enrollment for '%s' (%s). Please look at the camera...",
            request.employee_id,
            request.full_name,
        )

        try:
            while len(collected_vectors) < settings.enrollment_frames_required:
                captured = camera.get_latest_frame(timeout=2.0)
                if captured is None:
                    logger.warning("No frame available from camera during enrollment; retrying.")
                    continue

                faces = self._detector.detect(captured.image, captured.metadata.frame_id)

                if show_preview:
                    preview = captured.image.copy()

                if len(faces) == 1:
                    embedding = self._extractor.extract(captured.image, faces[0].bbox)
                    if embedding is not None:
                        collected_vectors.append(embedding.vector)
                        if show_preview:
                            draw_face_box(
                                preview,
                                faces[0].bbox,
                                f"Captured {len(collected_vectors)}/{settings.enrollment_frames_required}",
                                color=(0, 200, 0),
                            )
                elif len(faces) > 1 and show_preview:
                    for face in faces:
                        draw_face_box(preview, face.bbox, "Only 1 face allowed", color=(0, 0, 255))
                elif show_preview:
                    cv2.putText(
                        preview,
                        "No face detected",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2,
                    )

                if show_preview:
                    cv2.imshow("Enrollment - press 'q' to abort", preview)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        raise EnrollmentError("Enrollment aborted by user.")

                time.sleep(0.05)
        finally:
            camera.stop()
            if show_preview:
                cv2.destroyAllWindows()

        if len(collected_vectors) == 0:
            raise EnrollmentError("Failed to capture any valid face embeddings for enrollment.")

        averaged_vector = np.mean(np.asarray(collected_vectors), axis=0).tolist()

        record = EmployeeRecord(
            employee_id=request.employee_id,
            full_name=request.full_name,
            embedding=averaged_vector,
        )
        self._db.upsert_employee(record)
        logger.info(
            "Enrollment complete for '%s' using %d reference frames.",
            request.employee_id,
            len(collected_vectors),
        )
        return record
