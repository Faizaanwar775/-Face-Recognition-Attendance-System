"""
capture/camera.py

Owns the OpenCV VideoCapture device. Responsible ONLY for pulling frames off
the camera and pairing each one with validated FrameMetadata — no detection,
no recognition logic lives here.

Error handling philosophy: a camera hiccup (dropped frame, momentary
disconnect) should degrade gracefully (skip frame, retry, log) rather than
raise and crash the whole app. A *sustained* disconnect (device gone) is
raised as CameraError so the caller can decide whether to retry, alert, or
exit.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Iterator, Optional

import cv2
import numpy as np

from config import settings
from models.schemas import FrameMetadata

logger = logging.getLogger(__name__)


class CameraError(RuntimeError):
    """Raised when the camera cannot be opened or has been disconnected for
    longer than the configured retry budget."""


class Camera:
    """
    Thin, defensive wrapper around cv2.VideoCapture.

    Usage:
        with Camera() as cam:
            for meta, frame in cam.frames():
                ...

    `frames()` is a generator so the caller (the capture thread in
    pipeline/) can simply iterate it; internally it retries transient
    read failures a bounded number of times before raising CameraError.
    """

    def __init__(
        self,
        camera_index: int = settings.CAMERA_INDEX,
        target_fps: int = settings.TARGET_FPS,
        max_consecutive_failures: int = 30,
    ) -> None:
        self.camera_index = camera_index
        self.target_fps = target_fps
        self.max_consecutive_failures = max_consecutive_failures
        self._cap: Optional[cv2.VideoCapture] = None
        self._frame_id: int = 0

    # -- lifecycle -----------------------------------------------------

    def open(self) -> None:
        # On Windows, CAP_DSHOW avoids the multi-second startup delay/hang
        # some MSMF backends have with certain USB webcams.
        self._cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        if not self._cap.isOpened():
            # Retry once without the backend hint in case CAP_DSHOW itself
            # is the problem on a given machine/driver combo.
            self._cap = cv2.VideoCapture(self.camera_index)
        if not self._cap.isOpened():
            raise CameraError(
                f"Could not open camera index {self.camera_index}. "
                "Check that no other application is using the webcam and "
                "that Windows camera privacy settings allow desktop apps."
            )
        self._cap.set(cv2.CAP_PROP_FPS, self.target_fps)
        logger.info("Camera %d opened.", self.camera_index)

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            logger.info("Camera %d released.", self.camera_index)

    def __enter__(self) -> "Camera":
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    # -- frame production ------------------------------------------------

    def frames(self) -> Iterator[tuple[FrameMetadata, np.ndarray]]:
        """
        Yields (FrameMetadata, frame) pairs indefinitely. Transient read
        failures are retried with a short backoff; if failures exceed
        max_consecutive_failures in a row, raises CameraError so the caller
        (main loop) can decide to reconnect or exit rather than spinning
        forever silently.
        """
        if self._cap is None:
            raise CameraError("Camera.frames() called before open()")

        consecutive_failures = 0
        while True:
            ok, frame = self._cap.read()

            if not ok or frame is None:
                consecutive_failures += 1
                logger.warning(
                    "Camera read failed (%d/%d consecutive).",
                    consecutive_failures,
                    self.max_consecutive_failures,
                )
                if consecutive_failures >= self.max_consecutive_failures:
                    raise CameraError(
                        "Camera appears disconnected: "
                        f"{consecutive_failures} consecutive read failures."
                    )
                time.sleep(0.05)
                continue

            consecutive_failures = 0

            try:
                height, width = frame.shape[0], frame.shape[1]
                meta = FrameMetadata(
                    frame_id=self._frame_id,
                    timestamp=datetime.utcnow(),
                    width=width,
                    height=height,
                    camera_index=self.camera_index,
                )
            except Exception:
                # A malformed/corrupted frame's metadata failed validation
                # (e.g. zero-size). Skip this frame rather than propagating
                # bad data downstream, and don't crash the app over it.
                logger.exception("Corrupted frame metadata; skipping frame.")
                continue

            self._frame_id += 1
            yield meta, frame
