"""
Camera capture (producer side of the producer/consumer pipeline).

`CameraStream` runs a dedicated thread that continuously reads frames
from OpenCV's `VideoCapture` and publishes the *latest* one into a
single-slot queue. If the consumer hasn't picked up the previous frame
yet, it is silently dropped and replaced -- this is the mechanism that
prevents backlog build-up described in the assignment: we always want
the newest frame, never an ever-growing queue of stale ones.

A camera disconnect or a corrupted read does not crash the thread; it is
logged, retried with a short backoff, and surfaced to the consumer via
`is_healthy`.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import cv2
import numpy as np

from src.config import settings
from src.logging_events.logger import get_logger
from src.storage.models import FrameMetadata

logger = get_logger(__name__)


@dataclass
class CapturedFrame:
    """A frame plus its validated metadata, held together in memory only."""

    image: np.ndarray
    metadata: FrameMetadata


class CameraStream:
    """Background thread that continuously captures frames from a camera."""

    def __init__(
        self,
        camera_index: int = settings.camera_index,
        width: int = settings.frame_width,
        height: int = settings.frame_height,
    ) -> None:
        self._camera_index = camera_index
        self._width = width
        self._height = height

        self._queue: "queue.Queue[CapturedFrame]" = queue.Queue(
            maxsize=settings.frame_queue_max_size
        )
        self._capture: Optional[cv2.VideoCapture] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._frame_counter = 0
        self._healthy = threading.Event()

        # A separately-maintained "latest frame" reference (peek, never
        # consumed) so the display loop can render at its own pace,
        # decoupled from whatever rate the recognition consumer is
        # pulling frames from `self._queue` at.
        self._latest_lock = threading.Lock()
        self._latest_frame: Optional[CapturedFrame] = None

    # ------------------------------------------------------------------ #
    def start(self) -> "CameraStream":
        self._open_capture()
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="CameraCaptureThread", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        if self._capture is not None:
            self._capture.release()
        logger.info("Camera stream stopped.")

    def is_healthy(self) -> bool:
        return self._healthy.is_set()

    def get_latest_frame(self, timeout: float = 1.0) -> Optional[CapturedFrame]:
        """Block briefly for the next available frame (consumes it from the
        processing queue); returns None on timeout. Intended for the
        recognition/processing consumer."""
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def peek_current_frame(self) -> Optional[CapturedFrame]:
        """Return the most recently captured frame WITHOUT consuming it.

        Intended for the display loop, so on-screen video stays smooth at
        its own pace regardless of how fast the recognition consumer is
        draining `self._queue`.
        """
        with self._latest_lock:
            return self._latest_frame

    # ------------------------------------------------------------------ #
    def _open_capture(self) -> None:
        capture = cv2.VideoCapture(self._camera_index)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        if not capture.isOpened():
            logger.error("Unable to open camera index %s", self._camera_index)
        self._capture = capture

    def _run(self) -> None:
        consecutive_failures = 0
        while not self._stop_event.is_set():
            if self._capture is None or not self._capture.isOpened():
                self._healthy.clear()
                logger.warning("Camera not open; attempting to reconnect...")
                time.sleep(1.0)
                self._open_capture()
                continue

            ok, frame = self._capture.read()
            if not ok or frame is None:
                consecutive_failures += 1
                self._healthy.clear()
                logger.warning(
                    "Failed to read frame (%d consecutive failures)", consecutive_failures
                )
                if consecutive_failures >= 10:
                    logger.error("Too many consecutive read failures; reopening camera device.")
                    if self._capture is not None:
                        self._capture.release()
                    self._capture = None
                    consecutive_failures = 0
                time.sleep(0.05)
                continue

            consecutive_failures = 0
            self._healthy.set()
            self._frame_counter += 1

            metadata = FrameMetadata(
                frame_id=self._frame_counter,
                timestamp=datetime.now(),
                width=frame.shape[1],
                height=frame.shape[0],
                source=f"camera:{self._camera_index}",
            )
            captured = CapturedFrame(image=frame, metadata=metadata)

            with self._latest_lock:
                self._latest_frame = captured

            # --- Backlog prevention -------------------------------------
            # Drop the previous unread frame (if any) so the queue never
            # grows beyond a single slot; the consumer always gets the
            # most recent frame rather than an ever-growing backlog.
            if self._queue.full():
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    pass
            try:
                self._queue.put_nowait(captured)
            except queue.Full:
                # Extremely unlikely race; just skip this frame.
                pass
