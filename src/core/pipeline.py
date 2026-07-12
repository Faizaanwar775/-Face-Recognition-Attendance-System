"""
Main attendance pipeline orchestrator.

Concurrency architecture (documented here, mirrored in the README):

  ┌────────────────┐      single-slot       ┌────────────────────┐
  │ Capture thread │ ───── frame queue ───► │ Processing thread  │
  │ (producer)     │      (maxsize=1,       │ (consumer)         │
  │                │       drops stale      │  - detect faces    │
  │                │       frames)          │  - track faces     │
  └───────┬────────┘                        │  - EAR / liveness  │
          │                                 │  - submit embedding│
          │ peek (non-consuming)            │    jobs to a       │
          ▼                                 │    ProcessPoolExecutor
  ┌────────────────┐                        │  - poll results,   │
  │  Display loop  │ ◄──── shared state ──── │    match, log      │
  │  (main thread) │        (locked)         └────────────────────┘
  └────────────────┘

Three concurrency mechanisms work together, matching the assignment's
allowed options:
  1. Threading producer/consumer: the capture thread never blocks on
     recognition; the processing thread never blocks the video feed.
  2. multiprocessing (ProcessPoolExecutor): the CPU-bound embedding
     extraction (a dlib ResNet forward pass) runs in worker
     *processes*, sidestepping the GIL entirely, so it cannot stall
     either the capture or processing threads even under load.
  3. The display loop reads a `peek`-style snapshot of the camera and a
     lock-protected dict of per-track overlay state -- it never waits
     on the processing thread, so the visible video stays smooth even
     if recognition is temporarily behind.

Backlog handling: the capture->processing hand-off queue has
`maxsize=1` and always drops the previous unread frame before pushing a
new one (see `capture/camera_stream.py`). The processing thread also
only ever has at most one *pending* embedding job per track at a time
(`_pending_futures`), so a slow model never causes an unbounded queue of
outstanding work -- new detections for a track are simply skipped until
the current job resolves.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import dataclass
from typing import Dict, Optional

import cv2
import numpy as np

import face_recognition

from src.capture.camera_stream import CameraStream
from src.config import settings
from src.core.tracker import FaceTracker
from src.detection.face_detector import FaceDetector
from src.embeddings.embedding_extractor import extract_embedding
from src.liveness.blink_tracker import BlinkLivenessTracker
from src.liveness.eye_aspect_ratio import average_ear_from_landmarks
from src.liveness.motion_liveness import MotionTextureLiveness
from src.logging_events.attendance_logger import AttendanceLogger
from src.logging_events.logger import get_logger
from src.matching.face_matcher import FaceMatcher
from src.storage.database import AttendanceDatabase
from src.storage.models import BoundingBox, DetectedFace, FaceEmbedding
from src.utils.drawing import COLOR_CONFIRMED, COLOR_PENDING, COLOR_UNKNOWN, draw_face_box, draw_hud

logger = get_logger(__name__)


@dataclass
class _TrackDisplayState:
    bbox: BoundingBox
    label: str
    color: tuple[int, int, int]


class AttendancePipeline:
    """Wires together capture, detection, liveness, matching and logging."""

    def __init__(self) -> None:
        self._db = AttendanceDatabase()
        self._detector = FaceDetector()
        self._tracker = FaceTracker()
        self._blink_tracker = BlinkLivenessTracker()
        self._motion_tracker = MotionTextureLiveness()
        self._matcher = FaceMatcher()
        self._attendance_logger = AttendanceLogger(self._db)

        self._camera = CameraStream()
        self._executor = ProcessPoolExecutor(max_workers=settings.embedding_process_pool_workers)

        self._pending_futures: Dict[int, "Future[Optional[list[float]]]"] = {}
        self._confirmed_tracks: set[int] = set()  # already logged this "visit"
        self._latest_liveness: dict = {}

        self._display_state: Dict[int, _TrackDisplayState] = {}
        self._state_lock = threading.Lock()

        self._stop_event = threading.Event()
        self._processing_thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    def start(self) -> None:
        self._camera.start()
        self._stop_event.clear()
        self._processing_thread = threading.Thread(
            target=self._processing_loop, name="RecognitionProcessingThread", daemon=True
        )
        self._processing_thread.start()
        logger.info("Attendance pipeline started.")

    def stop(self) -> None:
        self._stop_event.set()
        if self._processing_thread is not None:
            self._processing_thread.join(timeout=5)
        self._executor.shutdown(wait=False, cancel_futures=True)
        self._camera.stop()
        self._db.close()
        logger.info("Attendance pipeline stopped.")

    # ------------------------------------------------------------------ #
    # Processing thread (consumer): detection, tracking, liveness, matching
    # ------------------------------------------------------------------ #
    def _processing_loop(self) -> None:
        while not self._stop_event.is_set():
            captured = self._camera.get_latest_frame(timeout=0.5)
            if captured is None:
                continue

            bgr_image = captured.image
            frame_id = captured.metadata.frame_id

            raw_faces = self._detector.detect(bgr_image, frame_id)
            tracked_faces = self._tracker.update(raw_faces)
            active_ids = {f.track_id for f in tracked_faces if f.track_id is not None}

            self._blink_tracker.purge_stale(active_ids)
            self._motion_tracker.purge_stale(active_ids)
            self._prune_display_state(active_ids)

            rgb_image = bgr_image[:, :, ::-1]

            for face in tracked_faces:
                self._process_single_face(face, bgr_image, rgb_image)

            self._collect_completed_embeddings()

    def _process_single_face(
        self, face: DetectedFace, bgr_image: np.ndarray, rgb_image: np.ndarray
    ) -> None:
        assert face.track_id is not None
        track_id = face.track_id
        bbox = face.bbox

        # --- Liveness signals (computed every frame; cheap relative to
        #     embedding extraction, so kept on this thread rather than
        #     the process pool) ---------------------------------------
        try:
            landmarks_list = face_recognition.face_landmarks(
                rgb_image, face_locations=[bbox.as_tuple()], model="large"
            )
            landmarks = landmarks_list[0] if landmarks_list else {}
        except Exception:  # noqa: BLE001
            landmarks = {}

        ear = average_ear_from_landmarks(landmarks) if landmarks else None
        liveness_result = self._blink_tracker.update(track_id, ear)

        face_crop = bgr_image[bbox.top : bbox.bottom, bbox.left : bbox.right]
        motion_info = self._motion_tracker.update(track_id, face_crop, bbox)

        if motion_info["suspected_static_spoof"]:
            self._set_display_state(
                track_id, bbox, "SPOOF SUSPECTED (static)", COLOR_UNKNOWN
            )
            return

        # --- Already confirmed and logged this visit? Just keep showing
        #     green, don't re-submit embedding jobs or re-log. ---------
        if track_id in self._confirmed_tracks:
            state = self._display_state.get(track_id)
            label = state.label if state else "Confirmed"
            self._set_display_state(track_id, bbox, label, COLOR_CONFIRMED)
            return

        # --- Submit (at most one in-flight) embedding job for this track
        if track_id not in self._pending_futures:
            future = self._executor.submit(extract_embedding, rgb_image, bbox.as_tuple())
            self._pending_futures[track_id] = future
            self._set_display_state(
                track_id, bbox, f"Analyzing... (blinks {liveness_result.blink_count})", COLOR_PENDING
            )
        else:
            self._set_display_state(
                track_id, bbox, f"Analyzing... (blinks {liveness_result.blink_count})", COLOR_PENDING
            )

        # Stash the latest liveness verdict so `_collect_completed_embeddings`
        # can combine it with the (possibly still-pending) match result.
        self._latest_liveness[track_id] = liveness_result

    def _collect_completed_embeddings(self) -> None:
        done_track_ids = [tid for tid, fut in self._pending_futures.items() if fut.done()]
        for track_id in done_track_ids:
            future = self._pending_futures.pop(track_id)
            try:
                vector = future.result()
            except Exception:  # noqa: BLE001
                logger.exception("Embedding extraction worker raised for track %s", track_id)
                continue

            if vector is None:
                continue

            embedding = FaceEmbedding(vector=vector)
            enrolled = self._db.get_all_employees()
            match_result = self._matcher.match(embedding, enrolled)

            liveness = self._latest_liveness.get(track_id)
            is_live = bool(liveness and liveness.is_live)

            state = self._display_state.get(track_id)
            bbox = state.bbox if state else None
            if bbox is None:
                continue

            if not match_result.is_match:
                self._set_display_state(track_id, bbox, "Unknown", COLOR_UNKNOWN)
                continue

            if not is_live:
                blink_count = liveness.blink_count if liveness else 0
                self._set_display_state(
                    track_id,
                    bbox,
                    f"{match_result.full_name}: confirm liveness ({blink_count} blinks)",
                    COLOR_PENDING,
                )
                continue

            # Matched AND live -> log attendance.
            entry = self._attendance_logger.try_log_attendance(
                match_result.employee_id, match_result.confidence
            )
            if entry is not None:
                self._confirmed_tracks.add(track_id)
                self._set_display_state(
                    track_id,
                    bbox,
                    f"{match_result.full_name}: {entry.event_type.upper()} logged",
                    COLOR_CONFIRMED,
                )
            else:
                # Within cooldown -- already logged recently, just show confirmed.
                self._confirmed_tracks.add(track_id)
                self._set_display_state(
                    track_id, bbox, f"{match_result.full_name}: confirmed", COLOR_CONFIRMED
                )

    # ------------------------------------------------------------------ #
    # Shared display-state helpers
    # ------------------------------------------------------------------ #
    def _set_display_state(
        self, track_id: int, bbox: BoundingBox, label: str, color: tuple[int, int, int]
    ) -> None:
        with self._state_lock:
            self._display_state[track_id] = _TrackDisplayState(bbox=bbox, label=label, color=color)

    def _prune_display_state(self, active_ids: set[int]) -> None:
        with self._state_lock:
            for track_id in list(self._display_state.keys()):
                if track_id not in active_ids:
                    del self._display_state[track_id]
        self._confirmed_tracks &= active_ids
        for track_id in list(self._latest_liveness.keys()):
            if track_id not in active_ids:
                del self._latest_liveness[track_id]
        for track_id in list(self._pending_futures.keys()):
            if track_id not in active_ids:
                # Track disappeared while a job was in flight; let it
                # finish naturally but stop waiting on it for this track.
                self._pending_futures.pop(track_id, None)

    def _snapshot_display_state(self) -> Dict[int, _TrackDisplayState]:
        with self._state_lock:
            return dict(self._display_state)

    # ------------------------------------------------------------------ #
    # Display loop (main thread)
    # ------------------------------------------------------------------ #
    def run_display_loop(self) -> None:
        """Blocking display loop. Press 'q' to quit."""
        self.start()
        last_fps_time = time.monotonic()
        frame_counter = 0
        fps = 0.0

        try:
            while True:
                captured = self._camera.peek_current_frame()
                if captured is None:
                    time.sleep(0.01)
                    continue

                frame = captured.image.copy()
                for state in self._snapshot_display_state().values():
                    draw_face_box(frame, state.bbox, state.label, state.color)

                frame_counter += 1
                now = time.monotonic()
                if now - last_fps_time >= 1.0:
                    fps = frame_counter / (now - last_fps_time)
                    frame_counter = 0
                    last_fps_time = now

                draw_hud(frame, fps, self._camera.is_healthy())
                cv2.imshow("Khizex Attendance - press 'q' to quit", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            self.stop()
            cv2.destroyAllWindows()
