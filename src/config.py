"""
Central configuration for the Khizex Face Recognition Attendance System.

All tunable parameters live here so that behaviour can be adjusted without
touching business logic. Values can be overridden with environment
variables (useful for deployment / containerisation) while sane defaults
are provided for local development.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env_float(name: str, default: float) -> float:
    """Read a float from the environment, falling back to ``default``."""
    raw = os.getenv(name)
    try:
        return float(raw) if raw is not None else default
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    """Read an int from the environment, falling back to ``default``."""
    raw = os.getenv(name)
    try:
        return int(raw) if raw is not None else default
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    """Immutable application settings."""

    # --- Paths -----------------------------------------------------------
    base_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent)
    data_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent / "data")
    db_path: Path = field(
        default_factory=lambda: Path(__file__).resolve().parent.parent / "data" / "attendance.db"
    )

    # --- Camera ------------------------------------------------------------
    camera_index: int = _env_int("KHIZEX_CAMERA_INDEX", 0)
    frame_width: int = _env_int("KHIZEX_FRAME_WIDTH", 640)
    frame_height: int = _env_int("KHIZEX_FRAME_HEIGHT", 480)
    target_fps: int = _env_int("KHIZEX_TARGET_FPS", 30)

    # --- Frame queue / backlog handling ------------------------------------
    # Deliberately tiny: we always want the *latest* frame, not a backlog.
    frame_queue_max_size: int = _env_int("KHIZEX_FRAME_QUEUE_SIZE", 1)
    result_queue_max_size: int = _env_int("KHIZEX_RESULT_QUEUE_SIZE", 5)

    # --- Face detection / embedding ----------------------------------------
    detection_model: str = os.getenv("KHIZEX_DETECTION_MODEL", "hog")  # "hog" (CPU) or "cnn" (GPU)
    embedding_model: str = os.getenv("KHIZEX_EMBEDDING_MODEL", "small")  # "small" (5 pts) or "large" (68 pts)
    num_jitters: int = _env_int("KHIZEX_NUM_JITTERS", 1)
    embedding_dim: int = 128  # dlib ResNet embedding size used by face_recognition

    # Only run heavy detection/embedding every Nth captured frame to keep
    # the display loop smooth on modest hardware.
    process_every_n_frames: int = _env_int("KHIZEX_PROCESS_EVERY_N", 2)

    # --- Matching -----------------------------------------------------------
    # face_recognition returns a *distance* (0 = identical, ~0.6 is the
    # typical accept/reject boundary for dlib's 128-d embedding space).
    match_distance_threshold: float = _env_float("KHIZEX_MATCH_THRESHOLD", 0.55)

    # --- Liveness / anti-spoofing --------------------------------------------
    ear_threshold: float = _env_float("KHIZEX_EAR_THRESHOLD", 0.21)
    ear_consec_frames: int = _env_int("KHIZEX_EAR_CONSEC_FRAMES", 2)
    required_blinks: int = _env_int("KHIZEX_REQUIRED_BLINKS", 1)
    liveness_window_seconds: float = _env_float("KHIZEX_LIVENESS_WINDOW_SECONDS", 8.0)
    track_max_missed_frames: int = _env_int("KHIZEX_TRACK_MAX_MISSED", 15)
    track_match_iou_threshold: float = _env_float("KHIZEX_TRACK_IOU_THRESHOLD", 0.3)

    # --- Enrollment -----------------------------------------------------------
    enrollment_frames_required: int = _env_int("KHIZEX_ENROLLMENT_FRAMES", 15)

    # --- Attendance logging cooldown -------------------------------------------
    # Prevents duplicate clock-in/out events if a person lingers in frame.
    log_cooldown_seconds: float = _env_float("KHIZEX_LOG_COOLDOWN_SECONDS", 30.0)

    # --- Concurrency -----------------------------------------------------------
    embedding_process_pool_workers: int = _env_int("KHIZEX_EMBEDDING_WORKERS", 2)


settings = Settings()
settings.data_dir.mkdir(parents=True, exist_ok=True)
