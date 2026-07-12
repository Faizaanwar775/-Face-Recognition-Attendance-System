"""Small drawing helpers used to render the on-screen debug overlay.

Purely cosmetic -- none of this touches persistence. The preview window
is for demoing/debugging only; nothing shown here is ever written to
disk or the database.
"""

from __future__ import annotations

import cv2
import numpy as np

from src.storage.models import BoundingBox

COLOR_UNKNOWN = (0, 0, 255)      # red   - unrecognised face
COLOR_PENDING = (0, 200, 255)    # amber - recognised, awaiting liveness
COLOR_CONFIRMED = (0, 200, 0)    # green - matched + live -> logged


def draw_face_box(
    frame: np.ndarray,
    bbox: BoundingBox,
    label: str,
    color: tuple[int, int, int] = COLOR_UNKNOWN,
) -> None:
    """Draw a labelled bounding box on `frame` in-place."""
    cv2.rectangle(frame, (bbox.left, bbox.top), (bbox.right, bbox.bottom), color, 2)
    label_y = max(0, bbox.top - 10)
    cv2.putText(
        frame,
        label,
        (bbox.left, label_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
        cv2.LINE_AA,
    )


def draw_hud(frame: np.ndarray, fps: float, camera_healthy: bool) -> None:
    """Draw a small heads-up display with FPS and camera health."""
    status_color = (0, 200, 0) if camera_healthy else (0, 0, 255)
    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "CAMERA OK" if camera_healthy else "CAMERA ISSUE",
        (10, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        status_color,
        2,
        cv2.LINE_AA,
    )
