"""Unit tests for src.core.tracker.FaceTracker."""

from __future__ import annotations

from src.core.tracker import FaceTracker
from src.storage.models import BoundingBox, DetectedFace


def _face(frame_id: int, top: int, left: int, size: int = 100) -> DetectedFace:
    return DetectedFace(
        frame_id=frame_id,
        bbox=BoundingBox(top=top, left=left, right=left + size, bottom=top + size),
    )


def test_new_detection_gets_new_track_id() -> None:
    tracker = FaceTracker()
    result = tracker.update([_face(1, 0, 0)])
    assert len(result) == 1
    assert result[0].track_id == 1


def test_overlapping_detection_reuses_track_id() -> None:
    tracker = FaceTracker()
    first = tracker.update([_face(1, 0, 0)])
    # Slight shift, should still match via IoU
    second = tracker.update([_face(2, 5, 5)])
    assert first[0].track_id == second[0].track_id


def test_far_away_detection_gets_new_track_id() -> None:
    tracker = FaceTracker()
    first = tracker.update([_face(1, 0, 0)])
    second = tracker.update([_face(2, 900, 900)])
    assert first[0].track_id != second[0].track_id


def test_track_expires_after_max_missed_frames() -> None:
    tracker = FaceTracker(max_missed_frames=2)
    first = tracker.update([_face(1, 0, 0)])
    track_id = first[0].track_id

    # Miss it for more frames than allowed
    tracker.update([])
    tracker.update([])
    tracker.update([])

    # A face reappearing in the same spot should now get a *new* ID
    # since the old track was purged.
    reappeared = tracker.update([_face(5, 0, 0)])
    assert reappeared[0].track_id != track_id
