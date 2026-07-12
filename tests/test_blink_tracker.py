"""Unit tests for src.liveness.blink_tracker.BlinkLivenessTracker."""

from __future__ import annotations

from src.liveness.blink_tracker import BlinkLivenessTracker


def test_static_open_eye_never_confirms_liveness() -> None:
    """A static photo produces a constant EAR -- it should never blink."""
    tracker = BlinkLivenessTracker(ear_threshold=0.2, ear_consec_frames=2, required_blinks=1)
    result = None
    for _ in range(50):
        result = tracker.update(track_id=1, ear=0.35)  
    assert result is not None
    assert result.is_live is False
    assert result.blink_count == 0


def test_full_blink_cycle_confirms_liveness() -> None:
    tracker = BlinkLivenessTracker(ear_threshold=0.2, ear_consec_frames=2, required_blinks=1)
    ear_sequence = [0.35, 0.35, 0.10, 0.10, 0.35, 0.35]
    result = None
    for ear in ear_sequence:
        result = tracker.update(track_id=1, ear=ear)

    assert result is not None
    assert result.blink_count == 1
    assert result.is_live is True


def test_brief_dip_below_threshold_not_long_enough_does_not_count() -> None:
    tracker = BlinkLivenessTracker(ear_threshold=0.2, ear_consec_frames=3, required_blinks=1)
    ear_sequence = [0.35, 0.10, 0.35, 0.35]
    result = None
    for ear in ear_sequence:
        result = tracker.update(track_id=1, ear=ear)
    assert result is not None
    assert result.blink_count == 0
    assert result.is_live is False


def test_missing_landmarks_does_not_crash_or_falsely_confirm() -> None:
    tracker = BlinkLivenessTracker()
    result = tracker.update(track_id=1, ear=None)
    assert result.is_live is False
