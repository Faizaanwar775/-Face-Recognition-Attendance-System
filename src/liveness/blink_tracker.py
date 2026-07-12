"""
Multi-frame blink-based liveness tracking.

For each tracked face (identified by `track_id` from `core/tracker.py`),
this module maintains a small state machine over the EAR (Eye Aspect
Ratio) time series:

  1. EAR drops below `ear_threshold` for at least `ear_consec_frames`
     consecutive frames  -> the eye is considered "closed".
  2. EAR subsequently rises back above the threshold -> one full blink
     cycle is counted.
  3. Once `required_blinks` full cycles have been observed within a
     rolling `liveness_window_seconds` window, the track is confirmed
     "live".

This is a genuine multi-frame technique -- a single static photo can
never produce the closed -> open transition, because its EAR value never
changes at all. A phone/monitor screen displaying a *static* image is
rejected for the same reason.

KNOWN LIMITATIONS (documented honestly, not claiming perfection):
  * A high-quality video replay of a real person blinking (e.g. a
    recorded clip played on a phone/tablet held up to the camera) WILL
    fool pure blink-based liveness, since the EAR signal would look
    identical to a live person blinking. Combining this with the
    micro-movement/texture checks in `liveness/motion_liveness.py`
    mitigates but does not eliminate this failure mode.
  * Users wearing certain glasses (heavy frames, strong reflections) or
    with partially closed eyes naturally can produce noisy EAR signals
    and take longer to confirm liveness.
  * Very fast blinks faster than the camera's effective frame rate may
    be missed entirely (aliasing) -- mitigated by only requiring one
    full cycle within an 8 second window rather than a hard real-time
    deadline.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict

from src.config import settings
from src.storage.models import LivenessResult


@dataclass
class _BlinkState:
    consecutive_low_frames: int = 0
    blink_count: int = 0
    is_currently_closed: bool = False
    window_start: float = field(default_factory=time.monotonic)
    last_seen: float = field(default_factory=time.monotonic)


class BlinkLivenessTracker:
    """Tracks EAR-based blink state per `track_id` across frames."""

    def __init__(
        self,
        ear_threshold: float = settings.ear_threshold,
        ear_consec_frames: int = settings.ear_consec_frames,
        required_blinks: int = settings.required_blinks,
        window_seconds: float = settings.liveness_window_seconds,
    ) -> None:
        self._ear_threshold = ear_threshold
        self._ear_consec_frames = ear_consec_frames
        self._required_blinks = required_blinks
        self._window_seconds = window_seconds
        self._states: Dict[int, _BlinkState] = {}

    def update(self, track_id: int, ear: float | None) -> LivenessResult:
        """Feed a new EAR sample for `track_id` and return the current
        liveness verdict for that track."""
        now = time.monotonic()
        state = self._states.setdefault(track_id, _BlinkState())
        state.last_seen = now

        # Reset the rolling window if it has expired without success,
        # so a person can't "bank" a blink from ten minutes ago.
        if now - state.window_start > self._window_seconds:
            state.window_start = now
            state.blink_count = 0

        if ear is None:
            # No usable eye landmarks this frame (e.g. profile view) --
            # don't penalise, just skip the update.
            return self._result(track_id, state, reason="no eye landmarks detected this frame")

        if ear < self._ear_threshold:
            state.consecutive_low_frames += 1
            if state.consecutive_low_frames >= self._ear_consec_frames:
                state.is_currently_closed = True
        else:
            if state.is_currently_closed:
                # Eye just reopened after being closed for long enough:
                # that's one full blink cycle.
                state.blink_count += 1
                state.is_currently_closed = False
            state.consecutive_low_frames = 0

        if state.blink_count >= self._required_blinks:
            return self._result(track_id, state, reason="blink cycle confirmed", is_live=True)
        return self._result(
            track_id, state, reason=f"awaiting blink ({state.blink_count}/{self._required_blinks})"
        )

    def _result(
        self, track_id: int, state: _BlinkState, reason: str, is_live: bool = False
    ) -> LivenessResult:
        return LivenessResult(
            track_id=track_id, is_live=is_live, blink_count=state.blink_count, reason=reason
        )

    def purge_stale(self, active_track_ids: set[int]) -> None:
        """Drop tracker state for tracks no longer being seen (memory hygiene)."""
        for track_id in list(self._states.keys()):
            if track_id not in active_track_ids:
                del self._states[track_id]
