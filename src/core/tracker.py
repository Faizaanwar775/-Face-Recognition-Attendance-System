"""
Lightweight cross-frame face tracker.

Face recognition and matching happen per-frame, but liveness detection
(blink counting) requires correlating "this box in frame N" with "this
box in frame N+5" so that blinks can be accumulated for the *same*
physical face over time. This module implements a minimal
IoU-based tracker -- good enough for a single-camera attendance kiosk
with a handful of people in frame, without pulling in a full
DeepSORT-style dependency.

This is intentionally simple and documented as such: it is not designed
to survive long occlusions or fast crowds, which is an acceptable
trade-off for a doorway/desk attendance use case.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from src.config import settings
from src.storage.models import BoundingBox, DetectedFace


def _iou(a: BoundingBox, b: BoundingBox) -> float:
    """Intersection-over-union of two bounding boxes."""
    inter_left = max(a.left, b.left)
    inter_top = max(a.top, b.top)
    inter_right = min(a.right, b.right)
    inter_bottom = min(a.bottom, b.bottom)

    inter_w = max(0, inter_right - inter_left)
    inter_h = max(0, inter_bottom - inter_top)
    inter_area = inter_w * inter_h

    union_area = a.area() + b.area() - inter_area
    if union_area <= 0:
        return 0.0
    return inter_area / union_area


@dataclass
class _Track:
    track_id: int
    bbox: BoundingBox
    missed_frames: int = 0


class FaceTracker:
    """Assigns stable `track_id`s to detected faces across frames via IoU matching."""

    def __init__(
        self,
        iou_threshold: float = settings.track_match_iou_threshold,
        max_missed_frames: int = settings.track_max_missed_frames,
    ) -> None:
        self._iou_threshold = iou_threshold
        self._max_missed_frames = max_missed_frames
        self._tracks: Dict[int, _Track] = {}
        self._id_counter = itertools.count(1)

    def update(self, detections: List[DetectedFace]) -> List[DetectedFace]:
        """Match new detections to existing tracks (or create new ones).

        Returns the same detections with `track_id` populated.
        """
        unmatched_detections = list(detections)
        matched_track_ids: set[int] = set()
        results: List[DetectedFace] = []

        for detection in list(unmatched_detections):
            best_track_id: Optional[int] = None
            best_iou = self._iou_threshold
            for track in self._tracks.values():
                if track.track_id in matched_track_ids:
                    continue
                score = _iou(track.bbox, detection.bbox)
                if score > best_iou:
                    best_iou = score
                    best_track_id = track.track_id

            if best_track_id is not None:
                self._tracks[best_track_id].bbox = detection.bbox
                self._tracks[best_track_id].missed_frames = 0
                matched_track_ids.add(best_track_id)
                results.append(detection.model_copy(update={"track_id": best_track_id}))
            else:
                new_id = next(self._id_counter)
                self._tracks[new_id] = _Track(track_id=new_id, bbox=detection.bbox)
                matched_track_ids.add(new_id)
                results.append(detection.model_copy(update={"track_id": new_id}))

        # Age out tracks that went unmatched this frame; drop stale ones.
        for track_id in list(self._tracks.keys()):
            if track_id not in matched_track_ids:
                self._tracks[track_id].missed_frames += 1
                if self._tracks[track_id].missed_frames > self._max_missed_frames:
                    del self._tracks[track_id]

        return results
