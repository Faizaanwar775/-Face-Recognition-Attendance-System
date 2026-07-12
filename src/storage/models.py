"""
Pydantic data models.

Every piece of data that crosses a functional boundary in this system
(camera -> detector -> embedder -> matcher -> liveness -> logger) is
represented by one of these models. This guarantees that malformed or
corrupted payloads are rejected immediately (at construction time)
rather than silently propagating deeper into the pipeline.

No model in this file ever carries raw image bytes for storage purposes.
`FrameMetadata` intentionally excludes the pixel buffer itself (the numpy
array is passed alongside, in-memory only, and is never persisted).
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

EMBEDDING_DIM_DEFAULT = 128


class EventType(str, Enum):
    """Type of attendance event."""

    CLOCK_IN = "clock_in"
    CLOCK_OUT = "clock_out"


class BoundingBox(BaseModel):
    """Axis-aligned bounding box for a detected face, in pixel coordinates."""

    model_config = ConfigDict(frozen=True)

    top: int = Field(..., ge=0)
    right: int = Field(..., ge=0)
    bottom: int = Field(..., ge=0)
    left: int = Field(..., ge=0)

    @model_validator(mode="after")
    def _validate_box_geometry(self) -> "BoundingBox":
        # A model-level (post-construction) validator is used here rather
        # than order-dependent field_validators, since field_validators
        # only see fields declared *before* the one being validated via
        # `info.data` -- checking "right > left" from within a validator
        # on "right" would silently never fire, because "left" is
        # declared after "right" and wouldn't be in info.data yet.
        if self.bottom <= self.top:
            raise ValueError("bottom must be greater than top")
        if self.right <= self.left:
            raise ValueError("right must be greater than left")
        return self

    def as_tuple(self) -> tuple[int, int, int, int]:
        """Return as the (top, right, bottom, left) tuple face_recognition expects."""
        return (self.top, self.right, self.bottom, self.left)

    def area(self) -> int:
        return max(0, self.right - self.left) * max(0, self.bottom - self.top)

    def center(self) -> tuple[float, float]:
        return ((self.left + self.right) / 2.0, (self.top + self.bottom) / 2.0)


class FrameMetadata(BaseModel):
    """Metadata describing a single captured video frame.

    NOTE: The actual pixel buffer (numpy.ndarray) is passed separately,
    in-memory, between threads/processes. It is deliberately NOT a field
    here because Pydantic models in this system represent data that is
    validated, logged, and potentially persisted -- and raw pixels must
    never be persisted.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    frame_id: int = Field(..., ge=0)
    timestamp: datetime
    width: int = Field(..., gt=0)
    height: int = Field(..., gt=0)
    source: str = Field(default="webcam", min_length=1)


class DetectedFace(BaseModel):
    """A single detected face within a frame, before embedding extraction."""

    frame_id: int = Field(..., ge=0)
    track_id: Optional[int] = Field(
        default=None, description="Stable identity assigned by the cross-frame tracker"
    )
    bbox: BoundingBox


class FaceEmbedding(BaseModel):
    """A numeric embedding vector for a detected face. Never a raw image."""

    vector: list[float]
    model_name: str = Field(default="dlib_resnet_v1")

    @field_validator("vector")
    @classmethod
    def _validate_length(cls, v: list[float]) -> list[float]:
        if len(v) == 0:
            raise ValueError("embedding vector must not be empty")
        if len(v) > 1024:
            raise ValueError("embedding vector suspiciously large; refusing to store")
        return v


class EmployeeRecord(BaseModel):
    """A persisted employee identity. Stores an embedding, never a photo."""

    employee_id: str = Field(..., min_length=1, max_length=64)
    full_name: str = Field(..., min_length=1, max_length=128)
    embedding: list[float] = Field(..., min_length=1)
    enrolled_at: datetime = Field(default_factory=datetime.now)

    @field_validator("employee_id")
    @classmethod
    def _no_whitespace(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("employee_id cannot be blank")
        return stripped


class MatchResult(BaseModel):
    """Result of comparing a live embedding against enrolled employees."""

    employee_id: Optional[str] = None
    full_name: Optional[str] = None
    distance: float = Field(..., ge=0.0)
    confidence: float = Field(..., ge=0.0, le=1.0)
    is_match: bool = False


class LivenessResult(BaseModel):
    """Result of the multi-frame liveness check for a tracked face."""

    track_id: int
    is_live: bool
    blink_count: int = Field(..., ge=0)
    reason: str = Field(..., min_length=1)


class AttendanceLogEntry(BaseModel):
    """A single attendance event as persisted to the database.

    Deliberately contains NO imagery -- only identity, time, event type
    and the confidence of the recognition that produced it.
    """

    employee_id: str = Field(..., min_length=1)
    event_type: Literal["clock_in", "clock_out"]
    timestamp: datetime = Field(default_factory=datetime.now)
    confidence: float = Field(..., ge=0.0, le=1.0)


class EnrollmentRequest(BaseModel):
    """Input payload for enrolling a new employee."""

    employee_id: str = Field(..., min_length=1, max_length=64)
    full_name: str = Field(..., min_length=1, max_length=128)
