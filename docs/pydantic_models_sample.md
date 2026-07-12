# Sample Pydantic Models (Deliverable #4)

Full source: `src/storage/models.py`. Every data structure that crosses a
functional boundary in this system (camera → detector → embedder → matcher →
liveness → logger) is one of these models, so malformed data is rejected at
construction time rather than propagating silently.

## Embeddings (never raw images)

```python
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
```

## Employee record (as persisted — embedding only, no photo)

```python
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
```

## Match result (matcher → liveness/logger boundary)

```python
class MatchResult(BaseModel):
    """Result of comparing a live embedding against enrolled employees."""

    employee_id: Optional[str] = None
    full_name: Optional[str] = None
    distance: float = Field(..., ge=0.0)
    confidence: float = Field(..., ge=0.0, le=1.0)
    is_match: bool = False
```

## Attendance log entry (the only thing ever written for a logged event)

```python
class AttendanceLogEntry(BaseModel):
    """A single attendance event as persisted to the database.

    Deliberately contains NO imagery -- only identity, time, event type
    and the confidence of the recognition that produced it.
    """

    employee_id: str = Field(..., min_length=1)
    event_type: Literal["clock_in", "clock_out"]
    timestamp: datetime = Field(default_factory=datetime.now)
    confidence: float = Field(..., ge=0.0, le=1.0)
```

## Bounding box (with cross-field validation)

```python
class BoundingBox(BaseModel):
    """Axis-aligned bounding box for a detected face, in pixel coordinates."""

    model_config = ConfigDict(frozen=True)

    top: int = Field(..., ge=0)
    right: int = Field(..., ge=0)
    bottom: int = Field(..., ge=0)
    left: int = Field(..., ge=0)

    @field_validator("bottom")
    @classmethod
    def _bottom_gt_top(cls, v: int, info) -> int:
        top = info.data.get("top")
        if top is not None and v <= top:
            raise ValueError("bottom must be greater than top")
        return v
```

These four (plus `FrameMetadata`, `DetectedFace`, `LivenessResult`, and
`EnrollmentRequest` in the full file) cover every boundary crossing in the
pipeline described in the assignment brief.
