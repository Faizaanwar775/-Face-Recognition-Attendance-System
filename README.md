# Khizex | Face Recognition Attendance System

A biometric attendance system that watches a doorway/desk via webcam, recognizes
enrolled employees by face, confirms they are a genuine live person (not a photo
or a screen replay), and logs a secure, timestamped clock-in/clock-out event —
**without ever storing a raw face image**. Only numeric embeddings ever touch disk.

Built for the Khizex Python Engineering Internship — 1 Week Build Challenge.

---

## 1. Features

- **Full recognition pipeline**: capture → detect → embed → match → log.
- **Employee enrollment** via webcam, storing only an averaged 128-d embedding.
- **Multi-frame liveness detection**: EAR-based blink detection (primary) plus
  micro-movement / texture analysis (secondary, fast-rejecting) — see
  [`docs/liveness_and_concurrency.md`](docs/liveness_and_concurrency.md).
- **Non-blocking concurrency**: threaded producer/consumer capture pipeline with
  the CPU-heavy embedding step offloaded to a `ProcessPoolExecutor`, so the video
  feed never stutters.
- **Pydantic models on every boundary** — malformed data is rejected at
  construction time, not silently propagated.
- **Zero raw image storage** — the database only ever contains employee IDs,
  names, embedding vectors (JSON floats), and attendance log rows.
- **Modular, typed, defensively-coded** — see folder structure below.

---

## 2. Folder structure

```
khizex_face_attendance/
├── main.py                     # CLI entry point (enroll / run / list-employees / logs)
├── requirements.txt
├── pytest.ini
├── .env.example
├── data/                       # created at runtime: attendance.db, logs/
├── docs/
│   ├── liveness_and_concurrency.md   # Deliverable #2: written explanation
│   └── pydantic_models_sample.md     # Deliverable #4: sample Pydantic models
├── tests/                      # pytest unit tests (tracker, matcher, blink logic)
└── src/
    ├── config.py                # all tunables, env-var overridable
    ├── capture/
    │   └── camera_stream.py     # threaded producer, backlog-free frame queue
    ├── detection/
    │   └── face_detector.py     # face_recognition (dlib HOG/CNN) wrapper
    ├── embeddings/
    │   └── embedding_extractor.py  # picklable function for ProcessPoolExecutor
    ├── matching/
    │   └── face_matcher.py      # distance-threshold matching vs. enrolled roster
    ├── liveness/
    │   ├── eye_aspect_ratio.py  # EAR landmark math
    │   ├── blink_tracker.py     # multi-frame blink state machine (primary signal)
    │   └── motion_liveness.py   # micro-movement + texture heuristic (secondary)
    ├── storage/
    │   ├── models.py            # every Pydantic boundary contract
    │   └── database.py          # SQLite: employees + attendance_logs (no images)
    ├── logging_events/          # (named to avoid shadowing stdlib `logging`)
    │   ├── logger.py            # app-wide console + rotating file logging
    │   └── attendance_logger.py # clock-in/out decision + cooldown + persistence
    ├── core/
    │   ├── tracker.py           # IoU-based cross-frame face tracker
    │   ├── enrollment.py        # interactive enrollment flow
    │   └── pipeline.py          # orchestrator wiring everything together
    └── utils/
        └── drawing.py           # on-screen overlay helpers (debug preview only)
```

---

## 3. Installation

### 3.1 Prerequisites

- Python 3.10+
- A working webcam
- `face_recognition` depends on **dlib**, which needs **CMake** and a **C++
  compiler** to build from source on most platforms:
  - **Ubuntu/Debian**: `sudo apt-get install -y cmake build-essential libopenblas-dev`
  - **macOS**: `brew install cmake`, plus Xcode command line tools
    (`xcode-select --install`)
  - **Windows**: install Visual Studio Build Tools (C++ workload) and CMake,
    or use a prebuilt `dlib` wheel for your Python version.

### 3.2 Setup

```bash
# 1. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) run the unit tests
pytest
```

---

## 4. Usage

### 4.1 Enroll a new employee

```bash
python main.py enroll --id E001 --name "Jane Doe"
```

A preview window opens. Look directly at the camera; the system captures
`KHIZEX_ENROLLMENT_FRAMES` (default 15) good single-face frames, extracts an
embedding from each, and stores the **average** embedding vector for that
employee — never a photo. Press `q` to abort early.

Run headless (e.g. over SSH with no display) with `--no-preview`.

### 4.2 Run the live attendance loop

```bash
python main.py run
```

Opens a live preview window with bounding boxes:

| Color  | Meaning                                              |
|--------|-------------------------------------------------------|
| Red    | Unrecognized face, or suspected static-photo/screen spoof |
| Amber  | Recognized (or being analyzed) — awaiting liveness confirmation |
| Green  | Matched **and** live — attendance event logged        |

Press `q` to quit.

### 4.3 Inspect data

```bash
python main.py list-employees
python main.py logs --id E001 --limit 20
```

---

## 5. Liveness detection — approach & limitations

**Primary signal**: Eye Aspect Ratio (EAR) blink detection across a sliding
window of frames (`liveness/blink_tracker.py`). A full open→closed→open cycle
within `KHIZEX_LIVENESS_WINDOW_SECONDS` (default 8s) confirms liveness.

**Secondary signal**: micro-movement (bounding-box centroid variance) and
texture analysis (Laplacian variance of the face crop) for fast rejection of
an obviously static printed photo or monitor/phone screen
(`liveness/motion_liveness.py`).

**Tested spoof rejection**: a printed photo or phone/monitor held up to the
camera produces a constant EAR (no blink ever registers) and, typically, low
centroid variance and flatter texture — both signals agree it's not live, and
the box is drawn in red as "SPOOF SUSPECTED" / left as "Unknown" rather than
ever reaching a logged attendance event.

**Known limitations (honest, not claiming perfection)**:
- A high-quality **video replay** of a real person blinking can fool the
  blink signal, since its EAR time series looks like a genuine blink.
- Heavy eyeglass frames or strong glare can produce noisy EAR values and
  slow down confirmation.
- Very fast blinks may be missed if they occur between processed frames.
- The texture/motion heuristic is a soft signal tuned to avoid false
  rejection of legitimate, mostly-still users; it will not catch a very
  high quality printed photo waved gently in front of the camera.

Full write-up: [`docs/liveness_and_concurrency.md`](docs/liveness_and_concurrency.md).

---

## 6. Concurrency architecture (backlog handling)

```
Capture thread (producer) --peek--> Display loop (main thread)
        |
        | single-slot queue (maxsize=1, drops stale frames)
        v
Processing thread (consumer): detect -> track -> liveness (EAR/motion)
        |
        | submit CPU-bound work (at most 1 pending job per tracked face)
        v
ProcessPoolExecutor: embedding extraction (dlib ResNet forward pass)
        |
        v
Processing thread polls completed futures -> match -> log -> update overlay
```

- The capture thread never blocks on recognition. Its hand-off queue has
  `maxsize=1` and **always drops the previous unread frame** before pushing a
  new one — the consumer only ever sees the *latest* frame, never a growing
  backlog.
- The heaviest CPU-bound step (embedding extraction) runs in worker
  **processes** via `concurrent.futures.ProcessPoolExecutor`, which sidesteps
  the GIL entirely so it cannot stall the capture or display threads.
- Each tracked face has **at most one in-flight embedding job** at a time —
  new detections for a track that's already "analyzing" are simply skipped
  rather than queued, so a slow model never causes unbounded pending work.
- The display loop reads a lock-protected snapshot of the latest frame and
  overlay state; it never waits on the processing thread, so the on-screen
  video stays smooth even if recognition briefly lags behind.

Full write-up: [`docs/liveness_and_concurrency.md`](docs/liveness_and_concurrency.md).

---

## 7. Security of biometric data

- `storage/models.py`'s `FrameMetadata` deliberately has **no pixel-buffer
  field** — raw frames only ever exist as in-memory numpy arrays, passed
  between threads/processes, never serialized to the Pydantic layer that
  feeds persistence.
- `storage/database.py` has exactly two tables — `employees` (embedding
  stored as a JSON list of floats) and `attendance_logs` (IDs, timestamps,
  event type, confidence) — **no BLOB/image columns exist anywhere**.
- `core/enrollment.py` never calls `cv2.imwrite` or otherwise persists a
  frame; it only calls the embedding extractor and stores the resulting
  vector.

---

## 8. Type safety & validation

- Every function signature uses type hints; `mypy` is included in
  `requirements.txt` and the codebase is written to be `mypy`-clean.
- Every cross-boundary data structure (camera frame metadata, detected
  faces, embeddings, employee records, match results, liveness results,
  attendance log entries, enrollment requests) is a Pydantic model with
  field validators — see [`docs/pydantic_models_sample.md`](docs/pydantic_models_sample.md).
- `Any` is not used anywhere in the core pipeline; the few places that
  accept loosely-typed external data (e.g. SQLite rows) immediately
  reconstruct a validated Pydantic model.

---

## 9. Error handling

- **Camera disconnect**: `CameraStream._run` detects failed reads, retries
  with backoff, and attempts to reopen the device — the app keeps running
  and simply shows a "CAMERA ISSUE" HUD indicator rather than crashing.
- **Corrupted frame**: `FaceDetector.detect` returns an empty list on any
  detection exception rather than propagating it.
- **DB write failure**: `AttendanceDatabase` wraps every write in a
  transaction with rollback-on-error, and `AttendanceLogger` catches and
  logs any persistence exception rather than crashing the recognition loop.
- **Worker process failure**: embedding extraction failures are caught
  inside the picklable `extract_embedding` function itself, returning
  `None` rather than raising across the process boundary.

---

## 10. Running the tests

```bash
pytest
```

Covers: the IoU-based tracker's identity assignment/expiry, the face
matcher's distance-threshold logic, and the blink tracker's state machine
(including that a static, never-blinking signal correctly never confirms
liveness).

---

## 11. Deliverables checklist (per assignment spec)

- [x] Source code repository, modular structure, README (this file).
- [x] Written explanation of liveness approach + concurrency architecture:
      [`docs/liveness_and_concurrency.md`](docs/liveness_and_concurrency.md).
- [ ] Short demo recording (record locally after running `python main.py run`
      showing a successful recognition + a rejected spoof attempt — this is a
      live capture step only you can perform on your own hardware).
- [x] Sample Pydantic models: [`docs/pydantic_models_sample.md`](docs/pydantic_models_sample.md)
      (full definitions in `src/storage/models.py`).

---

## 12. Configuration reference

All tunables live in `src/config.py` and can be overridden via environment
variables — see `.env.example` for the full list (match threshold, EAR
threshold, camera index, queue sizes, cooldown period, etc.).
