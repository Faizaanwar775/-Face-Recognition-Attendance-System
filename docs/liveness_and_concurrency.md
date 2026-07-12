# Written Explanation: Liveness Detection & Concurrency Architecture

## Part A — Liveness detection

### Chosen approach

The primary liveness mechanism is **Eye Aspect Ratio (EAR) blink detection**,
implemented in `src/liveness/eye_aspect_ratio.py` and
`src/liveness/blink_tracker.py`.

For each detected face, `face_recognition.face_landmarks(...)` returns 6
ordered points around each eye. EAR is computed as:

```
EAR = (‖p2-p6‖ + ‖p3-p5‖) / (2 * ‖p1-p4‖)
```

This value stays roughly constant while an eye is open and drops sharply
during a blink, then recovers. `BlinkLivenessTracker` maintains a small
per-face state machine across frames:

1. EAR falls below `ear_threshold` (default 0.21) for at least
   `ear_consec_frames` (default 2) consecutive frames → eye considered
   "closed".
2. EAR subsequently rises back above the threshold → one full blink cycle
   is counted.
3. Once `required_blinks` (default 1) full cycles occur within a rolling
   `liveness_window_seconds` (default 8s) window, the tracked face is
   confirmed **live**.

This is deliberately a genuine **multi-frame** technique — the required
open→closed→open transition cannot occur in a single static image, because a
static photo's EAR value never changes at all across frames.

### Secondary signal (defense in depth)

`src/liveness/motion_liveness.py` adds a fast-rejecting secondary check,
used to flag "obviously static" spoofs before the full blink window would
otherwise elapse:

- **Micro-movement**: the variance of the face's bounding-box centroid over
  the last ~20 frames. A real person held (or standing) in front of a
  camera always shows small natural jitter; a photo taped to a wall or a
  screen on a stand tends to show suspiciously close-to-zero variance.
- **Texture/frequency analysis**: the variance of the Laplacian of the
  cropped face region (a standard blur/sharpness metric). Printed photos
  and screens tend to show flatter local texture (paper grain / print dot
  pattern, or a screen's pixel grid / moiré) than real skin at typical
  webcam resolution.

Both signals are combined only as a fast-path "suspected static spoof" flag
(`motion_info["suspected_static_spoof"]`) — the actual accept decision for
logging attendance always still requires the primary blink signal to
confirm `is_live = True`.

### Tested against spoofing (as required by the assignment)

Manual testing (holding a printed photo, and a phone displaying a static
photo, up to the camera) confirmed:

- EAR never showed a full closed→open cycle → `is_live` stayed `False`
  indefinitely for both spoof types.
- Centroid variance for a rigidly-held photo/phone was near zero, and
  texture variance was noticeably lower than a live face at the same
  distance/lighting, so `suspected_static_spoof` correctly triggered in
  most tested conditions, painting the box red rather than progressing
  to matching/logging.

### Known limitations (documented honestly)

- **Video replay attack**: a high-quality recorded video of a real person
  blinking, played back on a phone/tablet/monitor, produces an EAR signal
  indistinguishable from a genuine live blink. This is the hardest failure
  mode for any blink-only liveness system and is **not fully solved** here.
  A production system would add screen/moiré-specific CNN classifiers or
  challenge-response (e.g. "turn your head left") to further mitigate this.
- **Eyewear / lighting**: heavy glasses frames, strong reflections, or low
  light can produce noisy EAR values, slowing down (but not usually
  preventing) confirmation.
- **Fast blinks**: extremely quick blinks between two processed frames can
  be missed (temporal aliasing); the 8-second rolling window mitigates
  this by giving the user multiple chances rather than requiring a single
  perfectly-timed detection.
- **Texture heuristic sensitivity**: a very high resolution printed photo
  or an excellent display panel can produce texture variance close to
  real skin, especially at low camera resolution — this signal is a
  soft, complementary check, not a hard security boundary on its own.

---

## Part B — Concurrency architecture

### Design

Three cooperating mechanisms keep the video feed smooth while the CPU-heavy
face-embedding math runs off the critical path:

1. **Threading producer/consumer (capture ↔ processing)**
   `src/capture/camera_stream.py` runs a dedicated capture thread that
   continuously reads frames from `cv2.VideoCapture` and publishes them
   through two channels:
   - A **single-slot queue** (`maxsize=1`) for the recognition/processing
     thread. Before pushing a new frame, the capture thread always drops
     any previously unread frame from this queue. This is the mechanism
     that prevents backlog build-up: the processing thread only ever
     receives the *latest* frame, never a growing pile of stale ones.
   - A **peekable "latest frame" reference**, read (never consumed) by the
     main display thread, so on-screen video keeps refreshing at its own
     pace regardless of how far behind recognition currently is.

2. **`ProcessPoolExecutor` for CPU-bound embedding extraction**
   `src/embeddings/embedding_extractor.py` exposes `extract_embedding` as a
   free, picklable function (a hard requirement for `ProcessPoolExecutor`,
   since arguments and return values must cross a process boundary via
   pickling). The processing thread (`src/core/pipeline.py`) submits this
   function to a small worker-process pool
   (`KHIZEX_EMBEDDING_WORKERS`, default 2) rather than calling it inline.
   Running this in separate **processes** (not just threads) sidesteps the
   GIL entirely, so a slow embedding computation cannot stall Python
   bytecode execution on the capture or display threads even under load.

3. **Bounded per-track in-flight work**
   The processing thread tracks at most **one pending embedding job per
   tracked face** (`self._pending_futures`, keyed by `track_id` from
   `src/core/tracker.py`'s IoU-based tracker). While a job is in flight for
   a given face, new detections of that same face are simply displayed as
   "Analyzing..." rather than triggering additional submissions — this
   caps outstanding work at O(number of visible faces), not O(frames
   processed), which is the second half of backlog prevention.

### Why this satisfies "non-blocking"

- The capture thread's frame rate is unaffected by recognition speed: it
  drops stale frames rather than waiting for consumption.
- The display loop reads a lock-protected snapshot
  (`_snapshot_display_state()`) and the peekable latest camera frame; it
  never blocks on the processing thread or the process pool.
- Even if a single embedding extraction takes, say, 200ms, only that one
  face's box shows "Analyzing..." a little longer — the video itself does
  not stutter, and other tracked faces continue to be processed
  independently once their own jobs complete.

### Trade-offs acknowledged

- Overlay labels can lag the true recognition state by up to one
  processing cycle — acceptable for an attendance kiosk, since the
  authoritative event (the logged attendance row) is only written once
  matching **and** liveness are both confirmed, regardless of any
  momentary display lag.
- The simple IoU tracker (not a full multi-object tracker like DeepSORT)
  can lose an identity across long occlusions or very fast crowds — an
  acceptable trade-off for a single-camera doorway/desk use case, and
  documented in `src/core/tracker.py`.
