"""
embeddings/extractor.py

ArcFace embedding extraction only. Deliberately loads just the recognition
model (w600k_r50.onnx from the buffalo_l pack) directly via
insightface.model_zoo, rather than instantiating a full FaceAnalysis app.
FaceAnalysis hard-requires a detection model to be present internally
(it asserts on it), which would mean loading a second, unused detection
model into memory here. Since detection/detector.py already produced
5-point landmarks for each face, this module only needs: aligned crop ->
512-d embedding vector.
"""

from __future__ import annotations

import logging
import os
from types import SimpleNamespace
from typing import Optional

import numpy as np
from insightface.model_zoo import model_zoo

from config import settings
from models.schemas import DetectedFace, FaceEmbedding

logger = logging.getLogger(__name__)

# Filename shipped inside the buffalo_l pack. Hardcoded rather than
# discovered dynamically -- this is a known, stable filename for this pack
# version; if you switch model packs, update this constant.
RECOGNITION_MODEL_FILENAME = "w600k_r50.onnx"


class EmbeddingError(RuntimeError):
    """Raised for unrecoverable embedding-extraction failures."""


class EmbeddingExtractor:
    def __init__(self, model_pack: str = "buffalo_l", ctx_id: int = -1) -> None:
        # We deliberately build the path by hand rather than importing
        # insightface's internal ensure_available()/download helpers -- those
        # are private APIs that have moved between insightface versions.
        # This assumes FaceDetector has already run at least once (it
        # triggers the pack download to the default ~/.insightface root).
        model_dir = os.path.expanduser(os.path.join("~/.insightface/models", model_pack))
        model_path = os.path.join(model_dir, RECOGNITION_MODEL_FILENAME)

        if not os.path.isfile(model_path):
            raise EmbeddingError(
                f"Recognition model not found at {model_path}. "
                "Run the detector at least once first (it downloads the "
                f"'{model_pack}' pack), or check your internet connection."
            )

        try:
            self._model = model_zoo.get_model(model_path)
            if self._model is None:
                raise EmbeddingError(f"insightface could not load recognition model at {model_path}")
            self._model.prepare(ctx_id=ctx_id)
        except EmbeddingError:
            raise
        except Exception as exc:
            raise EmbeddingError("Failed to initialize the ArcFace recognition model.") from exc

        logger.info("EmbeddingExtractor ready (%s, ctx_id=%d).", RECOGNITION_MODEL_FILENAME, ctx_id)

    def extract(self, frame: np.ndarray, detected: DetectedFace) -> Optional[FaceEmbedding]:
        """
        Produces a validated FaceEmbedding for one detected face, using its
        5-point landmarks for alignment (unaligned crops measurably degrade
        ArcFace match quality). Returns None -- not an exception -- if this
        particular face lacks usable landmarks; that's an expected,
        recoverable case (e.g. an extreme-profile face the detector still
        flagged), not a bug to crash over.
        """
        if detected.landmarks_5pt is None or len(detected.landmarks_5pt) != 5:
            logger.debug(
                "Frame %d: face has no usable 5-point landmarks; skipping embedding.",
                detected.frame_id,
            )
            return None

        try:
            kps = np.array(detected.landmarks_5pt, dtype=np.float32)
            # ArcFaceONNX.get() only reads `.kps` off the object it's given,
            # so a minimal duck-typed namespace is sufficient here -- we
            # don't need insightface's full Face class.
            face_obj = SimpleNamespace(kps=kps)
            raw_vector = self._model.get(frame, face_obj)
        except Exception:
            logger.exception(
                "Embedding extraction failed for frame %d; skipping face.", detected.frame_id
            )
            return None

        vector = raw_vector.flatten().astype(float).tolist()

        try:
            return FaceEmbedding(
                frame_id=detected.frame_id,
                vector=vector,
                model_name=settings.EMBEDDING_MODEL_NAME,
                dim=len(vector),
            )
        except Exception:
            logger.exception(
                "Extracted embedding failed schema validation for frame %d.", detected.frame_id
            )
            return None
