# =============================================================================
# inference_engine.py — OptoScan Inference Engine
# =============================================================================
# Wraps ML model inference (via classifier.py) with a threshold-based
# heuristic fallback.  The UI only ever calls InferenceEngine.run(value).
# =============================================================================

from __future__ import annotations
import logging

import classifier
from config import (
    HEMATOMA_UPPER, NORMAL_UPPER,
    LABEL_NORMAL, LABEL_HEMATOMA, LABEL_FIBROUS,
    DEFAULT_MODEL_PATH,
)

logger = logging.getLogger(__name__)

# ── Inference result type ─────────────────────────────────────────────────────
class InferenceResult:
    __slots__ = ("label", "confidence", "mode", "raw_value")

    def __init__(self, label: str, confidence: float, mode: str, raw_value: int):
        self.label      = label
        self.confidence = confidence
        self.mode       = mode        # "ML Model" or "Heuristic"
        self.raw_value  = raw_value

    def as_dict(self) -> dict:
        return {
            "label":      self.label,
            "confidence": self.confidence,
            "mode":       self.mode,
            "raw_value":  self.raw_value,
        }


class InferenceEngine:
    """
    Unified inference entry point.

    Priority
    --------
    1. ML model (if loaded via load_model())
    2. Threshold heuristic (always available)

    Parameters
    ----------
    model_path : str
        Path to model file.  Pass empty string to force heuristic mode.
    """

    def __init__(self, model_path: str = DEFAULT_MODEL_PATH):
        self._model      = None
        self._model_path = ""
        if model_path:
            self.load_model(model_path)

    # ── Public API ─────────────────────────────────────────────────────────

    def load_model(self, path: str) -> bool:
        """
        Load (or reload) a model file at runtime.

        Returns True on success, False on failure.
        Also called from the Streamlit sidebar when the user uploads a new file.
        """
        model = classifier.load_model(path)
        if model is not None:
            self._model      = model
            self._model_path = path
            logger.info("ML model loaded: %s", path)
            return True
        # Keep previous model (or None) on failure
        return False

    def unload_model(self) -> None:
        """Revert to heuristic-only mode."""
        self._model      = None
        self._model_path = ""
        logger.info("ML model unloaded — heuristic mode active.")

    @property
    def model_loaded(self) -> bool:
        return self._model is not None

    @property
    def model_path(self) -> str:
        return self._model_path

    def run(self, raw_value: int) -> InferenceResult:
        """
        Classify a single ADC sample.

        Parameters
        ----------
        raw_value : int   (0–1023)

        Returns
        -------
        InferenceResult
        """
        if self._model is not None:
            return self._ml_infer(raw_value)
        return self._heuristic_infer(raw_value)

    # ── Private Methods ────────────────────────────────────────────────────

    def _ml_infer(self, raw_value: int) -> InferenceResult:
        try:
            label, confidence = classifier.predict(self._model, raw_value)
            return InferenceResult(
                label=label,
                confidence=confidence,
                mode="ML Model",
                raw_value=raw_value,
            )
        except Exception as exc:
            logger.error("ML inference failed (%s) — falling back to heuristic.", exc)
            return self._heuristic_infer(raw_value)

    def _heuristic_infer(self, raw_value: int) -> InferenceResult:
        """
        Threshold-based classifier derived from IR reflectance physics:

          Low ADC  (0–340)    → high absorption  → Hematoma
          Mid ADC  (341–680)  → baseline scatter → Normal
          High ADC (681–1023) → high reflectance → Fibrous / Scar

        Confidence expressed as distance from zone boundary (0.5–1.0 range).
        """
        if raw_value <= HEMATOMA_UPPER:
            label = LABEL_HEMATOMA
            # 0 is furthest from boundary (most certain), HEMATOMA_UPPER is boundary
            zone_fraction = 1.0 - (raw_value / HEMATOMA_UPPER)
            confidence = 0.50 + 0.50 * zone_fraction

        elif raw_value <= NORMAL_UPPER:
            label = LABEL_NORMAL
            zone_size   = NORMAL_UPPER - HEMATOMA_UPPER          # 340
            zone_centre = HEMATOMA_UPPER + zone_size / 2.0       # 510
            distance_from_centre = abs(raw_value - zone_centre) / (zone_size / 2.0)
            confidence = 0.50 + 0.30 * (1.0 - distance_from_centre)

        else:  # raw_value > NORMAL_UPPER
            label = LABEL_FIBROUS
            zone_fraction = (raw_value - NORMAL_UPPER) / (1023 - NORMAL_UPPER)
            confidence = 0.50 + 0.50 * zone_fraction

        return InferenceResult(
            label=label,
            confidence=round(confidence, 3),
            mode="Heuristic",
            raw_value=raw_value,
        )
