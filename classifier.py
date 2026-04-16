# =============================================================================
# classifier.py — OptoScan ML Model Interface
# =============================================================================
# ┌──────────────────────────────────────────────────────────────────────────┐
# │  KAGGLE MODEL IMPORT ZONE                                                │
# │  Drop your trained model file into the project folder, then configure    │
# │  the loader below to match your model format.                            │
# │                                                                          │
# │  Supported formats (one at a time):                                      │
# │    • scikit-learn  → .pkl  (recommended)                                 │
# │    • Keras / TF    → .h5 / SavedModel directory                          │
# │    • ONNX          → .onnx                                               │
# └──────────────────────────────────────────────────────────────────────────┘

from __future__ import annotations
import logging
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

# ── Label Mapping ─────────────────────────────────────────────────────────────
# Must match the integer targets your model was trained on.
# Adjust indices if your training used a different encoding.
LABEL_MAP: dict[int, str] = {
    0: "Hematoma",
    1: "Normal",
    2: "Fibrous / Scar",
}


def load_model(model_path: str):
    """
    Load a trained model from disk.

    Parameters
    ----------
    model_path : str
        Absolute or relative path to the model file.

    Returns
    -------
    model object or None
        Returns None if the file does not exist or loading fails.

    Notes
    -----
    Uncomment the block that matches your model format.
    """
    path = Path(model_path)
    if not path.exists():
        logger.warning("Model file not found: %s — falling back to heuristic mode.", path)
        return None

    suffix = path.suffix.lower()

    # ── scikit-learn .pkl ────────────────────────────────────────────────────
    if suffix in (".pkl", ".pickle", ".joblib"):
        try:
            # Try joblib first (more space-efficient for large sklearn models)
            from joblib import load as jload
            model = jload(path)
            logger.info("Loaded scikit-learn model from %s", path)
            return model
        except ImportError:
            pass
        try:
            import pickle
            with open(path, "rb") as f:
                model = pickle.load(f)
            logger.info("Loaded pickle model from %s", path)
            return model
        except Exception as e:
            logger.error("Failed to load .pkl model: %s", e)
            return None

    # ── Keras / TensorFlow .h5 ──────────────────────────────────────────────
    elif suffix == ".h5":
        try:
            import tensorflow as tf          # noqa: F401
            from tensorflow import keras
            model = keras.models.load_model(path)
            logger.info("Loaded Keras model from %s", path)
            return model
        except ImportError:
            logger.error("TensorFlow not installed. Install with: pip install tensorflow")
            return None
        except Exception as e:
            logger.error("Failed to load .h5 model: %s", e)
            return None

    # ── ONNX Runtime ────────────────────────────────────────────────────────
    elif suffix == ".onnx":
        try:
            import onnxruntime as ort
            session = ort.InferenceSession(str(path))
            logger.info("Loaded ONNX model from %s", path)
            return session          # ort.InferenceSession is the "model" object
        except ImportError:
            logger.error("onnxruntime not installed. Install with: pip install onnxruntime")
            return None
        except Exception as e:
            logger.error("Failed to load ONNX model: %s", e)
            return None

    else:
        logger.error("Unsupported model format: %s", suffix)
        return None


def predict(model, raw_value: int) -> tuple[str, float]:
    """
    Run inference on a single raw ADC value.

    For single-feature models (1 input):
        Feature = [raw_value / 1023.0]

    For spectral multi-feature models (N wavelength inputs, trained by
    train_model.py on NIR data):
        Feature = raw reflectance replicated across all N wavelength slots.
        The model was trained on real spectral curves; this maps the single
        ADC reading to a flat spectral approximation for live inference.
    """
    # ── Build feature vector ─────────────────────────────────────────────────
    n_features = getattr(model, "n_features_in_", 1)
    reflectance = raw_value / 1023.0

    if n_features == 1:
        feature = np.array([[reflectance]], dtype=np.float32)
    else:
        # Spectral model: broadcast single reflectance to all wavelength slots
        feature = np.full((1, n_features), reflectance, dtype=np.float32)

    # ── scikit-learn ─────────────────────────────────────────────────────────
    if hasattr(model, "predict_proba"):
        proba      = model.predict_proba(feature)[0]   # shape (n_classes,)
        class_idx  = int(np.argmax(proba))
        confidence = float(proba[class_idx])

        # Priority: custom attr (train_model.py) → sklearn classes_ → static map
        if hasattr(model, "class_names_"):
            label = str(model.class_names_[class_idx])
        elif hasattr(model, "classes_"):
            label = str(model.classes_[class_idx])
        else:
            label = LABEL_MAP.get(class_idx, "Unknown")
        return label, confidence

    if hasattr(model, "predict") and not hasattr(model, "run"):
        # sklearn model without predict_proba (e.g. SVC without probability=True)
        pred = model.predict(feature)[0]
        if hasattr(model, "class_names_"):
            label = str(model.class_names_[int(pred)])
        elif hasattr(model, "classes_"):
            label = str(model.classes_[int(pred)])
        else:
            label = LABEL_MAP.get(int(pred), "Unknown")
        return label, 1.0

    # ── Keras / TensorFlow ───────────────────────────────────────────────────
    if hasattr(model, "predict") and hasattr(model, "input_shape"):
        proba = model.predict(feature, verbose=0)[0]    # shape (n_classes,)
        class_idx = int(np.argmax(proba))
        confidence = float(proba[class_idx])
        label = LABEL_MAP.get(class_idx, "Unknown")
        return label, confidence

    # ── ONNX Runtime ─────────────────────────────────────────────────────────
    if hasattr(model, "run"):
        input_name = model.get_inputs()[0].name
        outputs = model.run(None, {input_name: feature})
        proba = outputs[0][0]                           # shape (n_classes,)
        class_idx = int(np.argmax(proba))
        confidence = float(proba[class_idx])
        label = LABEL_MAP.get(class_idx, "Unknown")
        return label, confidence

    logger.error("Unrecognised model type: %s", type(model))
    return "Unknown", 0.0
