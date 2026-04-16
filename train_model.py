# =============================================================================
# train_model.py — OptoScan Spectral Classification Training Pipeline
# =============================================================================
# Handles NIR spectral reflectance datasets (999–2500 nm, ~1557 wavelength
# features). Each CSV file in the project folder represents one tissue class —
# the filename (without extension) is used as the class label.
#
# ── Data format understood ────────────────────────────────────────────────────
# Files like:
#   BLC spectra data.csv       → class "BLC"
#   MSC Spectra data.csv       → class "MSC"
#   MSC BLC Spectra data.csv   → class "MSC_BLC"
#   raw 2.csv                  → class "raw_2"
#
# Columns are wavelength values (numeric), optionally prefixed by a
# row-index column ("No." / "No" / integer index).  Any non-numeric columns
# other than known index/label headers are dropped automatically.
#
# ── Alternatively: labelled CSV ───────────────────────────────────────────────
# If a CSV has an explicit label/variety column, that column is used instead
# of the filename.
# =============================================================================

from __future__ import annotations
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)

# Known index-only column names to drop before using features
INDEX_COLS = {"no.", "no", "index", "unnamed: 0", "#", "id", "sample"}

# Column names that carry an explicit label (checked lower-case)
LABEL_COL_CANDIDATES = [
    "label", "diagnosis", "class", "tissue", "category",
    "target", "type", "condition", "variety", "output", "result",
]


def scan_datasets(folder: str) -> list[str]:
    """Return paths of all CSV files found inside *folder*."""
    return sorted(str(p) for p in Path(folder).glob("*.csv"))


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _strip_index_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Drop known row-index columns that carry no spectral information."""
    drop = [c for c in df.columns if c.lower().strip() in INDEX_COLS]
    return df.drop(columns=drop)


def _find_label_col(df: pd.DataFrame) -> str | None:
    """Return name of an explicit label column if one exists."""
    cols_lower = {c.lower().strip(): c for c in df.columns}
    for cand in LABEL_COL_CANDIDATES:
        if cand in cols_lower:
            return cols_lower[cand]
    return None


def _load_one_csv(csv_path: str, default_label: str) -> pd.DataFrame:
    """
    Load one CSV and return a DataFrame with columns [*features*, '__label__'].

    If the CSV has an explicit label column it is used; otherwise every row
    gets *default_label* (derived from the filename).
    Rows with NaN values are dropped.
    """
    df = pd.read_csv(csv_path)
    df = _strip_index_cols(df)

    label_col = _find_label_col(df)
    if label_col:
        labels = df[label_col].astype(str).str.strip()
        df = df.drop(columns=[label_col])
    else:
        labels = pd.Series([default_label] * len(df), index=df.index)

    # Keep only numeric feature columns
    df = df.select_dtypes(include=[np.number])
    df["__label__"] = labels
    df = df.dropna()
    return df


def _filename_to_label(path: str) -> str:
    """Convert filename to a clean label string."""
    stem = Path(path).stem
    return stem.replace(" ", "_")


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def load_and_merge(csv_paths: list[str]) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """
    Load and merge multiple spectral CSVs into one feature matrix.

    Each file's label either comes from an explicit label column or from the
    filename stem.  All files must share the same wavelength columns (they are
    aligned by column name intersection).

    Returns
    -------
    X            : np.ndarray (n_samples, n_features)
    y            : np.ndarray (n_samples,) — integer-encoded labels
    class_names  : list[str]  — human-readable class names
    feature_cols : list[str]  — wavelength column names used as features
    """
    frames: list[pd.DataFrame] = []
    for path in csv_paths:
        label = _filename_to_label(path)
        df = _load_one_csv(path, label)
        if len(df) == 0:
            logger.warning("Skipping empty file: %s", path)
            continue
        frames.append(df)
        logger.info("Loaded %s → %d rows, label='%s'", Path(path).name, len(df), label)

    if not frames:
        raise ValueError("No usable data found in any CSV file.")

    # Align on common feature columns (intersection of wavelengths)
    feature_sets = [set(f.columns) - {"__label__"} for f in frames]
    common_cols  = sorted(feature_sets[0].intersection(*feature_sets[1:]),
                          key=lambda c: float(c) if _is_numeric_str(c) else 0)

    if not common_cols:
        raise ValueError(
            "CSV files have no wavelength columns in common. "
            "Ensure all files cover the same spectral range."
        )

    merged = pd.concat(
        [f[common_cols + ["__label__"]] for f in frames],
        ignore_index=True,
    ).dropna()

    # Drop classes with only 1 member (cannot stratify or classify reliably)
    class_counts = merged["__label__"].value_counts()
    valid_classes = class_counts[class_counts >= 2].index
    dropped = set(merged["__label__"].unique()) - set(valid_classes)
    if dropped:
        logger.warning("Dropping classes with < 2 samples: %s", dropped)
        merged = merged[merged["__label__"].isin(valid_classes)]

    if len(merged) == 0:
        raise ValueError("No classes with ≥ 2 samples found across all CSV files.")

    X = merged[common_cols].values.astype(np.float32)

    le = LabelEncoder()
    y  = le.fit_transform(merged["__label__"].values)
    class_names = list(le.classes_)

    logger.info(
        "Merged dataset: %d samples × %d features | classes=%s",
        len(X), X.shape[1], class_names,
    )
    return X, y, class_names, common_cols


def _is_numeric_str(s: str) -> bool:
    try:
        float(s)
        return True
    except (ValueError, TypeError):
        return False


def train_and_save_merged(csv_paths: list[str], output_path: str) -> dict:
    """
    Train a RandomForestClassifier on all CSVs merged and save to *output_path*.

    Returns a result dict with accuracy, class names, sample counts, etc.
    """
    X, y, class_names, feature_cols = load_and_merge(csv_paths)

    if len(X) < 5:
        raise ValueError(f"Only {len(X)} usable samples — need at least 5.")

    n_classes = len(np.unique(y))
    if n_classes < 2:
        raise ValueError(
            f"Only 1 class found ({class_names}). Need at least 2 classes to train. "
            "Ensure multiple CSV files are present (each file = one tissue class)."
        )

    test_size = 0.20 if len(X) >= 50 else 0.10
    # Use stratify only when every class has at least 2 members in test split
    min_class_count = int(np.bincount(y).min())
    can_stratify    = min_class_count >= 2
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=42,
        stratify=y if can_stratify else None,
    )

    logger.info("Training RandomForestClassifier on %d samples …", len(X_train))
    model = RandomForestClassifier(
        n_estimators=300,
        max_features="sqrt",
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred   = model.predict(X_test)
    accuracy = float(accuracy_score(y_test, y_pred))
    report   = classification_report(
        y_test, y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_test, y_pred).tolist()

    logger.info("Test accuracy: %.4f | classes=%s", accuracy, class_names)

    # Attach metadata so classifier.py and inference_engine.py can read labels
    model.class_names_   = class_names
    model.feature_cols_  = feature_cols     # wavelength column names
    model.n_features_in_ = X.shape[1]

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out)
    logger.info("Model saved → %s", out)

    return {
        "accuracy":         accuracy,
        "class_names":      class_names,
        "n_samples":        len(X),
        "n_train":          len(X_train),
        "n_test":           len(X_test),
        "n_features":       X.shape[1],
        "report":           report,
        "confusion_matrix": cm,
        "model_path":       str(out),
        "feature_cols":     feature_cols,
    }


# Keep the single-CSV API for backward compatibility
def train_and_save(csv_path: str, output_path: str) -> dict:
    """Train on a single CSV (convenience wrapper)."""
    return train_and_save_merged([csv_path], output_path)
