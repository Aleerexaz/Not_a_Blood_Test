#!/usr/bin/env python3
"""
Offline trainer: trains/upgrades the base ML model from feedback_data.jsonl so the
RandomForest captures campus data patterns. Writes a new joblib bundle compatible with the API.

- Loads existing model bundle (model_rf.joblib) to get feature_cols and current LabelEncoder classes
- Reads feedback_data.jsonl (appended JSON objects)
- Builds X/y using the same feature order; computes engineered features when missing:
  * orientation_entropy from ori_bin_0..7
  * quality_pass from skeleton_pixels, orientation_entropy, contrast
- Normalizes labels: ensure Rh suffix (+/-) is present when possible
- Trains a class-balanced RandomForest (or re-fits existing estimator)
- Saves model_rf_trained.joblib with {'model','le','feature_cols','X_train','y_train'}

Run: python train_offline.py
"""
import os
import json
import joblib
import numpy as np
from typing import List, Dict, Any, Tuple

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

BASE_DIR = os.path.dirname(__file__)
DEFAULT_BUNDLE = os.path.join(BASE_DIR, "model_rf.joblib")
FEEDBACK_FILE = os.path.join(BASE_DIR, "feedback_data.jsonl")
OUTPUT_BUNDLE = os.path.join(BASE_DIR, "model_rf_trained.joblib")


def orientation_entropy(sample: Dict[str, float]) -> float:
    bins = [float(sample.get(f"ori_bin_{i}", 0.0) or 0.0) for i in range(8)]
    arr = np.array(bins, dtype=float)
    s = arr.sum()
    if s <= 0:
        return 0.0
    p = arr / s
    p = np.clip(p, 1e-12, 1.0)
    return float(-(p * np.log2(p)).sum())


def compute_quality_pass(sample: Dict[str, float]) -> int:
    sk = float(sample.get("skeleton_pixels", 0.0) or 0.0)
    ent = orientation_entropy(sample)
    contrast = float(sample.get("contrast", 0.0) or 0.0)
    return int((sk >= 1500.0) and (ent <= 2.6) and (contrast >= 60.0))


def load_existing_bundle(path: str) -> Tuple[Any, LabelEncoder, List[str]]:
    bundle = joblib.load(path)
    model = bundle.get("model")
    le = bundle.get("le")
    feature_cols = list(bundle.get("feature_cols"))
    if model is None or feature_cols is None:
        raise RuntimeError("Existing bundle missing model or feature_cols")
    return model, le, feature_cols


def normalize_label(raw_label: str, valid_labels: List[str]) -> str:
    raw_label = (raw_label or "").strip()
    if raw_label in valid_labels:
        return raw_label
    # try adding +/-
    if raw_label + "+" in valid_labels:
        return raw_label + "+"
    if raw_label + "-" in valid_labels:
        return raw_label + "-"
    # if still invalid, return as-is (encoder will expand classes)
    return raw_label


def load_feedback(feature_cols: List[str], le_existing: LabelEncoder = None) -> Tuple[np.ndarray, np.ndarray, LabelEncoder]:
    if not os.path.exists(FEEDBACK_FILE):
        raise FileNotFoundError(f"No feedback file found at {FEEDBACK_FILE}")

    rows: List[List[float]] = []
    labels: List[str] = []

    with open(FEEDBACK_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except Exception:
                continue
            feats = entry.get("features", {})
            true_label = entry.get("true_label")
            if not feats or not true_label:
                continue

            # Build row in same order
            row: List[float] = []
            for col in feature_cols:
                if col in feats:
                    row.append(float(feats.get(col, 0.0) or 0.0))
                elif col == "orientation_entropy":
                    row.append(float(orientation_entropy(feats)))
                elif col == "quality_pass":
                    row.append(float(compute_quality_pass(feats)))
                else:
                    row.append(0.0)

            rows.append(row)
            labels.append(true_label)

    X = np.array(rows, dtype=float)
    if X.shape[0] == 0:
        raise RuntimeError("Feedback file parsed but no valid samples found")

    # Determine label set
    if le_existing is not None:
        valid_labels = list(le_existing.classes_)
    else:
        valid_labels = sorted(set(labels))

    # Normalize labels against known set if possible
    labels_norm = [normalize_label(lbl, valid_labels) for lbl in labels]

    # Fit/extend LabelEncoder
    le = LabelEncoder()
    # Ensure that if there was an existing le, we keep its class order and extend as needed
    if le_existing is not None:
        existing = list(le_existing.classes_)
        new = sorted(set(labels_norm) - set(existing))
        le.fit(existing + new)
    else:
        le.fit(sorted(set(labels_norm)))

    y = le.transform(labels_norm)
    return X, y, le


def train_model(X: np.ndarray, y: np.ndarray, random_state: int = 42) -> Any:
    # Class-balanced RandomForest; depth limited a bit to reduce overfit
    clf = RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced",
        n_jobs=-1,
        random_state=random_state,
    )
    clf.fit(X, y)
    return clf


def main():
    print("[TRAIN] Loading existing bundle…")
    base_model, le_existing, feature_cols = load_existing_bundle(DEFAULT_BUNDLE)
    print(f"[TRAIN] Features={len(feature_cols)}, Existing classes={list(le_existing.classes_) if le_existing else 'unknown'}")

    print("[TRAIN] Reading feedback_data.jsonl…")
    X, y, le = load_feedback(feature_cols, le_existing)
    print(f"[TRAIN] Loaded feedback: X={X.shape}, classes={list(le.classes_)}")

    # Quick holdout to sanity check
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
    print("[TRAIN] Training RandomForest (balanced)…")
    model = train_model(Xtr, ytr)

    if Xte.shape[0] > 0:
        try:
            yhat = model.predict(Xte)
            # Align target names to labels present in yte
            present = np.unique(yte)
            target_names = [str(le.classes_[i]) for i in present]
            print("[TRAIN] Holdout classification report:\n", classification_report(yte, yhat, labels=present, target_names=target_names, zero_division=0))
        except Exception as e:
            print(f"[TRAIN] Skipping holdout report due to: {e}")

    bundle = {
        "model": model,
        "le": le,
        "feature_cols": feature_cols,
        "X_train": X,
        "y_train": y,
    }
    joblib.dump(bundle, OUTPUT_BUNDLE)
    print(f"[TRAIN] Saved new bundle to {OUTPUT_BUNDLE}")


if __name__ == "__main__":
    main()
