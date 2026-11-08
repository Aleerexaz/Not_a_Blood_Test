#!/usr/bin/env python3
"""
Evaluate heuristic vs ML vs ensemble on a features CSV and suggest ABO_ENSEMBLE_ALPHA.

Usage (PowerShell):
  python tools/evaluate_ensemble.py --features .\features_clean_dataset.csv \
    --model .\model_rf.joblib --calibrate .\calibrate_data --grid 0.3 0.5 0.6 0.7

Outputs accuracy for:
- Heuristics only
- ML only
- Ensemble(alpha) across the grid; reports the best alpha.

Notes:
- Heuristic uses normalize_and_predict from bloodgroup_ABO_calibrated on the feature vector (keys_order)
  with centroids built from --calibrate.
- ML uses model.feature_cols to build X; if 'orientation_entropy' missing, compute from ori_bin_*
  and if 'quality_pass' missing, compute from skeleton_pixels, orientation_entropy, contrast.
- Evaluation uses stratified 80/20 split by ABO (collapsing Rh ±), repeated once.
"""
import argparse
import json
from pathlib import Path
from typing import List, Dict, Any

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder

import sys
THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import bloodgroup_ABO_calibrated as model


ABO_LABELS = ["A","B","AB","O"]


def map_to_abo(lbl: str) -> str:
    s = str(lbl)
    if s.startswith("AB"): return "AB"
    if s.startswith("A"): return "A"
    if s.startswith("B"): return "B"
    if s.startswith("O"): return "O"
    return s


def orientation_entropy_from_row(row: pd.Series) -> float:
    bins = [row.get(f"ori_bin_{i}", 0.0) for i in range(8)]
    arr = np.array(bins, dtype=float)
    s = float(arr.sum())
    if s <= 0: return 0.0
    p = np.clip(arr / s, 1e-12, 1.0)
    return float(-(p * np.log2(p)).sum())


def quality_pass_from_row(row: pd.Series) -> int:
    sk = float(row.get("skeleton_pixels", 0.0) or 0.0)
    ent = float(row.get("orientation_entropy", np.nan))
    if np.isnan(ent):
        ent = orientation_entropy_from_row(row)
    contrast = float(row.get("contrast", 0.0) or 0.0)
    return int((sk >= 1500.0) and (ent <= 2.6) and (contrast >= 60.0))


def build_reference(cal_dir: str) -> Dict[str, np.ndarray]:
    return model.build_reference(cal_dir)


def heuristic_probs_from_row(row: pd.Series, ref: Dict[str, np.ndarray]) -> Dict[str,float]:
    v = np.array([float(row[k]) for k in model.keys_order], dtype=float)
    _, probs = model.normalize_and_predict(v, ref, use_weights=True)
    return probs


def ml_probs_from_row(row: pd.Series, bundle: Dict[str, Any]) -> Dict[str,float]:
    m = bundle["model"]; cols: List[str] = list(bundle["feature_cols"])  # le optional
    Xrow: List[float] = []
    for c in cols:
        if c in row:
            Xrow.append(float(row[c]))
        elif c == "orientation_entropy":
            Xrow.append(orientation_entropy_from_row(row))
        elif c == "quality_pass":
            Xrow.append(float(quality_pass_from_row(row)))
        else:
            Xrow.append(0.0)
    X = np.array([Xrow], dtype=float)
    proba = m.predict_proba(X)[0]
    # Resolve class names with label encoder if present
    le = bundle.get("le")
    if hasattr(m, "classes_"):
        cls_vals = list(m.classes_)
        try:
            if le is not None:
                classes = [str(x) for x in le.inverse_transform(np.array(cls_vals))]
            else:
                classes = [str(c) for c in cls_vals]
        except Exception:
            classes = [str(c) for c in cls_vals]
    else:
        classes = list(le.classes_) if le is not None else [str(i) for i in range(len(proba))]
    abo = {k:0.0 for k in ABO_LABELS}
    for cls, p in zip(classes, proba):
        abo[map_to_abo(cls)] += float(p)
    # normalize
    s = sum(abo.values()) or 1.0
    return {k:v/s for k,v in abo.items()}


def accuracy(y_true: List[str], y_pred: List[str]) -> float:
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return float((y_true == y_pred).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--features', required=True)
    ap.add_argument('--model', required=True)
    ap.add_argument('--calibrate', default=str(PROJECT_ROOT / 'calibrate_data'))
    ap.add_argument('--grid', nargs='*', type=float, default=[0.3, 0.5, 0.6, 0.7])
    args = ap.parse_args()

    df = pd.read_csv(args.features)
    if 'label' not in df.columns:
        raise SystemExit('CSV missing label column')

    # Map labels to ABO
    df['abo'] = df['label'].apply(map_to_abo)

    ref = build_reference(args.calibrate)
    bundle = joblib.load(args.model)

    # stratified split by ABO
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    (train_idx, test_idx) = next(splitter.split(df, df['abo']))
    test_df = df.iloc[test_idx].reset_index(drop=True)

    y_true = test_df['abo'].tolist()

    # Heuristic predictions
    h_probs = [heuristic_probs_from_row(r, ref) for _, r in test_df.iterrows()]
    h_pred = [max(p.items(), key=lambda x: x[1])[0] for p in h_probs]

    # ML predictions
    m_probs = [ml_probs_from_row(r, bundle) for _, r in test_df.iterrows()]
    m_pred = [max(p.items(), key=lambda x: x[1])[0] for p in m_probs]

    acc_h = accuracy(y_true, h_pred)
    acc_m = accuracy(y_true, m_pred)

    results = {
        'heuristic_acc': acc_h,
        'ml_acc': acc_m,
        'grid': {},
    }

    best_alpha = None
    best_acc = -1.0
    for a in args.grid:
        ens_pred = []
        for hp, mp in zip(h_probs, m_probs):
            final = {k: a*hp.get(k,0.0) + (1.0-a)*mp.get(k,0.0) for k in ABO_LABELS}
            ens_pred.append(max(final.items(), key=lambda x: x[1])[0])
        acc = accuracy(y_true, ens_pred)
        results['grid'][a] = acc
        if acc > best_acc:
            best_acc = acc
            best_alpha = a

    results['best_alpha'] = best_alpha
    results['best_acc'] = best_acc

    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
