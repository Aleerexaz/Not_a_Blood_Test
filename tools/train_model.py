#!/usr/bin/env python3
"""
Train a scikit-learn classifier on extracted features CSV.

Usage (PowerShell):
  python tools/train_model.py --features .\features_clean_dataset_sample.csv --out .\model_rf.joblib

The CSV must contain a 'label' column and numeric feature columns.
We will automatically exclude non-numeric columns like 'filename'.
"""
import argparse
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--features', required=True)
    ap.add_argument('--out', default='model_rf.joblib')
    ap.add_argument('--n-est', type=int, default=300)
    ap.add_argument('--max-depth', type=int, default=None)
    args = ap.parse_args()

    df = pd.read_csv(args.features)

    # Basic cleaning
    if 'label' not in df.columns:
        raise SystemExit("CSV must contain a 'label' column")

    # Remove obviously non-numeric columns
    drop_cols = [c for c in ('filename',) if c in df.columns]
    X_df = df.drop(columns=drop_cols)

    # Keep only numeric columns except 'label'
    num_cols = [c for c in X_df.columns if c != 'label' and pd.api.types.is_numeric_dtype(X_df[c])]
    X = X_df[num_cols].values
    y = df['label'].astype(str).values

    # Remove rows with NaN
    mask = ~np.isnan(X).any(axis=1)
    X = X[mask]
    y = y[mask]

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    clf = RandomForestClassifier(n_estimators=args.n_est, max_depth=args.max_depth, random_state=42, n_jobs=-1)
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    scores = cross_val_score(clf, X, y_enc, cv=cv, scoring='accuracy', n_jobs=-1)
    print('CV accuracy mean/std:', scores.mean(), scores.std())

    clf.fit(X, y_enc)
    joblib.dump({'model': clf, 'le': le, 'feature_cols': num_cols}, args.out)
    print('Saved model to', args.out)


if __name__ == '__main__':
    main()
