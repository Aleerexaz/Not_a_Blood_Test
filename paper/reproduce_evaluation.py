"""Recompute the numbers and figures used in the manuscript.

Run from the repository root with the Python environment that has
scikit-learn, pandas, and matplotlib:

    C:\\Python313\\python.exe paper\\reproduce_evaluation.py
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

ABO_ORDER = ["A", "B", "AB", "O"]
EIGHT = ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"]
PAT = re.compile(r"(?:augmented_)?cluster_(\d+)_(\d+)")


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    p = k / n
    den = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / den
    return float(centre - half), float(centre + half)


def to_abo(label: str) -> str:
    s = str(label or "").strip().upper().replace(" ", "")
    if s.startswith("AB"):
        return "AB"
    if s[:1] in {"A", "B", "O"}:
        return s[0]
    return s


def source_id(filename: str) -> str:
    text = str(filename).replace("\\", "/")
    match = PAT.search(text)
    if not match:
        return text
    return f"{match.group(1)}_{match.group(2)}"


def bmp_size(path: Path) -> tuple[int, int] | None:
    try:
        with path.open("rb") as handle:
            header = handle.read(26)
        if header[:2] != b"BM" or len(header) < 26:
            return None
        width = int.from_bytes(header[18:22], "little", signed=True)
        height = int.from_bytes(header[22:26], "little", signed=True)
        return abs(width), abs(height)
    except OSError:
        return None


def fit_scores(X, y, groups, grouped: bool, seed: int = 42):
    splitter = (
        GroupShuffleSplit(n_splits=3, test_size=0.2, random_state=seed)
        if grouped
        else StratifiedShuffleSplit(n_splits=3, test_size=0.2, random_state=seed)
    )
    splits = splitter.split(X, y, groups) if grouped else splitter.split(X, y)
    accs = []
    last = None
    for train_idx, test_idx in splits:
        model = RandomForestClassifier(
            n_estimators=120,
            max_depth=20,
            min_samples_leaf=2,
            class_weight="balanced",
            n_jobs=-1,
            random_state=seed,
        )
        model.fit(X[train_idx], y[train_idx])
        pred = model.predict(X[test_idx])
        accs.append(float(accuracy_score(y[test_idx], pred)))
        last = (y[test_idx], pred)
    return accs, last


def style_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=9)
    ax.grid(axis="y", color="#e6e6e6", linewidth=0.6)
    ax.set_axisbelow(True)


def main():
    df = pd.read_csv(ROOT / "features_clean_dataset.csv")
    numeric = [
        c
        for c in df.columns
        if c not in {"label", "filename"} and pd.api.types.is_numeric_dtype(df[c])
    ]
    df["source"] = df["filename"].map(source_id)
    df["abo"] = df["label"].map(to_abo)
    X = df[numeric].to_numpy(float)
    y8 = df["label"].astype(str).to_numpy()
    y4 = df["abo"].astype(str).to_numpy()
    groups = df["source"].to_numpy()

    # Exact-copy distances for sources that have more than one row.
    buckets: dict[str, list[int]] = {}
    for i, src in enumerate(groups):
        buckets.setdefault(src, []).append(i)
    distances = []
    for idxs in buckets.values():
        if len(idxs) < 2:
            continue
        a = X[idxs[0]]
        b = X[idxs[1]]
        distances.append(float(np.linalg.norm(a - b) / (np.linalg.norm(a) + 1e-12)))

    acc8_rand, _ = fit_scores(X, y8, groups, grouped=False)
    acc8_group, conf8 = fit_scores(X, y8, groups, grouped=True)
    acc4_rand, _ = fit_scores(X, y4, groups, grouped=False)
    acc4_group, conf4 = fit_scores(X, y4, groups, grouped=True)

    # Field log.
    rows = []
    with (ROOT / "feedback_data.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    n = len(rows)
    user_ok = sum(1 for r in rows if r.get("is_correct"))
    pred = [to_abo(r.get("predicted_label")) for r in rows]
    true = [to_abo(r.get("true_label")) for r in rows]
    abo_ok = sum(p == t for p, t in zip(pred, true))
    true_counts = Counter(true)
    pred_counts = Counter(pred)
    always_o = true_counts["O"]
    cm = confusion_matrix(true, pred, labels=ABO_ORDER)

    # Image geometry and class inventory from the folder, not only the CSV.
    inventory = {}
    sizes = Counter()
    for label in EIGHT:
        folder = ROOT / "clean_dataset" / label
        files = [p for p in folder.iterdir() if p.suffix.lower() == ".bmp"] if folder.exists() else []
        sources = {source_id(p.name) for p in files}
        inventory[label] = {"files": len(files), "unique_sources": len(sources)}
        for path in files[:: max(1, len(files) // 40)]:
            size = bmp_size(path)
            if size:
                sizes[f"{size[0]}x{size[1]}"] += 1

    def pack(accs):
        return {
            "mean": float(np.mean(accs)),
            "std": float(np.std(accs, ddof=0)),
            "folds": [round(a, 6) for a in accs],
        }

    user_ci = wilson(user_ok, n)
    abo_ci = wilson(abo_ok, n)
    results = {
        "feature_rows": int(len(df)),
        "n_features": len(numeric),
        "unique_sources": int(df["source"].nunique()),
        "sources_with_a_copy": int(sum(1 for v in buckets.values() if len(v) >= 2)),
        "copy_relative_l2_median": float(np.median(distances)) if distances else None,
        "copy_relative_l2_p90": float(np.percentile(distances, 90)) if distances else None,
        "eight_class_random": pack(acc8_rand),
        "eight_class_grouped": pack(acc8_group),
        "abo_random": pack(acc4_rand),
        "abo_grouped": pack(acc4_group),
        "eight_majority_baseline": float(pd.Series(y8).value_counts(normalize=True).iloc[0]),
        "abo_majority_baseline": float(pd.Series(y4).value_counts(normalize=True).iloc[0]),
        "field_n": n,
        "field_user_correct": user_ok,
        "field_user_correct_rate": user_ok / n,
        "field_user_wilson95": user_ci,
        "field_abo_correct": abo_ok,
        "field_abo_rate": abo_ok / n,
        "field_abo_wilson95": abo_ci,
        "field_always_predict_O": always_o,
        "field_always_predict_O_rate": always_o / n,
        "field_true_abo": {k: int(true_counts[k]) for k in ABO_ORDER},
        "field_pred_abo": {k: int(pred_counts[k]) for k in ABO_ORDER},
        "field_confusion_true_rows_pred_cols": {
            "labels": ABO_ORDER,
            "matrix": cm.tolist(),
        },
        "binom_abo_vs_uniform_quarter": {
            "p_value": float(binomtest(abo_ok, n, 0.25, alternative="two-sided").pvalue)
        },
        "binom_user_vs_one_eighth": {
            "p_value": float(binomtest(user_ok, n, 0.125, alternative="two-sided").pvalue)
        },
        "inventory": inventory,
        "bmp_size_sample_counts": dict(sizes),
        "model": "RandomForestClassifier n_estimators=120 max_depth=20 min_samples_leaf=2 class_weight=balanced",
        "splits": "3 x 80/20, random_state=42; grouped split keeps every copy of a source id together",
    }
    (OUT / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    from render_figures import main as render_main
    render_main()
    print('field ABO', round(results['field_abo_rate'], 3))


if __name__ == '__main__':
    main()
