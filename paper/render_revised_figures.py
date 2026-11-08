"""Figures for the eight-page revision. Numbers come from results.json."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

OUT = Path(__file__).resolve().parent
FIG = OUT / "figures"
FIG.mkdir(exist_ok=True)
R = json.loads((OUT / "results.json").read_text(encoding="utf-8"))

INK = "#0f172a"
MUTED = "#64748b"
LINE = "#e2e8f0"
STONE = "#d6dbe3"
INDIGO = "#1e3a8a"
CRIMSON = "#be123c"
TEAL = "#0f766e"
SLATE = "#94a3b8"

plt.rcParams.update(
    {
        "font.family": "Segoe UI",
        "font.size": 9,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": LINE,
        "axes.labelcolor": MUTED,
        "xtick.color": MUTED,
        "ytick.color": INK,
        "text.color": INK,
    }
)


def clean(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(LINE)
    ax.spines["bottom"].set_color(LINE)
    ax.tick_params(length=0)
    ax.grid(axis="y", color="#f1f5f9", zorder=0)
    ax.set_axisbelow(True)


def save(fig, name):
    fig.savefig(FIG / name, dpi=200, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def copies():
    order = ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"]
    files = [R["inventory"][k]["files"] for k in order]
    uniq = [R["inventory"][k]["unique_sources"] for k in order]
    ratio = [f / u for f, u in zip(files, uniq)]
    fig, ax = plt.subplots(figsize=(7.2, 2.55))
    x = np.arange(len(order))
    w = 0.36
    ax.bar(x - w / 2, files, w, color=STONE, label="Files on disk", zorder=2)
    ax.bar(x + w / 2, uniq, w, color=INDIGO, label="Unique source prints", zorder=2)
    for i, (f, q) in enumerate(zip(files, ratio)):
        ax.text(i - w / 2, f + 30, f"{q:.2f}×", ha="center", va="bottom", fontsize=7, color=MUTED)
    ax.set_xticks(x, order)
    ax.set_ylim(0, 1750)
    ax.set_ylabel("Count")
    ax.legend(frameon=False, ncol=2, loc="upper right", fontsize=8)
    clean(ax)
    fig.tight_layout()
    save(fig, "rev_copies.png")


def accuracy():
    labels = [
        "8 classes, random split",
        "8 classes, same print held out",
        "8 classes, largest class",
        "ABO, random split",
        "ABO, same print held out",
        "ABO, largest class",
        "Live: user accepted",
        "Live: ABO letter matched",
        "Live: always answer O",
    ]
    means = [
        R["eight_class_random"]["mean"] * 100,
        R["eight_class_grouped"]["mean"] * 100,
        R["eight_majority_baseline"] * 100,
        R["abo_random"]["mean"] * 100,
        R["abo_grouped"]["mean"] * 100,
        R["abo_majority_baseline"] * 100,
        R["field_user_correct_rate"] * 100,
        R["field_abo_rate"] * 100,
        R["field_always_predict_O_rate"] * 100,
    ]
    errors = [
        R["eight_class_random"]["std"] * 100,
        R["eight_class_grouped"]["std"] * 100,
        0,
        R["abo_random"]["std"] * 100,
        R["abo_grouped"]["std"] * 100,
        0, 0, 0, 0,
    ]
    colors = [STONE, INDIGO, SLATE, STONE, INDIGO, SLATE, CRIMSON, CRIMSON, TEAL]
    hatches = ["", "///", "", "", "///", "", "", "", ""]
    fig, ax = plt.subplots(figsize=(7.2, 3.15))
    y = np.arange(len(labels))[::-1]
    bars = ax.barh(y, means, color=colors, height=0.68, zorder=2)
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)
        bar.set_edgecolor("white")
    ax.errorbar(means, y, xerr=errors, fmt="none", ecolor=INK, elinewidth=0.6, capsize=1.5, zorder=3)
    for yi, value in zip(y, means):
        ax.text(value + 1.3, yi, f"{value:.1f}", va="center", fontsize=7.5, color=INK)
    ax.set_yticks(y, labels, fontsize=8)
    ax.set_xlim(0, 112)
    ax.set_xlabel("Accuracy (%)")
    ax.axvline(25, color=LINE, lw=0.8, ls=(0, (2, 2)))
    clean(ax)
    ax.grid(axis="x", color="#f1f5f9")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    save(fig, "rev_accuracy.png")


def field():
    labels = list(R["field_true_abo"])
    pred = [R["field_pred_abo"][k] for k in labels]
    true = [R["field_true_abo"][k] for k in labels]
    cm = np.array(R["field_confusion_true_rows_pred_cols"]["matrix"], dtype=float)
    recall = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, axes = plt.subplots(1, 2, figsize=(7.3, 3.15), gridspec_kw={"width_ratios": [1.05, 1]})
    ax = axes[0]
    x = np.arange(4)
    w = 0.36
    b1 = ax.bar(x - w / 2, pred, w, color=CRIMSON, label="System prediction", zorder=2)
    b2 = ax.bar(x + w / 2, true, w, color=INDIGO, label="Stated group", zorder=2)
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 3, f"{int(h)}", ha="center", va="bottom", fontsize=7, color=INK)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 260)
    ax.set_ylabel("Scans")
    ax.set_title("(a)  Predicted vs. stated letter", loc="left", fontsize=9, color=INK)
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    clean(ax)

    ax = axes[1]
    cmap = LinearSegmentedColormap.from_list("f", ["#f8fafc", "#fecdd3", "#fb7185", "#be123c"])
    ax.imshow(recall, cmap=cmap, vmin=0, vmax=1)
    ax.set_xticks(range(4), labels)
    ax.set_yticks(range(4), labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Stated")
    ax.set_title("(b)  Row-normalised confusion", loc="left", fontsize=9, color=INK)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    for i in range(4):
        for j in range(4):
            dark = recall[i, j] > 0.55
            ax.text(j, i, f"{recall[i, j]*100:.0f}%", ha="center", va="center", fontsize=8,
                    color="white" if dark else INK, fontweight="semibold")
    fig.tight_layout()
    save(fig, "rev_field.png")


def chance():
    labels = [
        "Live ABO letter match",
        "Expected from the margins",
        "Uniform guess, 1 in 4",
        "Always O, Sindh prior",
        "Always O, on this log",
    ]
    vals = [28.0, 28.6, 25.0, 37.8, 47.2]
    colors = [CRIMSON, SLATE, STONE, "#99f6e4", TEAL]
    fig, ax = plt.subplots(figsize=(7.2, 2.15))
    y = np.arange(len(labels))[::-1]
    ax.barh(y, vals, color=colors, height=0.62, zorder=2)
    for yi, v in zip(y, vals):
        ax.text(v + 0.6, yi, f"{v:.1f}", va="center", fontsize=8, color=INK)
    ax.set_yticks(y, labels, fontsize=8)
    ax.set_xlim(0, 58)
    ax.set_xlabel("Accuracy on an ABO letter (%)")
    clean(ax)
    ax.grid(axis="x", color="#f1f5f9")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    save(fig, "rev_chance.png")


if __name__ == "__main__":
    copies()
    accuracy()
    field()
    chance()
    print("ok", FIG)
