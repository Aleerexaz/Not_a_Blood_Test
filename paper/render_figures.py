"""Draw the manuscript figures from paper/results.json.

The palette is intentionally small: stone for a leaked random split, indigo
for a same-print split, crimson for the live scanner, teal for the
one-line baseline. Numbers are not recomputed here.
"""
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
FIG.mkdir(parents=True, exist_ok=True)

INK = "#0f172a"
MUTED = "#64748b"
LINE = "#e2e8f0"
STONE = "#cbd5e1"
INDIGO = "#1e3a8a"
CRIMSON = "#be123c"
TEAL = "#0f766e"
SLATE = "#94a3b8"

plt.rcParams.update(
    {
        "font.family": "Segoe UI",
        "font.size": 11,
        "axes.unicode_minus": False,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": LINE,
        "axes.labelcolor": INK,
        "axes.titlecolor": INK,
        "xtick.color": MUTED,
        "ytick.color": INK,
        "text.color": INK,
        "axes.titleweight": "semibold",
    }
)


def finish(ax, xlabel=None, ylabel=None):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(LINE)
    ax.spines["bottom"].set_color(LINE)
    ax.tick_params(length=0, labelsize=10)
    ax.grid(axis="y", color="#f1f5f9", linewidth=0.8)
    ax.set_axisbelow(True)
    if xlabel:
        ax.set_xlabel(xlabel, color=MUTED, fontsize=10, labelpad=8)
    if ylabel:
        ax.set_ylabel(ylabel, color=MUTED, fontsize=10, labelpad=8)


def title(ax, text):
    ax.set_title(text, loc="left", fontsize=13.5, pad=12, color=INK, fontweight="semibold")


def save(fig, name):
    fig.savefig(FIG / f"{name}.png", dpi=220, bbox_inches="tight", pad_inches=0.28)
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight", pad_inches=0.28)
    plt.close(fig)


def figure_accuracy(results):
    labels = [
        "8 classes, random split",
        "8 classes, same print held out",
        "8 classes, largest-class baseline",
        "ABO, random split",
        "ABO, same print held out",
        "ABO, largest-class baseline",
        "Live scans, user accepted the answer",
        "Live scans, ABO letter matched",
        "Live scans, always answer O",
    ]
    means = np.array(
        [
            results["eight_class_random"]["mean"] * 100,
            results["eight_class_grouped"]["mean"] * 100,
            results["eight_majority_baseline"] * 100,
            results["abo_random"]["mean"] * 100,
            results["abo_grouped"]["mean"] * 100,
            results["abo_majority_baseline"] * 100,
            results["field_user_correct_rate"] * 100,
            results["field_abo_rate"] * 100,
            results["field_always_predict_O_rate"] * 100,
        ]
    )
    errors = np.array(
        [
            results["eight_class_random"]["std"] * 100,
            results["eight_class_grouped"]["std"] * 100,
            0,
            results["abo_random"]["std"] * 100,
            results["abo_grouped"]["std"] * 100,
            0,
            0,
            0,
            0,
        ]
    )
    colors = [STONE, INDIGO, SLATE, STONE, INDIGO, SLATE, CRIMSON, CRIMSON, TEAL]
    fig, ax = plt.subplots(figsize=(9.4, 5.6))
    y = np.arange(len(labels))[::-1]
    ax.barh(y, means, color=colors, height=0.62, zorder=2)
    ax.errorbar(
        means,
        y,
        xerr=errors,
        fmt="none",
        ecolor=INK,
        elinewidth=0.8,
        capsize=2,
        zorder=3,
    )
    for yi, value in zip(y, means):
        ax.text(value + 1.6, yi, f"{value:.1f}", va="center", ha="left", fontsize=10, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10.5)
    ax.set_xlim(0, 112)
    ax.set_xlabel("Accuracy (%)", color=MUTED, fontsize=10, labelpad=8)
    ax.axvline(25, color=LINE, linewidth=0.8, linestyle=(0, (2, 3)), zorder=0)
    title(ax, "Accuracy depends on which question is asked")
    handles = [
        plt.Line2D([0], [0], color=STONE, lw=8, solid_capstyle="butt", label="Random split"),
        plt.Line2D([0], [0], color=INDIGO, lw=8, solid_capstyle="butt", label="Same print held out"),
        plt.Line2D([0], [0], color=CRIMSON, lw=8, solid_capstyle="butt", label="Live scanner"),
        plt.Line2D([0], [0], color=TEAL, lw=8, solid_capstyle="butt", label="Always answer O"),
        plt.Line2D([0], [0], color=SLATE, lw=8, solid_capstyle="butt", label="Largest class"),
    ]
    ax.legend(
        handles=handles,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.42, -0.16),
        ncols=3,
        fontsize=8.5,
        labelcolor=MUTED,
    )
    finish(ax)
    ax.grid(axis="x", color="#f1f5f9", linewidth=0.8)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    save(fig, "fig1_protocol_accuracy")


def figure_bias(results):
    labels = results["field_true_abo"].keys()
    labels = list(labels)
    pred = [results["field_pred_abo"][k] for k in labels]
    true = [results["field_true_abo"][k] for k in labels]
    fig, ax = plt.subplots(figsize=(7.6, 4.5))
    x = np.arange(len(labels))
    w = 0.34
    b1 = ax.bar(x - w / 2, pred, w, color=CRIMSON, label="System prediction", zorder=2)
    b2 = ax.bar(x + w / 2, true, w, color=INDIGO, label="Stated blood group", zorder=2)
    for bars in (b1, b2):
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 4,
                f"{int(height)}",
                ha="center",
                va="bottom",
                fontsize=9,
                color=INK,
            )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12, color=INK)
    ax.set_ylim(0, 260)
    ax.legend(frameon=False, loc="upper left", ncols=2, fontsize=10)
    title(ax, "The scanner answered B. People mostly reported O.")
    finish(ax, ylabel="Scans")
    fig.tight_layout()
    save(fig, "fig2_field_bias")


def figure_confusion(results):
    labels = results["field_confusion_true_rows_pred_cols"]["labels"]
    cm = np.array(results["field_confusion_true_rows_pred_cols"]["matrix"], dtype=float)
    recall = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    cmap = LinearSegmentedColormap.from_list("field", ["#f8fafc", "#fecdd3", "#fb7185", "#be123c"])
    fig, ax = plt.subplots(figsize=(6.3, 5.3))
    ax.imshow(recall, cmap=cmap, vmin=0, vmax=1)
    ax.set_xticks(range(4), labels, fontsize=12)
    ax.set_yticks(range(4), labels, fontsize=12)
    ax.set_xlabel("Predicted ABO group", color=MUTED, fontsize=10, labelpad=8)
    ax.set_ylabel("Stated ABO group", color=MUTED, fontsize=10, labelpad=8)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for i in range(4):
        for j in range(4):
            dark = recall[i, j] > 0.55
            ax.text(
                j,
                i - 0.08,
                f"{recall[i, j] * 100:.0f}%",
                ha="center",
                va="center",
                fontsize=13,
                fontweight="semibold",
                color="white" if dark else INK,
            )
            ax.text(
                j,
                i + 0.22,
                f"{int(cm[i, j])}",
                ha="center",
                va="center",
                fontsize=8,
                color="#ffe4e6" if dark else MUTED,
            )
    title(ax, "Every stated group was mostly called B")
    fig.tight_layout()
    save(fig, "fig3_field_confusion")


def figure_copies(results):
    order = ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"]
    files = [results["inventory"][k]["files"] for k in order]
    uniq = [results["inventory"][k]["unique_sources"] for k in order]
    fig, ax = plt.subplots(figsize=(8.6, 4.5))
    x = np.arange(len(order))
    w = 0.36
    ax.bar(x - w / 2, files, w, color=STONE, label="Files on disk", zorder=2)
    ax.bar(x + w / 2, uniq, w, color=INDIGO, label="Unique source prints", zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels(order, fontsize=11, color=INK)
    ax.set_ylim(0, 1600)
    ax.legend(frameon=False, ncols=2, loc="upper right", fontsize=10)
    title(ax, "Extra files are the same print, stored again")
    finish(ax, ylabel="Count")
    fig.tight_layout()
    save(fig, "fig4_duplicate_inflation")


def main():
    results = json.loads((OUT / "results.json").read_text(encoding="utf-8"))
    figure_accuracy(results)
    figure_bias(results)
    figure_confusion(results)
    figure_copies(results)
    print("wrote figures to", FIG)


if __name__ == "__main__":
    main()
