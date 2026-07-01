#!/usr/bin/env python
"""
analyze_modality_confusion.py
=============================
Generate confusion matrices and per-class/per-group accuracy analysis
for a SINGLE BlockGCN modality from its saved test-score pickle file.

Usage (run after training, using the saved .pkl):
    python analyze_modality_confusion.py \
        --pkl work_dir/ntu60/xsub/joint/epoch139_test_score.pkl \
        --split xsub \
        --modality joint \
        --output analysis/xsub/joint

Output files created:
    {output}/
        confusion_matrix.png          -- full 60x60 heatmap
        confusion_matrix_grouped.png  -- 3x3 group-level heatmap
        per_class_accuracy.png        -- bar chart of accuracy per action
        per_group_accuracy.png        -- bar chart of accuracy per group
        confusion_matrix.npy          -- raw 60x60 matrix
        per_class_accuracy.csv        -- table of class-level metrics
        per_group_accuracy.csv        -- table of group-level metrics
        worst_predictions.txt         -- most confused class pairs
"""

import argparse
import csv
import os
import pickle

import numpy as np
from tqdm import tqdm

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTS = True
except ImportError:
    HAS_PLOTS = False
    print("[WARN] matplotlib/seaborn not available. Skipping plots.")

from ntu60_action_groups import (
    NTU60_ACTIONS, ACTION_GROUPS, get_action_name, get_short_name, get_group_name
)


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze confusion matrix for one modality")
    parser.add_argument("--pkl", required=True, help="Path to epoch*_test_score.pkl")
    parser.add_argument("--split", required=True, choices=["xsub", "xview"],
                        help="NTU60 split")
    parser.add_argument("--modality", required=True,
                        choices=["joint", "bone", "vel", "bone_vel"],
                        help="Modality name for labels")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--alpha", nargs=4, type=float, default=None,
                        help="If provided, multiply scores by this weight (for ensemble debugging)")
    return parser.parse_args()


def load_labels(split):
    """Load ground-truth test labels from the NTU60 NPZ file."""
    if split == "xsub":
        npz_data = np.load("./data/ntu60/NTU60_CS.npz")
    else:
        npz_data = np.load("./data/ntu60/NTU60_CV.npz")
    labels = np.where(npz_data["y_test"] > 0)[1]
    return labels


def load_scores(pkl_path, weight=None):
    """Load prediction scores from a pickle file. Optionally apply a weight."""
    with open(pkl_path, "rb") as f:
        items = list(pickle.load(f).items())
    # Sort by sample name to ensure consistent ordering
    items.sort(key=lambda x: x[0])
    scores = np.array([item[1] for item in items])
    if weight is not None:
        scores = scores * weight
    return scores


def compute_confusion_matrix(predictions, labels, num_classes=60):
    """Compute confusion matrix from predictions and labels."""
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for pred, true in zip(predictions, labels):
        cm[true, pred] += 1
    return cm


def compute_group_confusion(cm):
    """Aggregate a 60x60 class-level confusion into a 3x3 group-level confusion."""
    groups = ["Daily", "Health", "Mutual"]
    group_cm = np.zeros((3, 3), dtype=np.int64)
    for gi, gname_i in enumerate(groups):
        indices_i = ACTION_GROUPS[gname_i]
        for gj, gname_j in enumerate(groups):
            indices_j = ACTION_GROUPS[gname_j]
            group_cm[gi, gj] = cm[np.ix_(indices_i, indices_j)].sum()
    return group_cm, groups


def compute_per_class_accuracy(cm):
    """Return per-class accuracy (diagonal / row_sum)."""
    diag = np.diag(cm)
    row_sum = cm.sum(axis=1)
    acc = np.zeros(len(cm), dtype=np.float64)
    mask = row_sum > 0
    acc[mask] = diag[mask] / row_sum[mask]
    return acc, diag, row_sum


def compute_per_group_accuracy(cm):
    """Return per-group accuracy."""
    groups = ["Daily", "Health", "Mutual"]
    result = {}
    for gname in groups:
        indices = ACTION_GROUPS[gname]
        sub_cm = cm[np.ix_(indices, indices)]
        correct = np.diag(sub_cm).sum()
        total = sub_cm.sum()
        result[gname] = {
            "correct": int(correct),
            "total": int(total),
            "accuracy": correct / total if total > 0 else 0.0,
        }
    return result


def find_worst_confusions(cm, top_k=20):
    """Find the most frequent off-diagonal confusions."""
    confusions = []
    n = cm.shape[0]
    for i in range(n):
        for j in range(n):
            if i != j and cm[i, j] > 0:
                confusions.append({
                    "true": i,
                    "pred": j,
                    "true_name": get_action_name(i),
                    "pred_name": get_action_name(j),
                    "count": int(cm[i, j]),
                })
    confusions.sort(key=lambda x: x["count"], reverse=True)
    return confusions[:top_k]


# ---------------------------------------------------------------------------
# Plotting functions
# ---------------------------------------------------------------------------

def plot_confusion_matrix(cm, output_path, title="Confusion Matrix"):
    """Plot a 60x60 confusion matrix heatmap."""
    if not HAS_PLOTS:
        return
    fig, ax = plt.subplots(figsize=(22, 18))
    # Normalize by row for better visibility
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_norm = cm / row_sums

    im = ax.imshow(cm_norm, cmap="YlOrRd", aspect="auto")

    # Labels
    short_labels = [get_short_name(i) for i in range(60)]
    ax.set_xticks(np.arange(60))
    ax.set_yticks(np.arange(60))
    ax.set_xticklabels(short_labels, rotation=90, fontsize=6)
    ax.set_yticklabels(short_labels, fontsize=6)

    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")

    # Add group boundary lines
    ax.axhline(y=26.5, color="blue", linewidth=1.5, linestyle="--")
    ax.axhline(y=40.5, color="blue", linewidth=1.5, linestyle="--")
    ax.axvline(x=26.5, color="blue", linewidth=1.5, linestyle="--")
    ax.axvline(x=40.5, color="blue", linewidth=1.5, linestyle="--")

    # Group labels
    ax.text(13, -3, "Daily", ha="center", fontsize=10, color="blue", fontweight="bold")
    ax.text(34, -3, "Health", ha="center", fontsize=10, color="blue", fontweight="bold")
    ax.text(50, -3, "Mutual", ha="center", fontsize=10, color="blue", fontweight="bold")

    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved: {output_path}")


def plot_group_confusion(group_cm, group_names, output_path, title="Group Confusion Matrix"):
    """Plot a 3x3 group-level confusion heatmap."""
    if not HAS_PLOTS:
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    # Normalize
    row_sums = group_cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    group_cm_norm = group_cm / row_sums

    im = ax.imshow(group_cm_norm, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(np.arange(3))
    ax.set_yticks(np.arange(3))
    ax.set_xticklabels(group_names, fontsize=11)
    ax.set_yticklabels(group_names, fontsize=11)
    ax.set_xlabel("Predicted Group", fontsize=12)
    ax.set_ylabel("True Group", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")

    # Annotate with counts and percentages
    for i in range(3):
        for j in range(3):
            count = group_cm[i, j]
            pct = group_cm_norm[i, j] * 100
            ax.text(j, i, f"{count}\n({pct:.1f}%)",
                    ha="center", va="center", fontsize=12, fontweight="bold",
                    color="white" if group_cm_norm[i, j] > 0.5 else "black")

    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved: {output_path}")


def plot_per_class_accuracy(acc, output_path, title="Per-Class Accuracy"):
    """Bar chart of per-class accuracy with group colors."""
    if not HAS_PLOTS:
        return
    fig, ax = plt.subplots(figsize=(22, 8))
    x = np.arange(60)
    colors = []
    for i in range(60):
        g = get_group_name(i)
        if g == "Daily":
            colors.append("#3498db")
        elif g == "Health":
            colors.append("#e74c3c")
        else:
            colors.append("#2ecc71")

    bars = ax.bar(x, acc * 100, color=colors, edgecolor="black", linewidth=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels([get_short_name(i) for i in range(60)], rotation=90, fontsize=7)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_xlabel("Action Class", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylim([0, 105])
    ax.axhline(y=np.mean(acc) * 100, color="red", linestyle="--", linewidth=1.5, label=f"Mean: {np.mean(acc)*100:.1f}%")

    # Group boundary lines
    ax.axvline(x=26.5, color="blue", linewidth=1.5, linestyle="--", alpha=0.5)
    ax.axvline(x=40.5, color="blue", linewidth=1.5, linestyle="--", alpha=0.5)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#3498db", label="Daily"),
        Patch(facecolor="#e74c3c", label="Health"),
        Patch(facecolor="#2ecc71", label="Mutual"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved: {output_path}")


def plot_per_group_accuracy(group_result, output_path, title="Per-Group Accuracy"):
    """Bar chart of per-group accuracy."""
    if not HAS_PLOTS:
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    groups = ["Daily", "Health", "Mutual"]
    values = [group_result[g]["accuracy"] * 100 for g in groups]
    colors = ["#3498db", "#e74c3c", "#2ecc71"]
    bars = ax.bar(groups, values, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylim([0, 105])

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{val:.1f}%", ha="center", fontsize=12, fontweight="bold")

    # Add counts
    for i, g in enumerate(groups):
        r = group_result[g]
        ax.text(i, 5, f"{r['correct']}/{r['total']}", ha="center", fontsize=10, color="white", fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    print("=" * 60)
    print(f"BlockGCN Confusion Matrix Analysis")
    print(f"Split: {args.split} | Modality: {args.modality}")
    print(f"PKL:   {args.pkl}")
    print("=" * 60)

    # 1. Load data
    print("\n[1/6] Loading labels and scores...")
    labels = load_labels(args.split)
    scores = load_scores(args.pkl, weight=args.alpha[0] if args.alpha else None)
    predictions = np.argmax(scores, axis=1)
    print(f"      Samples: {len(labels)} | Classes: {len(np.unique(labels))}")

    # 2. Compute confusion matrix
    print("[2/6] Computing confusion matrix...")
    cm = compute_confusion_matrix(predictions, labels)
    np.save(os.path.join(args.output, "confusion_matrix.npy"), cm)
    print(f"      Matrix shape: {cm.shape}")

    # 3. Per-class accuracy
    print("[3/6] Computing per-class accuracy...")
    acc, diag, row_sum = compute_per_class_accuracy(cm)
    mean_acc = np.mean(acc)
    print(f"      Mean accuracy: {mean_acc*100:.2f}%")

    # Save per-class CSV
    csv_path = os.path.join(args.output, "per_class_accuracy.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class_idx", "action_name", "group", "correct", "total", "accuracy"])
        for i in range(60):
            writer.writerow([i, get_action_name(i), get_group_name(i),
                             int(diag[i]), int(row_sum[i]), f"{acc[i]:.6f}"])
    print(f"[OK] Saved: {csv_path}")

    # 4. Per-group accuracy
    print("[4/6] Computing per-group accuracy...")
    group_result = compute_per_group_accuracy(cm)
    for gname, gdata in group_result.items():
        print(f"      {gname:10s}: {gdata['accuracy']*100:.2f}% ({gdata['correct']}/{gdata['total']})")

    csv_path = os.path.join(args.output, "per_group_accuracy.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["group", "correct", "total", "accuracy"])
        for gname, gdata in group_result.items():
            writer.writerow([gname, gdata["correct"], gdata["total"], f"{gdata['accuracy']:.6f}"])
    print(f"[OK] Saved: {csv_path}")

    # 5. Group confusion matrix
    print("[5/6] Computing group-level confusion...")
    group_cm, group_names = compute_group_confusion(cm)

    # 6. Find worst confusions
    print("[6/6] Finding most confused pairs...")
    worst = find_worst_confusions(cm, top_k=20)
    worst_path = os.path.join(args.output, "worst_predictions.txt")
    with open(worst_path, "w") as f:
        f.write(f"Top Confused Pairs for {args.modality} on NTU60 {args.split}\n")
        f.write("=" * 70 + "\n")
        for item in worst:
            f.write(f"{item['count']:4d}x  {item['true_name']:30s} -> {item['pred_name']}\n")
    print(f"[OK] Saved: {worst_path}")

    # Print top 10 to console
    print("\n      Top 10 confused pairs:")
    for item in worst[:10]:
        print(f"      {item['count']:4d}x  {item['true_name']:30s} -> {item['pred_name']}")

    # 7. Plots
    if HAS_PLOTS:
        print("\n[*] Generating plots...")
        plot_confusion_matrix(
            cm,
            os.path.join(args.output, "confusion_matrix.png"),
            title=f"{args.modality.upper()} | NTU60 {args.split.upper()} | Confusion Matrix"
        )
        plot_group_confusion(
            group_cm, group_names,
            os.path.join(args.output, "confusion_matrix_grouped.png"),
            title=f"{args.modality.upper()} | NTU60 {args.split.upper()} | Group Confusion"
        )
        plot_per_class_accuracy(
            acc,
            os.path.join(args.output, "per_class_accuracy.png"),
            title=f"{args.modality.upper()} | NTU60 {args.split.upper()} | Per-Class Accuracy"
        )
        plot_per_group_accuracy(
            group_result,
            os.path.join(args.output, "per_group_accuracy.png"),
            title=f"{args.modality.upper()} | NTU60 {args.split.upper()} | Per-Group Accuracy"
        )

    print("\n" + "=" * 60)
    print(f"Analysis complete. Results in: {args.output}")
    print("=" * 60)


if __name__ == "__main__":
    main()
