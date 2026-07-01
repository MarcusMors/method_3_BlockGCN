#!/usr/bin/env python
"""
analyze_ensemble_confusion.py
=============================
Generate confusion matrices and detailed per-action-group analysis for
the **ensemble** of all 4 BlockGCN modalities on NTU60.

This script replicates the ensemble fusion logic from ensemble_ntu60.py but
adds full confusion matrix export, per-group analysis, and visualizations.

Usage:
    python analyze_ensemble_confusion.py \
        --split xsub \
        --joint-dir work_dir/ntu60/xsub/joint \
        --bone-dir work_dir/ntu60/xsub/bone \
        --joint-motion-dir work_dir/ntu60/xsub/vel \
        --bone-motion-dir work_dir/ntu60/xsub/bone_vel \
        --joint-epoch 139 \
        --bone-epoch 138 \
        --joint-motion-epoch 137 \
        --bone-motion-epoch 117 \
        --alpha 0.6 0.7 0.35 0.2 \
        --output analysis/xsub/ensemble

Optional: omit --*-epoch to auto-find best epoch from log.txt (up to --max-epoch).

Output files:
    {output}/
        confusion_matrix.png              -- 60x60 heatmap
        confusion_matrix_grouped.png      -- 3x3 group-level heatmap
        confusion_matrix.npy              -- raw 60x60 numpy matrix
        per_class_accuracy.png            -- bar chart per action
        per_group_accuracy.png            -- bar chart per group
        modality_comparison.png           -- side-by-side group accuracy for all modalities + ensemble
        per_class_accuracy.csv            -- full class-level metrics
        per_group_accuracy.csv            -- group-level metrics
        worst_predictions.txt             -- most confused pairs
        modality_comparison.csv           -- group accuracy comparison across modalities
"""

import argparse
import csv
import os
import pickle
import re

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


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Ensemble confusion matrix analysis for BlockGCN")
    parser.add_argument("--split", required=True, choices=["xsub", "xview"],
                        help="NTU60 split")
    parser.add_argument("--joint-dir", required=True)
    parser.add_argument("--bone-dir", required=True)
    parser.add_argument("--joint-motion-dir", required=True)
    parser.add_argument("--bone-motion-dir", required=True)

    parser.add_argument("--joint-epoch", type=int, default=None)
    parser.add_argument("--bone-epoch", type=int, default=None)
    parser.add_argument("--joint-motion-epoch", type=int, default=None)
    parser.add_argument("--bone-motion-epoch", type=int, default=None)
    parser.add_argument("--max-epoch", type=int, default=140,
                        help="Cap for auto-finding best epoch")

    parser.add_argument("--alpha", nargs=4, type=float, default=[0.6, 0.7, 0.35, 0.2],
                        help="Fusion weights: joint bone joint-motion bone-motion")
    parser.add_argument("--output", required=True, help="Output directory")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers (same as ensemble_ntu60.py)
# ---------------------------------------------------------------------------

def find_best_epoch(log_path, max_epoch):
    if not os.path.exists(log_path):
        return None
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
    epoch_pattern = re.compile(r"Eval epoch:\s+(\d+)")
    top1_pattern = re.compile(r"Top1:\s+([\d.]+)%")
    results = []
    for line in lines:
        m = epoch_pattern.search(line)
        if m:
            results.append({"epoch": int(m.group(1)), "top1": None})
        m1 = top1_pattern.search(line)
        if m1 and results:
            results[-1]["top1"] = float(m1.group(1))
    valid = [r for r in results if r["top1"] is not None and r["epoch"] <= max_epoch]
    if not valid:
        return None
    return max(valid, key=lambda x: x["top1"])["epoch"]


def resolve_epoch(directory, provided_epoch, max_epoch):
    if provided_epoch is not None:
        return provided_epoch
    log_path = os.path.join(directory, "log.txt")
    best = find_best_epoch(log_path, max_epoch)
    if best is None:
        raise ValueError(f"Could not auto-find best epoch in {directory}")
    return best


def load_pkl(directory, epoch):
    path = os.path.join(directory, f"epoch{epoch}_test_score.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Score file not found: {path}")
    with open(path, "rb") as f:
        items = list(pickle.load(f).items())
    items.sort(key=lambda x: x[0])
    return np.array([item[1] for item in items])


def load_labels(split):
    if split == "xsub":
        npz_data = np.load("./data/ntu60/NTU60_CS.npz")
    else:
        npz_data = np.load("./data/ntu60/NTU60_CV.npz")
    return np.where(npz_data["y_test"] > 0)[1]


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------

def compute_confusion_matrix(predictions, labels, num_classes=60):
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for pred, true in zip(predictions, labels):
        cm[true, pred] += 1
    return cm


def compute_group_confusion(cm):
    groups = ["Daily", "Health", "Mutual"]
    group_cm = np.zeros((3, 3), dtype=np.int64)
    for gi, gi_name in enumerate(groups):
        for gj, gj_name in enumerate(groups):
            group_cm[gi, gj] = cm[np.ix_(ACTION_GROUPS[gi_name], ACTION_GROUPS[gj_name])].sum()
    return group_cm, groups


def compute_per_class_accuracy(cm):
    diag = np.diag(cm)
    row_sum = cm.sum(axis=1)
    acc = np.zeros(len(cm), dtype=np.float64)
    mask = row_sum > 0
    acc[mask] = diag[mask] / row_sum[mask]
    return acc, diag, row_sum


def compute_per_group_accuracy(cm):
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
    confusions = []
    n = cm.shape[0]
    for i in range(n):
        for j in range(n):
            if i != j and cm[i, j] > 0:
                confusions.append({
                    "true": i, "pred": j,
                    "true_name": get_action_name(i),
                    "pred_name": get_action_name(j),
                    "count": int(cm[i, j]),
                })
    confusions.sort(key=lambda x: x["count"], reverse=True)
    return confusions[:top_k]


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_confusion_matrix(cm, output_path, title="Ensemble Confusion Matrix"):
    if not HAS_PLOTS:
        return
    fig, ax = plt.subplots(figsize=(22, 18))
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_norm = cm / row_sums
    im = ax.imshow(cm_norm, cmap="YlOrRd", aspect="auto")
    short_labels = [get_short_name(i) for i in range(60)]
    ax.set_xticks(np.arange(60))
    ax.set_yticks(np.arange(60))
    ax.set_xticklabels(short_labels, rotation=90, fontsize=6)
    ax.set_yticklabels(short_labels, fontsize=6)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.axhline(y=26.5, color="blue", linewidth=1.5, linestyle="--")
    ax.axhline(y=40.5, color="blue", linewidth=1.5, linestyle="--")
    ax.axvline(x=26.5, color="blue", linewidth=1.5, linestyle="--")
    ax.axvline(x=40.5, color="blue", linewidth=1.5, linestyle="--")
    ax.text(13, -3, "Daily", ha="center", fontsize=10, color="blue", fontweight="bold")
    ax.text(34, -3, "Health", ha="center", fontsize=10, color="blue", fontweight="bold")
    ax.text(50, -3, "Mutual", ha="center", fontsize=10, color="blue", fontweight="bold")
    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved: {output_path}")


def plot_group_confusion(group_cm, group_names, output_path, title="Group Confusion"):
    if not HAS_PLOTS:
        return
    fig, ax = plt.subplots(figsize=(8, 6))
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
    if not HAS_PLOTS:
        return
    fig, ax = plt.subplots(figsize=(22, 8))
    x = np.arange(60)
    colors = []
    for i in range(60):
        g = get_group_name(i)
        colors.append("#3498db" if g == "Daily" else "#e74c3c" if g == "Health" else "#2ecc71")
    ax.bar(x, acc * 100, color=colors, edgecolor="black", linewidth=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels([get_short_name(i) for i in range(60)], rotation=90, fontsize=7)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_xlabel("Action Class", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylim([0, 105])
    ax.axhline(y=np.mean(acc) * 100, color="red", linestyle="--", linewidth=1.5,
               label=f"Mean: {np.mean(acc)*100:.1f}%")
    ax.axvline(x=26.5, color="blue", linewidth=1.5, linestyle="--", alpha=0.5)
    ax.axvline(x=40.5, color="blue", linewidth=1.5, linestyle="--", alpha=0.5)
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor="#3498db", label="Daily"),
                       Patch(facecolor="#e74c3c", label="Health"),
                       Patch(facecolor="#2ecc71", label="Mutual")]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved: {output_path}")


def plot_per_group_accuracy(group_result, output_path, title="Per-Group Accuracy"):
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
    for i, g in enumerate(groups):
        r = group_result[g]
        ax.text(i, 5, f"{r['correct']}/{r['total']}", ha="center", fontsize=10, color="white", fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved: {output_path}")


def plot_modality_comparison(group_results, output_path, title="Modality Comparison"):
    """Side-by-side bar chart comparing all modalities + ensemble per group."""
    if not HAS_PLOTS:
        return
    modalities = list(group_results.keys())
    groups = ["Daily", "Health", "Mutual"]
    group_colors = ["#3498db", "#e74c3c", "#2ecc71"]

    x = np.arange(len(groups))
    width = 0.15
    n = len(modalities)

    fig, ax = plt.subplots(figsize=(12, 7))
    for mi, mod in enumerate(modalities):
        offset = (mi - n/2 + 0.5) * width
        values = [group_results[mod][g]["accuracy"] * 100 for g in groups]
        ax.bar(x + offset, values, width, label=mod, alpha=0.85)

    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylim([0, 105])
    ax.legend(title="Modality", fontsize=9, title_fontsize=10)
    ax.grid(axis="y", alpha=0.3)

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

    print("=" * 70)
    print("BlockGCN ENSEMBLE Confusion Matrix Analysis")
    print("=" * 70)

    # Resolve epochs
    joint_ep = resolve_epoch(args.joint_dir, args.joint_epoch, args.max_epoch)
    bone_ep = resolve_epoch(args.bone_dir, args.bone_epoch, args.max_epoch)
    jm_ep = resolve_epoch(args.joint_motion_dir, args.joint_motion_epoch, args.max_epoch)
    bm_ep = resolve_epoch(args.bone_motion_dir, args.bone_motion_epoch, args.max_epoch)

    print(f"Split:       ntu60/{args.split}")
    print(f"Joint:       epoch {joint_ep}")
    print(f"Bone:        epoch {bone_ep}")
    print(f"Joint Vel:   epoch {jm_ep}")
    print(f"Bone Vel:    epoch {bm_ep}")
    print(f"Alpha:       {args.alpha}")
    print("=" * 70)

    # Load data
    print("\n[1/5] Loading labels and scores...")
    labels = load_labels(args.split)
    s_joint = load_pkl(args.joint_dir, joint_ep)
    s_bone = load_pkl(args.bone_dir, bone_ep)
    s_jm = load_pkl(args.joint_motion_dir, jm_ep)
    s_bm = load_pkl(args.bone_motion_dir, bm_ep)
    print(f"      Samples: {len(labels)}")

    # Also compute per-modality group accuracy for comparison
    modality_scores = {
        "Joint": s_joint,
        "Bone": s_bone,
        "JointVel": s_jm,
        "BoneVel": s_bm,
    }
    modality_group_results = {}
    for mod_name, mod_scores in modality_scores.items():
        mod_preds = np.argmax(mod_scores, axis=1)
        mod_cm = compute_confusion_matrix(mod_preds, labels)
        modality_group_results[mod_name] = compute_per_group_accuracy(mod_cm)

    # Fuse scores
    print("[2/5] Fusing ensemble scores...")
    alpha = args.alpha
    fused_scores = (s_joint * alpha[0] + s_bone * alpha[1] +
                    s_jm * alpha[2] + s_bm * alpha[3])
    predictions = np.argmax(fused_scores, axis=1)

    # Ensemble accuracy
    top1 = np.mean(predictions == labels)
    top5 = np.mean(np.argsort(fused_scores, axis=1)[:, -5:] == labels[:, None], axis=1).sum() / len(labels)
    print(f"      Ensemble Top-1: {top1*100:.2f}% | Top-5: {top5*100:.2f}%")

    # Confusion matrix
    print("[3/5] Computing confusion matrix...")
    cm = compute_confusion_matrix(predictions, labels)
    np.save(os.path.join(args.output, "confusion_matrix.npy"), cm)

    # Per-class accuracy
    print("[4/5] Computing per-class and per-group accuracy...")
    acc, diag, row_sum = compute_per_class_accuracy(cm)
    group_result = compute_per_group_accuracy(cm)
    modality_group_results["Ensemble"] = group_result

    # Save per-class CSV
    csv_path = os.path.join(args.output, "per_class_accuracy.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class_idx", "action_name", "group", "correct", "total", "accuracy"])
        for i in range(60):
            writer.writerow([i, get_action_name(i), get_group_name(i),
                             int(diag[i]), int(row_sum[i]), f"{acc[i]:.6f}"])
    print(f"[OK] Saved: {csv_path}")

    # Save per-group CSV
    csv_path = os.path.join(args.output, "per_group_accuracy.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["group", "correct", "total", "accuracy"])
        for gname, gdata in group_result.items():
            writer.writerow([gname, gdata["correct"], gdata["total"], f"{gdata['accuracy']:.6f}"])
    print(f"[OK] Saved: {csv_path}")

    # Modality comparison CSV
    csv_path = os.path.join(args.output, "modality_comparison.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["modality", "group", "correct", "total", "accuracy"])
        for mod_name, mod_groups in modality_group_results.items():
            for gname, gdata in mod_groups.items():
                writer.writerow([mod_name, gname, gdata["correct"], gdata["total"], f"{gdata['accuracy']:.6f}"])
    print(f"[OK] Saved: {csv_path}")

    # Worst confusions
    print("[5/5] Finding worst confusions...")
    worst = find_worst_confusions(cm, top_k=20)
    worst_path = os.path.join(args.output, "worst_predictions.txt")
    with open(worst_path, "w") as f:
        f.write(f"Top Confused Pairs — Ensemble on NTU60 {args.split}\n")
        f.write("=" * 70 + "\n")
        for item in worst:
            f.write(f"{item['count']:4d}x  {item['true_name']:30s} -> {item['pred_name']}\n")
    print(f"[OK] Saved: {worst_path}")

    # Print summary
    print("\n" + "-" * 50)
    print("Per-Group Accuracy Summary")
    print("-" * 50)
    print(f"{'Group':<12} {'Correct':>8} {'Total':>8} {'Accuracy':>10}")
    print("-" * 50)
    for gname, gdata in group_result.items():
        print(f"{gname:<12} {gdata['correct']:>8} {gdata['total']:>8} {gdata['accuracy']*100:>9.2f}%")
    print("-" * 50)
    print(f"{'Overall':<12} {sum(g['correct'] for g in group_result.values()):>8} "
          f"{sum(g['total'] for g in group_result.values()):>8} {top1*100:>9.2f}%")

    # Modality comparison table
    print("\n" + "-" * 70)
    print("Modality Comparison (Per-Group Accuracy)")
    print("-" * 70)
    print(f"{'Modality':<12} {'Daily':>8} {'Health':>8} {'Mutual':>8} {'Overall':>8}")
    print("-" * 70)
    for mod_name in ["Joint", "Bone", "JointVel", "BoneVel", "Ensemble"]:
        mg = modality_group_results[mod_name]
        # Compute overall
        total_correct = sum(mg[g]["correct"] for g in mg)
        total_samples = sum(mg[g]["total"] for g in mg)
        overall = total_correct / total_samples * 100 if total_samples > 0 else 0
        print(f"{mod_name:<12} {mg['Daily']['accuracy']*100:>7.1f}% "
              f"{mg['Health']['accuracy']*100:>7.1f}% {mg['Mutual']['accuracy']*100:>7.1f}% "
              f"{overall:>7.1f}%")

    # Plots
    if HAS_PLOTS:
        print("\n[*] Generating plots...")
        plot_confusion_matrix(
            cm, os.path.join(args.output, "confusion_matrix.png"),
            title=f"ENSEMBLE | NTU60 {args.split.upper()} | Confusion Matrix"
        )
        group_cm, group_names = compute_group_confusion(cm)
        plot_group_confusion(
            group_cm, group_names,
            os.path.join(args.output, "confusion_matrix_grouped.png"),
            title=f"ENSEMBLE | NTU60 {args.split.upper()} | Group Confusion"
        )
        plot_per_class_accuracy(
            acc, os.path.join(args.output, "per_class_accuracy.png"),
            title=f"ENSEMBLE | NTU60 {args.split.upper()} | Per-Class Accuracy"
        )
        plot_per_group_accuracy(
            group_result, os.path.join(args.output, "per_group_accuracy.png"),
            title=f"ENSEMBLE | NTU60 {args.split.upper()} | Per-Group Accuracy"
        )
        plot_modality_comparison(
            modality_group_results,
            os.path.join(args.output, "modality_comparison.png"),
            title=f"NTU60 {args.split.upper()} | Modality Comparison by Action Group"
        )

    print("\n" + "=" * 70)
    print(f"Ensemble analysis complete. Results in: {args.output}")
    print("=" * 70)


if __name__ == "__main__":
    main()
