#!/usr/bin/env python
"""
ensemble_ntu60.py
==================
NTU60-specific ensemble script.

Instead of hardcoding epoch1, this script accepts per-modality epoch arguments
so you can pick the best epoch (up to 140) from each stream.

Usage:
    python ensemble_ntu60.py \
        --split xsub \
        --joint-dir work_dir/ntu60/xsub/joint \
        --bone-dir work_dir/ntu60/xsub/bone \
        --joint-motion-dir work_dir/ntu60/xsub/vel \
        --bone-motion-dir work_dir/ntu60/xsub/bone_vel \
        --joint-epoch 139 \
        --bone-epoch 138 \
        --joint-motion-epoch 137 \
        --bone-motion-epoch 140

If --*-epoch is omitted, the script auto-finds the best epoch by parsing
log.txt (searching within the first --max-epoch epochs).
"""

import argparse
import os
import pickle
import re

import numpy as np
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Ensemble NTU60 BlockGCN results")
    parser.add_argument("--split", required=True, choices=["xsub", "xview"],
                        help="NTU60 split: xsub or xview")
    parser.add_argument("--joint-dir", required=True)
    parser.add_argument("--bone-dir", required=True)
    parser.add_argument("--joint-motion-dir", required=True)
    parser.add_argument("--bone-motion-dir", required=True)

    parser.add_argument("--joint-epoch", type=int, default=None,
                        help="Epoch to use from joint stream (default: auto-find best <= max-epoch)")
    parser.add_argument("--bone-epoch", type=int, default=None)
    parser.add_argument("--joint-motion-epoch", type=int, default=None)
    parser.add_argument("--bone-motion-epoch", type=int, default=None)

    parser.add_argument("--max-epoch", type=int, default=140,
                        help="Cap for auto-finding best epoch (default: 140)")
    parser.add_argument("--alpha", nargs=4, type=float, default=[0.6, 0.7, 0.35, 0.2],
                        help="Fusion weights: joint bone joint-motion bone-motion")

    return parser.parse_args()


def find_best_epoch(log_path, max_epoch):
    """Parse log.txt and return the epoch with highest Top-1 accuracy <= max_epoch."""
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
    """Return the epoch to use. If not provided, auto-find from log.txt."""
    if provided_epoch is not None:
        return provided_epoch
    log_path = os.path.join(directory, "log.txt")
    best = find_best_epoch(log_path, max_epoch)
    if best is None:
        raise ValueError(f"Could not auto-find best epoch in {directory}. "
                         f"Provide --*-epoch explicitly or ensure log.txt exists.")
    return best


def load_pkl(directory, epoch):
    """Load the test score pickle for a given epoch."""
    path = os.path.join(directory, f"epoch{epoch}_test_score.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Score file not found: {path}")
    with open(path, "rb") as f:
        return list(pickle.load(f).items())


def main():
    arg = parse_args()

    # ------------------------------------------------------------------
    # Resolve epochs
    # ------------------------------------------------------------------
    joint_ep = resolve_epoch(arg.joint_dir, arg.joint_epoch, arg.max_epoch)
    bone_ep = resolve_epoch(arg.bone_dir, arg.bone_epoch, arg.max_epoch)
    jm_ep = resolve_epoch(arg.joint_motion_dir, arg.joint_motion_epoch, arg.max_epoch)
    bm_ep = resolve_epoch(arg.bone_motion_dir, arg.bone_motion_epoch, arg.max_epoch)

    print("=" * 60)
    print("BlockGCN NTU60 Ensemble")
    print("=" * 60)
    print(f"Split:       ntu60/{arg.split}")
    print(f"Joint:       epoch {joint_ep}  ({arg.joint_dir})")
    print(f"Bone:        epoch {bone_ep}  ({arg.bone_dir})")
    print(f"Joint Vel:   epoch {jm_ep}  ({arg.joint_motion_dir})")
    print(f"Bone Vel:    epoch {bm_ep}  ({arg.bone_motion_dir})")
    print(f"Alpha:       {arg.alpha}")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Load labels
    # ------------------------------------------------------------------
    if arg.split == "xsub":
        npz_data = np.load("./data/ntu60/NTU60_CS.npz")
    else:
        npz_data = np.load("./data/ntu60/NTU60_CV.npz")
    label = np.where(npz_data["y_test"] > 0)[1]

    # ------------------------------------------------------------------
    # Load scores
    # ------------------------------------------------------------------
    r1 = load_pkl(arg.joint_dir, joint_ep)
    r2 = load_pkl(arg.bone_dir, bone_ep)
    r3 = load_pkl(arg.joint_motion_dir, jm_ep)
    r4 = load_pkl(arg.bone_motion_dir, bm_ep)

    alpha = arg.alpha

    # ------------------------------------------------------------------
    # Fuse and evaluate
    # ------------------------------------------------------------------
    right_num = right_num_5 = total_num = 0

    for i in tqdm(range(len(label)), desc="Ensembling"):
        l = label[i]
        _, r11 = r1[i]
        _, r22 = r2[i]
        _, r33 = r3[i]
        _, r44 = r4[i]

        r = r11 * alpha[0] + r22 * alpha[1] + r33 * alpha[2] + r44 * alpha[3]

        rank_5 = r.argsort()[-5:]
        right_num_5 += int(int(l) in rank_5)
        r = np.argmax(r)
        right_num += int(r == int(l))
        total_num += 1

    acc = right_num / total_num
    acc5 = right_num_5 / total_num

    print()
    print(f"Top-1 Accuracy: {acc * 100:.2f}%")
    print(f"Top-5 Accuracy: {acc5 * 100:.2f}%")
    print()
    print(f"Ensemble complete. {total_num} samples evaluated.")


if __name__ == "__main__":
    main()
