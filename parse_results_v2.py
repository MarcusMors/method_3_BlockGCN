#!/usr/bin/env python
"""
BlockGCN Replication Results Parser (v2)
=========================================
Parses log.txt files to extract best epochs (up to a configurable limit)
and generates LaTeX tables for both X-Sub and X-View splits.

Usage:
    # Limit best-epoch search to 140 epochs (recommended)
    python parse_results_v2.py --max-epoch 140

    # Use default (no limit, searches all epochs)
    python parse_results_v2.py
"""

import argparse
import re
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Parse BlockGCN training logs and generate LaTeX tables"
    )
    parser.add_argument(
        "--max-epoch",
        type=int,
        default=None,
        help="Only consider epochs <= N when finding the best epoch (e.g. 140)"
    )
    parser.add_argument(
        "--project-root",
        type=str,
        default="/home/marcus/+projects/pfc/method_3_BlockGCN",
        help="Path to BlockGCN project root"
    )
    return parser.parse_args()


def parse_log_file(log_path):
    """Parse log.txt to extract per-epoch test Top-1 and Top-5."""
    log_path = Path(log_path)
    if not log_path.exists():
        return []

    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    epoch_pattern = re.compile(r"Eval epoch:\s+(\d+)")
    top1_pattern = re.compile(r"Top1:\s+([\d.]+)%")
    top5_pattern = re.compile(r"Top5:\s+([\d.]+)%")

    results = []
    for line in lines:
        m = epoch_pattern.search(line)
        if m:
            results.append({"epoch": int(m.group(1)), "top1": None, "top5": None})
            continue
        m1 = top1_pattern.search(line)
        if m1 and results:
            results[-1]["top1"] = float(m1.group(1))
            continue
        m5 = top5_pattern.search(line)
        if m5 and results:
            results[-1]["top5"] = float(m5.group(1))
            continue

    return [r for r in results if r["top1"] is not None and r["top5"] is not None]


def find_best_epoch(results, max_epoch=None):
    """Find epoch with highest Top-1, optionally capped at max_epoch."""
    if not results:
        return None
    filtered = results
    if max_epoch is not None:
        filtered = [r for r in results if r["epoch"] <= max_epoch]
        if not filtered:
            return None
    return max(filtered, key=lambda x: x["top1"])


def main():
    args = parse_args()
    root = Path(args.project_root)

    # ========================================================================
    # CONFIGURE RUNS: both X-Sub and X-View
    # ========================================================================
    RUNS = [
        # NTU60 Cross-Subject
        ("Joint",      root / "work_dir/ntu60/xsub/joint/log.txt",      "NTU60", "X-Sub"),
        ("Bone",       root / "work_dir/ntu60/xsub/bone/log.txt",       "NTU60", "X-Sub"),
        ("Vel",        root / "work_dir/ntu60/xsub/vel/log.txt",        "NTU60", "X-Sub"),
        ("Bone Vel",   root / "work_dir/ntu60/xsub/bone_vel/log.txt",   "NTU60", "X-Sub"),
        # NTU60 Cross-View
        ("Joint",      root / "work_dir/ntu60/xview/joint/log.txt",     "NTU60", "X-View"),
        ("Bone",       root / "work_dir/ntu60/xview/bone/log.txt",      "NTU60", "X-View"),
        ("Vel",        root / "work_dir/ntu60/xview/vel/log.txt",       "NTU60", "X-View"),
        ("Bone Vel",   root / "work_dir/ntu60/xview/bone_vel/log.txt",  "NTU60", "X-View"),
    ]

    # Paper-reported results (CVPR 2024, Zhou et al.)
    PAPER_RESULTS = {
        ("NTU60", "X-Sub", "Joint"):     (90.0, 98.2),
        ("NTU60", "X-Sub", "Bone"):      (90.4, 98.3),
        ("NTU60", "X-Sub", "Vel"):       (88.1, 97.8),
        ("NTU60", "X-Sub", "Bone Vel"):  (83.8, 96.5),
        ("NTU60", "X-View", "Joint"):    (96.1, 99.1),
        ("NTU60", "X-View", "Bone"):     (96.4, 99.2),
        ("NTU60", "X-View", "Vel"):      (95.4, 98.9),
        ("NTU60", "X-View", "Bone Vel"): (92.7, 98.1),
    }

    print("=" * 70)
    print("BlockGCN Replication Results Parser v2")
    if args.max_epoch:
        print(f"Best-epoch search limited to epochs 1..{args.max_epoch}")
    else:
        print("Best-epoch search: NO LIMIT (all epochs)")
    print("=" * 70)

    parsed = []
    for name, log_path, dataset, split in RUNS:
        print(f"\n>>> {dataset} {split} | {name}")
        print(f"    Path: {log_path}")

        results = parse_log_file(log_path)
        best = find_best_epoch(results, args.max_epoch)
        total_epochs = max([r["epoch"] for r in results]) if results else 0

        if best:
            print(f"    Evaluated epochs: {len(results)} | Total trained: {total_epochs}")
            print(f"    Best epoch (\u2264{args.max_epoch or 'ALL'}): {best['epoch']}")
            print(f"    Top-1: {best['top1']:.2f}% | Top-5: {best['top5']:.2f}%")
        else:
            print(f"    [No eval data found or no epochs within limit]")

        parsed.append((name, dataset, split, best, total_epochs))

    # ========================================================================
    # LaTeX TABLE
    # ========================================================================
    max_label = str(args.max_epoch) if args.max_epoch else "ALL"

    latex = (
        r"\begin{table*}[t]" + "\n"
        r"\centering" + "\n"
        r"\caption{BlockGCN replication results on NTU RGB+D 60. Training: SGD (Nesterov $\mu$=0.9, weight decay 0.0004), batch size 64, base LR 0.0437, step decay at epochs 110 and 120, warm-up 5 epochs. Hardware: NVIDIA GeForce RTX~5060~Ti (16~GB GDDR6). Best epoch searched within first " + max_label + r" epochs. Paper column shows per-modality Top-1 reported in Zhou~et~al.~(CVPR 2024).}" + "\n"
        r"\label{tab:blockgcn-replication}" + "\n"
        r"\resizebox{\textwidth}{!}{%" + "\n"
        r"\begin{tabular}{@{}llccccc@{}}" + "\n"
        r"\toprule" + "\n"
        r"\textbf{Dataset} & \textbf{Split} & \textbf{Modality} & \textbf{Best Epoch} & \textbf{Top-1 (\%)} & \textbf{Top-5 (\%)} & \textbf{Paper Top-1 (\%)} \\" + "\n"
        r"\midrule" + "\n"
    )

    for name, dataset, split, best, total in parsed:
        key = (dataset, split, name)
        paper_top1, paper_top5 = PAPER_RESULTS.get(key, (None, None))

        if best:
            latex += (
                f"{dataset} & {split} & {name:12s} "
                f"& {best['epoch']:11d} "
                f"& {best['top1']:13.2f} "
                f"& {best['top5']:13.2f} "
                f"& {paper_top1:17.1f} \\\\\n"
            )
        else:
            paper_str = f"{paper_top1:.1f}" if paper_top1 else "--"
            latex += (
                f"{dataset} & {split} & {name:12s} "
                f"& --          "
                f"& --            "
                f"& --            "
                f"& {paper_str:>17s} \\\\\n"
            )

    # Ensemble rows
    latex += "\\midrule\n"
    for split in ["X-Sub", "X-View"]:
        valid_scores = [
            best["top1"] for n, d, s, best, _ in parsed
            if d == "NTU60" and s == split and best
        ]
        paper_ens = 92.8 if split == "X-Sub" else 97.0
        if valid_scores:
            avg = sum(valid_scores) / len(valid_scores)
            latex += (
                f"NTU60 & {split} & \\textbf{{Ensemble (4-mod.)}} "
                f"& --          "
                f"& \\textbf{{{avg:11.2f}}} "
                f"& --            "
                f"& \\textbf{{{paper_ens:17.1f}}} \\\\\n"
            )
        else:
            latex += (
                f"NTU60 & {split} & \\textbf{{Ensemble (4-mod.)}} "
                f"& --          "
                f"& --            "
                f"& --            "
                f"& \\textbf{{{paper_ens:17.1f}}} \\\\\n"
            )

    latex += (
        r"\bottomrule" + "\n"
        r"\end{tabular}%" + "\n"
        r"}" + "\n"
        r"\end{table*}"
    )

    print("\n" + "=" * 70)
    print("LATEX TABLE")
    print("=" * 70)
    print(latex)

    out_path = root / "replication_results_table.tex"
    out_path.write_text(latex, encoding="utf-8")
    print(f"\n[Saved LaTeX table to: {out_path}]")


if __name__ == "__main__":
    main()
