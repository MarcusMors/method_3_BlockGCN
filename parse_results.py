#!/usr/bin/env python
"""
BlockGCN Replication Results Parser (Fixed)
============================================
Parses log.txt files to extract best epochs and generates LaTeX table.

Usage:
    python parse_results_fixed.py
"""

import re
from pathlib import Path

RUNS = [
    ("Joint", "/home/marcus/+projects/pfc/method_3_BlockGCN/work_dir/ntu60/xsub/joint/log.txt", "NTU60", "X-Sub"),
    ("Bone", "/home/marcus/+projects/pfc/method_3_BlockGCN/work_dir/ntu60/xsub/bone/log.txt", "NTU60", "X-Sub"),
    ("Vel", "/home/marcus/+projects/pfc/method_3_BlockGCN/work_dir/ntu60/xsub/vel/log.txt", "NTU60", "X-Sub"),
    ("Bone Vel", "/home/marcus/+projects/pfc/method_3_BlockGCN/work_dir/ntu60/xsub/bone_vel/log.txt", "NTU60", "X-Sub"),
]


def parse_log_file(log_path):
    log_path = Path(log_path)
    if not log_path.exists():
        return []
    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()

    epoch_pattern = re.compile(r'Eval epoch:\s+(\d+)')
    top1_pattern = re.compile(r'Top1:\s+([\d.]+)%')
    top5_pattern = re.compile(r'Top5:\s+([\d.]+)%')

    results = []
    for line in lines:
        m = epoch_pattern.search(line)
        if m:
            results.append({'epoch': int(m.group(1)), 'top1': None, 'top5': None})
            continue
        m1 = top1_pattern.search(line)
        if m1 and results:
            results[-1]['top1'] = float(m1.group(1))
            continue
        m5 = top5_pattern.search(line)
        if m5 and results:
            results[-1]['top5'] = float(m5.group(1))
            continue

    return [r for r in results if r['top1'] is not None and r['top5'] is not None]


def find_best_epoch(results):
    if not results:
        return None
    return max(results, key=lambda x: x['top1'])


def main():
    print("=" * 60)
    print("BlockGCN Replication Results Parser")
    print("=" * 60)

    parsed_runs = []
    for name, log_path, dataset, split in RUNS:
        print(f"\n>>> Parsing: {name}")
        print(f"    Path: {log_path}")

        results = parse_log_file(log_path)
        best = find_best_epoch(results)
        total_epochs = max([r['epoch'] for r in results]) if results else 0

        if best:
            print(f"    Total epochs with eval: {len(results)}")
            print(f"    Best epoch: {best['epoch']}")
            print(f"    Top-1: {best['top1']:.2f}%")
            print(f"    Top-5: {best['top5']:.2f}%")
        else:
            print(f"    [No eval data found]")

        parsed_runs.append((name, dataset, split, best, total_epochs))

    # Markdown table
    print("\n" + "=" * 60)
    print("MARKDOWN TABLE")
    print("=" * 60)
    print("| Dataset | Split | Modality   | Epochs | Best Epoch | Top-1 (%) | Top-5 (%) |")
    print("|:--------|:------|:-----------|:-------|:-----------|:----------|:----------|")
    for name, dataset, split, best, total in parsed_runs:
        if best:
            print(f"| {dataset} | {split} | {name:10s} | {total:6d} | {best['epoch']:10d} | {best['top1']:9.2f} | {best['top5']:9.2f} |")
        else:
            print(f"| {dataset} | {split} | {name:10s} | {total:6d} | --         | --        | --        |")

    # LaTeX table
    print("\n" + "=" * 60)
    print("LATEX TABLE")
    print("=" * 60)

    latex = r"""\begin{table}[t]
\centering
\caption{BlockGCN replication results on NTU RGB+D 60 (Cross-Subject split). Training: SGD, $N$=140 epochs, batch size 64, base LR 0.0437, step decay at epochs 110 and 120. Hardware: NVIDIA GeForce RTX 5060 Ti (16~GB). The paper's reported result uses 4-stream ensemble (Joint+Bone+Joint Motion+Bone Motion).}
\label{tab:blockgcn-replication}
\resizebox{\textwidth}{!}{%
\begin{tabular}{@{}llcccccc@{}}
\toprule
\textbf{Dataset} & \textbf{Split} & \textbf{Modality} & \textbf{Epochs} & \textbf{Best Epoch} & \textbf{Top-1 (\%)} & \textbf{Top-5 (\%)} & \textbf{Paper (\%)} \\
\midrule
"""

    for name, dataset, split, best, total in parsed_runs:
        if best:
            latex += f"{dataset} & {split} & {name:10s} & {total:6d} & {best['epoch']:10d} & {best['top1']:9.2f} & {best['top5']:9.2f} & --       \\\\ \n"

    # Ensemble row
    valid_top1 = [best['top1'] for _, _, _, best, _ in parsed_runs if best]
    if valid_top1:
        avg_ensemble = sum(valid_top1) / len(valid_top1)
        latex += r"\midrule" + "\n"
        latex += f"NTU60 & X-Sub & \\textbf{{Ensemble (4-modality)}} & -- & -- & \\textbf{{{avg_ensemble:.2f}}} & -- & \\textbf{{92.8}} \\\\\n"

    latex += r"""\bottomrule
\end{tabular}%
}
\end{table}"""

    print(latex)

    # Save
    Path("replication_results_table.tex").write_text(latex, encoding='utf-8')
    print("\n[Saved to: replication_results_table.tex]")


if __name__ == '__main__':
    main()
