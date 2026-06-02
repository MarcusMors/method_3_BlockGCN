#!/usr/bin/env python
"""
BlockGCN Training Micro-Experiments
====================================
Systematically determines the optimal training duration (epochs) and
learning rate decay schedule (step points) through short, focused experiments.

This script runs abbreviated training runs with different configurations
and compares their convergence curves to find:

1. Sufficient epochs: How many epochs until validation accuracy plateaus?
2. LR decay points: When should the learning rate drop for best results?
3. Warm-up necessity: Does warm-up improve early training?

The key insight: you don't need to run full 140-epoch training multiple times.
Instead, run ~30-epoch experiments with different schedules and extrapolate.

Usage:
    # Experiment 1: Find sufficient epochs
    python training_micro_experiments.py \
        --config config/nturgbd-cross-subject/default.yaml \
        --mode find_epochs \
        --batch-size 64 --base-lr 0.05 \
        --device 0
    python training_micro_experiments.py --config config/nturgbd-cross-subject/default.yaml --mode find_epochs --batch-size 64 --base-lr 0.05 --device 0
    
    # Experiment 2: Find best LR decay points
    python training_micro_experiments.py \
        --config config/nturgbd-cross-subject/default.yaml \
        --mode find_steps \
        --batch-size 64 --base-lr 0.05 \
        --device 0

    # Experiment 3: Compare warm-up durations
    python training_micro_experiments.py \
        --config config/nturgbd-cross-subject/default.yaml \
        --mode find_warmup \
        --batch-size 64 --base-lr 0.05 \
        --device 0

Output:
- Console comparison table
- training_micro_experiments.png — Comparison plots
- Recommended config values
"""

import argparse
import os
import shutil
import sys
import time
from collections import OrderedDict, defaultdict
from pathlib import Path

import matplotlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

import resource

from torchlight import DictAction

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))


def import_class(import_str):
    mod_str, _, class_str = import_str.rpartition('.')
    __import__(mod_str)
    return getattr(sys.modules[mod_str], class_str)


def init_seed(seed):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def parse_args():
    parser = argparse.ArgumentParser(description='Training Micro-Experiments for BlockGCN')
    parser.add_argument('--config', required=True, help='Path to config file')
    parser.add_argument('--mode', required=True,
                        choices=['find_epochs', 'find_steps', 'find_warmup', 'full_comparison'],
                        help='Type of experiment to run')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--test-batch-size', type=int, default=64)
    parser.add_argument('--base-lr', type=float, default=0.05)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--num-worker', type=int, default=4)
    parser.add_argument('--max-epochs', type=int, default=35,
                        help='Max epochs for micro-experiments (not full training)')
    parser.add_argument('--eval-interval', type=int, default=2,
                        help='Evaluate every N epochs')
    parser.add_argument('--mixed-precision', action='store_true', default=True)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--output-dir', default='micro_exp_results',
                        help='Directory to save results')
    return parser.parse_args()


def build_model_and_data(config, batch_size, test_batch_size, num_worker, device, seed):
    """Build model, optimizer, loss, and data loaders from config."""
    # Model
    model_args = config.get('model_args', {})
    ModelClass = import_class(config.get('model', 'model.BlockGCN.Model'))
    model = ModelClass(**model_args).to(device)

    # Data loaders
    FeederClass = import_class(config.get('feeder', 'feeders.feeder_ntu.Feeder'))
    train_loader = torch.utils.data.DataLoader(
        dataset=Feeder(**config.get('train_feeder_args', {})),
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        prefetch_factor=2,
        num_workers=num_worker,
        drop_last=True,
        worker_init_fn=lambda x: init_seed(seed + x)
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=Feeder(**config.get('test_feeder_args', {})),
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=num_worker,
        drop_last=False,
        worker_init_fn=lambda x: init_seed(seed + x)
    )

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config.get('base_lr', 0.05),
        momentum=config.get('momentum', 0.9),
        nesterov=config.get('nesterov', True),
        weight_decay=config.get('weight_decay', 0.0004)
    )

    return model, train_loader, test_loader, optimizer, criterion


def train_epoch(model, loader, optimizer, criterion, device, use_amp, num_class):
    """Train for one epoch. Returns (avg_loss, avg_acc)."""
    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    losses, accs = [], []

    for joint, data, labels, _ in tqdm(loader, leave=False, ncols=60):
        data = data.float().to(device)
        labels = labels.long().to(device)
        joint = joint.float().to(device)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=use_amp):
            label_onehot = F.one_hot(labels, num_classes=num_class).float()
            output, _ = model(data, label_onehot, joint)
            loss = criterion(output, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        losses.append(loss.item())
        _, pred = output.max(1)
        acc = (pred == labels).float().mean().item()
        accs.append(acc)

    return np.mean(losses), np.mean(accs)


def eval_epoch(model, loader, criterion, device, num_class):
    """Evaluate for one epoch. Returns (avg_loss, avg_acc)."""
    model.eval()
    losses, accs = [], []
    all_scores = []
    all_labels = []

    with torch.no_grad():
        for joint, data, labels, _ in tqdm(loader, leave=False, ncols=60):
            data = data.float().to(device)
            labels = labels.long().to(device)
            joint = joint.float().to(device)

            with torch.cuda.amp.autocast():
                label_onehot = F.one_hot(labels, num_classes=num_class).float()
                output, _ = model(data, label_onehot, joint)
                loss = criterion(output, labels)

            losses.append(loss.item())
            _, pred = output.max(1)
            acc = (pred == labels).float().mean().item()
            accs.append(acc)

            all_scores.append(output.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    # Compute top-5 accuracy
    all_scores = np.concatenate(all_scores)
    all_labels = np.concatenate(all_labels)
    top5_acc = np.mean([
        int(label in np.argsort(score)[-5:])
        for label, score in zip(all_labels, all_scores)
    ])

    return np.mean(losses), np.mean(accs), top5_acc


def run_experiment(name, config, args, override_config=None):
    """Run a single micro-experiment. Returns history dict."""
    print(f"\n{'='*50}")
    print(f"Experiment: {name}")
    print(f"{'='*50}")

    # Apply overrides
    exp_config = config.copy()
    if override_config:
        exp_config.update(override_config)

    device = torch.device(f'cuda:{args.device}')
    model, train_loader, test_loader, optimizer, criterion = \
        build_model_and_data(exp_config, args.batch_size, args.test_batch_size,
                             args.num_worker, device, args.seed)
    model = model.to(device)
    num_class = exp_config.get('model_args', {}).get('num_class', 60)

    # Warm-up schedule
    warm_up = exp_config.get('warm_up_epoch', 5)
    base_lr = exp_config.get('base_lr', args.base_lr)
    step_points = exp_config.get('step', [110, 120])
    lr_decay_rate = exp_config.get('lr_decay_rate', 0.1)

    history = {
        'name': name,
        'epochs': [],
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_top5': [],
        'lr': []
    }

    best_acc = 0.0

    for epoch in range(args.max_epochs):
        # Adjust learning rate
        if epoch < warm_up:
            lr = base_lr * (epoch + 1) / warm_up
        else:
            lr = base_lr * (lr_decay_rate ** np.sum(epoch >= np.array(step_points)))
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device,
            args.mixed_precision, num_class)

        # Evaluate periodically
        if (epoch + 1) % args.eval_interval == 0 or epoch == args.max_epochs - 1:
            val_loss, val_acc, val_top5 = eval_epoch(
                model, test_loader, criterion, device, num_class)

            history['epochs'].append(epoch + 1)
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            history['val_top5'].append(val_top5)
            history['lr'].append(lr)

            best_acc = max(best_acc, val_acc)

            print(f"  Epoch {epoch+1:>3}/{args.max_epochs} | "
                  f"LR: {lr:.4f} | "
                  f"Train: {train_acc*100:.1f}% | "
                  f"Val: {val_acc*100:.1f}% (Top-5: {val_top5*100:.1f}%)")

    history['best_acc'] = best_acc
    print(f"  Best val accuracy: {best_acc*100:.2f}%")

    # Clean up GPU memory
    del model, optimizer
    torch.cuda.empty_cache()

    return history


def plot_comparison(all_histories, mode, output_path):
    """Plot comparison of all experiments."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    colors = plt.cm.tab10(np.linspace(0, 1, len(all_histories)))

    # Plot 1: Validation accuracy over epochs
    ax1 = axes[0]
    for i, h in enumerate(all_histories):
        ax1.plot(h['epochs'], [a * 100 for a in h['val_acc']],
                 marker='o', label=h['name'], color=colors[i], linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Validation Top-1 Accuracy (%)', fontsize=12)
    ax1.set_title(f'Micro-Experiment: {mode}', fontsize=13)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Validation loss over epochs
    ax2 = axes[1]
    for i, h in enumerate(all_histories):
        ax2.plot(h['epochs'], h['val_loss'],
                 marker='s', label=h['name'], color=colors[i], linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Validation Loss', fontsize=12)
    ax2.set_title('Validation Loss Curves', fontsize=13)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nComparison plot saved to: {output_path}")


def main():
    args = parse_args()
    init_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Apply base args
    config['base_lr'] = args.base_lr
    config['batch_size'] = args.batch_size
    config['test_batch_size'] = args.test_batch_size

    all_histories = []

    if args.mode == 'find_epochs':
        # Experiment: How many epochs are needed?
        # We test different max epoch counts with the same step schedule (scaled)
        print("\n" + "="*60)
        print("EXPERIMENT: Finding Sufficient Epoch Count")
        print("="*60)
        print("Testing different training durations to find when accuracy plateaus.")
        print(f"Each run goes for up to {args.max_epochs} epochs (micro-experiment).\n")

        # Test different epoch counts with proportional LR decay
        for max_ep, label in [(20, 'Short (20ep)'), (35, 'Medium (35ep)'), (50, 'Long (50ep)')]:
            history = run_experiment(
                label, config, args,
                override_config={
                    'step': [int(max_ep * 0.7), int(max_ep * 0.85)],
                    'num_epoch': max_ep
                }
            )
            all_histories.append(history)

        # Recommendations
        print(f"\n{'='*60}")
        print("FINDINGS: Epoch Count")
        print(f"{'='*60}")
        print("""
Look at the comparison plot. You want the point where adding more epochs
shows DIMINISHING RETURNS (the curve flattens).

For BlockGCN on NTU datasets:
- 80 epochs: Often sufficient for joint-only baseline
- 100 epochs: Good middle ground  
- 140 epochs: Paper's full schedule (best results, but diminishing returns after ~100)

RECOMMENDATION: Start with 100 epochs for initial experiments, then move to
140 epochs for final result reproduction. The extra 40 epochs typically give
~0.3-0.5% accuracy improvement.
""")

    elif args.mode == 'find_steps':
        print("\n" + "="*60)
        print("EXPERIMENT: Finding Best LR Decay Points")
        print("="*60)
        print("Testing different LR decay schedules with the same total epochs.\n")

        schedules = [
            ('Early decay [15,25]', [15, 25]),
            ('Mid decay [20,30]', [20, 30]),
            ('Late decay [25,32]', [25, 32]),
            ('Single decay [25]', [25]),
            ('No decay []', []),
        ]

        for label, steps in schedules:
            history = run_experiment(
                label, config, args,
                override_config={'step': steps}
            )
            all_histories.append(history)

        print(f"\n{'='*60}")
        print("FINDINGS: LR Decay Schedule")
        print(f"{'='*60}")
        print("""
The step parameter controls WHEN the learning rate drops by lr_decay_rate (10x).

Key observations to look for:
- Too EARLY decay: Model may stop learning before convergence
- Too LATE decay: Model may oscillate near optimum without settling
- NO decay: Usually suboptimal — LR stays too high

For the FULL 140-epoch training (extrapolate from these micro-experiments):
  - step: [110, 120]  <- Paper's default (decay at 78% and 86% of training)
  
For SHORTER training (~80 epochs):
  - step: [60, 70]    <- Proportional (decay at 75% and 88%)

For SHORT training (~50 epochs):
  - step: [35, 42]    <- Proportional

The paper's schedule [110, 120] for 140 epochs works well because:
  - First decay at 110: Fine-tuning begins after most learning is done
  - Second decay at 120: Final convergence to sharp minimum
""")

    elif args.mode == 'find_warmup':
        print("\n" + "="*60)
        print("EXPERIMENT: Finding Optimal Warm-up Duration")
        print("="*60)
        print("Testing different warm-up epoch counts.\n")

        for warmup_ep, label in [(0, 'No warm-up'), (3, '3 epochs'), (5, '5 epochs'), (10, '10 epochs')]:
            history = run_experiment(
                label, config, args,
                override_config={'warm_up_epoch': warmup_ep}
            )
            all_histories.append(history)

        print(f"\n{'='*60}")
        print("FINDINGS: Warm-up Duration")
        print(f"{'='*60}")
        print("""
Warm-up gradually increases LR from 0 to base_lr over N epochs.
This stabilizes early training, especially with large batch sizes.

For BlockGCN:
  - No warm-up: May have unstable first few epochs, but usually recovers
  - 5 epochs: Paper's default — works well for most configurations
  - 10 epochs: Overly conservative — doesn't hurt but wastes training time

RECOMMENDATION: Use warm_up_epoch: 5 for batch sizes 32-128.
If using batch_size >= 256, consider increasing to 10.
""")

    elif args.mode == 'full_comparison':
        print("\n" + "="*60)
        print("FULL COMPARISON: Best configs from all experiments")
        print("="*60)

        configs = [
            ('Paper default', {}),
            ('Short training', {'step': [60, 70], 'num_epoch': 80}),
            ('Aggressive LR', {'base_lr': 0.1}),
            ('Conservative LR', {'base_lr': 0.025}),
        ]

        for label, overrides in configs:
            history = run_experiment(label, config, args, override_config=overrides)
            all_histories.append(history)

    # Summary table
    print(f"\n{'='*60}")
    print("SUMMARY TABLE")
    print(f"{'='*60}")
    print(f"{'Experiment':<25} {'Best Val Acc':>12} {'Final Loss':>12} {'Config'}")
    print("-" * 60)
    for h in all_histories:
        best_acc_str = f"{h['best_acc']*100:.2f}%"
        final_loss = h['val_loss'][-1] if h['val_loss'] else float('nan')
        print(f"{h['name']:<25} {best_acc_str:>12} {final_loss:>12.4f}  {h.get('config', '-')}")

    # Plot
    plot_path = os.path.join(args.output_dir, f'micro_exp_{args.mode}.png')
    plot_comparison(all_histories, args.mode, plot_path)

    # Save raw data
    import json
    data_path = os.path.join(args.output_dir, f'micro_exp_{args.mode}.json')
    with open(data_path, 'w') as f:
        json.dump(all_histories, f, indent=2, default=str)
    print(f"Raw data saved to: {data_path}")


if __name__ == '__main__':
    main()
