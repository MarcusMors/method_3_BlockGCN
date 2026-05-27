#!/usr/bin/env python
"""
BlockGCN Learning Rate Range Test
==================================
Based on Leslie Smith's "Cyclical Learning Rates for Training Neural Networks"
(ICLR 2017). This script finds the optimal learning rate by:

1. Starting with a very small LR (1e-7)
2. Exponentially increasing LR after each mini-batch
3. Recording the loss at each step
4. Plotting loss vs. LR to find the "valley" — the LR just before loss explodes

The optimal LR is typically:
- 1/3 to 1/10 of the LR where minimum loss occurs
- At the steepest downward slope of the loss curve
- Well below where loss starts increasing ( divergence region )

Usage:
    python ./blockgcn_tools/lr_range_test.py \
        --config config/nturgbd-cross-subject/default.yaml \
        --batch-size 64 \
        --device 0

Output:
- lr_range_test.png — Plot of loss vs. learning rate
- Console output with recommended base_lr values
"""

import argparse
import sys
import time
import math
from pathlib import Path
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Fix: add torchlight package directory directly to sys.path to bypass
# broken editable install. The torchlight/ folder (containing the inner
# torchlight/ Python package) must be on the path before the import.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / 'torchlight'))   # torchlight pkg dir
sys.path.insert(0, str(_PROJECT_ROOT))                    # project root
sys.path.insert(0, str(Path(__file__).parent))            # blockgcn_tools

# from torchlight import DictAction
class DictAction(argparse.Action):
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        if nargs is not None:
            raise ValueError("nargs not allowed")
        super(DictAction, self).__init__(option_strings, dest, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        input_dict = eval(f'dict({values})')  #pylint: disable=W0123
        output_dict = getattr(namespace, self.dest)
        for k in input_dict:
            output_dict[k] = input_dict[k]
        setattr(namespace, self.dest, output_dict)


# resource module is Unix-only; skip on Windows
if sys.platform != 'win32':
    import resource
    try:
        rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))
    except (ValueError, OSError):
        pass


def import_class(import_str):
    mod_str, _, class_str = import_str.rpartition('.')
    __import__(mod_str)
    return getattr(sys.modules[mod_str], class_str)


def parse_args():
    parser = argparse.ArgumentParser(description='Learning Rate Range Test for BlockGCN')
    parser.add_argument('--config', required=True, help='Path to config file')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--device', type=int, default=0, help='GPU device ID')
    parser.add_argument('--num-worker', type=int, default=4, help='DataLoader workers')
    parser.add_argument('--lr-start', type=float, default=1e-7,
                        help='Starting learning rate (very small)')
    parser.add_argument('--lr-end', type=float, default=10.0,
                        help='Ending learning rate (where loss will definitely explode)')
    parser.add_argument('--num-iter', type=int, default=200,
                        help='Number of iterations for the test (200-500 recommended)')
    parser.add_argument('--smooth-factor', type=float, default=0.05,
                        help='Smoothing factor for loss curve (0=none, 1=full)')
    parser.add_argument('--mixed-precision', action='store_true', default=True,
                        help='Use AMP (should match training setup)')
    parser.add_argument('--output', default='lr_range_test.png',
                        help='Output plot filename')
    parser.add_argument('--seed', type=int, default=1, help='Random seed')
    return parser.parse_args()


def init_seed(seed):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


class LRFinder:
    """
    Implements the LR range test from Smith 2017.
    
    The algorithm:
    - For each iteration i from 0 to N:
        LR_i = start_lr * (end_lr / start_lr) ** (i / N)
        Run one training step with LR_i
        Record (LR_i, loss_i)
    - Plot loss vs. LR on log-log scale
    - The steepest descent region indicates good learning rates
    """

    def __init__(self, model, optimizer, criterion, device, use_amp=True):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.use_amp = use_amp
        self.scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

        # History
        self.history = {'lr': [], 'loss': [], 'smooth_loss': []}
        self.best_loss = float('inf')
        self.iteration = 0

    def range_test(self, data_loader, start_lr, end_lr, num_iter, smooth_factor=0.05):
        """Run the LR range test."""
        # Save initial model state so we can restore it
        initial_state = OrderedDict(
            (k, v.clone().detach())
            for k, v in self.model.state_dict().items()
        )

        # Learning rate multiplier per iteration
        lr_mult = (end_lr / start_lr) ** (1 / num_iter)
        current_lr = start_lr

        # Set initial LR
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = current_lr

        print(f"\nStarting LR Range Test:")
        print(f"  LR range: {start_lr:.1e} -> {end_lr:.1e}")
        print(f"  Iterations: {num_iter}")
        print(f"  LR multiplier per step: {lr_mult:.4f}")
        print(f"{'Iter':>6} {'LR':>12} {'Loss':>10} {'Smoothed':>10} {'Status'}")
        print("-" * 55)

        running_loss = 0.0
        data_iter = iter(data_loader)

        for iteration in range(num_iter):
            try:
                joint, data, labels, index = next(data_iter)
            except StopIteration:
                data_iter = iter(data_loader)
                joint, data, labels, index = next(data_iter)

            # Move to device
            data = data.float().to(self.device)
            labels = labels.long().to(self.device)
            joint = joint.float().to(self.device)

            # Forward + backward
            self.optimizer.zero_grad()

            try:
                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    label_onehot = F.one_hot(labels, num_classes=self.model.num_class).float()
                    output, _ = self.model(data, label_onehot, joint)
                    loss = self.criterion(output, labels)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                loss_value = loss.item()

                # Check for divergence (loss explosion or NaN)
                if math.isnan(loss_value) or loss_value > 4 * self.best_loss:
                    status = "DIVERGED"
                    print(f"{iteration:>6} {current_lr:>12.2e} {loss_value:>10.4f} {'--':>10} {status}")
                    print(f"\n  Loss diverged at iteration {iteration}. Stopping early.")
                    break

                # Track best
                if loss_value < self.best_loss:
                    self.best_loss = loss_value

                # Smooth the loss
                if iteration == 0:
                    running_loss = loss_value
                else:
                    running_loss = smooth_factor * loss_value + (1 - smooth_factor) * running_loss

                # Record
                self.history['lr'].append(current_lr)
                self.history['loss'].append(loss_value)
                self.history['smooth_loss'].append(running_loss)

                status = "OK"
                if iteration % 20 == 0 or iteration < 5:
                    print(f"{iteration:>6} {current_lr:>12.2e} {loss_value:>10.4f} "
                          f"{running_loss:>10.4f} {status}")

                # Update LR for next iteration
                current_lr *= lr_mult
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = current_lr

            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"\n  [OOM at iteration {iteration}]")
                    torch.cuda.empty_cache()
                    break
                raise

        # Restore model to initial state (we only wanted to find LR, not train)
        self.model.load_state_dict(initial_state)
        print(f"\nLR Range Test complete. {len(self.history['lr'])} iterations recorded.")

    def plot(self, output_path, skip_start=10, skip_end=5):
        """
        Plot the LR range test results with annotated recommendations.
        
        skip_start: Skip first N points (unstable warm-up)
        skip_end: Skip last N points (divergence region)
        """
        if len(self.history['lr']) < skip_start + skip_end + 10:
            print("[WARNING] Not enough data points for reliable analysis.")
            skip_start = max(0, skip_start)
            skip_end = max(0, skip_end)

        lr_history = self.history['lr'][skip_start:-skip_end if skip_end else None]
        smooth_loss = self.history['smooth_loss'][skip_start:-skip_end if skip_end else None]

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Plot 1: Loss vs. LR (log scale) — standard view
        ax1 = axes[0]
        ax1.plot(lr_history, smooth_loss)
        ax1.set_xscale('log')
        ax1.set_xlabel('Learning Rate (log scale)', fontsize=12)
        ax1.set_ylabel('Smoothed Loss', fontsize=12)
        ax1.set_title('LR Range Test: Loss vs. Learning Rate', fontsize=13)
        ax1.grid(True, alpha=0.3)

        # Find minimum loss point
        min_idx = np.argmin(smooth_loss)
        min_lr = lr_history[min_idx]
        ax1.axvline(min_lr, color='red', linestyle='--', alpha=0.5, label=f'Min loss @ {min_lr:.2e}')

        # Recommendations
        # Strategy 1: LR at minimum loss divided by 10
        rec_1 = min_lr / 10
        ax1.axvline(rec_1, color='green', linestyle='--', alpha=0.7,
                    label=f'Rec #1 (min/10): {rec_1:.2e}')

        # Strategy 2: Steepest gradient (minimum of d(loss)/d(log(lr)))
        log_lr = np.log10(lr_history)
        grad = np.gradient(smooth_loss, log_lr)
        steepest_idx = np.argmin(grad)
        rec_2 = lr_history[steepest_idx]
        ax1.axvline(rec_2, color='blue', linestyle='--', alpha=0.7,
                    label=f'Rec #2 (steepest): {rec_2:.2e}')

        # Strategy 3: Median between steepest and min
        rec_3 = 10 ** ((np.log10(rec_2) + np.log10(min_lr)) / 2)
        ax1.axvline(rec_3, color='purple', linestyle='--', alpha=0.7,
                    label=f'Rec #3 (median): {rec_3:.2e}')

        ax1.legend(loc='upper left', fontsize=9)

        # Plot 2: Rate of change (derivative) — helps identify the steepest region
        ax2 = axes[1]
        ax2.plot(lr_history[1:], grad[1:], color='darkblue')
        ax2.set_xscale('log')
        ax2.set_xlabel('Learning Rate (log scale)', fontsize=12)
        ax2.set_ylabel('d(Loss)/d(log10 LR)', fontsize=12)
        ax2.set_title('Rate of Loss Change (Steepest = Best Learning)', fontsize=13)
        ax2.axhline(0, color='black', linestyle='-', alpha=0.3)
        ax2.axvline(rec_2, color='blue', linestyle='--', alpha=0.7,
                    label=f'Steepest descent: {rec_2:.2e}')
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=10)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")

        return {
            'min_loss_lr': min_lr,
            'steepest_lr': rec_2,
            'recommended_1': rec_1,
            'recommended_2': rec_2,
            'recommended_3': rec_3,
        }


def main():
    args = parse_args()
    init_seed(args.seed)

    if not torch.cuda.is_available():
        print("[ERROR] CUDA required for LR range test.")
        sys.exit(1)

    device = torch.device(f'cuda:{args.device}')
    print(f"Device: {torch.cuda.get_device_name(args.device)}")

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Build model
    model_args = config.get('model_args', {})
    ModelClass = import_class(config.get('model', 'model.BlockGCN.Model'))
    model = ModelClass(**model_args).to(device)
    model.train()

    # Build data loader
    FeederClass = import_class(config.get('feeder', 'feeders.feeder_ntu.Feeder'))
    train_feeder = FeederClass(**config.get('train_feeder_args', {}))
    train_loader = torch.utils.data.DataLoader(
        dataset=train_feeder,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        prefetch_factor=2,
        num_workers=args.num_worker,
        drop_last=True,
        worker_init_fn=lambda x: init_seed(args.seed + x)
    )

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {num_params/1e6:.2f}M parameters")
    print(f"Train samples: {len(train_feeder)}")
    print(f"Batches per epoch: {len(train_loader)}")
    print(f"Effective test iterations: ~{min(args.num_iter, len(train_loader))}")

    # Setup optimizer and loss (SGD as in default config)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr_start, momentum=0.9, nesterov=True)
    criterion = nn.CrossEntropyLoss().to(device)

    # Run LR range test
    finder = LRFinder(model, optimizer, criterion, device, use_amp=args.mixed_precision)
    finder.range_test(
        train_loader,
        start_lr=args.lr_start,
        end_lr=args.lr_end,
        num_iter=args.num_iter,
        smooth_factor=args.smooth_factor
    )

    # Plot and get recommendations
    if len(finder.history['lr']) > 20:
        recommendations = finder.plot(args.output)

        print(f"\n{'='*60}")
        print("LEARNING RATE RECOMMENDATIONS")
        print(f"{'='*60}")
        print(f"\nBased on the LR range test plot ({args.output}):")
        print(f"  LR at minimum loss:     {recommendations['min_loss_lr']:.2e}")
        print(f"  LR at steepest descent: {recommendations['steepest_lr']:.2e}")
        print(f"\nRecommended base_lr values:")
        print(f"  1. Conservative (min_loss / 10):  {recommendations['recommended_1']:.2e}")
        print(f"  2. Aggressive (steepest point):   {recommendations['recommended_2']:.2e}")
        print(f"  3. Balanced (median):             {recommendations['recommended_3']:.2e}")

        print(f"\n{'='*60}")
        print("HOW TO USE THESE RESULTS")
        print(f"{'='*60}")
        print(f"""
1. Open {args.output} and visually inspect the curve.

2. The OPTIMAL learning rate is where the loss drops MOST STEEPLY — this is
   where the model learns fastest without diverging.

3. In the YAML config file, set:
   base_lr: <VALUE>

4. The paper's default of 0.05 is a reasonable starting point. If your
   LR range test suggests a very different value (e.g., 0.001 or 1.0),
   you may need to check your data pipeline first.

5. If you change batch_size, apply the LINEAR SCALING RULE:
   new_lr = base_lr * (new_batch_size / 64)
   
   Example: If optimal base_lr=0.05 at batch=64, and you use batch=128:
   new_lr = 0.05 * (128 / 64) = 0.10

6. For BlockGCN specifically, typical values are:
   - batch=32  -> base_lr=0.025
   - batch=64  -> base_lr=0.05 (default)
   - batch=128 -> base_lr=0.10
   - batch=256 -> base_lr=0.20
""")
    else:
        print("[WARNING] Not enough data collected. Try increasing --num-iter.")


if __name__ == '__main__':
    main()