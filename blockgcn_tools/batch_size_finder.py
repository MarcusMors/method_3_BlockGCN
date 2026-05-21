#!/usr/bin/env python
"""
BlockGCN Batch Size Finder
===========================
Incrementally tests batch sizes from small to large to find the maximum
that fits in your GPU VRAM without OOM errors.

Also measures throughput (samples/sec) to find the OPTIMAL batch size
(not just the maximum) — often these differ.

Run this BEFORE training to determine your hardware limits:
    python batch_size_finder.py --config config/nturgbd-cross-subject/default.yaml

The script will output:
1. Maximum batch size (where OOM occurs)
2. Optimal batch size (best throughput/efficiency tradeoff)
3. Recommended value with safety margin
"""

import argparse
import sys
import time
import traceback
from pathlib import Path

# Add project root to path (script is in blockgcn_tools/, parent is project root)
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import yaml

# resource module is Unix-only; skip on Windows
if sys.platform != 'win32':
    import resource
    try:
        rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))
    except (ValueError, OSError):
        pass


def import_class(import_str):
    """Dynamically import a class from module path string."""
    mod_str, _, class_str = import_str.rpartition('.')
    __import__(mod_str)
    return getattr(sys.modules[mod_str], class_str)


def parse_args():
    parser = argparse.ArgumentParser(description='Find optimal batch size for BlockGCN')
    parser.add_argument('--config', default='config/nturgbd-cross-subject/default.yaml',
                        help='Path to config file (for model architecture)')
    parser.add_argument('--device', type=int, default=0, help='GPU device ID')
    parser.add_argument('--start-batch', type=int, default=16,
                        help='Starting batch size to test')
    parser.add_argument('--max-batch', type=int, default=512,
                        help='Maximum batch size to test (hard stop)')
    parser.add_argument('--step', type=int, default=8,
                        help='Increment step between tests (use 8 or 16 for speed)')
    parser.add_argument('--num-batches', type=int, default=5,
                        help='Number of forward+backward passes per batch size (for throughput avg)')
    parser.add_argument('--seq-len', type=int, default=64, help='Sequence length')
    parser.add_argument('--num-joints', type=int, default=25, help='Number of joints')
    parser.add_argument('--num-persons', type=int, default=2, help='Number of persons')
    parser.add_argument('--mixed-precision', action='store_true', default=True,
                        help='Use Automatic Mixed Precision (AMP) - matches main.py')
    parser.add_argument('--safety-factor', type=float, default=0.85,
                        help='Safety margin: recommended = max * safety_factor')
    parser.add_argument('--find-throughput-optimum', action='store_true', default=True,
                        help='Also find throughput-optimal batch size')
    return parser.parse_args()


def create_dummy_batch(batch_size, seq_len, num_joints, num_persons, device):
    """Create a dummy data batch matching NTU dataset format."""
    # Shape: (N, C, T, V, M) where C=3 (x,y,z coordinates)
    data = torch.randn(batch_size, 3, seq_len, num_joints, num_persons,
                       device=device, dtype=torch.float32)
    labels = torch.randint(0, 60, (batch_size,), device=device)
    return data, labels


def test_batch_size(model, batch_size, seq_len, num_joints, num_persons,
                    device, num_batches, use_amp):
    """
    Test if a given batch size fits in GPU memory.
    Returns (success: bool, throughput: float, vram_used_mb: float, avg_time_ms: float)
    """
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # Warm-up CUDA (first allocation often slower)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)

    try:
        times = []
        for b in range(num_batches):
            data, labels = create_dummy_batch(
                batch_size, seq_len, num_joints, num_persons, device)

            # Simulate joint input (same shape as data in BlockGCN)
            joint = data.clone()

            # One-hot encoding for labels (as in main.py line 503)
            label_onehot = F.one_hot(labels, num_classes=model.num_class).float()

            torch.cuda.synchronize(device)
            t_start = time.perf_counter()

            with torch.cuda.amp.autocast(enabled=use_amp):
                output, _ = model(data, label_onehot, joint)
                loss = nn.CrossEntropyLoss()(output, labels)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            torch.cuda.synchronize(device)
            t_end = time.perf_counter()
            times.append(t_end - t_start)

            # Clean up
            del data, labels, joint, output, loss
            torch.cuda.empty_cache()

        vram_peak = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        avg_time = np.mean(times[1:])  # Skip first (warmup)
        throughput = batch_size / avg_time  # samples per second

        return True, throughput, vram_peak, avg_time * 1000

    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return False, 0.0, 0.0, 0.0
    except Exception as e:
        print(f"  [ERROR at batch={batch_size}]: {e}")
        torch.cuda.empty_cache()
        return False, 0.0, 0.0, 0.0


def find_max_batch_size(args, model):
    """Binary-search-like approach to find the exact max batch size."""
    device = f'cuda:{args.device}'
    print(f"\n{'='*60}")
    print("PHASE 1: Finding Maximum Batch Size")
    print(f"{'='*60}")
    print(f"Testing from {args.start_batch} to {args.max_batch} "
          f"(step={args.step}, AMP={'ON' if args.mixed_precision else 'OFF'})")
    print(f"{'Batch Size':<12} {'Status':<10} {'VRAM (MB)':<12} {'Time (ms)':<12} {'Throughput':<15}")
    print("-" * 60)

    results = []
    last_success = None

    # Phase 1: Coarse search with step increments
    for bs in range(args.start_batch, args.max_batch + 1, args.step):
        success, throughput, vram, avg_time = test_batch_size(
            model, bs, args.seq_len, args.num_joints, args.num_persons,
            device, args.num_batches, args.mixed_precision)

        status = "OK" if success else "OOM"
        print(f"{bs:<12} {status:<10} {vram:<12.0f} {avg_time:<12.1f} "
              f"{throughput:<12.1f} samp/s")

        if success:
            results.append({
                'batch_size': bs,
                'throughput': throughput,
                'vram_mb': vram,
                'time_ms': avg_time
            })
            last_success = bs
        else:
            # First OOM means we can stop coarse search
            break

    if not results:
        print("\n[ERROR] Even the smallest batch size OOM'd!")
        print("Try: reducing model size, enabling AMP, or closing other GPU apps.")
        return None, None

    # Phase 2: Fine search between last_success and last_success + step
    if last_success:
        fine_start = last_success + 1
        fine_end = min(last_success + args.step, args.max_batch)
        print(f"\nPhase 2: Fine search between {fine_start} and {fine_end}...")

        for bs in range(fine_start, fine_end + 1):
            success, throughput, vram, avg_time = test_batch_size(
                model, bs, args.seq_len, args.num_joints, args.num_persons,
                device, args.num_batches, args.mixed_precision)

            status = "OK" if success else "OOM"
            print(f"{bs:<12} {status:<10} {vram:<12.0f} {avg_time:<12.1f} "
                  f"{throughput:<12.1f} samp/s")

            if success:
                results.append({
                    'batch_size': bs,
                    'throughput': throughput,
                    'vram_mb': vram,
                    'time_ms': avg_time
                })
                last_success = bs
            else:
                break

    return results, last_success


def find_throughput_optimal(results):
    """Find the batch size with best throughput efficiency."""
    if not results:
        return None

    # Throughput usually plateaus; find where diminishing returns start
    throughputs = [r['throughput'] for r in results]
    batch_sizes = [r['batch_size'] for r in results]

    # Find where throughput improvement drops below 5%
    best_idx = 0
    for i in range(1, len(throughputs)):
        improvement = (throughputs[i] - throughputs[i - 1]) / throughputs[i - 1]
        if improvement > 0.05:  # Still improving by >5%
            best_idx = i
        else:
            break  # Diminishing returns

    return results[best_idx]


def main():
    args = parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Setup device
    if not torch.cuda.is_available():
        print("[ERROR] No CUDA available!")
        sys.exit(1)

    device = f'cuda:{args.device}'
    torch.cuda.set_device(args.device)
    total_vram = torch.cuda.get_device_properties(args.device).total_memory / (1024 ** 3)
    print(f"GPU: {torch.cuda.get_device_name(args.device)}")
    print(f"Total VRAM: {total_vram:.2f} GB")
    print(f"AMP (Mixed Precision): {'Enabled' if args.mixed_precision else 'Disabled'}")

    # Build model
    print("\nBuilding model...")
    model_args = config.get('model_args', {})
    ModelClass = import_class(config.get('model', 'model.BlockGCN.Model'))
    model = ModelClass(**model_args)
    model = model.to(device)
    model.train()  # Training mode (includes gradients)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,} ({num_params/1e6:.2f}M)")

    # Run batch size search
    results, max_bs = find_max_batch_size(args, model)

    if not results:
        sys.exit(1)

    # Analysis
    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")

    # Maximum batch size
    print(f"\n1. MAXIMUM batch size (before OOM): {max_bs}")
    max_result = [r for r in results if r['batch_size'] == max_bs][0]
    print(f"   VRAM at max: {max_result['vram_mb']:.0f} MB "
          f"({max_result['vram_mb']/1024:.2f} GB)")

    # Safe batch size with safety factor
    safe_bs = int(max_bs * args.safety_factor)
    # Round to nearest multiple of 8 (good for GPU utilization)
    safe_bs = (safe_bs // 8) * 8
    if safe_bs < 8:
        safe_bs = 8
    print(f"\n2. SAFE batch size (with {args.safety_factor*100:.0f}% margin): {safe_bs}")

    # Throughput-optimal
    if args.find_throughput_optimum:
        optimal = find_throughput_optimal(results)
        if optimal:
            print(f"\n3. THROUGHPUT-OPTIMAL batch size: {optimal['batch_size']}")
            print(f"   Throughput: {optimal['throughput']:.1f} samples/sec")
            print(f"   Per-batch time: {optimal['time_ms']:.1f} ms")

    # Power of 2 analysis
    print(f"\n{'='*60}")
    print("BATCH SIZE ANALYSIS: Powers of 2 vs. Arbitrary Values")
    print(f"{'='*60}")

    # Check nearby power-of-2 values
    po2_candidates = [2**i for i in range(4, 10)]  # 16, 32, 64, 128, 256, 512
    print(f"\nPower-of-2 candidates near max:")
    for p2 in po2_candidates:
        if p2 <= max_bs * 1.2:
            match = [r for r in results if r['batch_size'] == p2]
            if match:
                r = match[0]
                print(f"  {p2:<6} -> {r['throughput']:.1f} samp/s, {r['vram_mb']:.0f} MB VRAM")
            else:
                print(f"  {p2:<6} -> (not tested, but should fit)")

    # Recommendations
    print(f"\n{'='*60}")
    print("RECOMMENDATIONS")
    print(f"{'='*60}")

    # Find best power of 2 that fits with safety margin
    best_po2 = 2**int(np.log2(max_bs * args.safety_factor))
    if best_po2 < 8:
        best_po2 = 8

    print(f"""
For CONSERVATIVE training (maximum stability):
  --batch-size {safe_bs}

For BALANCED training (good speed + stability):
  --batch-size {best_po2}

For MAXIMUM SPEED (if you need faster training and can tolerate occasional OOM
  during validation with larger batches):
  --batch-size {max_bs}

IMPORTANT NOTES:
- Powers of 2 (16, 32, 64, 128, 256) are NOT strictly required but often give
  slightly better GPU utilization due to memory alignment. Values like 48, 96,
  192 are perfectly valid and may offer better VRAM utilization.
- BlockGCN is very lightweight. Your 16GB GPU can likely handle batch sizes
  of 256+ easily. The bottleneck will be training time, not memory.
- For your hardware, batch_size=64 (default) is conservative. Try 96 or 128
  for faster training with linear learning rate scaling.
""")

    print(f"{'='*60}")
    print("NEXT STEPS")
    print(f"{'='*60}")
    print("""
1. Use the recommended batch size above for your training runs.
2. Run lr_range_test.py to find the optimal learning rate:
      python lr_range_test.py --batch-size <YOUR_BS>
3. Run training_micro_experiments.py to tune epochs/steps:
      python training_micro_experiments.py --batch-size <YOUR_BS> --base-lr <LR_FROM_STEP2>
""")


if __name__ == '__main__':
    main()