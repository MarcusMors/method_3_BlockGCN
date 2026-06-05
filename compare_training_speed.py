#!/usr/bin/env python3
"""
BlockGCN Training Speed Comparison Tool
========================================
Compares training efficiency between Windows 11 and WSL Ubuntu
by analyzing log.txt, TensorBoard event files, and checkpoint patterns.

Usage:
    # Compare two directories (use forward slashes or double backslashes in Windows paths):
    python compare_training_speed.py --win-path "E:/pfc_methods/method_3_BlockGCN/work_dir/ntu60/xsub/joint" --wsl-path "/home/marcus/+projects/pfc/method_3_BlockGCN/work_dir/ntu60/xsub/vel"

    # Analyze a single directory:
    python compare_training_speed.py --path "E:/pfc_methods/method_3_BlockGCN/work_dir/ntu60/xsub/joint"

    # Specify output file:
    python compare_training_speed.py --win-path <path> --wsl-path <path> --output comparison_report.txt
"""

import os
import re
import sys
import glob
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from collections import OrderedDict

# Try to import tensorboard event reader
try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    HAS_TENSORBOARD = True
except ImportError:
    try:
        from tensorflow.python.summary.event_accumulator import EventAccumulator
        HAS_TENSORBOARD = True
    except ImportError:
        HAS_TENSORBOARD = False

# Try to import alternative tensorboard reader
try:
    import struct
    HAS_STRUCT = True
except ImportError:
    HAS_STRUCT = False


def print_section(title, char="="):
    """Print a formatted section header."""
    width = 70
    print(f"\n{char * width}")
    print(f" {title}")
    print(f"{char * width}\n")


def print_subsection(title):
    """Print a formatted subsection header."""
    print(f"\n  --- {title} ---")


def parse_checkpoint_info(directory):
    """
    Parse checkpoint filenames to extract epoch and step information.
    Format: runs-{epoch}-{step}.pt
    """
    ckpt_pattern = re.compile(r'runs-(\d+)-(\d+)\.pt$')
    checkpoints = []

    for f in os.listdir(directory):
        match = ckpt_pattern.match(f)
        if match:
            epoch = int(match.group(1))
            step = int(match.group(2))
            filepath = os.path.join(directory, f)
            mtime = os.path.getmtime(filepath)
            checkpoints.append({
                'file': f,
                'epoch': epoch,
                'step': step,
                'mtime': mtime,
                'datetime': datetime.fromtimestamp(mtime)
            })

    checkpoints.sort(key=lambda x: x['epoch'])
    return checkpoints


def parse_log_file(log_path):
    """
    Parse log.txt to extract timing and performance information.
    Handles common BlockGCN log formats.
    """
    if not os.path.exists(log_path):
        return None

    epochs = []
    train_times = []
    eval_times = []
    losses = []
    accuracies = []
    lr_values = []

    # Common patterns in training logs
    epoch_pattern = re.compile(r'(?:epoch|Epoch)\s*[:=]?\s*(\d+)', re.IGNORECASE)
    time_pattern = re.compile(r'(?:time|Time)\s*[:=]?\s*(\d+\.?\d*)\s*(?:s|sec|seconds)?', re.IGNORECASE)
    loss_pattern = re.compile(r'(?:loss|Loss)\s*[:=]?\s*(\d+\.?\d*)', re.IGNORECASE)
    acc_pattern = re.compile(r'(?:acc|accuracy|Accuracy|top1)\s*[:=]?\s*(\d+\.?\d*)', re.IGNORECASE)
    lr_pattern = re.compile(r'(?:lr|LR|learning rate)\s*[:=]?\s*(\d+\.?\d*(?:e-?\d+)?)', re.IGNORECASE)

    # Specific patterns for BlockGCN/torchlight style logs
    blockgcn_epoch_pattern = re.compile(r'(?:^|\s)(\d+)/\d+\s+\w+.*?(\d+\.\d+)\s*it/s')
    blockgcn_time_pattern = re.compile(r'(?:eta|ETA|time remaining|elapsed)\s*[:=]?\s*(\d+:\d+:\d+|\d+\.?\d*)', re.IGNORECASE)

    current_epoch = None

    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()

    for line_num, line in enumerate(lines, 1):
        line = line.strip()
        if not line:
            continue

        # Try to extract epoch info
        epoch_match = epoch_pattern.search(line)
        if epoch_match:
            current_epoch = int(epoch_match.group(1))

        # Extract time information
        time_match = time_pattern.search(line)
        if time_match and current_epoch:
            try:
                t = float(time_match.group(1))
                if t > 0 and t < 86400:  # reasonable range: 0 to 24 hours
                    train_times.append({'epoch': current_epoch, 'time': t, 'line': line_num})
            except ValueError:
                pass

        # Extract loss
        loss_match = loss_pattern.search(line)
        if loss_match and current_epoch:
            try:
                loss_val = float(loss_match.group(1))
                if 0 <= loss_val < 100:  # reasonable loss range
                    losses.append({'epoch': current_epoch, 'loss': loss_val, 'line': line_num})
            except ValueError:
                pass

        # Extract accuracy
        acc_match = acc_pattern.search(line)
        if acc_match and current_epoch:
            try:
                acc_val = float(acc_match.group(1))
                if 0 <= acc_val <= 100:  # accuracy percentage
                    accuracies.append({'epoch': current_epoch, 'accuracy': acc_val, 'line': line_num})
            except ValueError:
                pass

        # Extract learning rate
        lr_match = lr_pattern.search(line)
        if lr_match and current_epoch:
            try:
                lr_val = float(lr_match.group(1))
                lr_values.append({'epoch': current_epoch, 'lr': lr_val, 'line': line_num})
            except ValueError:
                pass

    # Extract timing from log timestamps if present
    log_timestamps = extract_log_timestamps(lines)

    return {
        'epochs_data': epochs,
        'train_times': train_times,
        'losses': losses,
        'accuracies': accuracies,
        'lr_values': lr_values,
        'log_timestamps': log_timestamps,
        'total_lines': len(lines),
        'raw_lines': lines
    }


def extract_log_timestamps(lines):
    """Extract timestamp information from log file lines."""
    timestamps = []
    # Common timestamp patterns
    ts_patterns = [
        re.compile(r'^(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})'),  # 2024-01-01 12:00:00
        re.compile(r'^(\d{2}/\d{2}/\d{4}\s+\d{2}:\d{2}:\d{2})'),   # 01/01/2024 12:00:00
        re.compile(r'^(\w{3}\s+\w{3}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}\s+\d{4})'),  # Mon Jan 1 12:00:00 2024
        re.compile(r'\[(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})\]'),  # [2024-01-01 12:00:00]
    ]

    for line_num, line in enumerate(lines, 1):
        for pattern in ts_patterns:
            match = pattern.search(line)
            if match:
                ts_str = match.group(1)
                try:
                    for fmt in ['%Y-%m-%d %H:%M:%S', '%m/%d/%Y %H:%M:%S', '%a %b %d %H:%M:%S %Y']:
                        try:
                            dt = datetime.strptime(ts_str, fmt)
                            timestamps.append({'line': line_num, 'timestamp': dt})
                            break
                        except ValueError:
                            continue
                except Exception:
                    pass
                break

    return timestamps


def read_tensorboard_events_simple(event_file):
    """
    Read TensorBoard event file without tensorflow dependency.
    Uses a basic parser for event files.
    """
    events_data = []

    if not HAS_STRUCT:
        return events_data

    try:
        from tensorboard.backend.event_processing.event_file_loader import RawEventFileLoader
        from tensorboard.compat.proto import event_pb2

        loader = RawEventFileLoader(event_file)
        for record in loader.Load():
            event = event_pb2.Event.FromString(record)
            if event.wall_time > 0:
                events_data.append({
                    'wall_time': event.wall_time,
                    'step': event.step,
                    'datetime': datetime.fromtimestamp(event.wall_time)
                })
    except Exception:
        # Fallback: try to extract wall_time using regex on binary file
        try:
            with open(event_file, 'rb') as f:
                content = f.read()
                # Look for wall_time doubles (8-byte floats)
                # This is a heuristic approach
                import struct
                pos = 0
                while pos < len(content) - 16:
                    try:
                        # Try reading a double at current position
                        val = struct.unpack('<d', content[pos:pos+8])[0]
                        # Check if it looks like a Unix timestamp (2000-2030 range)
                        if 946684800 < val < 1893456000:  # 2000-2030
                            step_val = struct.unpack('<q', content[pos+8:pos+16])[0]
                            if 0 <= step_val < 10000000:  # reasonable step range
                                events_data.append({
                                    'wall_time': val,
                                    'step': step_val,
                                    'datetime': datetime.fromtimestamp(val)
                                })
                        pos += 1
                    except Exception:
                        pos += 1
        except Exception:
            pass

    return events_data


def read_tensorboard_events_tf(event_file):
    """Read TensorBoard events using tensorflow's EventAccumulator."""
    events_data = []
    if not HAS_TENSORBOARD:
        return events_data

    try:
        ea = EventAccumulator(event_file)
        ea.Reload()

        # Get all scalar tags
        tags = ea.Tags().get('scalars', [])

        for tag in tags:
            events = ea.Scalars(tag)
            for e in events:
                events_data.append({
                    'wall_time': e.wall_time,
                    'step': e.step,
                    'value': e.value,
                    'tag': tag,
                    'datetime': datetime.fromtimestamp(e.wall_time)
                })
    except Exception as e:
        print(f"    Warning: Could not read TensorBoard events: {e}")

    return events_data


def find_tensorboard_events(directory):
    """Find all TensorBoard event files in a directory."""
    event_files = []

    # Look in runs/ subdirectory
    runs_dir = os.path.join(directory, 'runs')
    if os.path.isdir(runs_dir):
        for root, dirs, files in os.walk(runs_dir):
            for f in files:
                if f.startswith('events.out.tfevents'):
                    event_files.append(os.path.join(root, f))

    # Also check directly in the directory
    for f in os.listdir(directory):
        if f.startswith('events.out.tfevents'):
            event_files.append(os.path.join(directory, f))

    return event_files


def analyze_tensorboard_events(directory):
    """Analyze all TensorBoard event files for timing information."""
    event_files = find_tensorboard_events(directory)

    if not event_files:
        return None

    all_events = []
    for ef in event_files:
        # Try both methods
        events = read_tensorboard_events_tf(ef)
        if not events:
            events = read_tensorboard_events_simple(ef)

        rel_path = os.path.relpath(ef, directory)
        for e in events:
            e['source_file'] = rel_path
        all_events.extend(events)

    if not all_events:
        return None

    # Sort by wall_time
    all_events.sort(key=lambda x: x['wall_time'])

    # Calculate timing statistics
    wall_times = [e['wall_time'] for e in all_events]
    steps = [e.get('step', 0) for e in all_events]

    total_duration = wall_times[-1] - wall_times[0] if len(wall_times) > 1 else 0
    avg_step_time = total_duration / max(steps[-1] - steps[0], 1) if len(steps) > 1 else 0

    return {
        'events': all_events,
        'event_files': event_files,
        'start_time': datetime.fromtimestamp(wall_times[0]),
        'end_time': datetime.fromtimestamp(wall_times[-1]),
        'total_duration_seconds': total_duration,
        'total_duration_formatted': format_duration(total_duration),
        'num_events': len(all_events),
        'min_step': min(steps) if steps else 0,
        'max_step': max(steps) if steps else 0,
        'avg_step_time': avg_step_time
    }


def format_duration(seconds):
    """Format seconds into human-readable duration."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    elif seconds < 86400:
        return f"{seconds/3600:.2f}h"
    else:
        return f"{seconds/86400:.2f}d"


def calculate_epoch_timing(checkpoints):
    """Calculate timing between consecutive checkpoints."""
    if len(checkpoints) < 2:
        return []

    epoch_times = []
    for i in range(1, len(checkpoints)):
        prev = checkpoints[i-1]
        curr = checkpoints[i]
        time_diff = curr['mtime'] - prev['mtime']
        step_diff = curr['step'] - prev['step']
        epoch_diff = curr['epoch'] - prev['epoch']

        if epoch_diff > 0 and time_diff > 0:
            epoch_times.append({
                'from_epoch': prev['epoch'],
                'to_epoch': curr['epoch'],
                'epoch_diff': epoch_diff,
                'time_seconds': time_diff,
                'time_formatted': format_duration(time_diff),
                'step_diff': step_diff,
                'time_per_epoch': time_diff / epoch_diff,
                'steps_per_second': step_diff / time_diff if time_diff > 0 else 0
            })

    return epoch_times


def analyze_training_run(directory, label):
    """
    Comprehensive analysis of a single training run directory.
    Returns a dictionary with all extracted metrics.
    """
    results = {
        'label': label,
        'directory': directory,
        'exists': os.path.isdir(directory),
        'checkpoints': [],
        'checkpoint_timing': [],
        'log_data': None,
        'tensorboard_data': None,
        'summary': {}
    }

    if not results['exists']:
        return results

    print(f"\nAnalyzing: {label}")
    print(f"  Path: {directory}")

    # 1. Analyze checkpoints
    print_subsection("Checkpoints")
    checkpoints = parse_checkpoint_info(directory)
    results['checkpoints'] = checkpoints

    if checkpoints:
        print(f"    Found {len(checkpoints)} checkpoints")
        print(f"    Epoch range: {checkpoints[0]['epoch']} -> {checkpoints[-1]['epoch']}")
        print(f"    Step range: {checkpoints[0]['step']} -> {checkpoints[-1]['step']}")

        # Calculate timing between checkpoints
        ckpt_timing = calculate_epoch_timing(checkpoints)
        results['checkpoint_timing'] = ckpt_timing

        if ckpt_timing:
            avg_time_per_epoch = sum(t['time_per_epoch'] for t in ckpt_timing) / len(ckpt_timing)
            avg_steps_per_sec = sum(t['steps_per_second'] for t in ckpt_timing) / len(ckpt_timing)
            print(f"    Avg time per epoch: {format_duration(avg_time_per_epoch)}")
            print(f"    Avg steps/second: {avg_steps_per_sec:.2f}")
    else:
        print("    No checkpoints found")

    # 2. Parse log file
    print_subsection("Log File (log.txt)")
    log_path = os.path.join(directory, 'log.txt')
    log_data = parse_log_file(log_path)
    results['log_data'] = log_data

    if log_data:
        print(f"    Log lines: {log_data['total_lines']}")
        if log_data['log_timestamps']:
            ts_list = log_data['log_timestamps']
            print(f"    Log timestamps found: {len(ts_list)}")
            if len(ts_list) >= 2:
                log_duration = (ts_list[-1]['timestamp'] - ts_list[0]['timestamp']).total_seconds()
                print(f"    Log time span: {format_duration(log_duration)}")

        if log_data['train_times']:
            avg_time = sum(t['time'] for t in log_data['train_times']) / len(log_data['train_times'])
            print(f"    Avg train time/epoch (from logs): {format_duration(avg_time)}")

        if log_data['losses']:
            print(f"    Loss entries: {len(log_data['losses'])}")

        if log_data['accuracies']:
            print(f"    Accuracy entries: {len(log_data['accuracies'])}")
    else:
        print("    No log.txt found")

    # 3. Parse TensorBoard events
    print_subsection("TensorBoard Events")
    tb_data = analyze_tensorboard_events(directory)
    results['tensorboard_data'] = tb_data

    if tb_data:
        print(f"    Event files: {len(tb_data['event_files'])}")
        print(f"    Total events: {tb_data['num_events']}")
        print(f"    Start: {tb_data['start_time']}")
        print(f"    End: {tb_data['end_time']}")
        print(f"    Total duration: {tb_data['total_duration_formatted']}")
        print(f"    Step range: {tb_data['min_step']} -> {tb_data['max_step']}")
        if tb_data['avg_step_time'] > 0:
            print(f"    Avg time/step: {format_duration(tb_data['avg_step_time'])}")
    else:
        print("    No TensorBoard events found (install tensorboard for more details)")

    # 4. Compute summary statistics
    results['summary'] = compute_summary(results)

    return results


def compute_summary(results):
    """Compute summary statistics from all available data sources."""
    summary = {
        'total_epochs': 0,
        'total_steps': 0,
        'total_duration_seconds': 0,
        'time_per_epoch_seconds': 0,
        'time_per_step_seconds': 0,
        'epochs_per_hour': 0,
        'steps_per_second': 0,
        'reliability': 'unknown'  # How confident we are in the data
    }

    checkpoints = results['checkpoints']
    ckpt_timing = results['checkpoint_timing']
    tb_data = results['tensorboard_data']
    log_data = results['log_data']

    # Best source: TensorBoard events
    if tb_data and tb_data['total_duration_seconds'] > 0:
        summary['total_duration_seconds'] = tb_data['total_duration_seconds']
        summary['total_steps'] = tb_data['max_step']

        if checkpoints:
            summary['total_epochs'] = checkpoints[-1]['epoch']

        if summary['total_epochs'] > 0:
            summary['time_per_epoch_seconds'] = summary['total_duration_seconds'] / summary['total_epochs']
            summary['epochs_per_hour'] = 3600 / summary['time_per_epoch_seconds'] if summary['time_per_epoch_seconds'] > 0 else 0

        if summary['total_steps'] > 0:
            summary['time_per_step_seconds'] = summary['total_duration_seconds'] / summary['total_steps']
            summary['steps_per_second'] = 1 / summary['time_per_step_seconds'] if summary['time_per_step_seconds'] > 0 else 0

        summary['reliability'] = 'high (tensorboard)'

    # Second best: Checkpoint file timestamps
    elif ckpt_timing:
        avg_time_per_epoch = sum(t['time_per_epoch'] for t in ckpt_timing) / len(ckpt_timing)
        avg_steps_per_sec = sum(t['steps_per_second'] for t in ckpt_timing) / len(ckpt_timing)

        if checkpoints:
            summary['total_epochs'] = checkpoints[-1]['epoch']
            summary['total_steps'] = checkpoints[-1]['step']

        summary['time_per_epoch_seconds'] = avg_time_per_epoch
        summary['epochs_per_hour'] = 3600 / avg_time_per_epoch if avg_time_per_epoch > 0 else 0
        summary['steps_per_second'] = avg_steps_per_sec

        if summary['total_epochs'] > 0:
            summary['total_duration_seconds'] = avg_time_per_epoch * summary['total_epochs']

        if summary['total_steps'] > 0:
            summary['time_per_step_seconds'] = 1 / avg_steps_per_sec if avg_steps_per_sec > 0 else 0

        summary['reliability'] = 'medium (checkpoints)'

    # Third: Log file timestamps
    elif log_data and log_data['log_timestamps'] and len(log_data['log_timestamps']) >= 2:
        ts_list = log_data['log_timestamps']
        duration = (ts_list[-1]['timestamp'] - ts_list[0]['timestamp']).total_seconds()
        summary['total_duration_seconds'] = duration

        if checkpoints:
            summary['total_epochs'] = checkpoints[-1]['epoch']

        if summary['total_epochs'] > 0:
            summary['time_per_epoch_seconds'] = duration / summary['total_epochs']
            summary['epochs_per_hour'] = 3600 / summary['time_per_epoch_seconds']

        summary['reliability'] = 'low (log timestamps)'

    # Fallback: Just checkpoint count
    elif checkpoints:
        summary['total_epochs'] = checkpoints[-1]['epoch']
        summary['total_steps'] = checkpoints[-1]['step']
        summary['reliability'] = 'very low (file count only)'

    return summary


def print_comparison_table(results_list):
    """Print a formatted comparison table of all training runs."""
    print_section("COMPARISON SUMMARY", "=")

    # Header
    col_widths = [20, 15, 15, 18, 18, 15, 15, 12]
    headers = ['Environment', 'Epochs', 'Steps', 'Total Time', 'Time/Epoch', 'Epochs/Hour', 'Steps/Sec', 'Reliability']

    header_line = " | ".join(h.ljust(w) for h, w in zip(headers, col_widths))
    print(header_line)
    print("-" * len(header_line))

    for r in results_list:
        s = r['summary']
        label = r['label'][:col_widths[0]].ljust(col_widths[0])
        epochs = str(s.get('total_epochs', 0)).rjust(col_widths[1]-1).ljust(col_widths[1])
        steps = str(s.get('total_steps', 0)).rjust(col_widths[2]-1).ljust(col_widths[2])
        total_time = format_duration(s.get('total_duration_seconds', 0)).rjust(col_widths[3]-1).ljust(col_widths[3])
        time_epoch = format_duration(s.get('time_per_epoch_seconds', 0)).rjust(col_widths[4]-1).ljust(col_widths[4])
        epochs_hour = f"{s.get('epochs_per_hour', 0):.1f}".rjust(col_widths[5]-1).ljust(col_widths[5])
        steps_sec = f"{s.get('steps_per_second', 0):.2f}".rjust(col_widths[6]-1).ljust(col_widths[6])
        reliability = s.get('reliability', 'unknown')[:col_widths[7]].ljust(col_widths[7])

        row = f"{label} | {epochs} | {steps} | {total_time} | {time_epoch} | {epochs_hour} | {steps_sec} | {reliability}"
        print(row)

    print()

    # Determine winner if we have at least 2 results
    valid_results = [r for r in results_list if r['summary'].get('time_per_epoch_seconds', 0) > 0]

    if len(valid_results) >= 2:
        # Sort by time per epoch (fastest first)
        sorted_results = sorted(valid_results, key=lambda x: x['summary']['time_per_epoch_seconds'])

        fastest = sorted_results[0]
        slowest = sorted_results[-1]

        speedup = slowest['summary']['time_per_epoch_seconds'] / fastest['summary']['time_per_epoch_seconds']
        time_saved_per_epoch = slowest['summary']['time_per_epoch_seconds'] - fastest['summary']['time_per_epoch_seconds']

        print("=" * 70)
        print(f"  WINNER: {fastest['label']} is faster!")
        print(f"  Speedup: {speedup:.2f}x faster than {slowest['label']}")
        print(f"  Time saved per epoch: {format_duration(time_saved_per_epoch)}")

        if fastest['summary'].get('total_epochs', 0) > 0:
            total_time_saved = time_saved_per_epoch * fastest['summary']['total_epochs']
            print(f"  Estimated total time saved: {format_duration(total_time_saved)}")
        print("=" * 70)


def generate_report(results_list, output_file=None):
    """Generate a detailed text report and optionally save to file."""
    lines = []
    lines.append("=" * 78)
    lines.append(" " * 20 + "BlockGCN Training Speed Comparison Report")
    lines.append("=" * 78)
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    # Summary comparison
    lines.append("-" * 78)
    lines.append("SUMMARY COMPARISON")
    lines.append("-" * 78)

    for r in results_list:
        lines.append(f"\n  Environment: {r['label']}")
        lines.append(f"  Path: {r['directory']}")
        lines.append(f"  Exists: {r['exists']}")

        s = r['summary']
        lines.append(f"  Total Epochs: {s.get('total_epochs', 'N/A')}")
        lines.append(f"  Total Steps: {s.get('total_steps', 'N/A')}")
        lines.append(f"  Total Duration: {format_duration(s.get('total_duration_seconds', 0))}")
        lines.append(f"  Time per Epoch: {format_duration(s.get('time_per_epoch_seconds', 0))}")
        lines.append(f"  Epochs per Hour: {s.get('epochs_per_hour', 0):.2f}")
        lines.append(f"  Steps per Second: {s.get('steps_per_second', 0):.2f}")
        lines.append(f"  Data Reliability: {s.get('reliability', 'unknown')}")

    # Detailed breakdown per environment
    for r in results_list:
        lines.append("")
        lines.append("-" * 78)
        lines.append(f"DETAILED ANALYSIS: {r['label']}")
        lines.append("-" * 78)
        lines.append(f"  Directory: {r['directory']}")

        # Checkpoints
        lines.append(f"\n  Checkpoints: {len(r['checkpoints'])}")
        if r['checkpoints']:
            lines.append(f"    First: {r['checkpoints'][0]['file']} "
                        f"(epoch {r['checkpoints'][0]['epoch']}, "
                        f"saved {r['checkpoints'][0]['datetime'].strftime('%Y-%m-%d %H:%M:%S')})")
            lines.append(f"    Last:  {r['checkpoints'][-1]['file']} "
                        f"(epoch {r['checkpoints'][-1]['epoch']}, "
                        f"saved {r['checkpoints'][-1]['datetime'].strftime('%Y-%m-%d %H:%M:%S')})")

        # Checkpoint timing
        if r['checkpoint_timing']:
            lines.append(f"\n  Checkpoint Timing Samples:")
            for t in r['checkpoint_timing'][:5]:  # Show first 5
                lines.append(f"    Epoch {t['from_epoch']} -> {t['to_epoch']}: "
                            f"{t['time_formatted']} ({t['steps_per_second']:.2f} steps/s)")
            if len(r['checkpoint_timing']) > 5:
                lines.append(f"    ... and {len(r['checkpoint_timing'])-5} more")

        # Log data
        if r['log_data']:
            ld = r['log_data']
            lines.append(f"\n  Log File Analysis:")
            lines.append(f"    Total lines: {ld['total_lines']}")
            lines.append(f"    Timestamps found: {len(ld['log_timestamps'])}")
            lines.append(f"    Train time entries: {len(ld['train_times'])}")
            lines.append(f"    Loss entries: {len(ld['losses'])}")
            lines.append(f"    Accuracy entries: {len(ld['accuracies'])}")

        # TensorBoard
        if r['tensorboard_data']:
            tb = r['tensorboard_data']
            lines.append(f"\n  TensorBoard Events:")
            lines.append(f"    Event files: {len(tb['event_files'])}")
            lines.append(f"    Total events: {tb['num_events']}")
            lines.append(f"    Duration: {tb['total_duration_formatted']}")

    # Recommendation
    valid_results = [r for r in results_list if r['summary'].get('time_per_epoch_seconds', 0) > 0]
    if len(valid_results) >= 2:
        sorted_results = sorted(valid_results, key=lambda x: x['summary']['time_per_epoch_seconds'])
        fastest = sorted_results[0]
        slowest = sorted_results[-1]
        speedup = slowest['summary']['time_per_epoch_seconds'] / fastest['summary']['time_per_epoch_seconds']

        lines.append("")
        lines.append("=" * 78)
        lines.append("RECOMMENDATION")
        lines.append("=" * 78)
        lines.append(f"  {fastest['label']} is the faster training environment.")
        lines.append(f"  It is {speedup:.2f}x faster than {slowest['label']}.")
        lines.append(f"  Recommendation: Train on {fastest['label']} for better efficiency.")
        lines.append("=" * 78)

    report_text = "\n".join(lines)

    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report_text)
        print(f"\nReport saved to: {output_file}")

    return report_text


def main():
    parser = argparse.ArgumentParser(
        description='Compare BlockGCN training speed between environments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare Windows vs WSL:
  python compare_training_speed.py
    --win-path "E:/pfc_methods/method_3_BlockGCN/work_dir/ntu60/xsub/joint"
    --wsl-path "/home/marcus/+projects/pfc/method_3_BlockGCN/work_dir/ntu60/xsub/vel"

  # Analyze single directory:
  python compare_training_speed.py --path "C:/path/to/training/output"

  # Save report to file:
  python compare_training_speed.py --win-path <p1> --wsl-path <p2> --output report.txt
        """
    )

    parser.add_argument('--win-path', type=str, help='Path to Windows training output directory')
    parser.add_argument('--wsl-path', type=str, help='Path to WSL training output directory')
    parser.add_argument('--path', type=str, help='Analyze a single training directory')
    parser.add_argument('--output', '-o', type=str, default=None, help='Save report to file')
    parser.add_argument('--labels', type=str, nargs='+', help='Custom labels for each path')

    args = parser.parse_args()

    # Determine which paths to analyze
    paths_to_analyze = []

    if args.path:
        label = args.labels[0] if args.labels else 'Single Run'
        paths_to_analyze.append((args.path, label))

    if args.win_path:
        label = args.labels[len(paths_to_analyze)] if args.labels and len(args.labels) > len(paths_to_analyze) else 'Windows 11'
        paths_to_analyze.append((args.win_path, label))

    if args.wsl_path:
        label = args.labels[len(paths_to_analyze)] if args.labels and len(args.labels) > len(paths_to_analyze) else 'WSL Ubuntu'
        paths_to_analyze.append((args.wsl_path, label))

    if not paths_to_analyze:
        print("Error: No paths specified. Use --path, --win-path, or --wsl-path.")
        parser.print_help()
        sys.exit(1)

    # Print banner
    print("=" * 78)
    print(" " * 15 + "BlockGCN Training Speed Comparison Tool")
    print("=" * 78)
    print(f"Analysis started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Platform: {sys.platform}")
    print(f"TensorBoard available: {HAS_TENSORBOARD}")
    print(f"Directories to analyze: {len(paths_to_analyze)}")

    # Analyze each directory
    results = []
    for path, label in paths_to_analyze:
        result = analyze_training_run(path, label)
        results.append(result)

    # Print comparison
    print_comparison_table(results)

    # Generate and print report
    print_section("DETAILED REPORT", "=")
    report = generate_report(results, output_file=args.output)
    print(report)

    # Return exit code based on comparison
    valid_results = [r for r in results if r['summary'].get('time_per_epoch_seconds', 0) > 0]
    if len(valid_results) >= 2:
        sorted_results = sorted(valid_results, key=lambda x: x['summary']['time_per_epoch_seconds'])
        fastest = sorted_results[0]
        print(f"\n{'='*78}")
        print(f"  CONCLUSION: Use {fastest['label']} for faster training.")
        print(f"{'='*78}\n")

    print(f"Analysis completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == '__main__':
    main()
