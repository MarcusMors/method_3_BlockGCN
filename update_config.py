#!/usr/bin/env python
"""
BlockGCN Config Updater
=======================
Updates batch_size, test_batch_size, num_epoch, and base_lr across all YAML config
files in the config/ directory tree.

Usage (any order of arguments):
    python update_config.py --batch_size 128
    python update_config.py --batch_size 128 --test_batch_size 128
    python update_config.py --batch_size 128 --test_batch_size 128 --num_epoch 140
    python update_config.py --num_epoch 200 --batch_size 96
    python update_config.py --num_epoch 200 --batch_size 96 --base_lr 4.37e02
    python update_config.py --num_epoch 200 --batch_size 96 --base_lr 0.05
    python update_config.py --num_epoch 200 --batch_size 96 --base_lr 0.003

Only modifies files that actually contain the key being changed.
"""

import argparse
import re
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Update BlockGCN YAML config files for your hardware"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Training batch size (e.g. 64, 96, 128)"
    )
    parser.add_argument(
        "--test_batch_size",
        type=int,
        default=None,
        help="Test/validation batch size (e.g. 64, 128)"
    )
    parser.add_argument(
        "--num_epoch",
        type=int,
        default=None,
        help="Number of training epochs (e.g. 140, 200)"
    )
    parser.add_argument(
        "--base_lr",
        type=float,
        default=None,
        help="Base learning rate (e.g. 0.05, 0.003, 4.37e02, 7.59e02)"
    )
    parser.add_argument(
        "--config-dir",
        type=str,
        default="config",
        help="Path to config directory (default: config)"
    )
    return parser.parse_args()


def update_yaml_file(filepath, updates):
    """
    Update specific keys in a YAML file.
    
    Args:
        filepath: Path to YAML file
        updates: Dict of {key: new_value}
    
    Returns:
        Tuple of (was_modified: bool, changes_made: list of str)
    """
    content = filepath.read_text(encoding="utf-8")
    original = content
    changes = []

    for key, new_value in updates.items():
        # Match lines like "batch_size: 64" with flexible whitespace
        pattern = rf"^([ \t]*{re.escape(key)}[ \t]*:)[ \t]*\S+"
        replacement = rf"\1 {new_value}"
        
        new_content, count = re.subn(pattern, replacement, content, flags=re.MULTILINE)
        
        if count > 0:
            content = new_content
            changes.append(f"  {key}: -> {new_value}")

    if changes:
        filepath.write_text(content, encoding="utf-8")
        return True, changes
    return False, []


def main():
    args = parse_args()

    # Build updates dict from provided arguments
    updates = {}
    if args.batch_size is not None:
        updates["batch_size"] = args.batch_size
    if args.test_batch_size is not None:
        updates["test_batch_size"] = args.test_batch_size
    if args.num_epoch is not None:
        updates["num_epoch"] = args.num_epoch
    if args.base_lr is not None:
        updates["base_lr"] = args.base_lr

    if not updates:
        print("[ERROR] No parameters provided. Nothing to update.")
        print("\nUsage examples:")
        print("  python update_config.py --batch_size 128")
        print("  python update_config.py --batch_size 128 --test_batch_size 128")
        print("  python update_config.py --batch_size 128 --test_batch_size 128 --num_epoch 140")
        print("  python update_config.py --num_epoch 200 --batch_size 96 --base_lr 0.05")
        sys.exit(1)

    # Find config directory
    config_dir = Path(args.config_dir)
    if not config_dir.exists():
        script_dir = Path(__file__).parent
        config_dir = script_dir / args.config_dir
        if not config_dir.exists():
            print(f"[ERROR] Config directory not found: {args.config_dir}")
            sys.exit(1)

    print(f"Config directory: {config_dir.absolute()}")
    print(f"Updates to apply:")
    for k, v in updates.items():
        print(f"  {k} = {v}")
    print()

    # Find all YAML files
    yaml_files = sorted(config_dir.rglob("*.yaml"))
    print(f"Scanning {len(yaml_files)} YAML files...\n")

    modified_count = 0

    for yaml_file in yaml_files:
        modified, changes = update_yaml_file(yaml_file, updates)
        
        if modified:
            print(f"[MODIFIED] {yaml_file.relative_to(config_dir)}")
            for change in changes:
                print(change)
            modified_count += 1
        else:
            print(f"[SKIPPED]  {yaml_file.relative_to(config_dir)}")

    print(f"\n{'='*50}")
    print(f"Summary: {modified_count} of {len(yaml_files)} files modified")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()