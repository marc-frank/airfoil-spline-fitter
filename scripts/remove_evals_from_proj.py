#!/usr/bin/env python3
"""
Remove Evaluation Numbers from Project Files
=============================================

Migrates old .asfproj files that have per-evaluation loss_history records
to the new iteration-only format. The app reads both formats; this only
makes old files smaller.

This script:
1. Loads the old project file (with 'eval' field in loss_history)
2. Keeps only unique iterations (deduplicates evaluations within same iteration)
3. Removes the 'eval' field from each record
4. Saves the cleaned project file

Usage:
    python scripts/remove_evals_from_proj.py <input.asfproj> [output.asfproj]
    
If output path is not specified, overwrites the input file (with .bak backup).

Example:
    python scripts/remove_evals_from_proj.py my_project.asfproj
    python scripts/remove_evals_from_proj.py my_project.asfproj cleaned_project.asfproj
"""

import sys
import os
import json
import shutil
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Tuple


def migrate_loss_history(loss_history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert per-evaluation loss history to per-iteration.
    
    For each iteration, keeps only the last (best) evaluation's error value.
    Removes the 'eval' field entirely.
    
    Args:
        loss_history: List of records with 'eval', 'iter', 'error', 'time'
        
    Returns:
        List of records with only 'iter', 'error', 'time' (one per iteration)
    """
    if not loss_history:
        return []
    
    # Group by iteration, keep last entry for each iteration
    by_iteration: Dict[int, Dict[str, Any]] = {}
    
    for record in loss_history:
        iter_num = record.get("iter", 0)
        # Keep the last record for each iteration (best error at end of iteration)
        by_iteration[iter_num] = {
            "iter": iter_num,
            "error": record.get("error", 0.0),
            "time": record.get("time", 0.0),
        }
    
    # Sort by iteration number
    return [by_iteration[k] for k in sorted(by_iteration.keys())]


def migrate_project(data: Dict[str, Any]) -> Tuple[Dict[str, Any], int, int]:
    """
    Migrate a project's loss history from per-evaluation to per-iteration.
    
    Args:
        data: Parsed JSON project data
        
    Returns:
        (migrated project data, records before, records after)
    """
    runs = data.get("optimization_runs", [])
    original_records = 0
    migrated_records = 0
    
    for run in runs:
        old_history = run.get("loss_history", [])
        original_records += len(old_history)
        
        # Check if already migrated (no 'eval' field)
        if old_history and "eval" not in old_history[0]:
            migrated_records += len(old_history)
            continue
        
        # Migrate to iteration-only
        new_history = migrate_loss_history(old_history)
        run["loss_history"] = new_history
        migrated_records += len(new_history)
    
    return data, original_records, migrated_records


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    
    input_path = Path(sys.argv[1])
    
    if not input_path.exists():
        print(f"Error: File not found: {input_path}")
        sys.exit(1)
    
    if len(sys.argv) >= 3:
        output_path = Path(sys.argv[2])
    else:
        # Overwrite mode - create backup first
        output_path = input_path
        backup_path = input_path.with_suffix(input_path.suffix + ".bak")
        print(f"Creating backup: {backup_path}")
        shutil.copy2(input_path, backup_path)
    
    # Load project
    print(f"Loading: {input_path}")
    input_size = input_path.stat().st_size
    print(f"  Input size: {input_size / 1024 / 1024:.2f} MB")
    
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Migrate
    print("Migrating loss history from per-evaluation to per-iteration...")
    data, original_records, migrated_records = migrate_project(data)
    
    reduction = (1 - migrated_records / original_records) * 100 if original_records > 0 else 0
    print(f"  Original records: {original_records:,}")
    print(f"  Migrated records: {migrated_records:,}")
    print(f"  Reduction: {reduction:.1f}%")
    
    # Save
    print(f"Saving: {output_path}")
    # Write a temporary file and swap it in, so a failure leaves the target intact
    text = json.dumps(data, indent=2, ensure_ascii=False)
    fd, temp_path = tempfile.mkstemp(prefix=".", suffix=".tmp", dir=output_path.resolve().parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            f.write(text)
        os.replace(temp_path, output_path)
    except BaseException:
        os.remove(temp_path)
        raise
    
    output_size = output_path.stat().st_size
    size_reduction = (1 - output_size / input_size) * 100 if input_size > 0 else 0
    print(f"  Output size: {output_size / 1024 / 1024:.2f} MB")
    print(f"  Size reduction: {size_reduction:.1f}%")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
