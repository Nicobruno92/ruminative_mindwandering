#!/usr/bin/env python3
"""
Parallel batch processing script for PKL creation.
Divides the work into chunks for SLURM array jobs.

Usage:
    python batch_create_pkl_parallel.py --task-id <N> --n-jobs <TOTAL>
    
This script is designed to be called from a SLURM array job.
"""

import argparse
import sys
from pathlib import Path

# Import the batch processor
from batch_create_pkl_from_pipeline import (
    parse_elements_file, 
    process_element,
    DEFAULT_ELEMENTS_FILE
)


def get_chunk_for_task(elements: list, task_id: int, n_jobs: int) -> list:
    """
    Divide elements into chunks and return the chunk for this task.
    
    Parameters
    ----------
    elements : list
        List of (subject, task) tuples
    task_id : int
        Current task ID (1-indexed for SLURM)
    n_jobs : int
        Total number of parallel jobs
        
    Returns
    -------
    list
        Subset of elements for this task
    """
    total = len(elements)
    chunk_size = (total + n_jobs - 1) // n_jobs  # Ceiling division
    
    start_idx = (task_id - 1) * chunk_size
    end_idx = min(start_idx + chunk_size, total)
    
    chunk = elements[start_idx:end_idx]
    
    print(f"Task {task_id}/{n_jobs}: Processing elements {start_idx+1} to {end_idx} (total: {len(chunk)})")
    return chunk


def main():
    parser = argparse.ArgumentParser(description="Parallel PKL creation worker")
    parser.add_argument("--task-id", type=int, required=True,
                       help="SLURM array task ID (1-indexed)")
    parser.add_argument("--n-jobs", type=int, required=True,
                       help="Total number of parallel jobs")
    parser.add_argument("--elements-file", type=Path, default=DEFAULT_ELEMENTS_FILE,
                       help="Path to elements file")
    parser.add_argument("--desc", type=str, default="both",
                       choices=["evoked", "state", "both"],
                       help="Description type to process")
    parser.add_argument("--force", action="store_true",
                       help="Overwrite existing PKL files")
    
    args = parser.parse_args()
    
    # Parse all elements
    print(f"Reading elements from: {args.elements_file}")
    elements = parse_elements_file(args.elements_file)
    print(f"Total elements: {len(elements)}")
    
    # Determine which desc types to process
    if args.desc == "both":
        desc_types = ["evoked", "state"]
    else:
        desc_types = [args.desc]
    
    # Expand elements to include desc types
    all_combinations = []
    for subject, task in elements:
        for desc in desc_types:
            all_combinations.append((subject, task, desc))
    
    print(f"Total combinations (element × desc): {len(all_combinations)}")
    
    # Get this task's chunk
    chunk = get_chunk_for_task(all_combinations, args.task_id, args.n_jobs)
    
    if not chunk:
        print("No elements assigned to this task")
        return 0
    
    # Process the chunk
    success_count = 0
    error_count = 0
    
    for i, (subject, task, desc) in enumerate(chunk, 1):
        print(f"\n{'='*60}")
        print(f"Processing {i}/{len(chunk)}: {subject} / {task} / {desc}")
        print(f"{'='*60}")
        
        try:
            success = process_element(subject, task, desc, 
                                     dry_run=False, force=args.force)
            if success:
                success_count += 1
            else:
                error_count += 1
                
        except Exception as e:
            print(f"ERROR: Failed to process {subject}/{task}/{desc}: {e}")
            error_count += 1
            continue
    
    # Summary
    print(f"\n{'='*60}")
    print(f"Task {args.task_id} Summary:")
    print(f"  Total processed: {len(chunk)}")
    print(f"  Success: {success_count}")
    print(f"  Errors: {error_count}")
    print(f"{'='*60}")
    
    return 0 if error_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
