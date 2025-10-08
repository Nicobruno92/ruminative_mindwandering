#!/usr/bin/env python3
"""
Batch process pipeline outputs to create PKL files.

This script processes all elements from the Junifer pipeline, reading:
1. Original FIF files from BIDS derivatives directory
2. Output H5 files from Junifer features directory
3. Generates PKL files for each element

Usage:
    python batch_create_pkl_from_pipeline.py [--elements-file ELEMENTS] [--desc DESC]
    
Arguments:
    --elements-file: Path to elements file (default: junifer_jobs/CYBERSART_features/elements)
    --desc: Description type to process (evoked or state, default: both)
    --output-dir: Output directory for PKL files (default: features/junifer/pkl)
    --dry-run: Print what would be processed without actually processing
"""

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

# Import the conversion function
from create_pkl_from_h5_fif import create_pkl_from_h5_fif


# Configuration
BIDS_ROOT = Path("/network/iss/cenir/analyse/meeg/CYBERSART/BIDS")
DERIVATIVES_DIR = BIDS_ROOT / "derivatives"
FEATURES_DIR = BIDS_ROOT / "features" / "junifer"
DEFAULT_ELEMENTS_FILE = Path("/network/iss/home/nicolas.bruno/Junifer/junifer_jobs/CYBERSART_features/elements")


def parse_elements_file(elements_file: Path) -> List[Tuple[str, str]]:
    """Parse elements file to get list of (subject, task) tuples."""
    elements = []
    
    with open(elements_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split(',')
            if len(parts) >= 2:
                subject = parts[0].strip()
                task = parts[1].strip()
                elements.append((subject, task))
            else:
                print(f"Warning: Skipping malformed line: {line}")
    
    return elements


def construct_paths(subject: str, task: str, desc: str) -> Tuple[Path, Path, Path]:
    """
    Construct file paths for FIF input, H5 input, and PKL output.
    
    Args:
        subject: Subject ID (e.g., 'sub-31')
        task: Task name (e.g., 'Sart4')
        desc: Description type ('evoked' or 'state')
    
    Returns:
        Tuple of (fif_path, h5_path, pkl_path)
    """
    # FIF input: derivatives/{subject}/eeg/{subject}_task-{task}_desc-{desc}_epo.fif
    fif_path = DERIVATIVES_DIR / subject / "eeg" / f"{subject}_task-{task}_desc-{desc}_epo.fif"
    
    # H5 input: features/junifer/markers.h5 (single file with element-specific data)
    # Note: The actual H5 storage structure stores data with element keys internally
    h5_path = FEATURES_DIR / "markers.h5"
    
    # PKL output: BIDS format - features/{subject}/eeg/junifer/{subject}_task-{task}_desc-{desc}_markers.pkl
    pkl_path = (BIDS_ROOT / "features" / subject / "eeg" / "junifer" / 
                f"{subject}_task-{task}_desc-{desc}_markers.pkl")
    
    return fif_path, h5_path, pkl_path


def process_element(subject: str, task: str, desc: str, dry_run: bool = False) -> bool:
    """
    Process a single element to create PKL file.
    
    Returns:
        True if successful, False otherwise
    """
    fif_path, h5_path, pkl_path = construct_paths(subject, task, desc)
    
    print(f"\n{'[DRY RUN] ' if dry_run else ''}Processing: {subject}, {task}, desc={desc}")
    print(f"  FIF: {fif_path}")
    print(f"  H5:  {h5_path}")
    print(f"  PKL: {pkl_path}")
    
    # Check if input files exist
    if not fif_path.exists():
        print(f"  ⚠️  FIF file not found: {fif_path}")
        return False
    
    if not h5_path.exists():
        print(f"  ⚠️  H5 file not found: {h5_path}")
        return False
    
    # Check if output already exists
    if pkl_path.exists():
        print(f"  ℹ️  PKL file already exists, skipping: {pkl_path}")
        return True
    
    if dry_run:
        print(f"  ✓  Would process this element")
        return True
    
    # Create output directory
    pkl_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Construct element dict for H5 reading
    # The element dict needs to match what was used during storage
    element = {
        'subject': subject,
        'task': task,
        'desc': desc
    }
    
    # Process the files
    try:
        create_pkl_from_h5_fif(str(h5_path), str(fif_path), str(pkl_path), element=element)
        print(f"  ✓  Successfully created PKL file")
        return True
    except Exception as e:
        print(f"  ❌  Error processing element: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main batch processing function."""
    parser = argparse.ArgumentParser(
        description="Batch process pipeline outputs to create PKL files"
    )
    parser.add_argument(
        "--elements-file",
        type=Path,
        default=DEFAULT_ELEMENTS_FILE,
        help="Path to elements file"
    )
    parser.add_argument(
        "--desc",
        choices=["evoked", "state", "both"],
        default="both",
        help="Description type to process"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory for PKL files (overrides default)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be processed without actually processing"
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of elements to process (for testing)"
    )
    
    args = parser.parse_args()
    
    # Override output directory if specified
    global FEATURES_DIR
    if args.output_dir:
        FEATURES_DIR = args.output_dir
    
    # Check if elements file exists
    if not args.elements_file.exists():
        print(f"Error: Elements file not found: {args.elements_file}")
        sys.exit(1)
    
    # Parse elements file
    print(f"Reading elements from: {args.elements_file}")
    elements = parse_elements_file(args.elements_file)
    print(f"Found {len(elements)} elements")
    
    # Determine which descriptions to process
    desc_types = ["evoked", "state"] if args.desc == "both" else [args.desc]
    
    # Apply limit if specified
    if args.limit:
        elements = elements[:args.limit]
        print(f"Limited to {len(elements)} elements for testing")
    
    # Process each element
    total = len(elements) * len(desc_types)
    success_count = 0
    skip_count = 0
    fail_count = 0
    
    print(f"\nProcessing {total} total element-desc combinations...")
    print(f"{'=' * 80}")
    
    for i, (subject, task) in enumerate(elements, 1):
        for desc in desc_types:
            result = process_element(subject, task, desc, dry_run=args.dry_run)
            if result:
                success_count += 1
            else:
                fail_count += 1
    
    # Summary
    print(f"\n{'=' * 80}")
    print(f"{'DRY RUN ' if args.dry_run else ''}Summary:")
    print(f"  Total elements: {len(elements)}")
    print(f"  Total combinations: {total}")
    print(f"  Successful: {success_count}")
    print(f"  Failed: {fail_count}")
    
    if args.dry_run:
        print(f"\nThis was a dry run. Run without --dry-run to actually process files.")
    
    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
