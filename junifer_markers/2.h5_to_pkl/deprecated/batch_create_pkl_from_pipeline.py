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

print("[DEBUG] Script starting - imports beginning", flush=True)

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

print("[DEBUG] Standard imports completed", flush=True)

# Import the conversion function
print("[DEBUG] Importing create_pkl_from_h5_fif...", flush=True)
from create_pkl_from_h5_fif import create_pkl_from_h5_fif
print("[DEBUG] create_pkl_from_h5_fif imported successfully", flush=True)


# Configuration
print("[DEBUG] Setting up configuration paths...", flush=True)
BIDS_ROOT = Path("/network/iss/cenir/analyse/meeg/CYBERSART/BIDS")
LOCAL_ROOT = Path("/network/iss/levy/analyze/valerocabre/analyse/nbruno/depressed_mindwandering")
DERIVATIVES_DIR = BIDS_ROOT / "derivatives"
H5_FEATURES_DIR = BIDS_ROOT / "features" / "junifer"
PKL_FEATURES_DIR = BIDS_ROOT / "features"
DEFAULT_ELEMENTS_FILE = LOCAL_ROOT / "junifer_markers" / "1.markers_h5_creation" / "elements"
print(f"[DEBUG] BIDS_ROOT: {BIDS_ROOT}", flush=True)
print(f"[DEBUG] H5_FEATURES_DIR: {H5_FEATURES_DIR}", flush=True)
print(f"[DEBUG] PKL_FEATURES_DIR: {PKL_FEATURES_DIR}", flush=True)
print(f"[DEBUG] DEFAULT_ELEMENTS_FILE: {DEFAULT_ELEMENTS_FILE}", flush=True)
print("[DEBUG] Configuration complete", flush=True)

def parse_elements_file(elements_file: Path) -> List[Tuple[str, str]]:
    """Parse elements file to get list of (subject, task) tuples."""
    print(f"[DEBUG] parse_elements_file called with: {elements_file}", flush=True)
    elements = []
    
    with open(elements_file, 'r') as f:
        print("[DEBUG] Elements file opened successfully", flush=True)
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            parts = line.split(',')
            if len(parts) >= 2:
                subject = parts[0].strip()
                task = parts[1].strip()
                elements.append((subject, task))
                if line_num <= 3:  # Show first 3 elements
                    print(f"[DEBUG] Parsed line {line_num}: {subject}, {task}", flush=True)
            else:
                print(f"Warning: Skipping malformed line {line_num}: {line}", flush=True)
    
    print(f"[DEBUG] Total elements parsed: {len(elements)}", flush=True)
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
    
    # H5 input: features/junifer/element_{subject}_{task}_{desc}_markers.h5
    h5_path = H5_FEATURES_DIR / f"element_{subject}_{task}_{desc}_markers.h5"
    
    # PKL output: BIDS format - features/{subject}/eeg/junifer/{subject}_task-{task}_desc-{desc}_markers.pkl
    pkl_path = PKL_FEATURES_DIR / subject / "eeg" / "junifer" / f"{subject}_task-{task}_desc-{desc}_markers.pkl"
    
    return fif_path, h5_path, pkl_path


def process_element(subject: str, task: str, desc: str, dry_run: bool = False,
                    force: bool = False) -> bool:
    """
    Process a single element to create PKL file.
    
    Parameters
    ----------
    subject : str
        Subject ID (e.g., 'sub-02')
    task : str
        Task name (e.g., 'Sart1')
    desc : str
        Description type ('evoked' or 'state')
    dry_run : bool
        If True, only print what would be done
    force : bool
        If True, overwrite existing PKL files
    
    Returns
    -------
    bool
        True if successful, False otherwise
    """
    
    print(f"[DEBUG] process_element called: {subject}, {task}, {desc}", flush=True)
    
    fif_path, h5_path, pkl_path = construct_paths(subject, task, desc)
    
    print(f"\n{'[DRY RUN] ' if dry_run else ''}Processing: {subject}, {task}, desc={desc}", flush=True)
    print(f"  FIF: {fif_path}", flush=True)
    print(f"  H5:  {h5_path}", flush=True)
    print(f"  PKL: {pkl_path}", flush=True)
    
    # Support both old and new Junifer H5 naming conventions.
    # Old pattern (original pipeline):
    #   element_sub-XX_SartY_state_markers.h5
    # New pattern (split configs, e.g. markers_state.h5):
    #   element_sub-XX_SartY_state_markers_state.h5
    alt_h5_path = H5_FEATURES_DIR / f"element_{subject}_{task}_{desc}_markers_{desc}.h5"
    if (not h5_path.exists()) and alt_h5_path.exists():
        print(f"[DEBUG] Primary H5 not found, using alternative H5 path", flush=True)
        print(f"  H5 (alt): {alt_h5_path}", flush=True)
        h5_path = alt_h5_path
    
    # Check if input files exist
    print(f"[DEBUG] Checking FIF file existence", flush=True)
    if not fif_path.exists():
        print(f"  ⚠️  FIF file not found: {fif_path}", flush=True)
        return False
    print(f"[DEBUG] FIF file exists", flush=True)
    
    print(f"[DEBUG] Checking H5 file existence", flush=True)
    if not h5_path.exists():
        print(f"  ⚠️  H5 file not found: {h5_path}", flush=True)
        return False
    print(f"[DEBUG] H5 file exists", flush=True)
    
    # Check if output already exists
    print(f"[DEBUG] Checking PKL file existence", flush=True)
    if pkl_path.exists() and not force:
        print(f"  ℹ️  PKL file already exists, skipping: {pkl_path}", flush=True)
        return True
    
    if pkl_path.exists() and force:
        print(f"  🔄  Overwriting existing PKL file", flush=True)
    
    if dry_run:
        print(f"  ✓  Would process this element", flush=True)
        return True
    
    # Create output directory
    print(f"[DEBUG] Creating output directory: {pkl_path.parent}", flush=True)
    pkl_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[DEBUG] Output directory ready", flush=True)
    
    # Construct element dict for H5 reading
    element = {
        'subject': subject,
        'task': task,
        'desc': desc
    }
    print(f"[DEBUG] Element dict: {element}", flush=True)
    
    # Process the files
    try:
        print(f"[DEBUG] Calling create_pkl_from_h5_fif...", flush=True)
        create_pkl_from_h5_fif(str(h5_path), str(fif_path), str(pkl_path), element=element)
        print(f"[DEBUG] create_pkl_from_h5_fif completed", flush=True)
        print(f"  ✓  Successfully created PKL file", flush=True)
        return True
    except Exception as e:
        print(f"  ❌  Error processing element: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main batch processing function."""
    
    print("[DEBUG] Starting main() function", flush=True)
    
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
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing PKL files"
    )
    
    print("[DEBUG] Parsing arguments", flush=True)
    args = parser.parse_args()
    print(f"[DEBUG] Arguments parsed: {args}", flush=True)
    
    # Override output directory if specified
    global PKL_FEATURES_DIR
    if args.output_dir:
        PKL_FEATURES_DIR = args.output_dir
        print(f"[DEBUG] Using custom output dir: {PKL_FEATURES_DIR}", flush=True)
    else:
        print(f"[DEBUG] Using default output dir: {PKL_FEATURES_DIR}", flush=True)
    
    # Check if elements file exists
    print(f"[DEBUG] Checking elements file: {args.elements_file}", flush=True)
    if not args.elements_file.exists():
        print(f"Error: Elements file not found: {args.elements_file}")
        sys.exit(1)
    print(f"[DEBUG] Elements file exists", flush=True)
    
    # Parse elements file
    print(f"[DEBUG] Reading elements from: {args.elements_file}", flush=True)
    elements = parse_elements_file(args.elements_file)
    print(f"[DEBUG] Found {len(elements)} elements", flush=True)
    if elements:
        print(f"[DEBUG] First element: {elements[0]}", flush=True)
    
    # Determine which descriptions to process
    desc_types = ["evoked", "state"] if args.desc == "both" else [args.desc]
    print(f"[DEBUG] Processing desc types: {desc_types}", flush=True)
    
    # Apply limit if specified
    if args.limit:
        elements = elements[:args.limit]
        print(f"[DEBUG] Limited to {len(elements)} elements for testing", flush=True)
    
    # Process each element
    total = len(elements) * len(desc_types)
    success_count = 0
    fail_count = 0
    
    print(f"\n[INFO] Processing {total} total element-desc combinations...", flush=True)
    print(f"{'=' * 80}", flush=True)
    
    for i, (subject, task) in enumerate(elements, 1):
        print(f"\n[{i}/{len(elements)}] Element: {subject}, {task}", flush=True)
        
        for desc in desc_types:
            print(f"[DEBUG] About to process: {subject}, {task}, {desc}", flush=True)
            result = process_element(subject, task, desc,
                                   dry_run=args.dry_run,
                                   force=args.force)
            print(f"[DEBUG] Process result: {result}", flush=True)
            if result:
                success_count += 1
            else:
                fail_count += 1
    
    # Summary
    print(f"\n{'=' * 80}", flush=True)
    print(f"{'DRY RUN ' if args.dry_run else ''}Summary:", flush=True)
    print(f"  Total elements: {len(elements)}", flush=True)
    print(f"  Total combinations: {total}", flush=True)
    print(f"  Successful: {success_count}", flush=True)
    print(f"  Failed: {fail_count}", flush=True)
    
    if args.dry_run:
        print(f"\nThis was a dry run. Run without --dry-run to actually process files.", flush=True)
    
    print(f"[DEBUG] Exiting with code: {0 if fail_count == 0 else 1}", flush=True)
    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
