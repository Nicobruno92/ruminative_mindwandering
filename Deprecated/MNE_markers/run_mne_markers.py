#!/usr/bin/env python3
"""
Simple runner script for computing MNE-based markers.
Usage: 
  From project root: python MNE_markers/run_mne_markers.py --subject 02 --task Sart1
  From MNE_markers/: python run_mne_markers.py --subject 02 --task Sart1
"""

import argparse
import sys
import os

# Get the script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))

# Get the project root directory (parent of script_dir)
project_root = os.path.abspath(os.path.join(script_dir, '..'))

# Add script directory to path for imports
sys.path.insert(0, script_dir)

# Import the processing function
try:
    from compute_mne_markers import process_one_subject
except ImportError:
    print("Error importing compute_mne_markers. Check your Python path.")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description='Run MNE marker computation for a subject/task'
    )
    parser.add_argument('--subject', type=str, default='02',
                        help='Subject ID (e.g., 02)')
    parser.add_argument('--task', type=str, default='Sart1',
                        help='Task name (e.g., Sart1)')
    parser.add_argument('--root', type=str, 
                        default='/network/iss/cenir/analyse/meeg/CYBERSART/',
                        help='Root data directory')
    parser.add_argument('--output-dir', type=str,
                        default='./results/mne_markers',
                        help='Output directory')
    parser.add_argument('--tmin', type=float, default=0,
                        help='Start time for analysis')
    parser.add_argument('--tmax', type=float, default=2,
                        help='End time for analysis')
    
    args = parser.parse_args()
    
    # Handle output directory path
    # If it's a relative path, make it relative to the project root
    if args.output_dir.startswith('./') or not os.path.isabs(args.output_dir):
        output_dir = os.path.abspath(os.path.join(project_root, args.output_dir.lstrip('./')))
    else:
        output_dir = args.output_dir
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Computing MNE markers for subject {args.subject}, task {args.task}")
    print(f"Output will be saved to: {output_dir}")
    
    process_one_subject(
        root=args.root,
        subject=args.subject,
        task=args.task,
        data_type='eeg',
        desc='autoPreproc',
        output_dir=output_dir,
        tmin=args.tmin,
        tmax=args.tmax
    )
    
    print("Done!")


if __name__ == '__main__':
    main() 