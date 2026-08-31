#!/usr/bin/env python
"""
Helper script to get list of markers from config.

Usage:
    python get_marker_list.py --config config_andrillon.yaml
    
This is useful for:
1. Checking which markers will be analyzed
2. Updating the MARKER_LIST in submit_andrillon_array.sh
3. Setting the correct --array parameter in SLURM script
"""

import sys
import argparse
from pathlib import Path
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from andrillon_pipeline import get_marker_list


def main():
    parser = argparse.ArgumentParser(
        description="Get list of markers from configuration"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config_andrillon.yaml",
        help="Path to configuration YAML file"
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["list", "bash", "count"],
        default="list",
        help="Output format: list (one per line), bash (array format), count (number only)"
    )
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Get markers
    markers = get_marker_list(config)
    
    # Output in requested format
    if args.format == "list":
        for marker in markers:
            print(marker)
    elif args.format == "bash":
        print("MARKER_LIST=(")
        for marker in markers:
            print(f'    "{marker}"')
        print(")")
    elif args.format == "count":
        print(len(markers))
    
    # Print summary to stderr
    print(f"\nTotal markers: {len(markers)}", file=sys.stderr)
    print(f"SLURM array parameter: --array=0-{len(markers)-1}", file=sys.stderr)


if __name__ == "__main__":
    main()
