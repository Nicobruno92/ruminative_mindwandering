#!/usr/bin/env python
"""
Run Andrillon 2020 Pipeline - Local Execution Script

Usage:
    # Analyze all markers
    python run_andrillon_pipeline.py --config config_andrillon.yaml
    
    # Analyze specific marker
    python run_andrillon_pipeline.py --config config_andrillon.yaml --marker alpha
"""

import sys
import argparse
import logging
from pathlib import Path

# Configure logging BEFORE importing andrillon_pipeline
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    force=True  # Override any existing configuration
)

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from andrillon_pipeline import run_andrillon_pipeline


def main():
    parser = argparse.ArgumentParser(
        description="Run Andrillon 2020 cluster-permutation pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze all markers
  python run_andrillon_pipeline.py --config config_andrillon.yaml
  
  # Analyze specific marker
  python run_andrillon_pipeline.py --config config_andrillon.yaml --marker EEG_psd_bands_spectralpower_alpha
  
  # Use custom config
  python run_andrillon_pipeline.py --config my_custom_config.yaml
        """
    )
    
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration YAML file"
    )
    
    parser.add_argument(
        "--marker",
        type=str,
        default=None,
        help="Specific marker to analyze (optional). If not provided, analyzes all markers."
    )
    
    parser.add_argument(
        "--marker-index",
        type=int,
        default=None,
        help="Marker index for array job processing (alternative to --marker)"
    )
    
    args = parser.parse_args()
    
    # Validate config file exists
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Configuration file not found: {config_path}")
        sys.exit(1)
    
    # Determine marker name from index if provided
    marker_name = args.marker
    if args.marker_index is not None:
        import yaml
        from andrillon_pipeline import get_marker_list
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        markers = get_marker_list(config)
        
        if args.marker_index < 0 or args.marker_index >= len(markers):
            print(f"Error: Marker index {args.marker_index} out of range (0-{len(markers)-1})")
            print(f"Available markers: {len(markers)}")
            sys.exit(1)
        
        marker_name = markers[args.marker_index]
        print(f"Marker index {args.marker_index} -> {marker_name}")
    
    # Run pipeline
    try:
        results = run_andrillon_pipeline(
            config_path=str(config_path),
            marker_name=marker_name
        )
        
        print("\n" + "="*80)
        print("SUCCESS!")
        print("="*80)
        print(f"Analyzed {len(results)} marker(s)")
        
        total_clusters = sum(len(r['clusters']) for r in results.values())
        print(f"Total significant clusters: {total_clusters}")
        
    except Exception as e:
        print(f"\nError running pipeline: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
