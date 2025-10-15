#!/usr/bin/env python3
"""
Example usage of the updated reader module for LMM-based spatial cluster permutation testing.

This script demonstrates how to:
1. Load all aggregated probe marker data
2. Prepare data for LMM analysis for specific markers
3. Set up data for cluster permutation testing across channels
"""

import sys
from pathlib import Path
import numpy as np

# Add Statistics directory to path to import reader
sys.path.append(str(Path(__file__).parent))

from reader import (
    load_all_probe_data, 
    prepare_data_for_lmm, 
    get_available_markers,
    get_channel_names,
    prepare_channel_data
)


def main():
    """Example usage of the reader module using existing config.yaml."""
    
    print("=== EEG Mind-Wandering LMM Data Reader Example ===\n")
    
    # Load configuration from existing config.yaml
    import yaml
    config_path = Path("Statistics/config.yaml")
    
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        features_root = config['project']['features_root']
        subjects = config['project'].get('subjects', None)
        tasks = config['project'].get('tasks', None)
        marker_types = config['project'].get('marker_types', None)
        
        print(f"✓ Loaded configuration from {config_path}")
        print(f"  Features root: {features_root}")
        print(f"  Subjects: {subjects[:5] if subjects else 'All'}{'...' if subjects and len(subjects) > 5 else ''}")
        print(f"  Tasks: {tasks}")
        print(f"  Marker types: {marker_types}")
        print()
    else:
        print(f"✗ Configuration file not found: {config_path}")
        print("Please ensure Statistics/config.yaml exists or update the paths below:")
        features_root = "/network/iss/levy/analyze/valerocabre/analyse/nbruno/depressed_mindwandering/features"
        subjects = ["02", "03", "04"]  # Example subjects
        tasks = ["Sart1", "Sart2"]  # Example tasks
        marker_types = ["evoked", "state"]  # Marker types to include
        print()
    
    # Step 1: Load all aggregated probe data
    print("1. Loading all aggregated probe data...")
    try:
        df_all = load_all_probe_data(
            features_root=features_root,
            subjects=subjects,
            tasks=tasks,
            marker_types=marker_types,
            verbose=True
        )
        print(f"✓ Loaded {len(df_all)} rows of data")
    except FileNotFoundError as e:
        print(f"✗ Error loading data: {e}")
        print("Please update the features_root path in this script.")
        return
    
    # Step 2: Explore available markers
    print("\n2. Available markers:")
    available_markers = get_available_markers(features_root, marker_types)
    for marker_type, markers in available_markers.items():
        print(f"  {marker_type}: {len(markers)} markers")
        for marker in markers[:3]:  # Show first 3 markers
            print(f"    - {marker}")
        if len(markers) > 3:
            print(f"    ... and {len(markers) - 3} more")
    
    # Step 3: Get available channels
    print("\n3. Available channels:")
    channels = get_channel_names(features_root)
    print(f"  Total channels: {len(channels)}")
    print(f"  First 10: {channels[:10]}")
    print(f"  Last 10: {channels[-10:]}")
    
    # Step 4: Prepare data for a specific marker (example)
    print("\n4. Preparing data for LMM analysis...")
    
    # Choose a marker to analyze (use first available marker as example)
    if available_markers:
        first_marker_type = list(available_markers.keys())[0]
        first_marker = available_markers[first_marker_type][0]
        
        print(f"  Analyzing marker: {first_marker}")
        
        # Define LMM formula (using config if available)
        if config_path.exists():
            formula = config['lmm']['formula']
        else:
            formula = "power ~ onoff + (1|subject)"
        
        try:
            # Prepare data for this marker
            power_data, df_behavioral, marker_channels = prepare_data_for_lmm(
                df=df_all,
                marker_name=first_marker,
                formula=formula,
                include_channels=channels[:10],  # Use first 10 channels as example
                exclude_channels=None
            )
            
            print(f"  ✓ Data prepared successfully")
            print(f"    Observations: {power_data.shape[0]}")
            print(f"    Channels: {power_data.shape[1]}")
            print(f"    Behavioral variables: {list(df_behavioral.columns)}")
            
            # Check for missing values
            n_missing = np.isnan(power_data).sum()
            total_values = power_data.size
            print(f"    Missing values: {n_missing}/{total_values} ({100*n_missing/total_values:.1f}%)")
            
        except ValueError as e:
            print(f"  ✗ Error preparing data: {e}")
    
    # Step 5: Demonstrate channel-specific data preparation
    print("\n5. Preparing data for individual channels...")
    
    if 'power_data' in locals() and 'df_behavioral' in locals():
        # Example: prepare data for first channel
        channel_idx = 0
        channel_data = prepare_channel_data(
            power_data=power_data,
            df_behavioral=df_behavioral,
            channel_idx=channel_idx,
            channels=marker_channels
        )
        
        print(f"  Channel {channel_idx} ({marker_channels[channel_idx]}): {len(channel_data)} observations")
        print(f"    Formula variables: {[col for col in channel_data.columns if col in formula]}")
        
        # Show sample of data
        print(f"    Sample data:")
        print(f"      onoff range: [{channel_data['onoff'].min():.1f}, {channel_data['onoff'].max():.1f}]")
        print(f"      subjects: {sorted(channel_data['subject'].unique())}")
        print(f"      power range: {channel_data['power'].min():.3f} to {channel_data['power'].max():.3f}")
    
    print("\n=== Example completed ===")
    print("\nNext steps for LMM cluster permutation testing:")
    print("1. Your config.yaml is already set up and ready to use!")
    print("2. Run single marker: python Statistics/run_pipeline.py")
    print("3. Run all markers: python Statistics/run_pipeline.py --all-markers")
    print("4. Run specific markers: python Statistics/run_pipeline.py --markers 'EEG_psd_bands_spectralpower_alpha'")
    print("5. Check results in the output directory specified in config.yaml")
    print("\nResults will be saved with marker type prefixes:")
    print("- evoked_EEG_psd_bands_spectralpower_alpha_results.pkl")
    print("- state_EEG_psd_bands_spectralpower_alpha_results.pkl")
    print("- Plus organized summaries by marker type")
    print("\nFor SLURM cluster usage:")
    print("- sbatch Statistics/run_lmm_pipeline.sh --all-markers")
    print("- Check logs in the logs/ directory")


if __name__ == "__main__":
    main()
