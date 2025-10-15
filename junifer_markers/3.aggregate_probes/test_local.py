#!/usr/bin/env python3
"""
Test script to verify the aggregation pipeline with local PKL files.
"""

import os
import sys
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from aggregate_markers_by_probe import (
    load_marker_pkl, 
    aggregate_marker_epochs,
    save_aggregated_markers
)

def test_aggregation():
    """Test the aggregation pipeline with local PKL files."""
    
    # Use the PKL files that are already available
    evoked_pkl = "junifer_markers/3.aggregate_probes/sub-17_task-Sart1_desc-evoked_markers.pkl"
    state_pkl = "junifer_markers/3.aggregate_probes/sub-17_task-Sart1_desc-state_markers.pkl"
    
    print("=" * 80)
    print("TESTING LOCAL PKL AGGREGATION")
    print("=" * 80)
    
    # Test loading evoked markers
    if os.path.exists(evoked_pkl):
        print(f"\n📁 Loading evoked markers: {evoked_pkl}")
        evoked_data = load_marker_pkl(evoked_pkl)
        evoked_markers = evoked_data.get("markers", {})
        print(f"✅ Loaded {len(evoked_markers)} evoked markers")
        
        # Test aggregation on first marker
        if evoked_markers:
            first_marker_name = list(evoked_markers.keys())[0]
            first_marker = evoked_markers[first_marker_name]
            
            # Get channels
            channels = []
            if 'metadata' in evoked_data:
                metadata = evoked_data['metadata']
                fif_info = metadata.get('fif_info', {})
                channels = fif_info.get('channel_names', [])
            
            if not channels and hasattr(first_marker, 'channel_names'):
                channels = first_marker.channel_names
            
            if channels:
                print(f"🔌 Testing aggregation on {first_marker_name}")
                print(f"   Channels: {len(channels)} ({channels[:3]}...)")
                
                # Test with first 5 epochs
                test_epochs = [0, 1, 2, 3, 4]
                print(f"   Test epochs: {test_epochs}")
                
                aggregated = aggregate_marker_epochs(
                    marker_data=first_marker,
                    marker_name=first_marker_name,
                    epoch_indices=test_epochs,
                    channels=channels[:5],  # Test with first 5 channels
                    use_trimmean=True,
                    trimmean_percent=20.0,
                )
                
                print(f"   ✅ Aggregation successful!")
                print(f"   📊 Result structure: {list(aggregated.keys())}")
                
                for band_name, band_data in aggregated.items():
                    n_channels = len(band_data)
                    n_nans = sum(1 for v in band_data.values() if str(v) == 'nan')
                    print(f"      {band_name}: {n_channels} channels, {n_nans} NaNs")
            else:
                print("❌ No channels found")
    else:
        print(f"❌ Evoked PKL not found: {evoked_pkl}")
    
    # Test loading state markers
    if os.path.exists(state_pkl):
        print(f"\n📁 Loading state markers: {state_pkl}")
        state_data = load_marker_pkl(state_pkl)
        state_markers = state_data.get("markers", {})
        print(f"✅ Loaded {len(state_markers)} state markers")
        
        # Test aggregation on first marker
        if state_markers:
            first_marker_name = list(state_markers.keys())[0]
            first_marker = state_markers[first_marker_name]
            
            # Get channels
            channels = []
            if 'metadata' in state_data:
                metadata = state_data['metadata']
                fif_info = metadata.get('fif_info', {})
                channels = fif_info.get('channel_names', [])
            
            if not channels and hasattr(first_marker, 'channel_names'):
                channels = first_marker.channel_names
            
            if channels:
                print(f"🔌 Testing aggregation on {first_marker_name}")
                print(f"   Channels: {len(channels)} ({channels[:3]}...)")
                
                # Test with first 5 epochs
                test_epochs = [0, 1, 2, 3, 4]
                print(f"   Test epochs: {test_epochs}")
                
                aggregated = aggregate_marker_epochs(
                    marker_data=first_marker,
                    marker_name=first_marker_name,
                    epoch_indices=test_epochs,
                    channels=channels[:5],  # Test with first 5 channels
                    use_trimmean=True,
                    trimmean_percent=20.0,
                )
                
                print(f"   ✅ Aggregation successful!")
                print(f"   📊 Result structure: {list(aggregated.keys())}")
                
                for band_name, band_data in aggregated.items():
                    n_channels = len(band_data)
                    n_nans = sum(1 for v in band_data.values() if str(v) == 'nan')
                    print(f"      {band_name}: {n_channels} channels, {n_nans} NaNs")
            else:
                print("❌ No channels found")
    else:
        print(f"❌ State PKL not found: {state_pkl}")
    
    print("\n" + "=" * 80)
    print("✅ LOCAL TEST COMPLETED")
    print("=" * 80)

if __name__ == "__main__":
    test_aggregation()
