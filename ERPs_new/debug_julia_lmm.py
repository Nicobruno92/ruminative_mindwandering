#!/usr/bin/env python3
"""
Debug script for Julia LMM windowed analysis.

This script helps debug the Julia LMM windowed analysis issue by examining
the data conversion and Julia execution step by step.
"""

import os
import sys
import pandas as pd
from pathlib import Path

# Add current directory to path for imports
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

from helpers import load_yaml_config
from lmm_analysis import collect_probe_data_for_lmm
from unfold_bridge import JuliaLMMBridge, run_julia_windowed_lmm_analysis

# =============================================================================
# CONFIGURATION
# =============================================================================
CONFIG_PATH = "/Volumes/levy/analyze/valerocabre/analyse/nbruno/depressed_mindwandering/ERPs_new/config.yaml"
# =============================================================================


def main():
    """Debug Julia LMM windowed analysis."""
    print("🔍 Debugging Julia LMM Windowed Analysis")
    print("=" * 60)
    
    # Load configuration
    print("📋 Loading configuration...")
    cfg = load_yaml_config(CONFIG_PATH)
    
    # Check if LMM analysis is enabled
    if not cfg.get("lmm_analysis", {}).get("enabled", False):
        print("❌ LMM analysis not enabled in config")
        return
    
    # Collect windowed probe data 
    print("\n📊 Collecting windowed probe data...")
    windowed_data = collect_probe_data_for_lmm(cfg)
    
    if windowed_data.empty:
        print("❌ No windowed data collected")
        return
    
    print(f"✅ Collected windowed data: {len(windowed_data)} rows")
    print(f"📊 Columns: {list(windowed_data.columns)}")
    print(f"📊 Windows: {windowed_data['window'].unique()}")
    print(f"📊 ROIs: {windowed_data['roi'].unique()}")
    print(f"📊 Subjects: {windowed_data['subject'].nunique()}")
    print(f"📊 Conditions: {windowed_data['condition'].unique()}")
    
    # Show sample data by window and ROI
    print("\n📊 Data distribution by ROI and window:")
    summary = windowed_data.groupby(['roi', 'window']).agg({
        'subject': 'nunique',
        'condition': lambda x: len(x.unique()),
        'amplitude': 'count'
    }).rename(columns={
        'subject': 'n_subjects',
        'condition': 'n_conditions', 
        'amplitude': 'n_rows'
    })
    print(summary)
    
    # Test Julia bridge initialization
    print("\n🔧 Testing Julia bridge...")
    lmm_cfg = cfg.get("lmm_analysis", {})
    bridge = JuliaLMMBridge(
        julia_cmd=lmm_cfg.get("julia_cmd"),
        cluster_module=lmm_cfg.get("cluster_module"),
    )
    
    # Check Julia environment
    if not bridge.check_julia_environment():
        print("❌ Julia environment not ready")
        return
    
    # Test with small subset of data first
    print("\n🧪 Testing with small data subset...")
    # Take first 2 ROIs and first 2 windows for testing
    test_rois = windowed_data['roi'].unique()[:2]
    test_windows = windowed_data['window'].unique()[:2] 
    
    test_data = windowed_data[
        windowed_data['roi'].isin(test_rois) & 
        windowed_data['window'].isin(test_windows)
    ].copy()
    
    print(f"📊 Test data: {len(test_data)} rows")
    print(f"📊 Test ROIs: {test_data['roi'].unique()}")
    print(f"📊 Test windows: {test_data['window'].unique()}")
    
    # Test Julia windowed analysis with small dataset
    print("\n🚀 Running Julia windowed analysis on test data...")
    julia_results = run_julia_windowed_lmm_analysis(test_data, cfg)
    
    if julia_results.empty:
        print("❌ Julia test failed - no results")
        
        # Let's manually test the data conversion
        print("\n🔍 Manual data conversion test...")
        julia_data = []
        for _, row in test_data.iterrows():
            window_center = (row['window_start'] + row['window_end']) / 2
            julia_data.append({
                'subject': row['subject'],
                'condition': row['condition'],
                'roi': row['roi'],
                'time_point': window_center,
                'amplitude': row['amplitude'],
                'baseline': row['baseline'],
            })
        
        julia_df = pd.DataFrame(julia_data)
        print(f"📊 Converted data: {len(julia_df)} rows")
        
        # Check data distribution per time point
        time_point_summary = julia_df.groupby(['roi', 'time_point']).agg({
            'subject': 'nunique',
            'condition': lambda x: len(x.unique()),
            'amplitude': 'count'
        })
        print("📊 Data per time point:")
        print(time_point_summary)
        
        # Save debug data
        debug_dir = "/tmp/julia_lmm_debug"
        os.makedirs(debug_dir, exist_ok=True)
        julia_df.to_csv(f"{debug_dir}/debug_julia_data.csv", index=False)
        print(f"💾 Debug data saved: {debug_dir}/debug_julia_data.csv")
        
    else:
        print(f"✅ Julia test successful: {len(julia_results)} results")
        print(julia_results.head())
    
    print("\n🎯 Debug complete!")


if __name__ == "__main__":
    main()