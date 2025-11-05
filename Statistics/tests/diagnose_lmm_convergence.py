#!/usr/bin/env python3
"""
Diagnostic: Why are LMMs not converging?

This script tests LMM fitting on a single channel to understand convergence issues.
"""

import pandas as pd
import numpy as np
import warnings
from statsmodels.regression.mixed_linear_model import MixedLM
from statsmodels.tools.sm_exceptions import ConvergenceWarning

# Configuration
FEATURES_ROOT = "/Volumes/cenir/analyse/meeg/CYBERSART/BIDS/features/"
TEST_MARKER = "PE_gamma"
MARKER_TYPE = "evoked"
SUBJECTS = ["02", "03", "04", "05"]
TASKS = ["Sart1", "Sart2"]
TEST_CHANNEL_IDX = 0  # Test first channel

def load_sample_data():
    """Load sample data for testing."""
    from pathlib import Path
    
    data_list = []
    for subject in SUBJECTS:
        for task in TASKS:
            subject_dir = Path(FEATURES_ROOT) / f"sub-{subject}" / "eeg" / "junifer"
            pattern = f"sub-{subject}_task-{task}_desc-probe-*_{MARKER_TYPE}_aggMarkers.csv"
            probe_files = list(subject_dir.glob(pattern))
            
            for file_path in probe_files:
                df = pd.read_csv(file_path)
                if 'marker' in df.columns and 'value' in df.columns:
                    df_marker = df[df['marker'] == TEST_MARKER].copy()
                    if len(df_marker) > 0:
                        data_list.append(df_marker)
    
    if not data_list:
        raise ValueError(f"No data found for {TEST_MARKER}")
    
    df_all = pd.concat(data_list, ignore_index=True)
    
    # Pivot to wide format
    df_wide = df_all.pivot_table(
        index=['subject', 'task', 'probe_number', 'onoff'],
        columns='channel',
        values='value'
    ).reset_index()
    
    return df_wide

def test_lmm_convergence():
    """Test LMM convergence with different settings."""
    
    print("="*70)
    print("LMM CONVERGENCE DIAGNOSTIC")
    print("="*70)
    
    # Load data
    print("\nLoading data...")
    df = load_sample_data()
    print(f"✓ Loaded {len(df)} observations from {df['subject'].nunique()} subjects")
    
    # Get channel names
    channel_cols = [col for col in df.columns if col not in ['subject', 'task', 'probe_number', 'onoff']]
    test_channel = channel_cols[TEST_CHANNEL_IDX]
    
    print(f"\nTesting channel: {test_channel}")
    print(f"Total channels available: {len(channel_cols)}")
    
    # Prepare data for LMM
    df_lmm = df[['subject', 'onoff', test_channel]].copy()
    df_lmm.columns = ['subject', 'onoff', 'power']
    df_lmm = df_lmm.dropna()
    
    print(f"\nData summary:")
    print(f"  Observations: {len(df_lmm)}")
    print(f"  Subjects: {df_lmm['subject'].nunique()}")
    print(f"  Onoff range: [{df_lmm['onoff'].min():.1f}, {df_lmm['onoff'].max():.1f}]")
    print(f"  Power range: [{df_lmm['power'].min():.4f}, {df_lmm['power'].max():.4f}]")
    print(f"  Power mean: {df_lmm['power'].mean():.4f}")
    print(f"  Power std: {df_lmm['power'].std():.4f}")
    
    # Test different configurations
    configs = [
        {"method": "powell", "maxiter": 2000, "desc": "Current config (powell, 2000 iter)"},
        {"method": "lbfgs", "maxiter": 2000, "desc": "LBFGS optimizer (2000 iter)"},
        {"method": "lbfgs", "maxiter": 5000, "desc": "LBFGS optimizer (5000 iter)"},
        {"method": "nm", "maxiter": 5000, "desc": "Nelder-Mead (5000 iter)"},
    ]
    
    for i, cfg in enumerate(configs, 1):
        print(f"\n{'='*70}")
        print(f"TEST {i}: {cfg['desc']}")
        print(f"{'='*70}")
        
        try:
            # Fit model
            model = MixedLM.from_formula(
                "power ~ onoff",
                data=df_lmm,
                groups=df_lmm["subject"]
            )
            
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                result = model.fit(
                    method=cfg['method'],
                    maxiter=cfg['maxiter'],
                    disp=False
                )
                
                # Check for warnings
                conv_warnings = [warning for warning in w if issubclass(warning.category, ConvergenceWarning)]
                other_warnings = [warning for warning in w if not issubclass(warning.category, ConvergenceWarning)]
            
            # Print results
            if conv_warnings:
                print(f"⚠️  CONVERGENCE WARNING:")
                for warning in conv_warnings:
                    print(f"    {warning.message}")
            else:
                print(f"✓ Model converged successfully")
            
            if other_warnings:
                print(f"\nOther warnings ({len(other_warnings)}):")
                for warning in other_warnings:
                    print(f"    {warning.category.__name__}: {warning.message}")
            
            print(f"\nModel results:")
            print(f"  Converged: {result.converged}")
            print(f"  Log-likelihood: {result.llf:.2f}")
            if hasattr(result, 'aic'):
                print(f"  AIC: {result.aic:.2f}")
            
            if 'onoff' in result.tvalues.index:
                print(f"\n  Coefficient for 'onoff':")
                print(f"    Estimate: {result.params['onoff']:.6f}")
                print(f"    t-value: {result.tvalues['onoff']:.4f}")
                print(f"    p-value: {result.pvalues['onoff']:.4f}")
            
        except Exception as e:
            print(f"❌ ERROR: {type(e).__name__}: {e}")
    
    print(f"\n{'='*70}")
    print("RECOMMENDATIONS")
    print(f"{'='*70}")
    print("""
If all tests show convergence warnings:
  → The model specification may be too complex for the data
  → Try simplifying the model or checking data quality
  
If LBFGS works better than powell:
  → Change config.yaml: method: "lbfgs"
  
If increasing maxiter helps:
  → Change config.yaml: maxiter: 5000
    """)

if __name__ == "__main__":
    test_lmm_convergence()
