#!/usr/bin/env python3
"""
Diagnostic Script: Check for Normalization-Induced Bias

This script checks if subject-level normalization is creating spurious
correlations with the predictor variable (onoff).

The issue: If onoff varies systematically within subjects, z-score normalization
can remove the very effect we're trying to detect.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================
# Path to aggregated probe data
FEATURES_ROOT = "/Volumes/cenir/analyse/meeg/CYBERSART/BIDS/features/"
OUTPUT_DIR = "./normalization_diagnostics"

# Marker to test (use non-normalized version from raw data)
TEST_MARKER = "power_delta"  # Use raw power, not normalized
MARKER_TYPE = "state"

# Subjects and tasks to include
SUBJECTS = ["02", "03", "04", "05"]  # Test with a few subjects first
TASKS = ["Sart1", "Sart2"]

# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def load_sample_data():
    """Load a sample of aggregated probe data."""
    data_list = []
    
    for subject in SUBJECTS:
        for task in TASKS:
            # Find all probe files for this subject/task
            subject_dir = Path(FEATURES_ROOT) / f"sub-{subject}" / "eeg" / "junifer"
            pattern = f"sub-{subject}_task-{task}_desc-probe-*_{MARKER_TYPE}_aggMarkers.csv"
            
            if not subject_dir.exists():
                print(f"  Directory not found: {subject_dir}")
                continue
            
            probe_files = list(subject_dir.glob(pattern))
            
            if not probe_files:
                print(f"  No files found for sub-{subject} task-{task} (pattern: {pattern})")
                continue
            
            print(f"  Loading {len(probe_files)} probe files for sub-{subject} task-{task}...")
            
            for file_path in probe_files:
                df = pd.read_csv(file_path)
                
                # Data is in long format with 'marker' column
                # Filter for the specific marker and average across channels
                if 'marker' in df.columns and 'value' in df.columns:
                    df_marker = df[df['marker'] == TEST_MARKER].copy()
                    
                    if len(df_marker) > 0:
                        # Average across channels for each probe
                        df_agg = df_marker.groupby(['subject', 'task', 'probe_number', 'onoff']).agg({
                            'value': 'mean'
                        }).reset_index()
                        df_agg.columns = ['subject', 'task', 'probe_number', 'onoff', 'power']
                        data_list.append(df_agg)
                else:
                    print(f"    Warning: Expected 'marker' and 'value' columns not found in {file_path.name}")
                    print(f"    Available columns: {list(df.columns)[:10]}")
                    break  # Only show once per subject/task
    
    if not data_list:
        raise ValueError(f"No data found for marker {TEST_MARKER}. Check paths and marker name.")
    
    return pd.concat(data_list, ignore_index=True)


def check_onoff_power_relationship(df):
    """Check relationship between onoff and power before/after normalization."""
    
    print("\n" + "="*70)
    print("DIAGNOSTIC: ONOFF vs POWER RELATIONSHIP")
    print("="*70)
    
    # 1. Overall correlation (raw data)
    corr_raw = df[['onoff', 'power']].corr().iloc[0, 1]
    print(f"\n1. Raw data correlation (onoff vs power): {corr_raw:.4f}")
    
    # 2. Within-subject correlations
    print("\n2. Within-subject correlations:")
    subject_corrs = []
    for subject in df['subject'].unique():
        df_sub = df[df['subject'] == subject]
        if len(df_sub) > 2:  # Need at least 3 points for correlation
            corr = df_sub[['onoff', 'power']].corr().iloc[0, 1]
            subject_corrs.append(corr)
            print(f"   Subject {subject}: {corr:.4f}")
    
    print(f"\n   Mean within-subject correlation: {np.mean(subject_corrs):.4f}")
    print(f"   Std within-subject correlation: {np.std(subject_corrs):.4f}")
    
    # 3. Check if onoff varies within subjects
    print("\n3. Onoff variation within subjects:")
    for subject in df['subject'].unique():
        df_sub = df[df['subject'] == subject]
        onoff_std = df_sub['onoff'].std()
        onoff_range = df_sub['onoff'].max() - df_sub['onoff'].min()
        print(f"   Subject {subject}: std={onoff_std:.2f}, range={onoff_range:.2f}")
    
    # 4. Apply z-score normalization and check again
    df['power_normalized'] = df.groupby('subject')['power'].transform(
        lambda x: (x - x.mean()) / x.std()
    )
    
    corr_norm = df[['onoff', 'power_normalized']].corr().iloc[0, 1]
    print(f"\n4. After z-score normalization:")
    print(f"   Correlation (onoff vs normalized power): {corr_norm:.4f}")
    print(f"   Change in correlation: {corr_norm - corr_raw:.4f}")
    
    return df


def visualize_normalization_effect(df):
    """Create visualizations showing normalization effect."""
    
    Path(OUTPUT_DIR).mkdir(exist_ok=True)
    
    # Plot 1: Raw data - onoff vs power
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Raw data
    for subject in df['subject'].unique():
        df_sub = df[df['subject'] == subject]
        axes[0].scatter(df_sub['onoff'], df_sub['power'], alpha=0.6, label=f'Sub {subject}')
    
    axes[0].set_xlabel('Onoff (mind-wandering)')
    axes[0].set_ylabel('Power (raw)')
    axes[0].set_title('Raw Data: Onoff vs Power')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Normalized data
    for subject in df['subject'].unique():
        df_sub = df[df['subject'] == subject]
        axes[1].scatter(df_sub['onoff'], df_sub['power_normalized'], 
                       alpha=0.6, label=f'Sub {subject}')
    
    axes[1].set_xlabel('Onoff (mind-wandering)')
    axes[1].set_ylabel('Power (z-scored within subject)')
    axes[1].set_title('After Normalization: Onoff vs Power')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/normalization_effect.png", dpi=150)
    print(f"\n✓ Saved visualization to {OUTPUT_DIR}/normalization_effect.png")
    
    # Plot 2: Distribution of onoff within subjects
    fig, ax = plt.subplots(figsize=(10, 6))
    df.boxplot(column='onoff', by='subject', ax=ax)
    ax.set_xlabel('Subject')
    ax.set_ylabel('Onoff value')
    ax.set_title('Distribution of Onoff Within Each Subject')
    plt.suptitle('')  # Remove default title
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/onoff_distribution_by_subject.png", dpi=150)
    print(f"✓ Saved visualization to {OUTPUT_DIR}/onoff_distribution_by_subject.png")


def main():
    """Run diagnostic analysis."""
    
    print("\n" + "="*70)
    print("NORMALIZATION DIAGNOSTIC ANALYSIS")
    print("="*70)
    print(f"Marker: {TEST_MARKER}")
    print(f"Marker type: {MARKER_TYPE}")
    print(f"Subjects: {SUBJECTS}")
    print(f"Tasks: {TASKS}")
    
    # Load data
    print("\nLoading data...")
    df = load_sample_data()
    print(f"✓ Loaded {len(df)} observations from {df['subject'].nunique()} subjects")
    
    # Check relationships
    df = check_onoff_power_relationship(df)
    
    # Visualize
    visualize_normalization_effect(df)
    
    print("\n" + "="*70)
    print("INTERPRETATION GUIDE")
    print("="*70)
    print("""
If you see:
1. **High raw correlation** + **Low normalized correlation**:
   → Normalization is removing the effect you're trying to detect
   → This explains why you see uniform t-values
   
2. **Low within-subject onoff variation**:
   → Not enough variation within subjects to detect effects
   → Between-subject effects are being removed by normalization
   
3. **Similar correlations before/after normalization**:
   → Normalization is appropriate
   → The uniform t-values have a different cause

RECOMMENDATION:
- If (1) or (2), consider **NOT normalizing** or using **channel-wise** normalization
- If (3), investigate other causes (e.g., data quality, model specification)
    """)
    print("="*70)


if __name__ == "__main__":
    main()
