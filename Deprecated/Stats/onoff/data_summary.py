#!/usr/bin/env python3
"""
Data Summary Script for Simple LMM Analysis

This script provides a summary of the data to help understand
the distribution and identify potential issues.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
CSV_FILE = 'results/aggregated_mne_markers/aggregated_mne_markers_onoff_5trials_go_correct_iqr_probe.csv'
MARKERS = ['a', 'a_n', 't', 't_n']

# ROI definitions
FRONTAL_CHANNELS = ['Fp1', 'Fp2', 'AF3', 'AF4', 'AF7', 'AF8', 'AFz', 'F1', 'F2', 'F3', 'F4', 
                   'F5', 'F6', 'F7', 'F8', 'Fz', 'FC1', 'FC2', 'FC3', 'FC4', 'FC5', 'FC6']

POSTERIOR_CHANNELS = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'Pz', 'PO3', 'PO4', 
                     'PO7', 'PO8', 'POz', 'O1', 'O2', 'Oz', 'CP1', 'CP2', 'CP3', 'CP4', 
                     'CP5', 'CP6', 'CPz']

def load_and_prepare_data(csv_file):
    """Load and prepare the data for analysis."""
    print("Loading data...")
    df = pd.read_csv(csv_file)
    
    # Filter for markers of interest
    df = df[df['marker'].isin(MARKERS)].copy()
    
    # Map onoff_label to condition names
    df['condition'] = df['onoff_label'].map({'high': 'ontask', 'low': 'offtask'})
    
    # Add ROI information
    df['roi'] = df['channel'].apply(lambda x: 'frontal' if x in FRONTAL_CHANNELS 
                                   else 'posterior' if x in POSTERIOR_CHANNELS 
                                   else 'other')
    
    # Filter for ROI channels only
    df = df[df['roi'].isin(['frontal', 'posterior'])].copy()
    
    return df

def summarize_data(df):
    """Provide detailed summary of the data."""
    print("="*60)
    print("DATA SUMMARY")
    print("="*60)
    
    print(f"Total observations: {len(df)}")
    print(f"Unique subjects: {df['subject_id'].nunique()}")
    print(f"Markers: {df['marker'].unique()}")
    print(f"Conditions: {df['condition'].value_counts().to_dict()}")
    print(f"ROIs: {df['roi'].value_counts().to_dict()}")
    
    print("\nData by marker and condition:")
    for marker in MARKERS:
        marker_data = df[df['marker'] == marker]
        print(f"\n{marker}:")
        for condition in ['ontask', 'offtask']:
            cond_data = marker_data[marker_data['condition'] == condition]
            print(f"  {condition}: {len(cond_data)} observations, {cond_data['subject_id'].nunique()} subjects")
            print(f"    Mean: {cond_data['mean'].mean():.3e}, Std: {cond_data['mean'].std():.3e}")
            print(f"    Range: [{cond_data['mean'].min():.3e}, {cond_data['mean'].max():.3e}]")
    
    print("\nData by ROI:")
    for roi in ['frontal', 'posterior']:
        roi_data = df[df['roi'] == roi]
        print(f"\n{roi}:")
        print(f"  Channels: {sorted(roi_data['channel'].unique())}")
        print(f"  Observations: {len(roi_data)}")
        
        for condition in ['ontask', 'offtask']:
            cond_data = roi_data[roi_data['condition'] == condition]
            print(f"  {condition}: Mean = {cond_data['mean'].mean():.3e}, Std = {cond_data['mean'].std():.3e}")

def aggregate_and_summarize(df):
    """Aggregate by ROI and summarize."""
    print("\n" + "="*60)
    print("AGGREGATED DATA SUMMARY")
    print("="*60)
    
    # Aggregate by ROI
    agg_df = df.groupby(['subject_id', 'marker', 'roi', 'condition']).agg({
        'mean': 'mean',
        'count': 'sum'
    }).reset_index()
    
    print(f"Aggregated data shape: {agg_df.shape}")
    
    print("\nAggregated data by marker and condition:")
    for marker in MARKERS:
        marker_data = agg_df[agg_df['marker'] == marker]
        print(f"\n{marker}:")
        
        for roi in ['frontal', 'posterior']:
            roi_data = marker_data[marker_data['roi'] == roi]
            print(f"  {roi} ROI:")
            
            for condition in ['ontask', 'offtask']:
                cond_data = roi_data[roi_data['condition'] == condition]
                if len(cond_data) > 0:
                    print(f"    {condition}: {len(cond_data)} observations")
                    print(f"      Mean: {cond_data['mean'].mean():.3e} ± {cond_data['mean'].std():.3e}")
                    print(f"      Range: [{cond_data['mean'].min():.3e}, {cond_data['mean'].max():.3e}]")
                else:
                    print(f"    {condition}: No data")
    
    return agg_df

def check_effect_sizes(agg_df):
    """Calculate effect sizes for each marker-ROI combination."""
    print("\n" + "="*60)
    print("EFFECT SIZES (Cohen's d)")
    print("="*60)
    
    for marker in MARKERS:
        marker_data = agg_df[agg_df['marker'] == marker]
        print(f"\n{marker}:")
        
        for roi in ['frontal', 'posterior']:
            roi_data = marker_data[marker_data['roi'] == roi]
            
            ontask_data = roi_data[roi_data['condition'] == 'ontask']['mean']
            offtask_data = roi_data[roi_data['condition'] == 'offtask']['mean']
            
            if len(ontask_data) > 0 and len(offtask_data) > 0:
                # Calculate Cohen's d
                pooled_std = np.sqrt(((len(ontask_data) - 1) * ontask_data.var() + 
                                     (len(offtask_data) - 1) * offtask_data.var()) / 
                                    (len(ontask_data) + len(offtask_data) - 2))
                
                if pooled_std > 0:
                    cohens_d = (ontask_data.mean() - offtask_data.mean()) / pooled_std
                    print(f"  {roi}: Cohen's d = {cohens_d:.3f}")
                    
                    # Interpret effect size
                    if abs(cohens_d) < 0.2:
                        interpretation = "negligible"
                    elif abs(cohens_d) < 0.5:
                        interpretation = "small"
                    elif abs(cohens_d) < 0.8:
                        interpretation = "medium"
                    else:
                        interpretation = "large"
                    
                    print(f"    Interpretation: {interpretation} effect")
                else:
                    print(f"  {roi}: Cannot calculate effect size (zero variance)")
            else:
                print(f"  {roi}: Insufficient data")

def main():
    """Main function."""
    print("📊 DATA SUMMARY FOR SIMPLE LMM ANALYSIS 📊")
    
    # Load data
    df = load_and_prepare_data(CSV_FILE)
    
    # Summarize raw data
    summarize_data(df)
    
    # Aggregate and summarize
    agg_df = aggregate_and_summarize(df)
    
    # Check effect sizes
    check_effect_sizes(agg_df)

if __name__ == '__main__':
    main() 