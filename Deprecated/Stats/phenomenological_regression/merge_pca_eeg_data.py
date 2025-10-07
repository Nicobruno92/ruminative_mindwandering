#!/usr/bin/env python3
"""
Merge PCA results with EEG markers data

This script merges the PCA behavioral data with the EEG markers data
to create a combined dataset for continuous PC1 analysis.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import argparse


def merge_pca_eeg_data(pca_file, eeg_file, output_file):
    """
    Merge PCA results with EEG markers data.
    
    Parameters
    ----------
    pca_file : str
        Path to PCA results CSV file
    eeg_file : str
        Path to EEG markers CSV file  
    output_file : str
        Path for merged output CSV file
    """
    print("Loading PCA results...")
    pca_df = pd.read_csv(pca_file)
    print(f"PCA data shape: {pca_df.shape}")
    
    print("Loading EEG markers data...")
    eeg_df = pd.read_csv(eeg_file)
    print(f"EEG data shape: {eeg_df.shape}")
    
    # Check column overlap and prepare merge keys
    print("\nPCA columns:", pca_df.columns.tolist())
    print("\nEEG columns:", eeg_df.columns.tolist())
    
    # Prepare merge keys - both datasets have subject_id, task, probe_number
    merge_keys = ['subject_id', 'task', 'probe_number']
    
    # Check if merge keys exist in both datasets
    missing_pca = [k for k in merge_keys if k not in pca_df.columns]
    missing_eeg = [k for k in merge_keys if k not in eeg_df.columns]
    
    if missing_pca:
        print(f"Missing keys in PCA data: {missing_pca}")
    if missing_eeg:
        print(f"Missing keys in EEG data: {missing_eeg}")
    
    # Display unique values for merge keys to check compatibility
    for key in merge_keys:
        if key in pca_df.columns and key in eeg_df.columns:
            pca_unique = set(pca_df[key].unique())
            eeg_unique = set(eeg_df[key].unique())
            print(f"\n{key}:")
            print(f"  PCA unique values: {len(pca_unique)} ({sorted(list(pca_unique))[:5]}...)")
            print(f"  EEG unique values: {len(eeg_unique)} ({sorted(list(eeg_unique))[:5]}...)")
            print(f"  Overlap: {len(pca_unique.intersection(eeg_unique))} values")
    
    # Perform merge
    print(f"\nMerging datasets on: {merge_keys}")
    
    # Keep only relevant columns from PCA (PC1, PC2, PC3, group info)
    pca_cols_to_keep = merge_keys + ['PC1', 'PC2', 'PC3', 'group', 'inclusion_exclusion', 'sex', 'age']
    pca_cols_to_keep = [col for col in pca_cols_to_keep if col in pca_df.columns]
    pca_subset = pca_df[pca_cols_to_keep].copy()
    
    # Merge datasets
    merged_df = eeg_df.merge(pca_subset, on=merge_keys, how='left', suffixes=('', '_pca'))
    
    print(f"Merged data shape: {merged_df.shape}")
    print(f"Rows with PC1 data: {merged_df['PC1'].notna().sum()}")
    print(f"Rows without PC1 data: {merged_df['PC1'].isna().sum()}")
    
    # Check PC1 distribution
    if 'PC1' in merged_df.columns:
        pc1_stats = merged_df['PC1'].describe()
        print(f"\nPC1 statistics:\n{pc1_stats}")
    
    # Save merged dataset
    print(f"\nSaving merged dataset to: {output_file}")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    merged_df.to_csv(output_file, index=False)
    
    # Summary report
    print("\n" + "="*60)
    print("MERGE SUMMARY")
    print("="*60)
    print(f"Original EEG rows: {len(eeg_df)}")
    print(f"Original PCA rows: {len(pca_df)}")
    print(f"Merged rows: {len(merged_df)}")
    print(f"Rows with PC1: {merged_df['PC1'].notna().sum()}")
    print(f"Unique subjects: {merged_df['subject_id'].nunique()}")
    print(f"Unique markers: {merged_df['marker'].nunique()}")
    print(f"Unique channels: {merged_df['channel'].nunique()}")
    
    if 'group' in merged_df.columns:
        print(f"Groups: {merged_df['group'].value_counts().to_dict()}")
    
    return merged_df


def main():
    """Main function with argument parsing."""
    
    parser = argparse.ArgumentParser(description='Merge PCA results with EEG markers data')
    parser.add_argument('--pca-file', type=str, 
                       default='results/Behavior/probe_data/pca_results.csv',
                       help='Path to PCA results CSV file')
    parser.add_argument('--eeg-file', type=str,
                       default='results/aggregated_mne_markers/aggregated_mne_markers_onoff_5trials_go_correct_iqr_probe.csv',
                       help='Path to EEG markers CSV file')
    parser.add_argument('--output-file', type=str,
                       default='results/aggregated_mne_markers/merged_pca_eeg_markers.csv',
                       help='Path for merged output CSV file')
    
    args = parser.parse_args()
    
    # Check input files exist
    if not os.path.exists(args.pca_file):
        print(f"Error: PCA file not found: {args.pca_file}")
        return
        
    if not os.path.exists(args.eeg_file):
        print(f"Error: EEG file not found: {args.eeg_file}")
        return
    
    # Perform merge
    merged_df = merge_pca_eeg_data(args.pca_file, args.eeg_file, args.output_file)
    
    print(f"\nMerge completed successfully!")
    print(f"Merged dataset saved to: {args.output_file}")


if __name__ == '__main__':
    main() 