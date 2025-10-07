#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script para agregar los resultados de todos los análisis de cluster mixedlm.
Este script debe ejecutarse después de que todos los trabajos en el cluster hayan finalizado.
"""

import os
import sys
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Get the script's directory and project root
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
RESULTS_DIR = os.path.join(project_root, 'results/cluster_permutation_tests_mixedlm')
OUTPUT_DIR = os.path.join(project_root, 'results/cluster_permutation_interpretation')
ALPHA = 0.05  # significance level

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

def aggregate_cluster_results():
    """
    Aggregate results from all markers and create summary files.
    """
    print("Aggregating cluster results...")
    
    # Find all cluster result files
    cluster_files = glob.glob(os.path.join(RESULTS_DIR, '*_cluster_results_clusters.csv'))
    
    if not cluster_files:
        print(f"No cluster result files found in {RESULTS_DIR}")
        return
    
    print(f"Found {len(cluster_files)} cluster result files")
    
    # Read and combine all cluster results
    all_clusters = []
    for file in cluster_files:
        try:
            df = pd.read_csv(file)
            if not df.empty:
                all_clusters.append(df)
        except Exception as e:
            print(f"Error reading {file}: {e}")
    
    if not all_clusters:
        print("No valid cluster data found")
        return
    
    # Combine all cluster results
    cluster_df = pd.concat(all_clusters, ignore_index=True)
    
    # Save combined results
    combined_file = os.path.join(OUTPUT_DIR, 'all_cluster_results.csv')
    cluster_df.to_csv(combined_file, index=False)
    print(f"Saved combined cluster results to {combined_file}")
    
    # Create summary of significant clusters by marker
    sig_clusters = cluster_df[cluster_df['significant']]
    
    if sig_clusters.empty:
        print("No significant clusters found")
        return
    
    # Group by marker and count significant clusters
    marker_summary = (
        sig_clusters
        .groupby('marker')
        .agg({
            'cluster_id': 'count',
            'p_value': 'min',
            'cluster_size': ['mean', 'max'],
            'max_abs_t': ['mean', 'max']
        })
        .reset_index()
    )
    
    # Flatten multi-level columns
    marker_summary.columns = [
        '_'.join(col).strip('_') for col in marker_summary.columns.values
    ]
    
    # Rename columns for clarity
    marker_summary = marker_summary.rename(columns={
        'cluster_id_count': 'num_sig_clusters',
        'p_value_min': 'min_p_value',
        'cluster_size_mean': 'avg_cluster_size',
        'cluster_size_max': 'max_cluster_size',
        'max_abs_t_mean': 'avg_max_t',
        'max_abs_t_max': 'overall_max_t'
    })
    
    # Sort by number of significant clusters and then by min p-value
    marker_summary = marker_summary.sort_values(
        ['num_sig_clusters', 'min_p_value'], 
        ascending=[False, True]
    )
    
    # Save marker summary
    summary_file = os.path.join(OUTPUT_DIR, 'marker_effects_summary.csv')
    marker_summary.to_csv(summary_file, index=False)
    print(f"Saved marker summary to {summary_file}")
    
    # Create channel-level summary
    # Find all t-statistics files
    t_stat_files = glob.glob(os.path.join(RESULTS_DIR, '*_t_statistics.csv'))
    
    all_t_stats = []
    for file in t_stat_files:
        try:
        
            df = pd.read_csv(file)
            if not df.empty:
                all_t_stats.append(df)
        except Exception as e:
            print(f"Error reading {file}: {e}")
    
    if not all_t_stats:
        print("No valid t-statistic data found")
        return
    
    # Combine all t-statistics
    t_stats_df = pd.concat(all_t_stats, ignore_index=True)
    
    # Get significant clusters to identify significant channels
    sig_channels = set()
    for _, row in sig_clusters.iterrows():
        marker = row['marker']
        channels = row['channels'].split('; ')
        for channel in channels:
            sig_channels.add((marker, channel))
    
    # Add significance flag to t-statistics
    t_stats_df['in_sig_cluster'] = t_stats_df.apply(
        lambda x: (x['marker'], x['channel']) in sig_channels, 
        axis=1
    )
    
    # Group by channel and count significant markers
    channel_summary = (
        t_stats_df[t_stats_df['in_sig_cluster']]
        .groupby('channel')
        .agg({
            'marker': 'nunique',
            'T_statistic': ['mean', 'std', 'min', 'max']
        })
        .reset_index()
    )
    
    # Flatten multi-level columns
    channel_summary.columns = [
        '_'.join(col).strip('_') for col in channel_summary.columns.values
    ]
    
    # Rename columns for clarity
    channel_summary = channel_summary.rename(columns={
        'marker_nunique': 'num_sig_markers',
        'T_statistic_mean': 'avg_t_stat',
        'T_statistic_std': 'std_t_stat',
        'T_statistic_min': 'min_t_stat',
        'T_statistic_max': 'max_t_stat'
    })
    
    # Sort by number of significant markers
    channel_summary = channel_summary.sort_values('num_sig_markers', ascending=False)
    
    # Save channel summary
    channel_file = os.path.join(OUTPUT_DIR, 'all_channel_effects.csv')
    channel_summary.to_csv(channel_file, index=False)
    print(f"Saved channel summary to {channel_file}")
    
    # Create visualization of results
    create_summary_visualization(marker_summary, channel_summary)
    
    print("Aggregation complete!")


def create_summary_visualization(marker_summary, channel_summary):
    """
    Create visualization of the aggregated results.
    
    Parameters
    ----------
    marker_summary : pandas.DataFrame
        Summary of significant clusters by marker
    channel_summary : pandas.DataFrame
        Summary of significant effects by channel
    """
    # Create a summary plot
    plt.figure(figsize=(12, 8))
    
    # Plot marker effects
    plt.subplot(2, 1, 1)
    sns.barplot(x='marker', y='num_sig_clusters', data=marker_summary.head(15))
    plt.title('Number of Significant Clusters by Marker (Top 15)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Plot channel effects
    plt.subplot(2, 1, 2)
    sns.barplot(x='channel', y='num_sig_markers', data=channel_summary.head(15))
    plt.title('Number of Significant Markers by Channel (Top 15)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(os.path.join(OUTPUT_DIR, 'cluster_results_summary.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved summary visualization to {os.path.join(OUTPUT_DIR, 'cluster_results_summary.png')}")


if __name__ == '__main__':
    aggregate_cluster_results() 