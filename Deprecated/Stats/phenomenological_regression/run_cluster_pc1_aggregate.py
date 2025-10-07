#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to aggregate PC1 cluster permutation test results from all markers.
This script should be run after all SLURM jobs have finished.
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
RESULTS_DIR = os.path.join(project_root, 'results/cluster_permutation_tests_pc1')
OUTPUT_DIR = os.path.join(project_root, 'results/cluster_permutation_pc1_interpretation')
ALPHA = 0.05  # significance level

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

def aggregate_pc1_cluster_results():
    """
    Aggregate PC1 cluster results from all markers and create summary files.
    """
    print("Aggregating PC1 cluster results...")
    
    # Find all PC1 cluster result files
    cluster_files = glob.glob(os.path.join(RESULTS_DIR, '*_pc1_cluster_results_clusters.csv'))
    
    if not cluster_files:
        print(f"No PC1 cluster result files found in {RESULTS_DIR}")
        return
    
    print(f"Found {len(cluster_files)} PC1 cluster result files")
    
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
        print("No valid PC1 cluster data found")
        return
    
    # Combine all cluster results
    cluster_df = pd.concat(all_clusters, ignore_index=True)
    
    # Save combined results
    combined_file = os.path.join(OUTPUT_DIR, 'all_pc1_cluster_results.csv')
    cluster_df.to_csv(combined_file, index=False)
    print(f"Saved combined PC1 cluster results to {combined_file}")
    
    # Create summary of significant clusters by marker
    sig_clusters = cluster_df[cluster_df['significant']]
    
    if sig_clusters.empty:
        print("No significant PC1 clusters found")
        return
    
    # Group by marker and count significant clusters
    marker_summary = (
        sig_clusters
        .groupby('marker')
        .agg({
            'cluster_id': 'count',
            'p_value': 'min',
            'cluster_size': ['mean', 'max'],
            'max_abs_t': ['mean', 'max'],
            'mean_t': ['mean', 'std']
        })
        .reset_index()
    )
    
    # Flatten multi-level columns
    marker_summary.columns = [
        '_'.join(col).strip('_') for col in marker_summary.columns.values
    ]
    
    # Rename columns for clarity
    marker_summary = marker_summary.rename(columns={
        'cluster_id_count': 'num_sig_pc1_clusters',
        'p_value_min': 'min_p_value',
        'cluster_size_mean': 'avg_cluster_size',
        'cluster_size_max': 'max_cluster_size',
        'max_abs_t_mean': 'avg_max_t',
        'max_abs_t_max': 'overall_max_t',
        'mean_t_mean': 'avg_mean_t',
        'mean_t_std': 'std_mean_t'
    })
    
    # Sort by number of significant clusters and then by min p-value
    marker_summary = marker_summary.sort_values(
        ['num_sig_pc1_clusters', 'min_p_value'], 
        ascending=[False, True]
    )
    
    # Save marker summary
    summary_file = os.path.join(OUTPUT_DIR, 'pc1_marker_effects_summary.csv')
    marker_summary.to_csv(summary_file, index=False)
    print(f"Saved PC1 marker summary to {summary_file}")
    
    # Create channel-level summary
    # Find all t-statistics files
    t_stat_files = glob.glob(os.path.join(RESULTS_DIR, '*_pc1_cluster_results_t_statistics.csv'))
    
    all_t_stats = []
    for file in t_stat_files:
        try:
            df = pd.read_csv(file)
            if not df.empty:
                all_t_stats.append(df)
        except Exception as e:
            print(f"Error reading {file}: {e}")
    
    if not all_t_stats:
        print("No valid PC1 t-statistic data found")
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
    t_stats_df['in_sig_pc1_cluster'] = t_stats_df.apply(
        lambda x: (x['marker'], x['channel']) in sig_channels, 
        axis=1
    )
    
    # Group by channel and count significant markers
    channel_summary = (
        t_stats_df[t_stats_df['in_sig_pc1_cluster']]
        .groupby('channel')
        .agg({
            'marker': 'nunique',
            'PC1_t_statistic': ['mean', 'std', 'min', 'max']
        })
        .reset_index()
    )
    
    # Flatten multi-level columns
    channel_summary.columns = [
        '_'.join(col).strip('_') for col in channel_summary.columns.values
    ]
    
    # Rename columns for clarity
    channel_summary = channel_summary.rename(columns={
        'marker_nunique': 'num_sig_pc1_markers',
        'PC1_t_statistic_mean': 'avg_pc1_t_stat',
        'PC1_t_statistic_std': 'std_pc1_t_stat',
        'PC1_t_statistic_min': 'min_pc1_t_stat',
        'PC1_t_statistic_max': 'max_pc1_t_stat'
    })
    
    # Sort by number of significant markers
    channel_summary = channel_summary.sort_values('num_sig_pc1_markers', ascending=False)
    
    # Save channel summary
    channel_file = os.path.join(OUTPUT_DIR, 'all_pc1_channel_effects.csv')
    channel_summary.to_csv(channel_file, index=False)
    print(f"Saved PC1 channel summary to {channel_file}")
    
    # Create PC1 effect direction analysis
    pc1_direction_analysis(t_stats_df, sig_clusters)
    
    # Create visualization of results
    create_pc1_summary_visualization(marker_summary, channel_summary)
    
    print("PC1 cluster aggregation complete!")


def pc1_direction_analysis(t_stats_df, sig_clusters):
    """
    Analyze the direction of PC1 effects (positive vs negative).
    
    Parameters
    ----------
    t_stats_df : pandas.DataFrame
        Combined t-statistics for all markers
    sig_clusters : pandas.DataFrame
        Significant clusters
    """
    print("\nAnalyzing PC1 effect directions...")
    
    # Analyze overall PC1 effect directions
    pc1_effects = t_stats_df.copy()
    pc1_effects['pc1_effect_direction'] = np.where(
        pc1_effects['PC1_t_statistic'] > 0, 'Positive', 'Negative'
    )
    
    # Count positive vs negative effects by marker
    direction_summary = (
        pc1_effects.groupby(['marker', 'pc1_effect_direction'])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    
    if 'Positive' not in direction_summary.columns:
        direction_summary['Positive'] = 0
    if 'Negative' not in direction_summary.columns:
        direction_summary['Negative'] = 0
        
    direction_summary['total_channels'] = direction_summary['Positive'] + direction_summary['Negative']
    direction_summary['positive_ratio'] = direction_summary['Positive'] / direction_summary['total_channels']
    direction_summary['dominant_direction'] = np.where(
        direction_summary['positive_ratio'] > 0.5, 'Positive', 'Negative'
    )
    
    # Save direction analysis
    direction_file = os.path.join(OUTPUT_DIR, 'pc1_effect_directions.csv')
    direction_summary.to_csv(direction_file, index=False)
    print(f"Saved PC1 effect directions to {direction_file}")
    
    # Analyze significant cluster directions
    if not sig_clusters.empty:
        sig_direction_summary = []
        
        for _, cluster in sig_clusters.iterrows():
            marker = cluster['marker']
            mean_t = cluster['mean_t']
            direction = 'Positive' if mean_t > 0 else 'Negative'
            
            sig_direction_summary.append({
                'marker': marker,
                'cluster_id': cluster['cluster_id'],
                'mean_t': mean_t,
                'direction': direction,
                'cluster_size': cluster['cluster_size'],
                'p_value': cluster['p_value']
            })
        
        sig_direction_df = pd.DataFrame(sig_direction_summary)
        
        # Count significant clusters by direction
        sig_direction_counts = (
            sig_direction_df.groupby(['marker', 'direction'])
            .size()
            .unstack(fill_value=0)
            .reset_index()
        )
        
        sig_direction_file = os.path.join(OUTPUT_DIR, 'pc1_significant_cluster_directions.csv')
        sig_direction_df.to_csv(sig_direction_file, index=False)
        print(f"Saved significant PC1 cluster directions to {sig_direction_file}")


def create_pc1_summary_visualization(marker_summary, channel_summary):
    """
    Create visualization of the PC1 cluster results.
    
    Parameters
    ----------
    marker_summary : pandas.DataFrame
        Summary of significant clusters by marker
    channel_summary : pandas.DataFrame
        Summary of significant effects by channel
    """
    # Create a summary plot
    plt.figure(figsize=(16, 12))
    
    # Plot 1: Number of significant PC1 clusters by marker
    plt.subplot(3, 2, 1)
    top_markers = marker_summary.head(15)
    bars1 = plt.bar(range(len(top_markers)), top_markers['num_sig_pc1_clusters'])
    plt.title('Number of Significant PC1 Clusters by Marker (Top 15)', fontsize=12, fontweight='bold')
    plt.xlabel('Marker')
    plt.ylabel('Number of Significant PC1 Clusters')
    plt.xticks(range(len(top_markers)), top_markers['marker'], rotation=45, ha='right')
    
    # Add value labels on bars
    for i, bar in enumerate(bars1):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{int(height)}', ha='center', va='bottom', fontsize=8)
    
    # Plot 2: PC1 effect strength by marker
    plt.subplot(3, 2, 2)
    bars2 = plt.bar(range(len(top_markers)), top_markers['overall_max_t'], alpha=0.7, color='orange')
    plt.title('Maximum PC1 T-statistic by Marker (Top 15)', fontsize=12, fontweight='bold')
    plt.xlabel('Marker')
    plt.ylabel('Maximum |T-statistic|')
    plt.xticks(range(len(top_markers)), top_markers['marker'], rotation=45, ha='right')
    
    # Plot 3: Number of significant PC1 markers by channel
    plt.subplot(3, 2, 3)
    top_channels = channel_summary.head(15)
    bars3 = plt.bar(range(len(top_channels)), top_channels['num_sig_pc1_markers'], color='green', alpha=0.7)
    plt.title('Number of Significant PC1 Markers by Channel (Top 15)', fontsize=12, fontweight='bold')
    plt.xlabel('Channel')
    plt.ylabel('Number of Significant PC1 Markers')
    plt.xticks(range(len(top_channels)), top_channels['channel'], rotation=45, ha='right')
    
    # Plot 4: Average PC1 cluster size
    plt.subplot(3, 2, 4)
    bars4 = plt.bar(range(len(top_markers)), top_markers['avg_cluster_size'], color='purple', alpha=0.7)
    plt.title('Average PC1 Cluster Size by Marker (Top 15)', fontsize=12, fontweight='bold')
    plt.xlabel('Marker')
    plt.ylabel('Average Cluster Size (channels)')
    plt.xticks(range(len(top_markers)), top_markers['marker'], rotation=45, ha='right')
    
    # Plot 5: P-value distribution
    plt.subplot(3, 2, 5)
    plt.hist(top_markers['min_p_value'], bins=10, alpha=0.7, color='red', edgecolor='black')
    plt.title('Distribution of Minimum P-values (Top 15 Markers)', fontsize=12, fontweight='bold')
    plt.xlabel('Minimum P-value')
    plt.ylabel('Frequency')
    plt.axvline(x=0.05, color='black', linestyle='--', alpha=0.7, label='α = 0.05')
    plt.legend()
    
    # Plot 6: Summary statistics text
    plt.subplot(3, 2, 6)
    plt.axis('off')
    
    total_markers = len(marker_summary)
    total_sig_clusters = marker_summary['num_sig_pc1_clusters'].sum()
    total_channels = len(channel_summary)
    avg_cluster_size = marker_summary['avg_cluster_size'].mean()
    
    summary_text = f"""PC1 Cluster Analysis Summary
    
Total markers with significant PC1 clusters: {total_markers}
Total significant PC1 clusters found: {total_sig_clusters}
Total channels with significant PC1 effects: {total_channels}
Average cluster size: {avg_cluster_size:.1f} channels

Top markers by PC1 cluster count:
{chr(10).join([f'  {row.marker}: {row.num_sig_pc1_clusters} clusters' 
               for _, row in marker_summary.head(5).iterrows()])}

Top channels by PC1 marker count:
{chr(10).join([f'  {row.channel}: {row.num_sig_pc1_markers} markers' 
               for _, row in channel_summary.head(5).iterrows()])}"""
    
    plt.text(0.05, 0.95, summary_text, transform=plt.gca().transAxes, 
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    
    # Save the plot
    plot_file = os.path.join(OUTPUT_DIR, 'pc1_cluster_results_summary.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved PC1 summary visualization to {plot_file}")
    
    # Create additional heatmap visualization
    create_pc1_heatmap_visualization(marker_summary, channel_summary)


def create_pc1_heatmap_visualization(marker_summary, channel_summary):
    """
    Create heatmap visualizations for PC1 effects.
    """
    if len(marker_summary) == 0 or len(channel_summary) == 0:
        print("Insufficient data for heatmap visualization")
        return
    
    plt.figure(figsize=(14, 10))
    
    # Create marker x metric heatmap
    plt.subplot(2, 1, 1)
    
    # Select top markers and relevant metrics
    top_markers = marker_summary.head(20)
    heatmap_data = top_markers[['num_sig_pc1_clusters', 'avg_cluster_size', 'overall_max_t']].T
    heatmap_data.columns = top_markers['marker']
    
    # Normalize each row to 0-1 scale for better visualization
    heatmap_data_norm = heatmap_data.div(heatmap_data.max(axis=1), axis=0)
    
    sns.heatmap(heatmap_data_norm, annot=False, cmap='viridis', cbar_kws={'label': 'Normalized Value'})
    plt.title('PC1 Effects Heatmap: Top 20 Markers', fontsize=14, fontweight='bold')
    plt.ylabel('Metrics')
    plt.xlabel('Markers')
    
    # Create channel effects heatmap
    plt.subplot(2, 1, 2)
    
    # Select top channels
    top_channels = channel_summary.head(20)
    channel_heatmap = top_channels[['num_sig_pc1_markers', 'avg_pc1_t_stat', 'max_pc1_t_stat']].T
    channel_heatmap.columns = top_channels['channel']
    
    # Normalize
    channel_heatmap_norm = channel_heatmap.div(channel_heatmap.max(axis=1), axis=0)
    
    sns.heatmap(channel_heatmap_norm, annot=False, cmap='plasma', cbar_kws={'label': 'Normalized Value'})
    plt.title('PC1 Channel Effects Heatmap: Top 20 Channels', fontsize=14, fontweight='bold')
    plt.ylabel('Metrics')
    plt.xlabel('Channels')
    
    plt.tight_layout()
    
    # Save heatmap
    heatmap_file = os.path.join(OUTPUT_DIR, 'pc1_effects_heatmap.png')
    plt.savefig(heatmap_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved PC1 effects heatmap to {heatmap_file}")


if __name__ == '__main__':
    aggregate_pc1_cluster_results() 