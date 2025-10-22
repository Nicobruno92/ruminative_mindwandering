"""
Visualization module for cluster permutation test results.

This module creates topographic plots showing t-statistics with
significant clusters highlighted. Updated to work with the probe marker
pipeline and marker-specific results.
"""

import numpy as np
import matplotlib.pyplot as plt
import mne
from typing import List, Optional
from pathlib import Path
import warnings


def validate_montage_and_channels(info: mne.Info, t_stats: np.ndarray) -> tuple:
    """
    Validate that the montage and channels are properly configured for plotting.
    
    Parameters
    ----------
    info : mne.Info
        MNE Info object with channel information
    t_stats : np.ndarray
        T-statistics array
        
    Returns
    -------
    tuple
        (validated_info, validated_t_stats) - Ensures proper montage setup
    """
    # Check if montage is set
    if info.get_montage() is None:
        raise ValueError("No montage found in info object. Montage must be set for proper plotting.")
    
    # Validate channel count matches
    n_channels_info = len(info['ch_names'])
    n_channels_stats = len(t_stats)
    
    if n_channels_info != n_channels_stats:
        raise ValueError(f"Channel count mismatch: info has {n_channels_info} channels, "
                        f"t_stats has {n_channels_stats} channels")
    
    # Check for missing channel positions
    montage = info.get_montage()
    if montage is None:
        raise ValueError("Montage is None after validation")
    
    # Ensure all channels have positions
    missing_positions = []
    for ch_name in info['ch_names']:
        if ch_name not in montage.ch_names:
            missing_positions.append(ch_name)
    
    if missing_positions:
        warnings.warn(f"Channels missing from montage: {missing_positions}")
    
    return info, t_stats


def plot_cluster_topomap(
    t_stats: np.ndarray,
    clusters: List[np.ndarray],
    cluster_p_values: np.ndarray,
    info: mne.Info,
    alpha: float = 0.05,
    title: str = "Spatial Cluster Permutation Test",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cmap: str = 'RdBu_r',
    save_path: Optional[str] = None,
    dpi: int = 300
) -> plt.Figure:
    """
    Plot topographic map with t-statistics and significant clusters.
    
    Parameters
    ----------
    t_stats : np.ndarray
        T-statistics for each channel, shape (n_channels,)
    clusters : List[np.ndarray]
        List of channel index arrays for each cluster
    cluster_p_values : np.ndarray
        P-values for each cluster
    info : mne.Info
        MNE Info object with channel positions
    alpha : float
        Significance threshold
    title : str
        Plot title
    vmin : float, optional
        Minimum value for colormap
    vmax : float, optional
        Maximum value for colormap
    cmap : str
        Colormap name
    save_path : str, optional
        Path to save figure
    dpi : int
        Figure resolution
        
    Returns
    -------
    fig : plt.Figure
        Matplotlib figure object
    """
    # Validate montage and channel information
    info, t_stats = validate_montage_and_channels(info, t_stats)
    
    # Create mask for significant clusters
    mask = np.zeros(len(t_stats), dtype=bool)
    
    for cluster, pval in zip(clusters, cluster_p_values):
        if pval < alpha:
            mask[cluster] = True
    
    # Set colormap limits if not provided
    if vmin is None or vmax is None:
        # Use nanmax to handle NaN values properly
        abs_max = np.nanmax(np.abs(t_stats))
        if np.isnan(abs_max) or abs_max == 0:
            # Fallback to reasonable defaults if all values are NaN or zero
            abs_max = 1.0
        vmin = -abs_max
        vmax = abs_max
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # MNE's plot_topomap doesn't handle NaN values well
    # Replace NaN with 0 for plotting (won't affect significant clusters)
    t_stats_plot = np.copy(t_stats)
    nan_mask = np.isnan(t_stats_plot)
    t_stats_plot[nan_mask] = 0
    
    # Plot topographic map with proper montage
    # Show electrode markers only for significant clusters
    # Use mask to control which electrodes get markers
    im, _ = mne.viz.plot_topomap(
        t_stats_plot,
        info,
        axes=ax,
        show=False,
        cmap=cmap,
        vlim=(vmin, vmax),
        sensors=False,  # Don't show all sensors by default
        mask=mask,  # Only significant electrodes
        mask_params=dict(marker='o', markerfacecolor='k', 
                        markeredgecolor='k', linewidth=0, markersize=5),
        contours=6,
        ch_type='eeg',
        sphere='auto',
        outlines='head',
        extrapolate='head',  # Extrapolate to head boundary
        image_interp='cubic',
        border='mean',
        res=64
    )
    
    # Clip the image to a circular mask (head boundary)
    # Get the image extent and create circular mask
    from matplotlib.patches import Circle
    radius = 0.5  # Normalized radius for head circle
    circle = Circle((0.5, 0.5), radius, transform=ax.transAxes, 
                    facecolor='none', edgecolor='none')
    im.set_clip_path(circle)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('T-statistic', rotation=270, labelpad=20)
    
    # Add title
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Add text with cluster information
    n_sig_clusters = np.sum(cluster_p_values < alpha)
    n_total_clusters = len(clusters)
    fig.text(0.5, 0.02, f'Clusters: {n_sig_clusters}/{n_total_clusters} significant (α = {alpha})',
             ha='center', fontsize=10)
    
    plt.tight_layout()
    
    # Save figure if path provided
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
    
    return fig


def plot_cluster_details(
    clusters: List[np.ndarray],
    cluster_stats: np.ndarray,
    cluster_p_values: np.ndarray,
    ch_names: List[str],
    alpha: float = 0.05,
    save_path: Optional[str] = None,
    dpi: int = 300
) -> plt.Figure:
    """
    Create detailed bar plot of cluster statistics.
    
    Parameters
    ----------
    clusters : List[np.ndarray]
        List of channel index arrays
    cluster_stats : np.ndarray
        Cluster statistics
    cluster_p_values : np.ndarray
        Cluster p-values
    ch_names : List[str]
        Channel names
    alpha : float
        Significance threshold
    save_path : str, optional
        Path to save figure
    dpi : int
        Figure resolution
        
    Returns
    -------
    fig : plt.Figure
        Matplotlib figure object
    """
    n_clusters = len(clusters)
    
    if n_clusters == 0:
        print("No clusters to plot")
        return None
    
    # Create figure
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot 1: Cluster statistics
    ax1 = axes[0]
    colors = ['red' if p < alpha else 'gray' for p in cluster_p_values]
    bars = ax1.bar(range(1, n_clusters + 1), cluster_stats, color=colors, alpha=0.7)
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax1.set_xlabel('Cluster ID', fontsize=12)
    ax1.set_ylabel('Cluster Statistic (sum of t-values)', fontsize=12)
    ax1.set_title('Cluster Statistics', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    # Add significance stars
    for i, (bar, pval) in enumerate(zip(bars, cluster_p_values)):
        if pval < alpha:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    '*' if pval < 0.05 else '**' if pval < 0.01 else '***',
                    ha='center', va='bottom', fontsize=16, fontweight='bold')
    
    # Plot 2: P-values
    ax2 = axes[1]
    ax2.bar(range(1, n_clusters + 1), cluster_p_values, color=colors, alpha=0.7)
    ax2.axhline(y=alpha, color='red', linestyle='--', linewidth=2, label=f'α = {alpha}')
    ax2.set_xlabel('Cluster ID', fontsize=12)
    ax2.set_ylabel('P-value', fontsize=12)
    ax2.set_title('Cluster P-values', fontsize=14, fontweight='bold')
    ax2.set_ylim([0, 1])
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure if path provided
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
    
    return fig


def plot_t_statistics_distribution(
    t_stats: np.ndarray,
    threshold: float,
    save_path: Optional[str] = None,
    dpi: int = 300
) -> plt.Figure:
    """
    Plot distribution of t-statistics with threshold lines.
    
    Parameters
    ----------
    t_stats : np.ndarray
        T-statistics for each channel
    threshold : float
        Threshold used for cluster formation
    save_path : str, optional
        Path to save figure
    dpi : int
        Figure resolution
        
    Returns
    -------
    fig : plt.Figure
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Filter out NaN values for plotting
    t_stats_valid = t_stats[~np.isnan(t_stats)]
    
    if len(t_stats_valid) == 0:
        # No valid data to plot
        ax.text(0.5, 0.5, 'No valid t-statistics to display', 
                ha='center', va='center', transform=ax.transAxes, fontsize=14)
        return fig
    
    # Plot histogram
    ax.hist(t_stats_valid, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
    
    # Add threshold lines
    ax.axvline(x=threshold, color='red', linestyle='--', linewidth=2, 
               label=f'Threshold = ±{threshold:.2f}')
    ax.axvline(x=-threshold, color='red', linestyle='--', linewidth=2)
    ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
    
    # Labels and title
    ax.set_xlabel('T-statistic', fontsize=12)
    ax.set_ylabel('Number of channels', fontsize=12)
    ax.set_title('Distribution of T-statistics Across Channels', 
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    
    # Add statistics text using valid values only
    n_nan = np.sum(np.isnan(t_stats))
    stats_text = f'Mean: {np.mean(t_stats_valid):.3f}\nStd: {np.std(t_stats_valid):.3f}'
    if n_nan > 0:
        stats_text += f'\nNaN channels: {n_nan}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # Save figure if path provided
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
    
    return fig


def create_results_report(
    t_stats: np.ndarray,
    clusters: List[np.ndarray],
    cluster_stats: np.ndarray,
    cluster_p_values: np.ndarray,
    info: mne.Info,
    threshold: float,
    alpha: float,
    marker_name: str,
    output_dir: str,
    prefix: Optional[str] = None
) -> None:
    """
    Create comprehensive visualization report for probe marker analysis.
    
    Parameters
    ----------
    t_stats : np.ndarray
        T-statistics for each channel
    clusters : List[np.ndarray]
        List of channel index arrays
    cluster_stats : np.ndarray
        Cluster statistics
    cluster_p_values : np.ndarray
        Cluster p-values
    info : mne.Info
        MNE Info object with channel positions
    threshold : float
        Cluster formation threshold
    alpha : float
        Significance level
    marker_name : str
        Name of the marker being analyzed
    output_dir : str
        Output directory
    prefix : str, optional
        Filename prefix (defaults to marker_name)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create safe filename prefix from marker name
    if prefix is None:
        safe_marker_name = marker_name.replace('/', '_').replace(' ', '_')
        prefix = f"results_{safe_marker_name}"
    
    # Create title with marker name
    plot_title = f"Spatial Cluster Test - {marker_name}"
    
    # Plot 1: Topographic map
    fig1 = plot_cluster_topomap(
        t_stats=t_stats,
        clusters=clusters,
        cluster_p_values=cluster_p_values,
        info=info,
        alpha=alpha,
        title=plot_title,
        save_path=output_path / f"{prefix}_topomap.png"
    )
    plt.close(fig1)
    
    # Plot 2: Cluster details
    if len(clusters) > 0:
        fig2 = plot_cluster_details(
            clusters=clusters,
            cluster_stats=cluster_stats,
            cluster_p_values=cluster_p_values,
            ch_names=info['ch_names'],
            alpha=alpha,
            save_path=output_path / f"{prefix}_cluster_details.png"
        )
        if fig2:
            plt.close(fig2)
    
    # Plot 3: T-statistics distribution
    fig3 = plot_t_statistics_distribution(
        t_stats=t_stats,
        threshold=threshold,
        save_path=output_path / f"{prefix}_t_distribution.png"
    )
    plt.close(fig3)
    
    print(f"✓ Figures saved to {output_dir}")
    print(f"  - Topographic map: {prefix}_topomap.png")
    if len(clusters) > 0:
        print(f"  - Cluster details: {prefix}_cluster_details.png")
    print(f"  - T-statistics distribution: {prefix}_t_distribution.png")
