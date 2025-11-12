"""
Visualization module for cluster permutation test results.

This module creates topographic plots showing t-statistics with
significant clusters highlighted. Updated to work with the probe marker
pipeline and marker-specific results.
"""

import numpy as np
import matplotlib.pyplot as plt
import mne
from typing import List, Optional, Tuple
from pathlib import Path
import warnings
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages


def validate_montage_and_channels(info: mne.Info, values: np.ndarray) -> tuple:
    """
    Validate that the montage and channels are properly configured for plotting.
    
    Parameters
    ----------
    info : mne.Info
        MNE Info object with channel information
    values : np.ndarray
        Values array for each channel
        
    Returns
    -------
    tuple
        (validated_info, validated_values) - Ensures proper montage setup
    """
    # Check if montage is set
    if info.get_montage() is None:
        raise ValueError("No montage found in info object. Montage must be set for proper plotting.")
    
    # Validate channel count matches
    n_channels_info = len(info['ch_names'])
    n_channels_values = len(values)
    
    if n_channels_info != n_channels_values:
        raise ValueError(f"Channel count mismatch: info has {n_channels_info} channels, "
                        f"values has {n_channels_values} channels")
    
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
    
    return info, values


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
    dpi: int = 300,
    cluster_p_values_uncorrected: Optional[np.ndarray] = None
) -> plt.Figure:
    """
    Plot topographic map with t-statistics and significant clusters.
    
    Shows both uncorrected (empty circles) and corrected (filled black dots) 
    significant clusters when both p-values are provided.
    
    Parameters
    ----------
    t_stats : np.ndarray
        T-statistics for each channel, shape (n_channels,)
    clusters : List[np.ndarray]
        List of channel index arrays for each cluster
    cluster_p_values : np.ndarray
        P-values for each cluster (corrected if available)
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
    cluster_p_values_uncorrected : np.ndarray, optional
        Uncorrected p-values to show as empty circles
        
    Returns
    -------
    fig : plt.Figure
        Matplotlib figure object
    """
    # Validate montage and channel information
    info, t_stats = validate_montage_and_channels(info, t_stats)
    
    # Create masks for significant clusters
    # mask_corrected: filled black dots (primary display)
    # mask_uncorrected: empty circles (if provided)
    mask_corrected = np.zeros(len(t_stats), dtype=bool)
    
    for cluster, pval in zip(clusters, cluster_p_values):
        if pval < alpha:
            mask_corrected[cluster] = True
    
    # If uncorrected p-values provided, create separate mask
    mask_uncorrected = None
    if cluster_p_values_uncorrected is not None:
        mask_uncorrected = np.zeros(len(t_stats), dtype=bool)
        for cluster, pval in zip(clusters, cluster_p_values_uncorrected):
            if pval < alpha:
                mask_uncorrected[cluster] = True
    
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
    # Show corrected significant clusters as filled black dots
    im, _ = mne.viz.plot_topomap(
        t_stats_plot,
        info,
        axes=ax,
        show=False,
        cmap=cmap,
        vlim=(vmin, vmax),
        sensors=False,  # Don't show all sensors by default
        mask=mask_corrected,  # Corrected significant clusters
        mask_params=dict(marker='o', markerfacecolor='k', 
                        markeredgecolor='k', linewidth=0, markersize=6),
        contours=6,  # More contours for smoother appearance
        ch_type='eeg',
        sphere='auto',
        outlines='head',
        extrapolate='head',  # Extrapolate to head boundary
        image_interp='cubic',  # Cubic interpolation for smoother gradients
        border='mean',
        res=128  # Higher resolution for smoother interpolation
    )
    
    # If uncorrected p-values provided, overlay empty circles for uncorrected significant clusters
    if mask_uncorrected is not None:
        # Get channel positions using MNE's public API
        from mne.channels import find_layout
        layout = find_layout(info, ch_type='eeg')
        pos = layout.pos[:len(t_stats), :2]  # Get x, y positions
        
        # Plot empty circles for uncorrected significant clusters
        # Only plot if not already shown as corrected (to avoid overlap)
        uncorrected_only = mask_uncorrected & ~mask_corrected
        if np.any(uncorrected_only):
            ax.plot(pos[uncorrected_only, 0], pos[uncorrected_only, 1], 
                   'o', markerfacecolor='none', markeredgecolor='k', 
                   markeredgewidth=1.5, markersize=6, zorder=10)
    
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
    n_sig_corrected = np.sum(cluster_p_values < alpha)
    n_total_clusters = len(clusters)
    
    # Create cluster info text
    if cluster_p_values_uncorrected is not None:
        n_sig_uncorrected = np.sum(cluster_p_values_uncorrected < alpha)
        cluster_text = (f'Clusters: {n_sig_corrected}/{n_total_clusters} significant after MCC '
                       f'({n_sig_uncorrected} before MCC, α = {alpha})\n'
                       f'● = significant after MCC, ○ = significant before MCC only')
    else:
        cluster_text = f'Clusters: {n_sig_corrected}/{n_total_clusters} significant (α = {alpha})'
    
    fig.text(0.5, 0.02, cluster_text, ha='center', fontsize=9)
    
    plt.tight_layout()
    
    # Save figure if path provided
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)  # Close to free memory
        return None
    
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
    for bar, pval in zip(bars, cluster_p_values):
        stars = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else ''
        if stars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    stars, ha='center', va='bottom', fontsize=16, fontweight='bold')
    
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
    method: str = "threshold",
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
        Threshold used for cluster formation (only shown for threshold method)
    method : str
        Clustering method ("threshold" or "tfce")
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
    
    # Add threshold lines (only for threshold method)
    if method.lower() == "threshold":
        ax.axvline(x=threshold, color='red', linestyle='--', linewidth=2, 
                   label=f'Threshold = ±{threshold:.2f}')
        ax.axvline(x=-threshold, color='red', linestyle='--', linewidth=2)
    else:
        # For TFCE method, only show zero line
        ax.axvline(x=0, color='black', linestyle='-', linewidth=1, 
                   label='Zero line')
    
    # Labels and title
    ax.set_xlabel('T-statistic', fontsize=12)
    ax.set_ylabel('Number of channels', fontsize=12)
    
    # Method-specific title
    if method.lower() == "tfce":
        title = 'Distribution of T-statistics Across Channels (TFCE Method)'
    else:
        title = 'Distribution of T-statistics Across Channels (Threshold Method)'
    
    ax.set_title(title, fontsize=14, fontweight='bold')
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
    config: dict = None,
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
    # Get method from config or default to threshold
    method = 'threshold'  # Default
    if config is not None:
        method = config.get('clustering', {}).get('method', 'threshold')
    
    fig3 = plot_t_statistics_distribution(
        t_stats=t_stats,
        threshold=threshold,
        method=method,
        save_path=output_path / f"{prefix}_t_distribution.png"
    )
    plt.close(fig3)
    
    print(f"✓ Figures saved to {output_dir}")
    print(f"  - Topographic map: {prefix}_topomap.png")
    if len(clusters) > 0:
        print(f"  - Cluster details: {prefix}_cluster_details.png")
    print(f"  - T-statistics distribution: {prefix}_t_distribution.png")


def plot_raw_topomap(
    values: np.ndarray,
    info: mne.Info,
    title: str = "Topography",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cmap: str = 'RdBu_r',
    show_sensors: bool = True
) -> plt.Figure:
    """
    Plot raw topographic map without clusters or masks.
    
    Parameters
    ----------
    values : np.ndarray
        Values for each channel, shape (n_channels,)
    info : mne.Info
        MNE Info object with channel positions
    title : str
        Plot title
    vmin : float, optional
        Minimum value for colormap
    vmax : float, optional
        Maximum value for colormap
    cmap : str
        Colormap name
    show_sensors : bool
        Whether to show sensor positions
        
    Returns
    -------
    fig : plt.Figure
        Matplotlib figure object
    """
    # Validate montage and channel information
    info, values = validate_montage_and_channels(info, values)
    
    # Set colormap limits if not provided
    if vmin is None or vmax is None:
        abs_max = np.nanmax(np.abs(values))
        if np.isnan(abs_max) or abs_max == 0:
            abs_max = 1.0
        vmin = -abs_max
        vmax = abs_max
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Replace NaN with 0 for plotting
    values_plot = np.copy(values)
    nan_mask = np.isnan(values_plot)
    values_plot[nan_mask] = 0
    
    # Plot topographic map
    im, _ = mne.viz.plot_topomap(
        values_plot,
        info,
        axes=ax,
        show=False,
        cmap=cmap,
        vlim=(vmin, vmax),
        sensors=show_sensors,
        names=info['ch_names'] if show_sensors else None,  # Show electrode names
        contours=6,  # More contours for smoother appearance
        ch_type='eeg',
        sphere='auto',
        outlines='head',
        extrapolate='head',
        image_interp='cubic',  # Cubic interpolation for smoother gradients
        border='mean',
        res=128  # Higher resolution for smoother interpolation
    )
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Value', rotation=270, labelpad=20)
    
    # Add title
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    return fig


def create_raw_topography_report(
    power_data: np.ndarray,
    df_behavioral: pd.DataFrame,
    ch_names: List[str],
    info: mne.Info,
    marker_name: str,
    output_dir: str,
    subject_col: str = 'subject',
    subjects_per_page: int = 20,
    predictor_of_interest: str = None
) -> None:
    """
    Create PDF report with raw topographies for each subject and grand average.
    
    This function computes the average marker value per channel for each subject
    and across all subjects, then creates topographic plots saved to a PDF file.
    
    Parameters
    ----------
    power_data : np.ndarray
        Power values with shape (n_observations, n_channels)
    df_behavioral : pd.DataFrame
        Behavioral data with subject identifiers
    ch_names : List[str]
        List of channel names
    info : mne.Info
        MNE Info object with channel positions
    marker_name : str
        Name of the marker being analyzed
    output_dir : str
        Output directory for saving the PDF
    subject_col : str
        Column name containing subject identifiers
    subjects_per_page : int
        Number of subject topoplots to show per page (default: 20)
    predictor_of_interest : str, optional
        Name of the predictor variable to use for high/low split (e.g., 'onoff')
        If None, no high/low comparison will be created
        
    Notes
    -----
    Creates a PDF file named 'raw_topographies.pdf' in the output directory.
    The PDF contains:
    - Multiple topoplots per page (20 by default) showing average marker values per subject
    - One large topoplot showing grand average across all subjects
    - High/Low median split comparison page showing topographies for high, low, and difference
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    pdf_path = output_path / "raw_topographies.pdf"
    
    # Get unique subjects
    subjects = sorted(df_behavioral[subject_col].unique())
    n_subjects = len(subjects)
    
    print(f"\nCreating raw topography report for {marker_name}...")
    print(f"  Subjects: {n_subjects}")
    print(f"  Channels: {len(ch_names)}")
    print(f"  Subjects per page: {subjects_per_page}")
    
    # Compute average per subject
    subject_averages = {}
    for subject in subjects:
        subject_mask = df_behavioral[subject_col] == subject
        subject_data = power_data[subject_mask, :]
        
        # Compute mean across observations for this subject
        subject_mean = np.nanmean(subject_data, axis=0)
        subject_averages[subject] = subject_mean
    
    # Compute grand average
    grand_average = np.nanmean(power_data, axis=0)
    
    # Determine colormap limits from all data
    all_subject_values = np.array(list(subject_averages.values()))
    all_values = np.concatenate([all_subject_values.flatten(), grand_average])
    vmin = np.nanmin(all_values)
    vmax = np.nanmax(all_values)
    
    # Create PDF
    with PdfPages(pdf_path) as pdf:
        # Plot individual subjects (multiple per page)
        # Arrange in a grid: calculate rows and columns
        n_cols = 5  # 5 columns
        n_rows = 4  # 4 rows
        plots_per_page = n_rows * n_cols
        
        # Ensure subjects_per_page matches the grid size
        if subjects_per_page != plots_per_page:
            subjects_per_page = plots_per_page
        
        # Process subjects in batches
        for page_start in range(0, n_subjects, subjects_per_page):
            page_end = min(page_start + subjects_per_page, n_subjects)
            page_subjects = subjects[page_start:page_end]
            
            # Create figure with subplots
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 16))
            axes = axes.flatten()
            
            # Plot each subject in this page
            for idx, subject in enumerate(page_subjects):
                subject_values = subject_averages[subject]
                
                # Skip if all NaN
                if np.all(np.isnan(subject_values)):
                    continue
                
                ax = axes[idx]
                title = f"S{subject}"
                
                # Plot raw topomap on this subplot
                info_valid, values_valid = validate_montage_and_channels(info, subject_values)
                
                # Replace NaN with 0 for plotting
                values_plot = np.copy(subject_values)
                nan_mask = np.isnan(values_plot)
                values_plot[nan_mask] = 0
                
                # Plot topographic map
                im, _ = mne.viz.plot_topomap(
                    values_plot,
                    info_valid,
                    axes=ax,
                    show=False,
                    cmap='RdBu_r',
                    vlim=(vmin, vmax),
                    sensors=True,  # Show sensors for individual plots
                    names=info_valid['ch_names'],  # Show electrode names
                    contours=6,  # More contours for smoother appearance
                    ch_type='eeg',
                    sphere='auto',
                    outlines='head',
                    extrapolate='head',
                    image_interp='cubic',  # Cubic interpolation for smoother gradients
                    border='mean',
                    res=128  # Higher resolution for smoother interpolation
                )
                
                # Add title
                ax.set_title(title, fontsize=10, fontweight='bold')
            
            # Hide unused subplots
            for idx in range(len(page_subjects), len(axes)):
                axes[idx].axis('off')
            
            plt.suptitle(f"{marker_name} - Raw Topographies", fontsize=16, fontweight='bold', y=0.98)
            plt.tight_layout(rect=[0, 0, 1, 0.98])
            
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
            
            if page_end % subjects_per_page == 0 or page_end == n_subjects:
                print(f"  Processed {page_end}/{n_subjects} subjects...")
        
        # Plot grand average - larger size
        if not np.all(np.isnan(grand_average)):
            fig = plot_raw_topomap(
                values=grand_average,
                info=info,
                title=f"Grand Average (All Subjects) - {marker_name}",
                vmin=vmin,
                vmax=vmax,
                cmap='RdBu_r',
                show_sensors=True
            )
            
            # Make figure larger for grand average
            fig.set_size_inches(12, 10)
            
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
    
        # Create high/low median split comparison page
        if predictor_of_interest is not None:
            print("\n  Creating high/low median split comparison...")
            
            # Use predictor_of_interest to split the data
            # This is the behavioral variable (e.g., 'onoff'), not the EEG marker
            if predictor_of_interest in df_behavioral.columns:
                predictor_values = df_behavioral[predictor_of_interest].values
            
                # Calculate median
                median_value = np.nanmedian(predictor_values)
                
                # Split data into high and low groups
                high_mask = predictor_values >= median_value
                low_mask = predictor_values < median_value
            
                # Compute average topographies for high and low groups
                high_data = power_data[high_mask, :]
                low_data = power_data[low_mask, :]
                
                high_avg = np.nanmean(high_data, axis=0)
                low_avg = np.nanmean(low_data, axis=0)
                difference = high_avg - low_avg
                
                # Create figure with 3 subplots
                fig, axes = plt.subplots(1, 3, figsize=(18, 6))
                
                # Determine common colormap limits for high and low
                all_values = np.concatenate([high_avg, low_avg])
                vmin_common = np.nanmin(all_values)
                vmax_common = np.nanmax(all_values)
                
                # Determine colormap limits for difference
                diff_abs_max = np.nanmax(np.abs(difference))
                if np.isnan(diff_abs_max) or diff_abs_max == 0:
                    diff_abs_max = 1.0
                vmin_diff = -diff_abs_max
                vmax_diff = diff_abs_max
                
                # Plot HIGH values
                ax = axes[0]
                info_valid, values_valid = validate_montage_and_channels(info, high_avg)
                values_plot = np.copy(high_avg)
                nan_mask = np.isnan(values_plot)
                values_plot[nan_mask] = 0
                
                im, _ = mne.viz.plot_topomap(
                    values_plot,
                    info_valid,
                    axes=ax,
                    show=False,
                    cmap='RdBu_r',
                    vlim=(vmin_common, vmax_common),
                    sensors=True,
                    names=info_valid['ch_names'],  # Show electrode names
                    contours=6,
                    ch_type='eeg',
                    sphere='auto',
                    outlines='head',
                    extrapolate='head',
                    image_interp='cubic',
                    border='mean',
                    res=128
                )
                ax.set_title(f'High {predictor_of_interest}\n(≥ median = {median_value:.3f}, n={np.sum(high_mask)})', 
                            fontsize=12, fontweight='bold')
                cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cbar.set_label('Value', rotation=270, labelpad=20)
                
                # Plot LOW values
                ax = axes[1]
                info_valid, values_valid = validate_montage_and_channels(info, low_avg)
                values_plot = np.copy(low_avg)
                nan_mask = np.isnan(values_plot)
                values_plot[nan_mask] = 0
                
                im, _ = mne.viz.plot_topomap(
                    values_plot,
                    info_valid,
                    axes=ax,
                    show=False,
                    cmap='RdBu_r',
                    vlim=(vmin_common, vmax_common),
                    sensors=True,
                    names=info_valid['ch_names'],  # Show electrode names
                    contours=6,
                    ch_type='eeg',
                    sphere='auto',
                    outlines='head',
                    extrapolate='head',
                    image_interp='cubic',
                    border='mean',
                    res=128
                )
                ax.set_title(f'Low {predictor_of_interest}\n(< median = {median_value:.3f}, n={np.sum(low_mask)})', 
                            fontsize=12, fontweight='bold')
                cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cbar.set_label('Value', rotation=270, labelpad=20)
                
                # Plot DIFFERENCE (High - Low)
                ax = axes[2]
                info_valid, values_valid = validate_montage_and_channels(info, difference)
                values_plot = np.copy(difference)
                nan_mask = np.isnan(values_plot)
                values_plot[nan_mask] = 0
                
                im, _ = mne.viz.plot_topomap(
                    values_plot,
                    info_valid,
                    axes=ax,
                    show=False,
                    cmap='RdBu_r',
                    vlim=(vmin_diff, vmax_diff),
                    sensors=True,
                    names=info_valid['ch_names'],  # Show electrode names
                    contours=6,
                    ch_type='eeg',
                    sphere='auto',
                    outlines='head',
                    extrapolate='head',
                    image_interp='cubic',
                    border='mean',
                    res=128
                )
                ax.set_title(f'Difference\n(High - Low)', 
                            fontsize=12, fontweight='bold')
                cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cbar.set_label('Difference', rotation=270, labelpad=20)
                
                plt.suptitle(f"{marker_name} - High/Low {predictor_of_interest} Comparison", 
                            fontsize=16, fontweight='bold', y=0.98)
                plt.tight_layout(rect=[0, 0, 1, 0.96])
                
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)
                
                print(f"  ✓ High/Low comparison added for '{predictor_of_interest}' (median = {median_value:.3f})")
            else:
                print(f"  ⚠ Warning: Predictor '{predictor_of_interest}' not found in behavioral data. Skipping high/low comparison.")
    
    # Calculate number of pages
    n_pages_subjects = int(np.ceil(n_subjects / subjects_per_page))
    n_pages_total = n_pages_subjects + 2  # +1 for grand average, +1 for high/low comparison
    
    print(f"✓ Raw topography report saved to {pdf_path}")
    print(f"  Total pages: {n_pages_total} ({n_pages_subjects} pages with {subjects_per_page} subjects each + 1 grand average page + 1 high/low comparison page)")
