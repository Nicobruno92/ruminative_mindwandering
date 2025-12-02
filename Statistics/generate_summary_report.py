"""
Generate comprehensive summary report for LMM cluster analysis results.

This module creates a visual report that aggregates all marker results into:
1. Summary tables with all statistical results
2. Comparison topoplots showing all markers side-by-side
3. Easy visual identification of effects across markers

The report is generated after all markers in a run have been processed.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
import mne
from pathlib import Path
from typing import List, Dict
import pickle
from datetime import datetime

from plot_results import plot_cluster_topomap


# ============================================================================
# CONFIGURATION
# ============================================================================

# Figure settings
FIGURE_DPI = 300
TOPOMAP_SIZE = 3.5  # inches per topomap
COLORMAP = 'RdBu_r'


# ============================================================================
# REPORT GENERATION FUNCTIONS
# ============================================================================

def load_all_marker_results(model_dir: Path, verbose: bool = True) -> List[Dict]:
    """
    Load all marker results from a model directory.
    
    Parameters
    ----------
    model_dir : Path
        Path to model directory containing marker subdirectories
    verbose : bool
        Whether to print loading progress
        
    Returns
    -------
    List[Dict]
        List of result dictionaries, one per marker
    """
    results = []
    
    # Find all marker directories (contain results.pkl)
    marker_dirs = [d for d in model_dir.iterdir() if d.is_dir()]
    
    if verbose:
        print(f"Scanning {len(marker_dirs)} directories for results...")
    
    for marker_dir in sorted(marker_dirs):
        results_file = marker_dir / "results.pkl"
        
        if results_file.exists():
            try:
                with open(results_file, 'rb') as f:
                    result = pickle.load(f)
                    result['marker_dir'] = marker_dir
                    results.append(result)
                    
                    if verbose:
                        marker_name = result.get('marker_name', 'unknown')
                        marker_type = result.get('marker_type', 'unknown')
                        n_sig = result.get('n_sig_clusters', 0)
                        print(f"  ✓ Loaded: {marker_type}/{marker_name} ({n_sig} sig clusters)")
                        
            except Exception as e:
                if verbose:
                    print(f"  ✗ Failed to load {marker_dir.name}: {e}")
    
    if verbose:
        print(f"\n✓ Loaded {len(results)} marker results")
    
    return results


def create_summary_table(results: List[Dict], alpha: float = 0.05) -> pd.DataFrame:
    """
    Create comprehensive summary table from all results.
    
    Parameters
    ----------
    results : List[Dict]
        List of result dictionaries
    alpha : float
        Significance threshold
        
    Returns
    -------
    pd.DataFrame
        Summary table with all markers
    """
    summary_data = []
    
    for result in results:
        # Determine which cluster count to use (corrected if available)
        if 'cluster_p_values_corrected' in result:
            correction_alpha = result.get('correction_alpha', alpha)
            n_sig_corrected = np.sum(result['cluster_p_values_corrected'] <= correction_alpha)
        else:
            n_sig_corrected = result.get('n_sig_clusters', 0)
        
        summary_data.append({
            'Marker Type': result.get('marker_type', 'unknown'),
            'Marker Name': result.get('marker_name', 'unknown'),
            'N Subjects': result.get('n_subjects', 0),
            'N Observations': result.get('n_observations', 0),
            'N Channels': len(result.get('ch_names', [])),
            'Total Clusters': result.get('n_clusters', 0),
            'Sig Clusters (uncorr)': result.get('n_sig_clusters', 0),
            'Sig Clusters (corr)': n_sig_corrected,
            'Correction Method': result.get('correction_method', 'none'),
            'T-stat Min': np.nanmin(result.get('t_stats', [0])),
            'T-stat Max': np.nanmax(result.get('t_stats', [0])),
            'T-stat Mean (abs)': np.nanmean(np.abs(result.get('t_stats', [0]))),
            'Predictor': result.get('predictor_of_interest', 'unknown'),
            'Threshold': result.get('threshold', 0),
            'N Permutations': result.get('n_permutations', 0),
            'Analysis Date': result.get('analysis_timestamp', 'unknown')
        })
    
    df = pd.DataFrame(summary_data)
    
    # Sort by marker type and significance
    df = df.sort_values(
        by=['Marker Type', 'Sig Clusters (corr)', 'Marker Name'],
        ascending=[True, False, True]
    ).reset_index(drop=True)
    
    return df


def create_comparison_topoplots(
    results: List[Dict],
    output_path: Path,
    alpha: float = 0.05,
    max_per_page: int = 12,
    use_corrected: bool = True,
    verbose: bool = True
) -> None:
    """
    Create comparison topoplots showing all markers together.
    
    Parameters
    ----------
    results : List[Dict]
        List of result dictionaries
    output_path : Path
        Path to save the PDF
    alpha : float
        Significance threshold
    max_per_page : int
        Maximum topoplots per page
    use_corrected : bool
        Whether to use corrected p-values if available
    verbose : bool
        Whether to print progress
    """
    if len(results) == 0:
        print("No results to plot")
        return
    
    # Separate by marker type
    evoked_results = [r for r in results if r.get('marker_type') == 'evoked']
    state_results = [r for r in results if r.get('marker_type') == 'state']
    sleep_results = [r for r in results if r.get('marker_type') == 'sleep']
    
    with PdfPages(output_path) as pdf:
        # Plot evoked markers
        if evoked_results:
            if verbose:
                print(f"\nCreating topoplots for {len(evoked_results)} evoked markers...")
            _plot_marker_type_topoplots(
                evoked_results, pdf, "Evoked Markers", alpha, 
                max_per_page, use_corrected, verbose
            )
        
        # Plot state markers
        if state_results:
            if verbose:
                print(f"\nCreating topoplots for {len(state_results)} state markers...")
            _plot_marker_type_topoplots(
                state_results, pdf, "State Markers", alpha,
                max_per_page, use_corrected, verbose
            )
        
        # Plot sleep markers
        if sleep_results:
            if verbose:
                print(f"\nCreating topoplots for {len(sleep_results)} sleep markers...")
            _plot_marker_type_topoplots(
                sleep_results, pdf, "Sleep Markers", alpha,
                max_per_page, use_corrected, verbose
            )
    
    if verbose:
        print(f"\n✓ Topoplots saved to {output_path}")


def _plot_marker_type_topoplots(
    results: List[Dict],
    pdf: PdfPages,
    title: str,
    alpha: float,
    max_per_page: int,
    use_corrected: bool,
    verbose: bool
) -> None:
    """Plot topoplots for a specific marker type."""
    
    # Calculate number of pages needed
    n_markers = len(results)
    n_pages = int(np.ceil(n_markers / max_per_page))
    
    # Determine grid layout
    n_cols = 4
    n_rows = int(np.ceil(max_per_page / n_cols))
    
    for page_idx in range(n_pages):
        start_idx = page_idx * max_per_page
        end_idx = min(start_idx + max_per_page, n_markers)
        page_results = results[start_idx:end_idx]
        
        if verbose:
            print(f"  Page {page_idx + 1}/{n_pages}: markers {start_idx + 1}-{end_idx}")
        
        # Create figure
        fig = plt.figure(figsize=(n_cols * TOPOMAP_SIZE, n_rows * TOPOMAP_SIZE))
        fig.suptitle(f"{title} - Page {page_idx + 1}/{n_pages}", 
                     fontsize=16, fontweight='bold', y=0.98)
        
        # Create grid
        gs = gridspec.GridSpec(n_rows, n_cols, figure=fig, 
                              hspace=0.4, wspace=0.3,
                              top=0.95, bottom=0.05, left=0.05, right=0.95)
        
        # Plot each marker
        for plot_idx, result in enumerate(page_results):
            row = plot_idx // n_cols
            col = plot_idx % n_cols
            
            ax = fig.add_subplot(gs[row, col])
            
            _plot_single_topomap(
                result, ax, alpha, use_corrected
            )
        
        # Save page
        pdf.savefig(fig, dpi=FIGURE_DPI, bbox_inches='tight')
        plt.close(fig)


def _plot_single_topomap(
    result: Dict,
    ax: plt.Axes,
    alpha: float,
    use_corrected: bool
) -> None:
    """
    Plot a single topomap for a marker using plot_cluster_topomap.
    
    Shows both uncorrected (empty circles) and corrected (filled dots) significant clusters.
    """
    
    # Extract data
    t_stats = result.get('t_stats', np.array([]))
    clusters = result.get('clusters', [])
    cluster_p_values_uncorrected = result.get('cluster_p_values', np.array([]))
    info = result.get('info')
    marker_name = result.get('marker_name', 'Unknown')
    
    # Determine which p-values to use as primary
    cluster_p_values_corrected = None
    if use_corrected and 'cluster_p_values_corrected' in result:
        cluster_p_values = result['cluster_p_values_corrected']
        cluster_p_values_corrected = cluster_p_values_uncorrected  # For showing both
        alpha_to_use = result.get('correction_alpha', alpha)
        correction_label = f" (MCC)"
    else:
        cluster_p_values = cluster_p_values_uncorrected
        alpha_to_use = alpha
        correction_label = ""
    
    # Create title with marker name and significance info
    n_sig_clusters = np.sum(cluster_p_values < alpha_to_use)
    n_total_clusters = len(clusters)
    
    # Shorten marker name if too long
    display_name = marker_name
    if len(display_name) > 40:
        display_name = display_name[:37] + '...'
    
    title_text = f"{display_name}\n{n_sig_clusters}/{n_total_clusters} sig{correction_label}"
    
    try:
        # Validate montage and channel information
        from plot_results import validate_montage_and_channels
        info, t_stats = validate_montage_and_channels(info, t_stats)
        
        # Create masks for significant clusters
        mask_corrected = np.zeros(len(t_stats), dtype=bool)
        for cluster, pval in zip(clusters, cluster_p_values):
            if pval < alpha_to_use:
                mask_corrected[cluster] = True
        
        # If using corrected p-values, also create mask for uncorrected
        mask_uncorrected = None
        if cluster_p_values_corrected is not None:
            mask_uncorrected = np.zeros(len(t_stats), dtype=bool)
            for cluster, pval in zip(clusters, cluster_p_values_corrected):
                if pval < alpha_to_use:
                    mask_uncorrected[cluster] = True
        
        # Set colormap limits
        abs_max = np.nanmax(np.abs(t_stats))
        if np.isnan(abs_max) or abs_max == 0:
            abs_max = 1.0
        vmin, vmax = -abs_max, abs_max
        
        # Replace NaN with 0 for plotting
        t_stats_plot = np.copy(t_stats)
        t_stats_plot[np.isnan(t_stats_plot)] = 0
        
        # Plot topomap with corrected significant clusters
        im, _ = mne.viz.plot_topomap(
            t_stats_plot,
            info,
            axes=ax,
            show=False,
            cmap=COLORMAP,
            vlim=(vmin, vmax),
            sensors=False,
            mask=mask_corrected,
            mask_params=dict(
                marker='o',
                markerfacecolor='k',
                markeredgecolor='k',
                linewidth=0,
                markersize=4  # Slightly smaller for grid layout
            ),
            contours=6,
            ch_type='eeg',
            sphere='auto',
            outlines='head',
            extrapolate='head',
            image_interp='cubic',
            border='mean',
            res=128
        )
        
        # If uncorrected p-values provided, overlay empty circles
        if mask_uncorrected is not None:
            from mne.channels import find_layout
            layout = find_layout(info, ch_type='eeg')
            pos = layout.pos[:len(t_stats), :2]  # Get x, y positions
            uncorrected_only = mask_uncorrected & ~mask_corrected
            if np.any(uncorrected_only):
                ax.plot(pos[uncorrected_only, 0], pos[uncorrected_only, 1], 
                       'o', markerfacecolor='none', markeredgecolor='k', 
                       markeredgewidth=1.0, markersize=4, zorder=10)
        
        # Add colorbar (adjusted size for grid layout)
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, shrink=0.8)
        cbar.set_label('t-stat', rotation=270, labelpad=12, fontsize=8)
        cbar.ax.tick_params(labelsize=7)
        
        # Set title
        ax.set_title(title_text, fontsize=9, fontweight='bold' if n_sig_clusters > 0 else 'normal')
        
    except Exception as e:
        # If plotting fails, show error message
        ax.text(0.5, 0.5, f'Plot error:\n{str(e)[:50]}', 
                ha='center', va='center', transform=ax.transAxes,
                fontsize=8, color='red')


def create_detailed_results_table(
    results: List[Dict],
    output_path: Path,
    alpha: float = 0.05,
    use_corrected: bool = True
) -> None:
    """
    Create detailed Excel file with multiple sheets for different views.
    
    Parameters
    ----------
    results : List[Dict]
        List of result dictionaries
    output_path : Path
        Path to save Excel file
    alpha : float
        Significance threshold
    use_corrected : bool
        Whether to use corrected p-values
    """
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # Sheet 1: Overall summary
        summary_df = create_summary_table(results, alpha)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        # Sheet 2: Significant markers only
        if use_corrected:
            sig_mask = summary_df['Sig Clusters (corr)'] > 0
        else:
            sig_mask = summary_df['Sig Clusters (uncorr)'] > 0
        
        sig_df = summary_df[sig_mask].copy()
        sig_df.to_excel(writer, sheet_name='Significant Only', index=False)
        
        # Sheet 3: Evoked markers
        evoked_df = summary_df[summary_df['Marker Type'] == 'evoked'].copy()
        evoked_df.to_excel(writer, sheet_name='Evoked Markers', index=False)
        
        # Sheet 4: State markers
        state_df = summary_df[summary_df['Marker Type'] == 'state'].copy()
        state_df.to_excel(writer, sheet_name='State Markers', index=False)
        
        # Sheet 5: Sleep markers (if present)
        sleep_df = summary_df[summary_df['Marker Type'] == 'sleep'].copy()
        if not sleep_df.empty:
            sleep_df.to_excel(writer, sheet_name='Sleep Markers', index=False)
        
        # Sheet 6: Cluster details for significant markers
        cluster_details = []
        for result in results:
            marker_name = result.get('marker_name', 'unknown')
            marker_type = result.get('marker_type', 'unknown')
            clusters = result.get('clusters', [])
            cluster_stats = result.get('cluster_stats', np.array([]))
            cluster_p_values = result.get('cluster_p_values', np.array([]))
            ch_names = result.get('ch_names', [])
            
            # Use corrected p-values if available
            if use_corrected and 'cluster_p_values_corrected' in result:
                cluster_p_values = result['cluster_p_values_corrected']
                alpha_to_use = result.get('correction_alpha', alpha)
            else:
                alpha_to_use = alpha
            
            for i, (cluster, stat, pval) in enumerate(zip(clusters, cluster_stats, cluster_p_values)):
                if pval < alpha_to_use:
                    cluster_channels = [ch_names[idx] for idx in cluster]
                    cluster_details.append({
                        'Marker Type': marker_type,
                        'Marker Name': marker_name,
                        'Cluster ID': i + 1,
                        'N Channels': len(cluster),
                        'Cluster Stat': stat,
                        'P-value': pval,
                        'Channels': ', '.join(cluster_channels)
                    })
        
        if cluster_details:
            cluster_df = pd.DataFrame(cluster_details)
            cluster_df = cluster_df.sort_values(by=['Marker Type', 'Marker Name', 'P-value'])
            cluster_df.to_excel(writer, sheet_name='Cluster Details', index=False)
    
    print(f"✓ Detailed results saved to {output_path}")


def generate_summary_report(
    model_dir: Path,
    alpha: float = 0.05,
    use_corrected: bool = True,
    verbose: bool = True
) -> None:
    """
    Generate comprehensive summary report for all markers in a model directory.
    
    This is the main function that creates:
    1. Summary CSV table
    2. Comparison topoplots PDF
    3. Detailed Excel file with multiple sheets
    
    Parameters
    ----------
    model_dir : Path
        Path to model directory (e.g., results/lmm_cluster/onoff/)
    alpha : float
        Significance threshold
    use_corrected : bool
        Whether to use corrected p-values if available
    verbose : bool
        Whether to print progress
    """
    print("="*80)
    print("GENERATING SUMMARY REPORT")
    print("="*80)
    print(f"Model directory: {model_dir}")
    print(f"Alpha: {alpha}")
    print(f"Use corrected p-values: {use_corrected}")
    print()
    
    # Load all results
    results = load_all_marker_results(model_dir, verbose=verbose)
    
    if len(results) == 0:
        print("\n⚠ No results found. Cannot generate report.")
        return
    
    # Create output paths
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_csv = model_dir / f"SUMMARY_REPORT_{timestamp}.csv"
    topoplots_pdf = model_dir / f"SUMMARY_TOPOPLOTS_{timestamp}.pdf"
    detailed_xlsx = model_dir / f"SUMMARY_DETAILED_{timestamp}.xlsx"
    
    # Generate summary table
    print("\n" + "-"*80)
    print("Creating summary table...")
    print("-"*80)
    summary_df = create_summary_table(results, alpha)
    summary_df.to_csv(summary_csv, index=False)
    print(f"✓ Summary table saved to {summary_csv}")
    
    # Display summary
    print("\nSummary Statistics:")
    print(f"  Total markers: {len(results)}")
    print(f"  Evoked markers: {len([r for r in results if r.get('marker_type') == 'evoked'])}")
    print(f"  State markers: {len([r for r in results if r.get('marker_type') == 'state'])}")
    print(f"  Sleep markers: {len([r for r in results if r.get('marker_type') == 'sleep'])}")
    
    if use_corrected:
        n_sig = summary_df['Sig Clusters (corr)'].sum()
        n_markers_with_sig = (summary_df['Sig Clusters (corr)'] > 0).sum()
    else:
        n_sig = summary_df['Sig Clusters (uncorr)'].sum()
        n_markers_with_sig = (summary_df['Sig Clusters (uncorr)'] > 0).sum()
    
    print(f"  Total significant clusters: {n_sig}")
    print(f"  Markers with ≥1 sig cluster: {n_markers_with_sig}/{len(results)}")
    
    # Generate comparison topoplots
    print("\n" + "-"*80)
    print("Creating comparison topoplots...")
    print("-"*80)
    create_comparison_topoplots(
        results, topoplots_pdf, alpha, 
        max_per_page=12, use_corrected=use_corrected, verbose=verbose
    )
    
    # Generate detailed Excel file
    print("\n" + "-"*80)
    print("Creating detailed Excel file...")
    print("-"*80)
    create_detailed_results_table(
        results, detailed_xlsx, alpha, use_corrected
    )
    
    # Final summary
    print("\n" + "="*80)
    print("REPORT GENERATION COMPLETED")
    print("="*80)
    print("\nGenerated files:")
    print(f"  1. Summary CSV:        {summary_csv.name}")
    print(f"  2. Topoplots PDF:      {topoplots_pdf.name}")
    print(f"  3. Detailed Excel:     {detailed_xlsx.name}")
    print(f"\nAll files saved to: {model_dir}")
    print()


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def main():
    """Command line interface for report generation."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate comprehensive summary report for LMM cluster analysis"
    )
    parser.add_argument(
        "model_dir",
        type=str,
        help="Path to model directory containing marker results"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance threshold (default: 0.05)"
    )
    parser.add_argument(
        "--use-uncorrected",
        action="store_true",
        help="Use uncorrected p-values instead of corrected (if available)"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress messages"
    )
    
    args = parser.parse_args()
    
    model_dir = Path(args.model_dir)
    
    if not model_dir.exists():
        print(f"Error: Directory not found: {model_dir}")
        return 1
    
    if not model_dir.is_dir():
        print(f"Error: Not a directory: {model_dir}")
        return 1
    
    generate_summary_report(
        model_dir=model_dir,
        alpha=args.alpha,
        use_corrected=not args.use_uncorrected,
        verbose=not args.quiet
    )
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
