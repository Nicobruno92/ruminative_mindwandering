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
import yaml
from pathlib import Path
from typing import Dict, List, Optional
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


# ====================================================================================
# CONFIG HELPERS
# ============================================================================

def flatten_config_dict(config: Dict, parent_key: str = "", sep: str = ".") -> Dict[str, object]:
    """Flatten a nested configuration dictionary into dotted-key format.

    Parameters
    ----------
    config : dict
        Nested configuration dictionary.
    parent_key : str
        Prefix for keys in nested calls.
    sep : str
        Separator between key levels.

    Returns
    -------
    dict
        Flattened dictionary mapping 'section.subsection.key' to values.
    """
    items: Dict[str, object] = {}
    for key, value in config.items():
        new_key = f"{parent_key}{sep}{key}" if parent_key else str(key)
        if isinstance(value, dict):
            items.update(flatten_config_dict(value, new_key, sep=sep))
        else:
            items[new_key] = value
    return items


def load_config_table(config_path: Path) -> pd.DataFrame:
    """Load YAML configuration file and convert to a flat table.

    Parameters
    ----------
    config_path : Path
        Path to the YAML configuration file.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns 'Parameter' and 'Value'.
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r") as f:
        config = yaml.safe_load(f)

    if config is None:
        data = []
    else:
        flat_config = flatten_config_dict(config)

        # Curated subset of keys that are most relevant for understanding a run
        include_keys = {
            # Project-level
            "project.features_root",
            "project.output_path",
            "project.montage_path",
            "project.pca_results_path",
            "project.qa_summary_path",
            "project.exclude_failed_qa",
            "project.marker_types",
            "project.markers",
            # Preprocessing
            "preprocessing.normalize_by_subject",
            "preprocessing.normalization_method",
            "preprocessing.channel_wise",
            # LMM
            "lmm.formula",
            "lmm.predictor_of_interest",
            "lmm.method",
            "lmm.maxiter",
            "lmm.random_state",
            # Clustering (general pipeline)
            "clustering.method",
            "clustering.permutation_method",
            "clustering.threshold",
            "clustering.tail",
            "clustering.stat_fun",
            "clustering.t_power",
            "clustering.tfce.E",
            "clustering.tfce.H",
            "clustering.tfce.n_steps",
            "clustering.n_permutations",
            "clustering.alpha",
            "clustering.seed",
            "clustering.n_jobs",
            # Andrillon-specific clustering (if present)
            "andrillon_clustering.cluster_alpha",
            "andrillon_clustering.n_permutations",
            "andrillon_clustering.n_jobs",
            "andrillon_clustering.montecarlo_alpha",
            "andrillon_clustering.stat_fun",
            "andrillon_clustering.bonferroni_correction",
            "andrillon_clustering.permutation_within",
            "andrillon_clustering.seed",
            # Multiple comparisons
            "multiple_comparisons.evoked",
            "multiple_comparisons.state",
            "multiple_comparisons.alpha",
            # Output
            "output.save_pickle",
            "output.save_csv",
            "output.save_figures",
            "output.fig_format",
            "output.fig_dpi",
            "output.overwrite",
        }

        # If none of the curated keys are present (e.g., future configs),
        # fall back to showing all flattened keys.
        intersection = {k: v for k, v in flat_config.items() if k in include_keys}
        items_to_use = intersection if intersection else flat_config

        def _is_effectively_unused(value: object) -> bool:
            """Return True for values that indicate a parameter is not used.

            Filters out None, empty strings, and empty containers.
            """
            if value is None:
                return True
            if isinstance(value, str) and value.strip() == "":
                return True
            if isinstance(value, (list, tuple, dict, set)) and len(value) == 0:
                return True
            return False

        data = [
            {"Parameter": key, "Value": value}
            for key, value in items_to_use.items()
            if not _is_effectively_unused(value)
        ]

    df = pd.DataFrame(data, columns=["Parameter", "Value"])
    if not df.empty:
        df = df.sort_values(by="Parameter").reset_index(drop=True)
    return df


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
    verbose: bool = True,
    config_df: Optional[pd.DataFrame] = None
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
    config_df : pd.DataFrame, optional
        Optional configuration table to add as the first page of the PDF.
    """
    if len(results) == 0:
        print("No results to plot")
        return
    
    # Separate by marker type
    evoked_results = [r for r in results if r.get('marker_type') == 'evoked']
    state_results = [r for r in results if r.get('marker_type') == 'state']
    sleep_results = [r for r in results if r.get('marker_type') == 'sleep']
    
    with PdfPages(output_path) as pdf:
        # Optional first page: configuration table
        if config_df is not None and not config_df.empty:
            fig = plt.figure(figsize=(11.69, 8.27))  # A4 landscape
            ax = fig.add_subplot(111)
            ax.axis('off')
            fig.suptitle("Analysis Configuration", fontsize=16, fontweight='bold', y=0.96)
            fig.subplots_adjust(top=0.9)

            # Build table data (limit very long values for readability)
            max_value_len = 80
            table_rows = []
            for _, row in config_df.iterrows():
                param = str(row.get("Parameter", ""))
                value = str(row.get("Value", ""))
                if len(value) > max_value_len:
                    value = value[: max_value_len - 3] + "..."
                table_rows.append([param, value])

            table = ax.table(
                cellText=table_rows,
                colLabels=["Parameter", "Value"],
                loc='center',
                cellLoc='left'
            )
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            table.scale(1.0, 1.2)

            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
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
    use_corrected: bool = True,
    config_df: Optional[pd.DataFrame] = None
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
    config_df : pd.DataFrame, optional
        Optional configuration table to include as an additional sheet.
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

        # Sheet 7: Configuration (optional)
        if config_df is not None and not config_df.empty:
            config_df.to_excel(writer, sheet_name='Config', index=False)
    
    print(f"✓ Detailed results saved to {output_path}")


def generate_pipeline_qa_html_report(
    model_dir: Path,
    successful_markers: List[str],
    failed_markers: List[Dict[str, str]],
    config: Dict,
    output_filename: str = "pipeline_qa_summary.html"
) -> str:
    """
    Generate a visual HTML summary report showing pipeline execution status.
    
    Parameters
    ----------
    model_dir : Path
        Path to model directory where report will be saved
    successful_markers : List[str]
        List of marker names that were processed successfully
    failed_markers : List[Dict[str, str]]
        List of dicts with 'marker' and 'error' keys for failed markers
    config : Dict
        Configuration dictionary used for the run
    output_filename : str
        Name of the output HTML file
        
    Returns
    -------
    str
        Path to the generated HTML report
    """
    output_path = model_dir / output_filename
    
    # Compute summary statistics
    n_total = len(successful_markers) + len(failed_markers)
    n_passed = len(successful_markers)
    n_failed = len(failed_markers)
    pass_rate = (n_passed / n_total * 100) if n_total > 0 else 0
    
    # Load results for successful markers to get significance info
    results = load_all_marker_results(model_dir, verbose=False)
    n_with_sig_clusters = sum(1 for r in results if r.get('n_sig_clusters', 0) > 0)
    
    # Count by marker type
    evoked_success = [m for m in successful_markers if 'evoked' in m.lower() or any(
        r.get('marker_type') == 'evoked' for r in results if r.get('marker_name') == m
    )]
    state_success = [m for m in successful_markers if 'state' in m.lower() or any(
        r.get('marker_type') == 'state' for r in results if r.get('marker_name') == m
    )]
    
    # Build HTML
    html_parts = [
        "<!DOCTYPE html>",
        "<html>",
        "<head>",
        "<meta charset='utf-8'>",
        "<title>LMM Pipeline Execution Summary</title>",
        "<style>",
        "body { font-family: 'Segoe UI', Arial, sans-serif; margin: 0; padding: 20px; background: #f8f9fa; }",
        ".container { max-width: 1400px; margin: 0 auto; }",
        "h1 { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }",
        "h2 { color: #34495e; margin-top: 30px; }",
        ".summary-cards { display: flex; gap: 20px; flex-wrap: wrap; margin: 20px 0; }",
        ".card { background: white; border-radius: 10px; padding: 20px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); min-width: 150px; text-align: center; }",
        ".card-value { font-size: 36px; font-weight: bold; }",
        ".card-label { color: #7f8c8d; font-size: 14px; margin-top: 5px; }",
        ".pass { color: #27ae60; }",
        ".fail { color: #e74c3c; }",
        ".warn { color: #f39c12; }",
        ".neutral { color: #3498db; }",
        ".progress-bar { background: #ecf0f1; border-radius: 10px; height: 30px; overflow: hidden; margin: 10px 0; display: flex; }",
        ".progress-fill { height: 100%; display: flex; align-items: center; justify-content: center; color: white; font-weight: bold; }",
        ".progress-pass { background: linear-gradient(90deg, #27ae60, #2ecc71); }",
        ".progress-fail { background: linear-gradient(90deg, #e74c3c, #c0392b); }",
        "table { width: 100%; border-collapse: collapse; background: white; border-radius: 10px; overflow: hidden; box-shadow: 0 2px 10px rgba(0,0,0,0.1); margin: 20px 0; }",
        "th { background: #3498db; color: white; padding: 12px 8px; text-align: left; font-size: 12px; }",
        "td { padding: 10px 8px; border-bottom: 1px solid #ecf0f1; font-size: 12px; }",
        "tr:hover { background: #f8f9fa; }",
        ".status-pass { background: #d5f4e6; color: #27ae60; padding: 4px 8px; border-radius: 4px; font-weight: bold; }",
        ".status-fail { background: #fde8e8; color: #e74c3c; padding: 4px 8px; border-radius: 4px; font-weight: bold; }",
        ".status-sig { background: #fff3cd; color: #856404; padding: 4px 8px; border-radius: 4px; font-weight: bold; }",
        ".config-section { background: white; border-radius: 10px; padding: 20px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); margin: 20px 0; }",
        ".config-item { display: flex; padding: 5px 0; border-bottom: 1px solid #ecf0f1; }",
        ".config-key { font-weight: bold; color: #34495e; width: 300px; }",
        ".config-value { color: #7f8c8d; }",
        ".timestamp { color: #95a5a6; font-size: 12px; margin-top: 30px; }",
        "</style>",
        "</head>",
        "<body>",
        "<div class='container'>",
        f"<h1>LMM Cluster Pipeline - Execution Summary</h1>",
        f"<p class='timestamp'>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>",
        "",
        "<div class='summary-cards'>",
        f"<div class='card'><div class='card-value neutral'>{n_total}</div><div class='card-label'>Total Markers</div></div>",
        f"<div class='card'><div class='card-value pass'>{n_passed}</div><div class='card-label'>Successful</div></div>",
        f"<div class='card'><div class='card-value fail'>{n_failed}</div><div class='card-label'>Failed</div></div>",
        f"<div class='card'><div class='card-value'>{pass_rate:.1f}%</div><div class='card-label'>Success Rate</div></div>",
        f"<div class='card'><div class='card-value warn'>{n_with_sig_clusters}</div><div class='card-label'>With Sig. Clusters</div></div>",
        "</div>",
        "",
        "<h2>Success Rate</h2>",
        "<div class='progress-bar'>",
        f"<div class='progress-fill progress-pass' style='width: {pass_rate}%'>{n_passed} passed</div>" if pass_rate > 0 else "",
        f"<div class='progress-fill progress-fail' style='width: {100-pass_rate}%'>{n_failed} failed</div>" if (100-pass_rate) > 0 else "",
        "</div>",
    ]
    
    # Configuration summary
    html_parts.append("<h2>Analysis Configuration</h2>")
    html_parts.append("<div class='config-section'>")
    
    key_configs = [
        ('Formula', config.get('lmm', {}).get('formula', 'N/A')),
        ('Predictor of Interest', config.get('lmm', {}).get('predictor_of_interest', 'auto')),
        ('Clustering Method', config.get('clustering', {}).get('method', 'threshold')),
        ('Permutation Method', config.get('clustering', {}).get('permutation_method', 'simple')),
        ('N Permutations', config.get('clustering', {}).get('n_permutations', 'N/A')),
        ('Alpha', config.get('clustering', {}).get('alpha', 0.05)),
        ('Threshold', config.get('clustering', {}).get('threshold', 'N/A')),
        ('Normalize by Subject', config.get('preprocessing', {}).get('normalize_by_subject', False)),
        ('QA Filtering', config.get('project', {}).get('exclude_failed_qa', False)),
    ]
    
    for key, value in key_configs:
        html_parts.append(f"<div class='config-item'><span class='config-key'>{key}</span><span class='config-value'>{value}</span></div>")
    
    html_parts.append("</div>")
    
    # Successful markers table
    html_parts.append("<h2>Successful Markers</h2>")
    if results:
        html_parts.append("<table>")
        html_parts.append("<tr><th>Marker Name</th><th>Type</th><th>N Subjects</th><th>N Obs</th><th>N Clusters</th><th>Sig Clusters</th><th>Status</th></tr>")
        
        for result in sorted(results, key=lambda r: (r.get('marker_type', ''), r.get('marker_name', ''))):
            marker_name = result.get('marker_name', 'unknown')
            marker_type = result.get('marker_type', 'unknown')
            n_subjects = result.get('n_subjects', 0)
            n_obs = result.get('n_observations', 0)
            n_clusters = result.get('n_clusters', 0)
            n_sig = result.get('n_sig_clusters', 0)
            
            status_class = 'status-sig' if n_sig > 0 else 'status-pass'
            status_text = f'{n_sig} SIG' if n_sig > 0 else 'OK'
            
            html_parts.append(
                f"<tr><td>{marker_name}</td><td>{marker_type}</td><td>{n_subjects}</td>"
                f"<td>{n_obs}</td><td>{n_clusters}</td><td>{n_sig}</td>"
                f"<td><span class='{status_class}'>{status_text}</span></td></tr>"
            )
        
        html_parts.append("</table>")
    else:
        html_parts.append("<p>No successful markers to display.</p>")
    
    # Failed markers section
    if failed_markers:
        html_parts.append("<h2>Failed Markers</h2>")
        html_parts.append("<table>")
        html_parts.append("<tr><th>Marker Name</th><th>Error</th><th>Status</th></tr>")
        
        for fail_info in failed_markers:
            marker_name = fail_info.get('marker', 'unknown')
            error = fail_info.get('error', 'Unknown error')
            # Truncate long error messages
            if len(error) > 200:
                error = error[:200] + "..."
            
            html_parts.append(
                f"<tr><td>{marker_name}</td><td>{error}</td>"
                f"<td><span class='status-fail'>FAILED</span></td></tr>"
            )
        
        html_parts.append("</table>")
    
    html_parts.extend([
        "</div>",
        "</body>",
        "</html>",
    ])
    
    # Write HTML
    with open(output_path, 'w') as f:
        f.write("\n".join(html_parts))
    
    return str(output_path)


def generate_summary_report(
    model_dir: Path,
    alpha: float = 0.05,
    use_corrected: bool = True,
    verbose: bool = True,
    config_path: Optional[Path] = None
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
    config_path : Path, optional
        Optional path to YAML configuration file. If provided, a
        flattened config table will be saved and added to the
        detailed Excel report.
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

    # Optionally load configuration and save as table
    config_df: Optional[pd.DataFrame] = None
    if config_path is not None:
        try:
            config_df = load_config_table(config_path)
            config_csv = model_dir / f"SUMMARY_CONFIG_{timestamp}.csv"
            config_df.to_csv(config_csv, index=False)
            print(f"✓ Config table saved to {config_csv}")
        except Exception as exc:
            print(f"⚠ Failed to load config from {config_path}: {exc}")
    
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
        max_per_page=12,
        use_corrected=use_corrected,
        verbose=verbose,
        config_df=config_df
    )
    
    # Generate detailed Excel file
    print("\n" + "-"*80)
    print("Creating detailed Excel file...")
    print("-"*80)
    create_detailed_results_table(
        results, detailed_xlsx, alpha, use_corrected, config_df
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
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional path to YAML config file to include in the report"
    )
    
    args = parser.parse_args()
    
    model_dir = Path(args.model_dir)
    config_path = Path(args.config) if args.config is not None else None
    
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
        verbose=not args.quiet,
        config_path=config_path
    )
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
