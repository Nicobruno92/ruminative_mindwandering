"""
Combined ERP and Beta Time-course Plotting Functions.

This module creates publication-quality plots that combine traditional ERP 
waveforms with statistical beta time-courses from temporal LMM analysis.

Author: AI Assistant
Date: $(date)
"""

import os
import sys
from typing import Dict, List, Tuple, Optional
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import seaborn as sns
from scipy.ndimage import gaussian_filter1d

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))

try:
    from helpers import load_yaml_config
except ImportError:
    warnings.warn("Could not import helpers - some functionality may be limited")

# =============================================================================
# CONFIGURATION - Modify these variables as needed
# =============================================================================
DEFAULT_COLORS = {
    'onTask': '#1f77b4',     # Blue
    'offTask': '#ff7f0e',    # Orange
    'beta_positive': '#2ca02c',  # Green
    'beta_negative': '#d62728',  # Red
    'significance': '#ff1493',   # Deep pink
    'confidence': '#87ceeb'      # Sky blue
}

DEFAULT_PLOT_PARAMS = {
    'figure_width': 16,
    'figure_height': 10,
    'dpi': 300,
    'font_size': 12,
    'line_width': 2.5,
    'alpha_fill': 0.3,
    'alpha_sig': 0.8
}
# =============================================================================


def load_erp_data(features_root: str, subjects: List[str], tasks: List[str], 
                  rois: Dict[str, List[str]], conditions: List[str] = ["onTask", "offTask"]) -> Dict:
    """
    Load traditional ERP data for plotting.
    
    This function loads the averaged ERP data from your existing pipeline.
    It tries to load real probe evoked files first, then falls back to synthetic data.
    
    Parameters
    ----------
    features_root : str
        Root directory for ERP features
    subjects : List[str]
        List of subject IDs
    tasks : List[str]
        List of task names
    rois : Dict[str, List[str]]
        Dictionary mapping ROI names to channel lists
    conditions : List[str]
        List of condition names
        
    Returns
    -------
    Dict
        Dictionary containing ERP data by ROI and condition
    """
    print("📊 Loading ERP data for combined plotting...")
    
    # Try to load real ERP data from your pipeline
    try:
        from lmm_analysis import _find_probe_evokeds, _split_by_label, _roi_picks, _load_and_roi_average
        import mne
        
        erp_data = {}
        
        for roi_name, roi_channels in rois.items():
            erp_data[roi_name] = {}
            
            # Collect data for each condition
            for condition in conditions:
                condition_erps = []
                condition_times = None
                
                # Group by subject first, then average across tasks within subject
                for subject in subjects:
                    subject_task_data = []
                    
                    for task in tasks:
                        # Find probe evoked files
                        probe_paths = _find_probe_evokeds(features_root, subject, task)
                        if not probe_paths:
                            continue
                        
                        # Split by condition
                        on_paths, off_paths = _split_by_label(probe_paths)
                        paths = on_paths if condition == "onTask" else off_paths
                        
                        if not paths:
                            continue
                        
                        # Load one file to get info for ROI picks
                        info_evoked = mne.read_evokeds(paths[0], condition=0, verbose=False).info
                        picks = _roi_picks(info_evoked, list(roi_channels))
                        
                        if picks.size == 0:
                            continue
                        
                        # Load and average all probes for this subject-task-condition
                        task_probe_data = []
                        for path in paths:
                            try:
                                times, roi_avg, ratings = _load_and_roi_average(path, picks)
                                task_probe_data.append(roi_avg)
                                if condition_times is None:
                                    condition_times = times
                            except Exception:
                                continue
                        
                        if task_probe_data:
                            # Average across probes for this task
                            task_avg = np.mean(task_probe_data, axis=0)
                            subject_task_data.append(task_avg)
                    
                    if subject_task_data:
                        # Average across tasks for this subject
                        subject_avg = np.mean(subject_task_data, axis=0)
                        condition_erps.append(subject_avg)
                
                if condition_erps and condition_times is not None:
                    # Store the data
                    condition_erps = np.array(condition_erps)
                    erp_data[roi_name][condition] = {
                        'times': condition_times,
                        'individual': condition_erps,
                        'mean': np.mean(condition_erps, axis=0),
                        'sem': np.std(condition_erps, axis=0) / np.sqrt(len(condition_erps)),
                        'n_subjects': len(condition_erps)
                    }
                    print(f"✅ Loaded real ERP data for {roi_name} {condition}: {len(condition_erps)} subjects")
                else:
                    print(f"❌ No data found for {roi_name} {condition}")
        
        # Check if we successfully loaded any real data
        if erp_data and any(roi_data.values() for roi_data in erp_data.values() if roi_data):
            print(f"✅ Successfully loaded real ERP data for {len(erp_data)} ROIs")
            return erp_data
        else:
            print("❌ No real ERP data found")
            print("   Possible reasons:")
            print("   1. No probe evoked files exist in features_root")
            print("   2. Need to run: python make_probe_evokeds.py --config config.yaml")
            print("   3. features_root path in config.yaml is incorrect")
            print("   4. ROI channels not found in the data")
            raise FileNotFoundError("No real ERP data found - cannot generate plots with synthetic data")
            
    except FileNotFoundError:
        # Re-raise this specific error to avoid fallback
        raise
    except Exception as e:
        print(f"❌ Failed to load real ERP data: {e}")
        import traceback
        traceback.print_exc()
        raise RuntimeError(f"Error loading ERP data: {e}") from e


def create_combined_erp_beta_plot(erp_data: Dict, beta_results: pd.DataFrame, 
                                  roi: str, output_path: str, 
                                  windowed_results: Optional[pd.DataFrame] = None,
                                  plot_params: Optional[Dict] = None) -> None:
    """
    Create combined ERP + Beta time-course plot for a single ROI.
    
    Parameters
    ----------
    erp_data : Dict
        ERP data dictionary from load_erp_data()
    beta_results : pd.DataFrame
        Beta time-course results from temporal LMM analysis
    roi : str
        ROI name to plot
    output_path : str
        Output file path for the plot
    windowed_results : pd.DataFrame, optional
        Windowed LMM results to show significant time windows on ERP plot
    plot_params : Dict, optional
        Plotting parameters dictionary
    """
    if plot_params is None:
        plot_params = DEFAULT_PLOT_PARAMS.copy()
    
    print(f"📊 Creating combined ERP+Beta plot for ROI: {roi}")
    
    # Filter beta results for this ROI
    roi_beta = beta_results[beta_results['roi'] == roi].copy()
    if roi_beta.empty:
        print(f"⚠️  No beta results found for ROI {roi}")
        return
    
    roi_beta = roi_beta.sort_values('time_point')
    
    # Get ERP data for this ROI
    if roi not in erp_data:
        print(f"⚠️  No ERP data found for ROI {roi}")
        return
    
    roi_erp = erp_data[roi]
    
    # Set up the figure with two subplots side by side (A and B)
    fig = plt.figure(figsize=(plot_params['figure_width'], plot_params['figure_height']))
    gs = GridSpec(1, 2, width_ratios=[1, 1], wspace=0.3)
    
    # Left panel (A): Traditional ERP waveforms
    ax_erp = fig.add_subplot(gs[0])
    
    # Plot ERP waveforms for each condition
    for condition in ['onTask', 'offTask']:
        if condition in roi_erp:
            times = roi_erp[condition]['times']
            mean_erp = roi_erp[condition]['mean']
            sem_erp = roi_erp[condition]['sem']
            
            # Main ERP line
            color = DEFAULT_COLORS[condition]
            ax_erp.plot(times, mean_erp, color=color, linewidth=plot_params['line_width'],
                       label=f'{condition.replace("onTask", "On-task").replace("offTask", "Off-task")}')
            
            # Standard error fill
            ax_erp.fill_between(times, mean_erp - sem_erp, mean_erp + sem_erp,
                               color=color, alpha=plot_params['alpha_fill'])
    
    # ERP panel formatting
    ax_erp.set_ylabel('Amplitude (µV)', fontsize=plot_params['font_size'])
    ax_erp.set_title(f'ROI {roi}: Traditional ERP Waveforms vs Statistical Beta Time-course',
                     fontsize=plot_params['font_size'] + 2, fontweight='bold')
    ax_erp.legend(loc='upper right')
    ax_erp.grid(True, alpha=0.3)
    ax_erp.axhline(0, color='black', linestyle='-', alpha=0.5, linewidth=0.8)
    ax_erp.axvline(0, color='black', linestyle='--', alpha=0.5, linewidth=0.8)
    
    # Highlight significant windowed time windows on ERP plot
    if windowed_results is not None:
        roi_windowed = windowed_results[
            (windowed_results['roi'] == roi) &
            windowed_results['significant']
        ]
        
        if not roi_windowed.empty:
            for _, row in roi_windowed.iterrows():
                window_start = row['window_start']
                window_end = row['window_end']
                window_name = row['window']
                p_val = row['p_value']
                
                # Add colored rectangle for significant window
                y_min, y_max = ax_erp.get_ylim()
                ax_erp.axvspan(window_start, window_end,
                              alpha=0.2, color='gold', zorder=1)
                
                # Add window label at the top
                window_center = (window_start + window_end) / 2
                ax_erp.text(window_center, y_max * 0.95,
                           f'{window_name}\np={p_val:.3f}',
                           fontsize=8, ha='center', va='top',
                           bbox=dict(boxstyle="round,pad=0.3",
                                    facecolor='gold', alpha=0.8,
                                    edgecolor='black', linewidth=1))
    
    # Add ERP component labels at appropriate latencies
    y_min, y_max = ax_erp.get_ylim()
    y_mid = (y_max + y_min) / 2
    
    # P1 (~100ms) - typically positive deflection
    if 0.08 <= times.max():
        ax_erp.text(0.10, y_mid + (y_max - y_min) * 0.25, 'P1', 
                   fontsize=9, ha='center', va='center',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray', alpha=0.7))
    
    # N1 (~150-180ms) - typically negative deflection  
    if 0.15 <= times.max():
        ax_erp.text(0.165, y_mid - (y_max - y_min) * 0.25, 'N1', 
                   fontsize=9, ha='center', va='center',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray', alpha=0.7))
    
    # P3 (~300-350ms) - typically positive deflection
    if 0.30 <= times.max():
        ax_erp.text(0.325, y_mid + (y_max - y_min) * 0.25, 'P3', 
                   fontsize=9, ha='center', va='center',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray', alpha=0.7))
    
    # Right panel (B): Beta time-course
    ax_beta = fig.add_subplot(gs[1])
    
    # Extract beta data
    beta_times = roi_beta['time_point'].values
    beta_values = roi_beta['beta'].values
    beta_ci_lower = roi_beta['ci_lower'].values
    beta_ci_upper = roi_beta['ci_upper'].values
    
    # Use the 'significant' column which already has cluster correction applied
    # This is better than using raw p_value < 0.05
    if 'significant' in roi_beta.columns:
        significant = roi_beta['significant'].values
    else:
        # Fallback if column doesn't exist
        beta_pvalues = roi_beta.get('p_corrected', roi_beta.get('p_value', np.ones(len(beta_times)))).values
        significant = beta_pvalues < 0.05
    
    # Check if cluster correction was used
    has_clusters = 'cluster_id' in roi_beta.columns and 'cluster_p_value' in roi_beta.columns
    
    # Apply smoothing if requested
    if len(beta_times) > 10:  # Only smooth if we have enough points
        from scipy.ndimage import gaussian_filter1d
        sigma = 2  # Smoothing parameter
        beta_values_smooth = gaussian_filter1d(beta_values, sigma=sigma)
        beta_ci_lower_smooth = gaussian_filter1d(beta_ci_lower, sigma=sigma)
        beta_ci_upper_smooth = gaussian_filter1d(beta_ci_upper, sigma=sigma)
    else:
        beta_values_smooth = beta_values
        beta_ci_lower_smooth = beta_ci_lower
        beta_ci_upper_smooth = beta_ci_upper
    
    # Plot confidence interval
    ax_beta.fill_between(beta_times, beta_ci_lower_smooth, beta_ci_upper_smooth,
                        color=DEFAULT_COLORS['confidence'], alpha=plot_params['alpha_fill'],
                        label='95% Confidence Interval')
    
    # Plot beta line
    ax_beta.plot(beta_times, beta_values_smooth, color='black', linewidth=plot_params['line_width'],
                label='Off-task - On-task Effect (β)')
    
    # Highlight significant time points
    if np.any(significant):
        if has_clusters:
            # Plot clusters with different colors
            cluster_ids = roi_beta['cluster_id'].values
            cluster_pvals = roi_beta['cluster_p_value'].values
            
            unique_clusters = np.unique(cluster_ids[significant])
            # plt already imported at module level
            cluster_colors = plt.cm.Set1(np.linspace(0, 1, len(unique_clusters)))
            
            for cluster_idx, cluster_color in zip(unique_clusters, cluster_colors):
                if cluster_idx == -1:
                    continue
                
                cluster_mask = (cluster_ids == cluster_idx) & significant
                cluster_times = beta_times[cluster_mask]
                cluster_betas = beta_values_smooth[cluster_mask]
                cluster_p = cluster_pvals[cluster_mask][0]
                
                if len(cluster_times) > 0:
                    # Highlight cluster region
                    ax_beta.axvspan(cluster_times[0], cluster_times[-1], 
                                   alpha=0.2, color=cluster_color, zorder=1)
                    
                    # Mark cluster points
                    ax_beta.scatter(cluster_times, cluster_betas, 
                                   c=[cluster_color], s=40, alpha=plot_params['alpha_sig'],
                                   label=f'Cluster {int(cluster_idx)} (p={cluster_p:.4f})',
                                   zorder=5, edgecolors='black', linewidths=0.5)
        else:
            # Standard marking for non-cluster methods
            sig_times = beta_times[significant]
            sig_betas = beta_values_smooth[significant]
            if len(sig_times) > 0:
                ax_beta.scatter(sig_times, sig_betas, color=DEFAULT_COLORS['significance'], 
                               s=40, alpha=plot_params['alpha_sig'], zorder=5,
                               label='Significant (cluster-corrected)')
        
        # Add significance bars at the top ONLY if no cluster correction
        if not has_clusters and np.any(significant):
            y_max = max(np.max(beta_ci_upper_smooth), np.max(beta_values_smooth)) * 1.1
            for i, (time, is_sig) in enumerate(zip(beta_times, significant)):
                if is_sig:
                    ax_beta.plot([time, time], [y_max * 0.95, y_max], 
                               color=DEFAULT_COLORS['significance'], linewidth=2, alpha=0.8)
    
    # Beta panel formatting
    ax_beta.set_xlabel('Time (s)', fontsize=plot_params['font_size'])
    ax_beta.set_ylabel('Beta Coefficient (µV)', fontsize=plot_params['font_size'])
    ax_beta.legend(loc='upper right')
    ax_beta.grid(True, alpha=0.3)
    ax_beta.axhline(0, color='black', linestyle='-', alpha=0.5, linewidth=0.8)
    ax_beta.axvline(0, color='black', linestyle='--', alpha=0.5, linewidth=0.8)
    
    # Synchronize x-axes
    xlim = (min(np.min(beta_times), np.min(roi_erp['onTask']['times'])),
            max(np.max(beta_times), np.max(roi_erp['onTask']['times'])))
    ax_erp.set_xlim(xlim)
    ax_beta.set_xlim(xlim)
    
    # Add text box with analysis info
    n_subjects = roi_erp['onTask']['n_subjects']
    n_timepoints = len(beta_times)
    n_significant = np.sum(significant)
    
    info_text = f"Subjects: {n_subjects}\nTime points: {n_timepoints}\nSignificant: {n_significant}"
    ax_beta.text(0.02, 0.98, info_text, transform=ax_beta.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", 
                facecolor='white', alpha=0.8))
    
    # Add method info at bottom
    method_text = "Left (A): Traditional ERP averages ± SEM | Right (B): LMM beta coefficients ± 95% CI"
    fig.text(0.5, 0.02, method_text, fontsize=9, ha='center', style='italic',
             bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.7))
    
    plt.tight_layout()
    
    # Save plot
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=plot_params['dpi'], bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight')  # Also save PDF
    plt.close()
    
    print(f"✅ Saved combined plot: {output_path}")
def run_combined_erp_beta_analysis(cfg: Dict) -> None:
    """
    Main function to run combined ERP+Beta analysis and create plots.
    
    Parameters
    ----------
    cfg : Dict
        Configuration dictionary from YAML file
    """
    print("🎨 Running Combined ERP + Beta Time-course Analysis")
    print("=" * 60)
    
    # Get configuration parameters
    proj_cfg = cfg.get("project", {})
    features_root = proj_cfg.get("features_root")
    results_root = proj_cfg.get("results_root", "results/ERPs_new")
    
    subjects = [str(s) for s in cfg.get("subjects", [])]
    tasks = [str(t) for t in cfg.get("tasks", [])]
    rois = cfg.get("erp_rois", {})
    
    if not features_root:
        print("❌ features_root not configured")
        return
    
    if not rois:
        print("❌ No ROIs configured")
        return
    
    # Create output directory
    output_dir = os.path.join(results_root, "combined_erp_beta")
    os.makedirs(output_dir, exist_ok=True)
    
    # Load ERP data
    erp_data = load_erp_data(features_root, subjects, tasks, rois)
    
    # LMM results directory
    lmm_dir = os.path.join(results_root, "lmm")
    
    # Load windowed LMM results for significance windows
    windowed_results = None
    windowed_file = os.path.join(lmm_dir, "lmm_results.csv")
    if os.path.exists(windowed_file):
        try:
            windowed_results = pd.read_csv(windowed_file)
            print(f"✅ Loaded windowed LMM results: {windowed_file}")
        except Exception as e:
            print(f"⚠️  Failed to load windowed results: {e}")
    
    # Load beta results (try to find existing temporal LMM results)
    beta_files = [
        os.path.join(lmm_dir, "lmm_temporal_results.csv"),
        os.path.join(lmm_dir, "unfold_lmm_results_combined.csv")
    ]
    
    beta_results = None
    for beta_file in beta_files:
        if os.path.exists(beta_file):
            try:
                beta_results = pd.read_csv(beta_file)
                print(f"✅ Loaded beta results: {beta_file}")
                break
            except Exception as e:
                print(f"⚠️  Failed to load {beta_file}: {e}")
    
    if beta_results is None or beta_results.empty:
        print("❌ No beta time-course results found")
        print("💡 Run temporal LMM analysis first:")
        print("   python lmm_analysis.py --config config.yaml")
        return
    
    # Create individual ROI plots
    for roi in rois.keys():
        output_path = os.path.join(output_dir, f"combined_erp_beta_roi_{roi}.png")
        create_combined_erp_beta_plot(erp_data, beta_results, roi, output_path,
                                     windowed_results=windowed_results)
    
    # Create summary plot by combining individual images
    print("\n🎨 Creating summary plot from individual images...")
    from create_summary_from_individual_plots import load_individual_plots, create_summary_from_images
    
    roi_list = list(rois.keys())
    images = load_individual_plots(output_dir, roi_list)
    
    if images:
        summary_path = os.path.join(output_dir, 'combined_erp_beta_summary.png')
        create_summary_from_images(images, roi_list, summary_path)
        print(f"✅ Saved combined summary: {summary_path}")
    else:
        print("⚠️  No individual plots found to create summary")
    
    print("✅ Combined ERP + Beta analysis completed!")
    print(f"📁 Plots saved to: {output_dir}")


if __name__ == "__main__":
    """
    Run combined ERP+Beta analysis with default configuration.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Create combined ERP and Beta time-course plots")
    parser.add_argument("--config", default="config.yaml", help="Path to configuration file")
    args = parser.parse_args()
    
    try:
        cfg = load_yaml_config(args.config)
        run_combined_erp_beta_analysis(cfg)
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
