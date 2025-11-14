"""
Visualization Module for Andrillon 2020 Results

Generate publication-quality figures showing:
- Topographic maps with significant clusters
- Cluster statistics
- Summary reports
"""

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import patches
from pathlib import Path
from typing import Dict, List, Optional
import mne
from mne.viz import plot_topomap


def load_results(results_path: str) -> Dict:
    """Load analysis results from pickle file."""
    with open(results_path, 'rb') as f:
        results = pickle.load(f)
    return results


def plot_cluster_topography(
    results: Dict,
    montage_path: str,
    output_path: Optional[str] = None,
    show: bool = True,
):
    """
    Plot topographic map with t-values and significant clusters highlighted.
    
    Parameters
    ----------
    results : dict
        Analysis results from andrillon_pipeline
    montage_path : str
        Path to montage file
    output_path : str, optional
        Path to save figure
    show : bool
        Whether to display figure
    """
    # Load montage
    montage = mne.channels.read_custom_montage(montage_path)
    
    # Get channel positions
    pos = np.array([montage.get_positions()['ch_pos'][ch] 
                    for ch in montage.ch_names])
    
    # Extract t-values
    real_stats = results['real_stats']
    electrode_ids = real_stats[:, 0].astype(int)
    t_values = real_stats[:, 2]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot topomap
    im, cn = plot_topomap(
        t_values,
        pos[:, :2],
        axes=ax,
        show=False,
        cmap='RdBu_r',
        vlim=(-np.max(np.abs(t_values)), np.max(np.abs(t_values))),
        contours=6,
    )
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('t-value', fontsize=12)
    
    # Highlight significant clusters
    clusters = results['clusters']
    if clusters:
        for i, cluster in enumerate(clusters):
            # Get electrode positions for this cluster
            cluster_electrodes = cluster.electrodes
            cluster_pos = pos[cluster_electrodes, :2]
            
            # Plot cluster electrodes
            color = 'red' if cluster.cluster_type == 'positive' else 'blue'
            ax.scatter(
                cluster_pos[:, 0], 
                cluster_pos[:, 1],
                s=200,
                facecolors='none',
                edgecolors=color,
                linewidths=3,
                label=f'{cluster.cluster_type.capitalize()} cluster {i+1}',
                zorder=10,
            )
    
    # Add title
    marker_name = results['marker_name']
    ax.set_title(
        f'{marker_name}\n'
        f'{len(clusters)} significant cluster(s) found',
        fontsize=14,
        fontweight='bold'
    )
    
    # Add legend if clusters exist
    if clusters:
        ax.legend(loc='upper left', bbox_to_anchor=(1.15, 1), fontsize=10)
    
    plt.tight_layout()
    
    # Save if requested
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {output_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


def plot_cluster_statistics(
    results: Dict,
    output_path: Optional[str] = None,
    show: bool = True,
):
    """
    Plot bar chart of cluster statistics.
    
    Parameters
    ----------
    results : dict
        Analysis results
    output_path : str, optional
        Path to save figure
    show : bool
        Whether to display figure
    """
    clusters = results['clusters']
    
    if not clusters:
        print("No significant clusters to plot")
        return None
    
    # Extract cluster info
    cluster_labels = []
    cluster_stats = []
    cluster_pvals = []
    cluster_colors = []
    
    for i, cluster in enumerate(clusters):
        cluster_labels.append(f"{cluster.cluster_type[:3].upper()}\n#{i+1}")
        cluster_stats.append(cluster.cluster_stat)
        cluster_pvals.append(cluster.p_value)
        cluster_colors.append('red' if cluster.cluster_type == 'positive' else 'blue')
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot cluster statistics
    bars1 = ax1.bar(
        range(len(cluster_stats)),
        cluster_stats,
        color=cluster_colors,
        alpha=0.7,
        edgecolor='black',
    )
    ax1.set_xlabel('Cluster', fontsize=12)
    ax1.set_ylabel('Cluster Statistic (sum of t-values)', fontsize=12)
    ax1.set_title('Cluster Statistics', fontsize=14, fontweight='bold')
    ax1.set_xticks(range(len(cluster_labels)))
    ax1.set_xticklabels(cluster_labels)
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax1.grid(axis='y', alpha=0.3)
    
    # Plot p-values
    bars2 = ax2.bar(
        range(len(cluster_pvals)),
        cluster_pvals,
        color=cluster_colors,
        alpha=0.7,
        edgecolor='black',
    )
    ax2.set_xlabel('Cluster', fontsize=12)
    ax2.set_ylabel('Monte Carlo p-value', fontsize=12)
    ax2.set_title('Cluster Significance', fontsize=14, fontweight='bold')
    ax2.set_xticks(range(len(cluster_labels)))
    ax2.set_xticklabels(cluster_labels)
    ax2.axhline(y=0.05, color='red', linestyle='--', linewidth=2, label='α=0.05')
    ax2.set_ylim(0, max(0.06, max(cluster_pvals) * 1.1))
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    plt.suptitle(
        f"{results['marker_name']}\n{len(clusters)} Significant Clusters",
        fontsize=16,
        fontweight='bold',
        y=1.02
    )
    
    plt.tight_layout()
    
    # Save if requested
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {output_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


def create_summary_report(
    results_dir: Path,
    output_path: Optional[str] = None,
):
    """
    Create HTML summary report for all markers.
    
    Parameters
    ----------
    results_dir : Path
        Directory containing result pickle files
    output_path : str, optional
        Path to save HTML report
    """
    # Find all result files
    result_files = list(results_dir.glob("**/*_results.pkl"))
    
    if not result_files:
        print(f"No result files found in {results_dir}")
        return
    
    # Collect summary data
    summary_data = []
    
    for result_file in result_files:
        results = load_results(str(result_file))
        
        marker_name = results['marker_name']
        n_clusters = len(results['clusters'])
        n_electrodes = results['n_electrodes']
        n_observations = results['n_observations']
        
        # Count positive and negative clusters
        n_pos = sum(1 for c in results['clusters'] if c.cluster_type == 'positive')
        n_neg = sum(1 for c in results['clusters'] if c.cluster_type == 'negative')
        
        summary_data.append({
            'Marker': marker_name,
            'Total Clusters': n_clusters,
            'Positive Clusters': n_pos,
            'Negative Clusters': n_neg,
            'N Electrodes': n_electrodes,
            'N Observations': n_observations,
        })
    
    # Create DataFrame
    df = pd.DataFrame(summary_data)
    df = df.sort_values('Total Clusters', ascending=False)
    
    # Create HTML report
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Andrillon 2020 Pipeline - Summary Report</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 40px;
                background-color: #f5f5f5;
            }}
            h1 {{
                color: #333;
                border-bottom: 3px solid #4CAF50;
                padding-bottom: 10px;
            }}
            h2 {{
                color: #555;
                margin-top: 30px;
            }}
            table {{
                border-collapse: collapse;
                width: 100%;
                background-color: white;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            th {{
                background-color: #4CAF50;
                color: white;
                padding: 12px;
                text-align: left;
            }}
            td {{
                padding: 10px;
                border-bottom: 1px solid #ddd;
            }}
            tr:hover {{
                background-color: #f5f5f5;
            }}
            .summary-box {{
                background-color: white;
                padding: 20px;
                margin: 20px 0;
                border-radius: 5px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            .stat {{
                display: inline-block;
                margin: 10px 20px;
                font-size: 18px;
            }}
            .stat-value {{
                font-size: 32px;
                font-weight: bold;
                color: #4CAF50;
            }}
        </style>
    </head>
    <body>
        <h1>Andrillon 2020 Cluster-Permutation Pipeline</h1>
        <h2>Summary Report</h2>
        
        <div class="summary-box">
            <div class="stat">
                <div class="stat-value">{len(df)}</div>
                <div>Markers Analyzed</div>
            </div>
            <div class="stat">
                <div class="stat-value">{df['Total Clusters'].sum()}</div>
                <div>Total Clusters</div>
            </div>
            <div class="stat">
                <div class="stat-value">{df['Positive Clusters'].sum()}</div>
                <div>Positive Clusters</div>
            </div>
            <div class="stat">
                <div class="stat-value">{df['Negative Clusters'].sum()}</div>
                <div>Negative Clusters</div>
            </div>
        </div>
        
        <h2>Results by Marker</h2>
        {df.to_html(index=False, classes='table')}
        
        <div class="summary-box">
            <h3>Configuration</h3>
            <p><strong>Analysis Date:</strong> {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p><strong>Pipeline:</strong> Andrillon et al. (2020)</p>
            <p><strong>Method:</strong> Cluster-permutation with LMM</p>
        </div>
    </body>
    </html>
    """
    
    # Save HTML
    if output_path is None:
        output_path = results_dir / "summary_report.html"
    
    with open(output_path, 'w') as f:
        f.write(html)
    
    print(f"Summary report saved to {output_path}")
    print(f"\nSummary:")
    print(df.to_string(index=False))
    
    return df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Visualize Andrillon 2020 pipeline results"
    )
    parser.add_argument(
        "--results",
        type=str,
        required=True,
        help="Path to results pickle file or directory"
    )
    parser.add_argument(
        "--montage",
        type=str,
        default="../Preprocessing_pipeline_new/CACS-64_REF.bvef",
        help="Path to montage file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save figures"
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Generate summary report for all markers in directory"
    )
    
    args = parser.parse_args()
    
    results_path = Path(args.results)
    
    if args.summary and results_path.is_dir():
        # Generate summary report
        create_summary_report(results_path, args.output_dir)
    elif results_path.is_file():
        # Plot single marker
        results = load_results(str(results_path))
        
        # Create output directory
        if args.output_dir:
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        else:
            output_dir = results_path.parent
        
        marker_name = results['marker_name']
        
        # Plot topography
        topo_path = output_dir / f"{marker_name}_topography.png"
        plot_cluster_topography(
            results,
            args.montage,
            output_path=str(topo_path),
            show=False
        )
        
        # Plot statistics
        if results['clusters']:
            stats_path = output_dir / f"{marker_name}_statistics.png"
            plot_cluster_statistics(
                results,
                output_path=str(stats_path),
                show=False
            )
        
        print(f"Figures saved to {output_dir}")
    else:
        print(f"Error: {results_path} not found")
