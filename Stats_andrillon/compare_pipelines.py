"""
Compare Results: Andrillon 2020 vs Current Pipeline

Compare cluster detection results between:
- Andrillon 2020 pipeline (this implementation)
- Current MNE-based pipeline

Generates comparison reports and visualizations.
"""

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple
import seaborn as sns


def load_andrillon_results(results_path: str) -> Dict:
    """Load Andrillon pipeline results."""
    with open(results_path, 'rb') as f:
        results = pickle.load(f)
    return results


def load_current_pipeline_results(results_path: str) -> Dict:
    """Load current pipeline results."""
    with open(results_path, 'rb') as f:
        results = pickle.load(f)
    return results


def extract_cluster_info(results: Dict, pipeline_type: str) -> pd.DataFrame:
    """
    Extract cluster information into DataFrame.
    
    Parameters
    ----------
    results : dict
        Results dictionary
    pipeline_type : str
        'andrillon' or 'current'
        
    Returns
    -------
    df : pd.DataFrame
        Cluster information
    """
    if pipeline_type == 'andrillon':
        clusters = results.get('clusters', [])
        
        data = []
        for i, cluster in enumerate(clusters):
            data.append({
                'cluster_id': i,
                'cluster_type': cluster.cluster_type,
                'n_electrodes': len(cluster.electrodes),
                'electrodes': set(cluster.electrodes),
                'cluster_stat': cluster.cluster_stat,
                'p_value': cluster.p_value,
                'pipeline': 'Andrillon 2020',
            })
        
        return pd.DataFrame(data)
    
    elif pipeline_type == 'current':
        # Adapt to current pipeline format
        # This will depend on the exact structure of current pipeline results
        clusters = results.get('clusters', [])
        
        data = []
        for i, cluster in enumerate(clusters):
            # Extract cluster info based on current pipeline structure
            # Adjust as needed
            data.append({
                'cluster_id': i,
                'cluster_type': 'positive' if cluster.get('stat', 0) > 0 else 'negative',
                'n_electrodes': len(cluster.get('channels', [])),
                'electrodes': set(cluster.get('channels', [])),
                'cluster_stat': cluster.get('stat', 0),
                'p_value': cluster.get('pval', 1.0),
                'pipeline': 'Current (MNE)',
            })
        
        return pd.DataFrame(data)
    
    else:
        raise ValueError(f"Unknown pipeline type: {pipeline_type}")


def compare_clusters(
    andrillon_results: Dict,
    current_results: Dict,
) -> Dict:
    """
    Compare clusters between pipelines.
    
    Parameters
    ----------
    andrillon_results : dict
        Andrillon pipeline results
    current_results : dict
        Current pipeline results
        
    Returns
    -------
    comparison : dict
        Comparison metrics
    """
    # Extract cluster info
    df_andrillon = extract_cluster_info(andrillon_results, 'andrillon')
    df_current = extract_cluster_info(current_results, 'current')
    
    # Basic statistics
    comparison = {
        'andrillon': {
            'n_clusters': len(df_andrillon),
            'n_positive': len(df_andrillon[df_andrillon['cluster_type'] == 'positive']),
            'n_negative': len(df_andrillon[df_andrillon['cluster_type'] == 'negative']),
            'mean_cluster_size': df_andrillon['n_electrodes'].mean() if len(df_andrillon) > 0 else 0,
        },
        'current': {
            'n_clusters': len(df_current),
            'n_positive': len(df_current[df_current['cluster_type'] == 'positive']),
            'n_negative': len(df_current[df_current['cluster_type'] == 'negative']),
            'mean_cluster_size': df_current['n_electrodes'].mean() if len(df_current) > 0 else 0,
        },
    }
    
    # Spatial overlap analysis
    if len(df_andrillon) > 0 and len(df_current) > 0:
        overlaps = []
        for _, andr_cluster in df_andrillon.iterrows():
            for _, curr_cluster in df_current.iterrows():
                # Calculate Jaccard similarity
                intersection = len(andr_cluster['electrodes'] & curr_cluster['electrodes'])
                union = len(andr_cluster['electrodes'] | curr_cluster['electrodes'])
                jaccard = intersection / union if union > 0 else 0
                
                if jaccard > 0:
                    overlaps.append({
                        'andrillon_cluster': andr_cluster['cluster_id'],
                        'current_cluster': curr_cluster['cluster_id'],
                        'jaccard_similarity': jaccard,
                        'n_overlap': intersection,
                    })
        
        comparison['overlaps'] = overlaps
        comparison['mean_jaccard'] = np.mean([o['jaccard_similarity'] for o in overlaps]) if overlaps else 0
    else:
        comparison['overlaps'] = []
        comparison['mean_jaccard'] = 0
    
    return comparison


def plot_comparison(
    andrillon_results: Dict,
    current_results: Dict,
    output_path: Optional[str] = None,
):
    """
    Create comparison visualization.
    
    Parameters
    ----------
    andrillon_results : dict
        Andrillon pipeline results
    current_results : dict
        Current pipeline results
    output_path : str, optional
        Path to save figure
    """
    comparison = compare_clusters(andrillon_results, current_results)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Number of clusters
    ax = axes[0, 0]
    pipelines = ['Andrillon 2020', 'Current (MNE)']
    n_clusters = [
        comparison['andrillon']['n_clusters'],
        comparison['current']['n_clusters']
    ]
    bars = ax.bar(pipelines, n_clusters, color=['#4CAF50', '#2196F3'], alpha=0.7)
    ax.set_ylabel('Number of Clusters', fontsize=12)
    ax.set_title('Total Clusters Detected', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # 2. Positive vs Negative
    ax = axes[0, 1]
    x = np.arange(2)
    width = 0.35
    
    pos_counts = [
        comparison['andrillon']['n_positive'],
        comparison['current']['n_positive']
    ]
    neg_counts = [
        comparison['andrillon']['n_negative'],
        comparison['current']['n_negative']
    ]
    
    ax.bar(x - width/2, pos_counts, width, label='Positive', color='red', alpha=0.7)
    ax.bar(x + width/2, neg_counts, width, label='Negative', color='blue', alpha=0.7)
    
    ax.set_ylabel('Number of Clusters', fontsize=12)
    ax.set_title('Cluster Polarity', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(pipelines)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # 3. Mean cluster size
    ax = axes[1, 0]
    mean_sizes = [
        comparison['andrillon']['mean_cluster_size'],
        comparison['current']['mean_cluster_size']
    ]
    bars = ax.bar(pipelines, mean_sizes, color=['#4CAF50', '#2196F3'], alpha=0.7)
    ax.set_ylabel('Mean Cluster Size (electrodes)', fontsize=12)
    ax.set_title('Average Cluster Size', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # 4. Spatial overlap
    ax = axes[1, 1]
    if comparison['overlaps']:
        jaccard_values = [o['jaccard_similarity'] for o in comparison['overlaps']]
        ax.hist(jaccard_values, bins=20, color='purple', alpha=0.7, edgecolor='black')
        ax.axvline(comparison['mean_jaccard'], color='red', linestyle='--', 
                   linewidth=2, label=f'Mean: {comparison["mean_jaccard"]:.2f}')
        ax.set_xlabel('Jaccard Similarity', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Spatial Overlap Between Pipelines', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No overlapping clusters', 
                ha='center', va='center', fontsize=14, transform=ax.transAxes)
        ax.set_title('Spatial Overlap Between Pipelines', fontsize=14, fontweight='bold')
    
    plt.suptitle(
        f"Pipeline Comparison: {andrillon_results['marker_name']}",
        fontsize=16,
        fontweight='bold',
        y=0.995
    )
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Comparison figure saved to {output_path}")
    
    plt.show()
    
    return fig


def generate_comparison_report(
    andrillon_dir: Path,
    current_dir: Path,
    output_path: Optional[str] = None,
):
    """
    Generate comprehensive comparison report.
    
    Parameters
    ----------
    andrillon_dir : Path
        Directory with Andrillon results
    current_dir : Path
        Directory with current pipeline results
    output_path : str, optional
        Path to save report
    """
    # Find matching markers
    andrillon_files = {f.stem.replace('_results', ''): f 
                       for f in andrillon_dir.glob("**/*_results.pkl")}
    current_files = {f.stem.replace('_results', ''): f 
                     for f in current_dir.glob("**/*_results.pkl")}
    
    common_markers = set(andrillon_files.keys()) & set(current_files.keys())
    
    if not common_markers:
        print("No common markers found between pipelines")
        return
    
    print(f"Found {len(common_markers)} common markers")
    
    # Compare each marker
    all_comparisons = []
    
    for marker in sorted(common_markers):
        print(f"Comparing {marker}...")
        
        andr_results = load_andrillon_results(str(andrillon_files[marker]))
        curr_results = load_current_pipeline_results(str(current_files[marker]))
        
        comparison = compare_clusters(andr_results, curr_results)
        comparison['marker'] = marker
        
        all_comparisons.append(comparison)
    
    # Create summary DataFrame
    summary_data = []
    for comp in all_comparisons:
        summary_data.append({
            'Marker': comp['marker'],
            'Andrillon Clusters': comp['andrillon']['n_clusters'],
            'Current Clusters': comp['current']['n_clusters'],
            'Andrillon Mean Size': comp['andrillon']['mean_cluster_size'],
            'Current Mean Size': comp['current']['mean_cluster_size'],
            'Mean Jaccard Similarity': comp['mean_jaccard'],
            'N Overlaps': len(comp['overlaps']),
        })
    
    df_summary = pd.DataFrame(summary_data)
    
    # Save report
    if output_path is None:
        output_path = "pipeline_comparison_report.html"
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Pipeline Comparison Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            h1 {{ color: #333; border-bottom: 3px solid #4CAF50; }}
            table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
            th {{ background-color: #4CAF50; color: white; padding: 12px; }}
            td {{ padding: 10px; border-bottom: 1px solid #ddd; }}
            tr:hover {{ background-color: #f5f5f5; }}
        </style>
    </head>
    <body>
        <h1>Pipeline Comparison Report</h1>
        <h2>Andrillon 2020 vs Current MNE Pipeline</h2>
        
        <h3>Summary Statistics</h3>
        <p><strong>Markers Compared:</strong> {len(common_markers)}</p>
        <p><strong>Mean Jaccard Similarity:</strong> {df_summary['Mean Jaccard Similarity'].mean():.3f}</p>
        
        <h3>Detailed Comparison</h3>
        {df_summary.to_html(index=False)}
        
        <h3>Methodology Differences</h3>
        <ul>
            <li><strong>Permutation:</strong> Andrillon permutes predictor labels; Current uses Freedman-Lane</li>
            <li><strong>Threshold:</strong> Andrillon uses p-value (0.025); Current uses t-value (3.0)</li>
            <li><strong>Cluster Statistic:</strong> Andrillon uses sum of t-values; Current uses sum of F-values</li>
            <li><strong>Correction:</strong> Andrillon uses Bonferroni; Current uses FDR</li>
        </ul>
    </body>
    </html>
    """
    
    with open(output_path, 'w') as f:
        f.write(html)
    
    print(f"\nComparison report saved to {output_path}")
    print("\nSummary:")
    print(df_summary.to_string(index=False))
    
    return df_summary


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Compare Andrillon 2020 vs Current pipeline results"
    )
    parser.add_argument(
        "--andrillon-results",
        type=str,
        required=True,
        help="Path to Andrillon results (file or directory)"
    )
    parser.add_argument(
        "--current-results",
        type=str,
        required=True,
        help="Path to current pipeline results (file or directory)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for report/figure"
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Generate full comparison report (for directories)"
    )
    
    args = parser.parse_args()
    
    andrillon_path = Path(args.andrillon_results)
    current_path = Path(args.current_results)
    
    if args.report and andrillon_path.is_dir() and current_path.is_dir():
        # Generate full report
        generate_comparison_report(andrillon_path, current_path, args.output)
    elif andrillon_path.is_file() and current_path.is_file():
        # Compare single marker
        andr_results = load_andrillon_results(str(andrillon_path))
        curr_results = load_current_pipeline_results(str(current_path))
        
        plot_comparison(andr_results, curr_results, args.output)
    else:
        print("Error: Provide either two files or two directories with --report flag")
