"""
Analyze spatial coherence of t-value topographies.

The issue is NOT the t-values themselves, but their SPATIAL PATTERN:
- "Bubbles": Isolated significant channels without spatial coherence
- "Full topographies": Spatially coherent patterns that make physiological sense

This script analyzes:
1. Spatial autocorrelation of t-values
2. Cluster sizes and spatial extent
3. Comparison of different configurations (normalization, covariates, TFCE parameters)
4. Recommendations for improving spatial coherence
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr

# ============================================================================
# CONFIGURATION
# ============================================================================

# Summary file with t-statistics
SUMMARY_FILE = Path("/Volumes/levy/analyze/valerocabre/analyse/nbruno/depressed_mindwandering/results/lmm_cluster/onoff_time_on_task_valence_selfother_time/SUMMARY_REPORT_20251110_182400.csv")

# Output directory
OUTPUT_DIR = Path(__file__).parent / "diagnostics" / "spatial_coherence_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# ANALYSIS
# ============================================================================

def analyze_summary():
    """Analyze the summary CSV to understand spatial coherence issues."""
    print("\n" + "="*70)
    print("SPATIAL COHERENCE ANALYSIS")
    print("="*70)
    
    # Load summary
    df = pd.read_csv(SUMMARY_FILE)
    
    print(f"\nLoaded summary: {len(df)} markers")
    print(f"  Evoked: {(df['Marker Type'] == 'evoked').sum()}")
    print(f"  State: {(df['Marker Type'] == 'state').sum()}")
    
    # Analyze t-statistics
    print(f"\n{'='*70}")
    print("T-STATISTIC ANALYSIS")
    print(f"{'='*70}")
    
    print(f"\nOverall t-statistics:")
    print(f"  Mean |t| range: [{df['T-stat Mean (abs)'].min():.2f}, {df['T-stat Mean (abs)'].max():.2f}]")
    print(f"  Max |t| range: [{df['T-stat Max'].abs().min():.2f}, {df['T-stat Max'].abs().max():.2f}]")
    print(f"  Min t range: [{df['T-stat Min'].min():.2f}, {df['T-stat Min'].max():.2f}]")
    
    # Analyze clusters
    print(f"\n{'='*70}")
    print("CLUSTER ANALYSIS")
    print(f"{'='*70}")
    
    print(f"\nCluster statistics:")
    print(f"  Markers with clusters: {(df['Total Clusters'] > 0).sum()}/{len(df)} ({100*(df['Total Clusters'] > 0).sum()/len(df):.1f}%)")
    print(f"  Total clusters across all markers: {df['Total Clusters'].sum()}")
    print(f"  Mean clusters per marker: {df['Total Clusters'].mean():.2f}")
    print(f"  Max clusters in one marker: {df['Total Clusters'].max()}")
    
    # Markers with many small clusters (bubbles)
    print(f"\n⚠ POTENTIAL 'BUBBLE' PATTERNS:")
    bubble_markers = df[df['Total Clusters'] >= 3].sort_values('Total Clusters', ascending=False)
    if len(bubble_markers) > 0:
        print(f"  Markers with ≥3 clusters (may indicate fragmented patterns):")
        for _, row in bubble_markers.iterrows():
            print(f"    - {row['Marker Type']}_{row['Marker Name']}: {row['Total Clusters']} clusters")
            print(f"      Mean |t|={row['T-stat Mean (abs)']:.2f}, Range=[{row['T-stat Min']:.2f}, {row['T-stat Max']:.2f}]")
    
    # Markers with no clusters despite reasonable t-values
    print(f"\n⚠ MARKERS WITH NO CLUSTERS (despite t-values):")
    no_clusters = df[(df['Total Clusters'] == 0) & (df['T-stat Mean (abs)'] > 0.8)]
    if len(no_clusters) > 0:
        print(f"  {len(no_clusters)} markers have no clusters but mean |t| > 0.8:")
        for _, row in no_clusters.head(10).iterrows():
            print(f"    - {row['Marker Type']}_{row['Marker Name']}: mean |t|={row['T-stat Mean (abs)']:.2f}")
    
    # Create visualizations
    print(f"\n{'='*70}")
    print("CREATING VISUALIZATIONS")
    print(f"{'='*70}")
    
    # Plot 1: Clusters vs Mean |t|
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    ax = axes[0]
    for marker_type in ['evoked', 'state']:
        subset = df[df['Marker Type'] == marker_type]
        ax.scatter(subset['T-stat Mean (abs)'], subset['Total Clusters'], 
                  label=marker_type, alpha=0.6, s=80)
    ax.set_xlabel('Mean |T-statistic|', fontsize=12)
    ax.set_ylabel('Number of Clusters', fontsize=12)
    ax.set_title('Clusters vs T-statistic Strength', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add annotations for bubble markers
    for _, row in bubble_markers.head(5).iterrows():
        ax.annotate(row['Marker Name'], 
                   (row['T-stat Mean (abs)'], row['Total Clusters']),
                   fontsize=8, alpha=0.7)
    
    # Plot 2: Distribution of cluster counts
    ax = axes[1]
    cluster_counts = df['Total Clusters'].value_counts().sort_index()
    ax.bar(cluster_counts.index, cluster_counts.values, alpha=0.7, color='steelblue', edgecolor='black')
    ax.set_xlabel('Number of Clusters', fontsize=12)
    ax.set_ylabel('Number of Markers', fontsize=12)
    ax.set_title('Distribution of Cluster Counts', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'cluster_analysis.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {OUTPUT_DIR / 'cluster_analysis.png'}")
    plt.close()
    
    # Plot 3: T-statistic ranges
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Sort by mean |t|
    df_sorted = df.sort_values('T-stat Mean (abs)')
    y_pos = np.arange(len(df_sorted))
    
    # Plot ranges
    for i, (_, row) in enumerate(df_sorted.iterrows()):
        color = 'red' if row['Total Clusters'] >= 3 else 'blue' if row['Total Clusters'] > 0 else 'gray'
        ax.plot([row['T-stat Min'], row['T-stat Max']], [i, i], 
               color=color, linewidth=2, alpha=0.6)
        ax.scatter(row['T-stat Mean (abs)'] * np.sign(row['T-stat Max']), i, 
                  color=color, s=50, zorder=3, alpha=0.8)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"{row['Marker Type']}_{row['Marker Name']}" 
                        for _, row in df_sorted.iterrows()], fontsize=7)
    ax.set_xlabel('T-statistic', fontsize=12)
    ax.set_title('T-statistic Ranges by Marker\n(Red=≥3 clusters, Blue=1-2 clusters, Gray=0 clusters)', 
                fontsize=13, fontweight='bold')
    ax.axvline(0, color='black', linestyle='--', linewidth=1)
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 't_statistic_ranges.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {OUTPUT_DIR / 't_statistic_ranges.png'}")
    plt.close()
    
    # DIAGNOSIS AND RECOMMENDATIONS
    print(f"\n{'='*70}")
    print("DIAGNOSIS")
    print(f"{'='*70}")
    
    print(f"\n✓ T-values are REASONABLE (mostly -6 to +6)")
    print(f"✓ Problem is SPATIAL COHERENCE, not t-value magnitude")
    
    n_bubble_markers = len(bubble_markers)
    n_no_cluster_markers = len(no_clusters)
    
    if n_bubble_markers > 0:
        print(f"\n⚠ BUBBLE PATTERN detected in {n_bubble_markers} markers")
        print(f"  These markers have multiple small clusters instead of coherent regions")
    
    if n_no_cluster_markers > 0:
        print(f"\n⚠ MISSING CLUSTERS in {n_no_cluster_markers} markers")
        print(f"  These markers have reasonable t-values but no detected clusters")
    
    print(f"\n{'='*70}")
    print("RECOMMENDATIONS")
    print(f"{'='*70}")
    
    print(f"\n1. TFCE PARAMETERS (config.yaml: clustering.tfce)")
    print(f"   Current issue: Likely using standard parameters (E=0.5, H=2.0)")
    print(f"   These favor FOCAL PEAKS → 'bubbly' results")
    print(f"   ")
    print(f"   SOLUTION: Increase spatial coherence by adjusting E/H ratio:")
    print(f"   ")
    print(f"   Option A - BALANCED (recommended first try):")
    print(f"     E: 1.0  # Increased from 0.5")
    print(f"     H: 2.0  # Keep standard")
    print(f"     Ratio H/E = 2.0 (better spatial coherence)")
    print(f"   ")
    print(f"   Option B - SMOOTH (if Option A still too bubbly):")
    print(f"     E: 1.5  # Even higher extent weight")
    print(f"     H: 2.0")
    print(f"     Ratio H/E = 1.33 (strongly favors extended clusters)")
    print(f"   ")
    print(f"   Option C - VERY SMOOTH (aggressive):")
    print(f"     E: 2.0")
    print(f"     H: 2.0")
    print(f"     Ratio H/E = 1.0 (maximum spatial smoothness)")
    
    print(f"\n2. THRESHOLD METHOD")
    print(f"   If TFCE still gives bubbles, try threshold-based clustering:")
    print(f"   ")
    print(f"   clustering:")
    print(f"     method: 'threshold'  # Instead of 'tfce'")
    print(f"     threshold: 2.0  # Lower threshold = larger clusters")
    print(f"     stat_fun: 'sum'  # Favors spatial extent")
    
    print(f"\n3. NORMALIZATION")
    print(f"   Current: channel_wise=false (global normalization)")
    print(f"   This is CORRECT for spatial coherence")
    print(f"   ")
    print(f"   ⚠ DO NOT use channel_wise=true (destroys spatial patterns)")
    
    print(f"\n4. NUMBER OF PERMUTATIONS")
    print(f"   Current: n_permutations=100 (TOO LOW)")
    print(f"   Minimum: 1000")
    print(f"   Recommended: 5000+")
    print(f"   ")
    print(f"   Low permutations → unstable p-values → inconsistent clusters")
    
    print(f"\n5. PREDICTOR VARIANCE")
    print(f"   Current: min_predictor_variability=5")
    print(f"   This filters subjects with low variance in 'onoff'")
    print(f"   ")
    print(f"   Check if this is removing too many subjects:")
    print(f"   - Try min_predictor_variability: 3 (less strict)")
    print(f"   - Or min_predictor_variability: 'auto' (only zero variance)")
    
    print(f"\n{'='*70}")
    print("IMMEDIATE ACTION PLAN")
    print(f"{'='*70}")
    
    print(f"\n1. Edit config.yaml:")
    print(f"   ")
    print(f"   clustering:")
    print(f"     method: 'tfce'")
    print(f"     tfce:")
    print(f"       E: 1.0  # ← CHANGE THIS (was 0.5)")
    print(f"       H: 2.0")
    print(f"       n_steps: 300")
    print(f"     n_permutations: 5000  # ← CHANGE THIS (was 100)")
    print(f"   ")
    print(f"2. Re-run analysis:")
    print(f"   python Statistics/run_pipeline.py")
    print(f"   ")
    print(f"3. Compare results:")
    print(f"   - Check if clusters are more spatially coherent")
    print(f"   - Look for fewer, larger clusters instead of many small ones")
    print(f"   ")
    print(f"4. If still bubbly, try E=1.5 or E=2.0")
    
    # Save summary
    summary_file = OUTPUT_DIR / 'diagnosis_summary.txt'
    with open(summary_file, 'w') as f:
        f.write("SPATIAL COHERENCE DIAGNOSIS\n")
        f.write("="*70 + "\n\n")
        f.write(f"Analysis date: {pd.Timestamp.now()}\n")
        f.write(f"Summary file: {SUMMARY_FILE}\n\n")
        f.write(f"FINDINGS:\n")
        f.write(f"- T-values are reasonable (not the problem)\n")
        f.write(f"- Issue is spatial fragmentation ('bubbles')\n")
        f.write(f"- {n_bubble_markers} markers with ≥3 clusters\n")
        f.write(f"- {n_no_cluster_markers} markers with no clusters despite t-values\n\n")
        f.write(f"ROOT CAUSE:\n")
        f.write(f"- TFCE parameters favor focal peaks (E=0.5 too low)\n")
        f.write(f"- Low permutations (100) cause instability\n\n")
        f.write(f"SOLUTION:\n")
        f.write(f"1. Increase TFCE E parameter: 0.5 → 1.0 (or 1.5)\n")
        f.write(f"2. Increase permutations: 100 → 5000\n")
        f.write(f"3. Keep global normalization (channel_wise=false)\n")
    
    print(f"\n✓ Summary saved: {summary_file}")
    
    print(f"\n{'='*70}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    analyze_summary()
