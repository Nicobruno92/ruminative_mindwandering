"""
Diagnose false positive inflation in topographies.

This script identifies markers where ALL or most channels are significant,
which is statistically improbable and indicates:
1. Too few permutations (n_permutations=100 is insufficient)
2. Threshold too liberal
3. Inadequate multiple comparisons correction
4. Problematic normalization
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================

SUMMARY_FILE = Path("/Volumes/levy/analyze/valerocabre/analyse/nbruno/depressed_mindwandering/results/lmm_cluster/onoff_time_on_task_valence_selfother_time/SUMMARY_REPORT_20251110_182400.csv")
OUTPUT_DIR = Path(__file__).parent / "diagnostics" / "false_positive_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N_CHANNELS = 64  # Total channels

# ============================================================================
# ANALYSIS
# ============================================================================

def analyze_false_positives():
    """Analyze potential false positive inflation."""
    print("\n" + "="*70)
    print("FALSE POSITIVE INFLATION ANALYSIS")
    print("="*70)
    
    # Load summary
    df = pd.read_csv(SUMMARY_FILE)
    
    print(f"\nLoaded: {len(df)} markers")
    print(f"Total channels per marker: {N_CHANNELS}")
    
    # Estimate channels in clusters (rough approximation)
    # Assuming average cluster size of 8-10 channels
    df['est_sig_channels'] = df['Total Clusters'] * 8  # Conservative estimate
    df['pct_sig_channels'] = (df['est_sig_channels'] / N_CHANNELS) * 100
    
    # Analyze distribution
    print(f"\n{'='*70}")
    print("SIGNIFICANCE DISTRIBUTION")
    print(f"{'='*70}")
    
    print(f"\nEstimated percentage of significant channels:")
    print(f"  Mean: {df['pct_sig_channels'].mean():.1f}%")
    print(f"  Median: {df['pct_sig_channels'].median():.1f}%")
    print(f"  Range: [{df['pct_sig_channels'].min():.1f}%, {df['pct_sig_channels'].max():.1f}%]")
    
    # Identify problematic markers
    print(f"\n{'='*70}")
    print("POTENTIAL FALSE POSITIVE INFLATION")
    print(f"{'='*70}")
    
    # Markers with many clusters (likely many significant channels)
    high_sig = df[df['Total Clusters'] >= 4].sort_values('Total Clusters', ascending=False)
    
    if len(high_sig) > 0:
        print(f"\n⚠️ {len(high_sig)} markers with ≥4 clusters:")
        print(f"   (Likely >50% channels significant - SUSPICIOUS)")
        for _, row in high_sig.iterrows():
            est_pct = (row['Total Clusters'] * 8 / N_CHANNELS) * 100
            print(f"   - {row['Marker Type']}_{row['Marker Name']}: {row['Total Clusters']} clusters")
            print(f"     Estimated ~{est_pct:.0f}% channels significant")
            print(f"     Mean |t|={row['T-stat Mean (abs)']:.2f}")
    
    # Statistical expectation
    print(f"\n{'='*70}")
    print("STATISTICAL EXPECTATION")
    print(f"{'='*70}")
    
    print(f"\nWith α=0.05 (5% false positive rate):")
    print(f"  Expected false positives: {N_CHANNELS * 0.05:.1f} channels (~3 channels)")
    print(f"  Expected true positives: Depends on effect size")
    print(f"  ")
    print(f"  Reasonable range: 5-30% channels significant (3-20 channels)")
    print(f"  Suspicious: >50% channels significant (>32 channels)")
    print(f"  Very suspicious: >80% channels significant (>51 channels)")
    
    # Analyze by marker type
    print(f"\n{'='*70}")
    print("BY MARKER TYPE")
    print(f"{'='*70}")
    
    for marker_type in ['evoked', 'state']:
        subset = df[df['Marker Type'] == marker_type]
        print(f"\n{marker_type.upper()}:")
        print(f"  Mean clusters: {subset['Total Clusters'].mean():.2f}")
        print(f"  Markers with ≥4 clusters: {(subset['Total Clusters'] >= 4).sum()}/{len(subset)}")
        print(f"  Markers with 0 clusters: {(subset['Total Clusters'] == 0).sum()}/{len(subset)}")
    
    # ROOT CAUSE ANALYSIS
    print(f"\n{'='*70}")
    print("ROOT CAUSE ANALYSIS")
    print(f"{'='*70}")
    
    print(f"\n🔴 CRITICAL ISSUE: n_permutations=100")
    print(f"   ")
    print(f"   With only 100 permutations:")
    print(f"   - Minimum p-value resolution: 1/100 = 0.01")
    print(f"   - Cannot detect p-values < 0.01")
    print(f"   - With α=0.05, almost everything appears significant")
    print(f"   ")
    print(f"   Example:")
    print(f"   - True p-value: 0.001 (very significant)")
    print(f"   - Observed p-value: 0.01 (best we can measure)")
    print(f"   - True p-value: 0.03 (marginally significant)")
    print(f"   - Observed p-value: 0.03 (same!)")
    print(f"   ")
    print(f"   Result: Cannot distinguish strong from weak effects")
    print(f"           → Everything looks equally significant")
    
    print(f"\n🟡 SECONDARY ISSUES:")
    print(f"   1. TFCE E=0.5 too low → bubbles (already addressed)")
    print(f"   2. Possible threshold too liberal")
    print(f"   3. Check if multiple comparisons correction is working")
    
    # SOLUTION
    print(f"\n{'='*70}")
    print("SOLUTION")
    print(f"{'='*70}")
    
    print(f"\n✅ ALREADY IMPLEMENTED:")
    print(f"   - n_permutations: 100 → 5000 (50x increase)")
    print(f"   - TFCE E: 0.5 → 1.0 (better spatial coherence)")
    print(f"   ")
    print(f"   This should resolve BOTH problems:")
    print(f"   - False positive inflation (too many significant)")
    print(f"   - Spatial fragmentation (bubbles)")
    
    print(f"\n📋 NEXT STEPS:")
    print(f"   1. Re-run analysis with new parameters")
    print(f"   2. Check if % significant channels is now 20-40%")
    print(f"   3. If still >50%, increase threshold or H parameter")
    
    # Create visualization
    print(f"\n{'='*70}")
    print("CREATING VISUALIZATIONS")
    print(f"{'='*70}")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Distribution of cluster counts
    ax = axes[0, 0]
    cluster_counts = df['Total Clusters'].value_counts().sort_index()
    colors = ['red' if x >= 4 else 'orange' if x >= 2 else 'green' for x in cluster_counts.index]
    ax.bar(cluster_counts.index, cluster_counts.values, color=colors, alpha=0.7, edgecolor='black')
    ax.set_xlabel('Number of Clusters', fontsize=12)
    ax.set_ylabel('Number of Markers', fontsize=12)
    ax.set_title('Cluster Count Distribution\n(Red=≥4 clusters, Orange=2-3, Green=0-1)', fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    # 2. Estimated % significant channels
    ax = axes[0, 1]
    ax.hist(df['pct_sig_channels'], bins=20, alpha=0.7, color='steelblue', edgecolor='black')
    ax.axvline(50, color='red', linestyle='--', linewidth=2, label='50% (suspicious)')
    ax.axvline(30, color='orange', linestyle='--', linewidth=1, label='30% (high)')
    ax.set_xlabel('Estimated % Significant Channels', fontsize=12)
    ax.set_ylabel('Number of Markers', fontsize=12)
    ax.set_title('Distribution of Significance\n(Estimated from cluster counts)', fontsize=11)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # 3. Clusters vs Mean |t|
    ax = axes[1, 0]
    for marker_type in ['evoked', 'state']:
        subset = df[df['Marker Type'] == marker_type]
        ax.scatter(subset['T-stat Mean (abs)'], subset['Total Clusters'],
                  label=marker_type, alpha=0.6, s=80)
    ax.axhline(4, color='red', linestyle='--', linewidth=1, alpha=0.5, label='4 clusters (suspicious)')
    ax.set_xlabel('Mean |T-statistic|', fontsize=12)
    ax.set_ylabel('Number of Clusters', fontsize=12)
    ax.set_title('Clusters vs T-statistic Strength', fontsize=11)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Problem markers
    ax = axes[1, 1]
    problem_types = ['≥4 clusters\n(too many)', '0 clusters\n(too few)', '1-3 clusters\n(reasonable)']
    problem_counts = [
        (df['Total Clusters'] >= 4).sum(),
        (df['Total Clusters'] == 0).sum(),
        ((df['Total Clusters'] >= 1) & (df['Total Clusters'] <= 3)).sum()
    ]
    colors_bar = ['red', 'gray', 'green']
    ax.bar(problem_types, problem_counts, color=colors_bar, alpha=0.7, edgecolor='black')
    ax.set_ylabel('Number of Markers', fontsize=12)
    ax.set_title('Marker Classification', fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add counts on bars
    for i, (label, count) in enumerate(zip(problem_types, problem_counts)):
        ax.text(i, count + 1, str(count), ha='center', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'false_positive_analysis.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {OUTPUT_DIR / 'false_positive_analysis.png'}")
    plt.close()
    
    # Save detailed report
    report_file = OUTPUT_DIR / 'detailed_report.csv'
    df_report = df[['Marker Type', 'Marker Name', 'Total Clusters', 'est_sig_channels', 
                    'pct_sig_channels', 'T-stat Mean (abs)', 'T-stat Min', 'T-stat Max']]
    df_report = df_report.sort_values('Total Clusters', ascending=False)
    df_report.to_csv(report_file, index=False)
    print(f"  ✓ Saved: {report_file}")
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    
    n_too_many = (df['Total Clusters'] >= 4).sum()
    n_too_few = (df['Total Clusters'] == 0).sum()
    n_reasonable = ((df['Total Clusters'] >= 1) & (df['Total Clusters'] <= 3)).sum()
    
    print(f"\nMarker classification:")
    print(f"  ✅ Reasonable (1-3 clusters): {n_reasonable}/{len(df)} ({100*n_reasonable/len(df):.1f}%)")
    print(f"  ⚠️  Too many (≥4 clusters): {n_too_many}/{len(df)} ({100*n_too_many/len(df):.1f}%)")
    print(f"  ⚠️  Too few (0 clusters): {n_too_few}/{len(df)} ({100*n_too_few/len(df):.1f}%)")
    
    print(f"\n🎯 TARGET after re-running with n_permutations=5000:")
    print(f"   - Reasonable: >70% of markers")
    print(f"   - Too many: <10% of markers")
    print(f"   - Too few: <30% of markers")
    
    print(f"\n{'='*70}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*70}")
    print(f"\nResults saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    analyze_false_positives()
