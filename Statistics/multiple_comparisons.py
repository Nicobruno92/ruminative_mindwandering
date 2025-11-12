"""
Multiple comparisons correction for cluster-based permutation testing.

This module provides functions to adjust p-values when testing multiple markers
(features) to control family-wise error rate (FWER) or false discovery rate (FDR).

Key concepts:
- When testing multiple markers, the probability of false positives increases
- Multiple comparisons correction adjusts p-values to maintain desired error rate
- Can be applied separately for evoked and state markers
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Literal, Union
import warnings
from mne.stats import fdr_correction as mne_fdr_correction
from mne.stats import bonferroni_correction as mne_bonferroni_correction


def apply_fdr_correction(
    p_values: np.ndarray,
    alpha: float = 0.05,
    method: str = 'indep'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply False Discovery Rate (FDR) correction to p-values using MNE implementation.
    
    FDR controls the expected proportion of false discoveries among rejected
    hypotheses. Less conservative than Bonferroni, more appropriate when
    testing many hypotheses.
    
    Parameters
    ----------
    p_values : np.ndarray
        Array of p-values to correct
    alpha : float
        Desired FDR level (typically 0.05)
    method : str
        FDR method: 'indep' (Benjamini-Hochberg) or 'negcorr' (Benjamini-Yekutieli)
        
    Returns
    -------
    rejected : np.ndarray
        Boolean array indicating which hypotheses are rejected at alpha level
    corrected_p_values : np.ndarray
        FDR-corrected p-values
        
    Notes
    -----
    - 'indep': Benjamini-Hochberg for independent or positively correlated tests
    - 'negcorr': Benjamini-Yekutieli for general or negatively correlated tests
    - NaN values are preserved in output
    - Uses MNE-Python's implementation for correctness
    
    References
    ----------
    Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery rate:
    a practical and powerful approach to multiple testing. Journal of the Royal
    Statistical Society: Series B, 57(1), 289-300.
    """
    # Handle NaN values
    valid_mask = ~np.isnan(p_values)
    n_tests = np.sum(valid_mask)
    
    if n_tests == 0:
        return np.zeros(len(p_values), dtype=bool), p_values.copy()
    
    # Initialize output arrays
    corrected_p = p_values.copy()
    rejected = np.zeros(len(p_values), dtype=bool)
    
    # Get valid p-values and their indices
    valid_p = p_values[valid_mask]
    valid_indices = np.where(valid_mask)[0]
    
    # Use MNE's FDR correction on valid p-values
    rejected_valid, corrected_valid = mne_fdr_correction(valid_p, alpha=alpha, method=method)
    
    # Place back into full arrays
    corrected_p[valid_indices] = corrected_valid
    rejected[valid_indices] = rejected_valid
    
    return rejected, corrected_p


def apply_bonferroni_correction(
    p_values: np.ndarray,
    alpha: float = 0.05
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply Bonferroni correction to p-values using MNE implementation.
    
    Bonferroni correction controls the family-wise error rate (FWER) by
    adjusting the significance threshold. Very conservative but guarantees
    strong control of Type I error.
    
    Parameters
    ----------
    p_values : np.ndarray
        Array of p-values to correct
    alpha : float
        Desired FWER level (typically 0.05)
        
    Returns
    -------
    rejected : np.ndarray
        Boolean array indicating which hypotheses are rejected at alpha level
    corrected_p_values : np.ndarray
        Bonferroni-corrected p-values (p * n_tests)
        
    Notes
    -----
    - Corrected p-value = original p-value × number of tests
    - Very conservative, may have low power when testing many hypotheses
    - NaN values are preserved in output
    - Uses MNE-Python's implementation for correctness
    
    References
    ----------
    Bonferroni, C. E. (1936). Teoria statistica delle classi e calcolo delle
    probabilità. Pubblicazioni del R Istituto Superiore di Scienze Economiche
    e Commerciali di Firenze, 8, 3-62.
    """
    # Handle NaN values
    valid_mask = ~np.isnan(p_values)
    n_tests = np.sum(valid_mask)
    
    if n_tests == 0:
        return np.zeros(len(p_values), dtype=bool), p_values.copy()
    
    # Initialize output arrays
    corrected_p = p_values.copy()
    rejected = np.zeros(len(p_values), dtype=bool)
    
    # Get valid p-values and their indices
    valid_p = p_values[valid_mask]
    valid_indices = np.where(valid_mask)[0]
    
    # Use MNE's Bonferroni correction on valid p-values
    rejected_valid, corrected_valid = mne_bonferroni_correction(valid_p, alpha=alpha)
    
    # Place back into full arrays
    corrected_p[valid_indices] = corrected_valid
    rejected[valid_indices] = rejected_valid
    
    return rejected, corrected_p


def correct_cluster_p_values(
    results_list: List[Dict],
    correction_method: Union[str, bool],
    alpha: float = 0.05,
    verbose: bool = True
) -> List[Dict]:
    """
    Apply multiple comparisons correction to cluster p-values across markers.
    
    This function takes results from multiple markers and applies correction
    to the cluster p-values to account for testing multiple features.
    
    Parameters
    ----------
    results_list : List[Dict]
        List of result dictionaries, each containing:
        - 'cluster_p_values': array of p-values for clusters
        - 'marker_name': name of the marker
        - Other metadata
    correction_method : str or bool
        Correction method to apply:
        - False or "none": No correction
        - "fdr_indep" or "fdr_bh": Benjamini-Hochberg FDR correction
        - "fdr_negcorr" or "fdr_by": Benjamini-Yekutieli FDR correction
        - "bonferroni": Bonferroni correction
    alpha : float
        Significance level for correction
    verbose : bool
        Whether to print correction summary
        
    Returns
    -------
    List[Dict]
        Updated results list with corrected p-values added to each result:
        - 'cluster_p_values_corrected': corrected p-values
        - 'cluster_rejected': boolean array of rejected hypotheses
        - 'correction_method': method used
        - 'correction_alpha': alpha level used
        
    Notes
    -----
    - Correction is applied across all clusters from all markers
    - Original p-values are preserved in 'cluster_p_values'
    - If a marker has no clusters, it is skipped
    - NaN p-values are preserved and excluded from correction
    
    Examples
    --------
    >>> results = [result1, result2, result3]  # Each has cluster_p_values
    >>> corrected_results = correct_cluster_p_values(
    ...     results, correction_method="fdr_bh", alpha=0.05
    ... )
    >>> # Each result now has 'cluster_p_values_corrected' field
    """
    # Handle no correction case
    if correction_method is False or correction_method == "none":
        if verbose:
            print("\nNo multiple comparisons correction applied")
        # Add uncorrected values as "corrected" for consistency
        for result in results_list:
            if 'cluster_p_values' in result:
                p_vals = result['cluster_p_values']
                result['cluster_p_values_corrected'] = p_vals.copy()
                result['cluster_rejected'] = p_vals <= alpha
                result['correction_method'] = 'none'
                result['correction_alpha'] = alpha
        return results_list
    
    # Collect all cluster p-values across markers
    all_p_values = []
    marker_cluster_counts = []
    valid_results_indices = []
    
    for idx, result in enumerate(results_list):
        if 'cluster_p_values' not in result:
            continue
        
        p_vals = result['cluster_p_values']
        if len(p_vals) == 0:
            continue
        
        all_p_values.extend(p_vals)
        marker_cluster_counts.append(len(p_vals))
        valid_results_indices.append(idx)
    
    if len(all_p_values) == 0:
        warnings.warn("No cluster p-values found in results")
        return results_list
    
    # Convert to array
    all_p_values = np.array(all_p_values)
    n_total_tests = len(all_p_values)
    
    if verbose:
        print(f"\nApplying multiple comparisons correction:")
        print(f"  Method: {correction_method}")
        print(f"  Total clusters across all markers: {n_total_tests}")
        print(f"  Number of markers: {len(valid_results_indices)}")
        print(f"  Alpha level: {alpha}")
    
    # Apply correction
    if correction_method in ['fdr_bh', 'fdr_indep', 'fdr_by', 'fdr_negcorr']:
        # Map to MNE method names
        if correction_method in ['fdr_bh', 'fdr_indep']:
            method = 'indep'
            correction_name = "FDR (Benjamini-Hochberg)"
        else:  # fdr_by or fdr_negcorr
            method = 'negcorr'
            correction_name = "FDR (Benjamini-Yekutieli)"
        rejected, corrected_p = apply_fdr_correction(all_p_values, alpha, method)
    elif correction_method == 'bonferroni':
        rejected, corrected_p = apply_bonferroni_correction(all_p_values, alpha)
        correction_name = "Bonferroni"
    else:
        raise ValueError(
            f"Unknown correction method: {correction_method}. "
            f"Use 'fdr_bh', 'fdr_indep', 'fdr_by', 'fdr_negcorr', 'bonferroni', or False"
        )
    
    # Split corrected p-values back to individual markers
    start_idx = 0
    for result_idx, n_clusters in zip(valid_results_indices, marker_cluster_counts):
        end_idx = start_idx + n_clusters
        
        # Extract corrected values for this marker
        marker_corrected_p = corrected_p[start_idx:end_idx]
        marker_rejected = rejected[start_idx:end_idx]
        
        # Add to result dictionary
        results_list[result_idx]['cluster_p_values_corrected'] = marker_corrected_p
        results_list[result_idx]['cluster_rejected'] = marker_rejected
        results_list[result_idx]['correction_method'] = correction_method
        results_list[result_idx]['correction_alpha'] = alpha
        
        start_idx = end_idx
    
    # Print summary
    if verbose:
        n_sig_before = np.sum(all_p_values <= alpha)
        n_sig_after = np.sum(rejected)
        
        print(f"\n  Correction summary:")
        print(f"    Significant before correction: {n_sig_before}/{n_total_tests} "
              f"({100*n_sig_before/n_total_tests:.1f}%)")
        print(f"    Significant after correction: {n_sig_after}/{n_total_tests} "
              f"({100*n_sig_after/n_total_tests:.1f}%)")
        
        # Per-marker summary
        print(f"\n  Per-marker summary:")
        start_idx = 0
        for result_idx, n_clusters in zip(valid_results_indices, marker_cluster_counts):
            end_idx = start_idx + n_clusters
            marker_name = results_list[result_idx].get('marker_name', f'Marker {result_idx}')
            
            orig_p = all_p_values[start_idx:end_idx]
            corr_rej = rejected[start_idx:end_idx]
            
            n_sig_orig = np.sum(orig_p <= alpha)
            n_sig_corr = np.sum(corr_rej)
            
            print(f"    {marker_name}: {n_sig_orig} → {n_sig_corr} significant clusters")
            
            start_idx = end_idx
    
    return results_list


def create_correction_summary(
    results_list: List[Dict],
    output_path: str = None
) -> pd.DataFrame:
    """
    Create summary DataFrame of multiple comparisons correction results.
    
    Parameters
    ----------
    results_list : List[Dict]
        List of result dictionaries with correction applied
    output_path : str, optional
        Path to save summary CSV file
        
    Returns
    -------
    pd.DataFrame
        Summary with columns:
        - marker_name: name of marker
        - marker_type: evoked or state
        - n_clusters: total number of clusters
        - n_sig_uncorrected: significant at alpha (uncorrected)
        - n_sig_corrected: significant at alpha (corrected)
        - correction_method: method used
        - correction_alpha: alpha level
    """
    summary_data = []
    
    for result in results_list:
        if 'cluster_p_values' not in result:
            continue
        
        p_vals = result['cluster_p_values']
        if len(p_vals) == 0:
            continue
        
        # Get correction info
        p_vals_corr = result.get('cluster_p_values_corrected', p_vals)
        alpha = result.get('correction_alpha', 0.05)
        method = result.get('correction_method', 'none')
        
        # Count significant clusters
        n_sig_uncorr = np.sum(p_vals <= alpha)
        # Use stored rejection array if available, otherwise recompute
        rejected_arr = result.get('cluster_rejected')
        if rejected_arr is not None:
            n_sig_corr = np.sum(rejected_arr)
        else:
            n_sig_corr = np.sum(p_vals_corr <= alpha)
        
        summary_data.append({
            'marker_name': result.get('marker_name', 'unknown'),
            'marker_type': result.get('marker_type', 'unknown'),
            'n_clusters': len(p_vals),
            'n_sig_uncorrected': n_sig_uncorr,
            'n_sig_corrected': n_sig_corr,
            'correction_method': method,
            'correction_alpha': alpha
        })
    
    summary_df = pd.DataFrame(summary_data)
    
    if output_path:
        summary_df.to_csv(output_path, index=False)
        print(f"✓ Correction summary saved to {output_path}")
    
    return summary_df
