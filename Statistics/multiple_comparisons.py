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


def _dispatch_correction(
    p_values: np.ndarray,
    correction_method: str,
    alpha: float
) -> Tuple[np.ndarray, np.ndarray, str]:
    """
    Route a p-value array to the requested correction implementation.

    Parameters
    ----------
    p_values : np.ndarray
        Uncorrected p-values forming one family.
    correction_method : str
        One of 'fdr_bh', 'fdr_indep', 'fdr_by', 'fdr_negcorr', 'bonferroni'.
    alpha : float
        Significance level.

    Returns
    -------
    rejected : np.ndarray
        Boolean rejection flags.
    corrected_p : np.ndarray
        Corrected p-values.
    correction_name : str
        Human-readable name of the applied correction.

    Raises
    ------
    ValueError
        If ``correction_method`` is not recognised.
    """
    if correction_method in ('fdr_bh', 'fdr_indep'):
        rejected, corrected_p = apply_fdr_correction(p_values, alpha, 'indep')
        return rejected, corrected_p, "FDR (Benjamini-Hochberg)"
    if correction_method in ('fdr_by', 'fdr_negcorr'):
        rejected, corrected_p = apply_fdr_correction(p_values, alpha, 'negcorr')
        return rejected, corrected_p, "FDR (Benjamini-Yekutieli)"
    if correction_method == 'bonferroni':
        rejected, corrected_p = apply_bonferroni_correction(p_values, alpha)
        return rejected, corrected_p, "Bonferroni"
    raise ValueError(
        f"Unknown correction method: {correction_method}. "
        f"Use 'fdr_bh', 'fdr_indep', 'fdr_by', 'fdr_negcorr', 'bonferroni', or False"
    )


def _marker_representative_p(result: Dict) -> float:
    """
    Return the single p-value that represents a marker in a family correction.

    The cluster p-values produced by the permutation test are already calibrated
    against a *max-statistic* null distribution, which controls the family-wise
    error rate across electrodes within the marker. Only the most extreme
    cluster is calibrated against that null; sub-maximal clusters carry a
    conservative p-value. The marker is therefore represented by its minimum
    cluster p-value — the max-statistic cluster.

    A marker with no clusters produced no candidate above the cluster-forming
    threshold, i.e. no evidence at all, and is represented by p = 1.0. Such
    markers are kept in the family: they were tested, and dropping them would
    shrink the family size post hoc and inflate significance.

    Parameters
    ----------
    result : Dict
        Marker result dictionary containing ``cluster_p_values``.

    Returns
    -------
    float
        Representative (uncorrected) p-value for this marker.

    Notes
    -----
    With ``separate_signs=True`` the permutation step builds two max-statistic
    nulls per marker (positive and negative). Taking the minimum across signs
    is mildly anti-conservative (by a factor of at most 2). This is left
    uncompensated deliberately: the family-level correction used here
    (Benjamini-Yekutieli) is itself conservative under arbitrary dependence and
    absorbs it. Documented so the choice is visible rather than implicit.
    """
    p_vals = np.asarray(result.get('cluster_p_values', []), dtype=float)
    valid = p_vals[~np.isnan(p_vals)]
    if valid.size == 0:
        return 1.0
    return float(np.min(valid))


def _correct_at_marker_level(
    results_list: List[Dict],
    correction_method: str,
    alpha: float,
    verbose: bool
) -> List[Dict]:
    """
    Correct across markers, using one representative p-value per marker.

    Each marker contributes exactly one hypothesis ("does this marker show a
    mind-wandering effect?"), so the family size is the number of markers
    tested, not the number of clusters they happen to produce.

    Cluster-level significance is then resolved by hierarchical gatekeeping:
    a cluster is significant only if (a) its marker survived the family-wise
    correction and (b) its own permutation p-value is below alpha. That rule is
    expressed as an adjusted per-cluster p-value,
    ``max(marker_p_corrected, cluster_p_raw)``, so every downstream consumer
    that thresholds ``cluster_p_values_corrected`` at alpha stays correct
    without modification.

    Parameters
    ----------
    results_list : List[Dict]
        All marker results belonging to one family (e.g. all 'state' markers).
    correction_method : str
        Correction to apply across markers.
    alpha : float
        Significance level.
    verbose : bool
        Print a per-marker summary.

    Returns
    -------
    List[Dict]
        The same list, with correction fields added in place.
    """
    marker_p = np.array([_marker_representative_p(r) for r in results_list], dtype=float)
    n_markers = len(marker_p)

    rejected, corrected_p, correction_name = _dispatch_correction(
        marker_p, correction_method, alpha
    )

    if verbose:
        print(f"\nApplying multiple comparisons correction:")
        print(f"  Method: {correction_name} ({correction_method})")
        print(f"  Unit: marker (one max-statistic p-value per marker)")
        print(f"  Family size (markers tested): {n_markers}")
        print(f"  Alpha level: {alpha}")

    for result, p_raw, p_corr, rej in zip(results_list, marker_p, corrected_p, rejected):
        cluster_p_raw = np.asarray(result.get('cluster_p_values', []), dtype=float)

        # Hierarchical gatekeeping expressed as an adjusted p-value.
        cluster_p_corrected = np.maximum(p_corr, cluster_p_raw)

        result['marker_p_value'] = float(p_raw)
        result['marker_p_value_corrected'] = float(p_corr)
        result['marker_rejected'] = bool(rej)
        result['cluster_p_values_corrected'] = cluster_p_corrected
        result['cluster_rejected'] = cluster_p_corrected <= alpha
        result['correction_method'] = correction_method
        result['correction_alpha'] = alpha
        result['correction_unit'] = 'marker'
        result['correction_family_size'] = n_markers

    if verbose:
        n_sig_before = int(np.sum(marker_p <= alpha))
        n_sig_after = int(np.sum(rejected))
        print(f"\n  Correction summary (markers):")
        print(f"    Significant before correction: {n_sig_before}/{n_markers}")
        print(f"    Significant after correction:  {n_sig_after}/{n_markers}")
        print(f"\n  Per-marker summary:")
        for result, p_raw, p_corr in zip(results_list, marker_p, corrected_p):
            name = result.get('marker_name', 'unknown')
            n_cl_before = int(np.sum(
                np.asarray(result.get('cluster_p_values', []), dtype=float) <= alpha
            ))
            n_cl_after = int(np.sum(result['cluster_rejected']))
            flag = "✓" if result['marker_rejected'] else " "
            print(f"    {flag} {name}: p={p_raw:.4f} → p_corr={p_corr:.4f} "
                  f"| clusters {n_cl_before} → {n_cl_after}")

    return results_list


def correct_cluster_p_values(
    results_list: List[Dict],
    correction_method: Union[str, bool],
    alpha: float = 0.05,
    unit: str = 'cluster',
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
    unit : str
        Unit of correction, i.e. what counts as one hypothesis in the family:
        - "marker": one representative (max-statistic) p-value per marker.
          Family size = number of markers tested. Cluster significance is then
          resolved by hierarchical gatekeeping. This is the statistically
          coherent choice, because the per-cluster p-values already come from a
          max-statistic null that controls FWER across electrodes within the
          marker — only the most extreme cluster is calibrated against it.
        - "cluster": every cluster from every marker is one hypothesis.
          Family size = total number of clusters. Preserved for backwards
          compatibility with the original Statistics/ pipeline.
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
        - 'correction_unit': unit the correction was applied over
        With unit="marker" the following are added as well:
        - 'marker_p_value': uncorrected max-statistic p-value for the marker
        - 'marker_p_value_corrected': its corrected counterpart
        - 'marker_rejected': whether the marker survived the family correction
        - 'correction_family_size': number of hypotheses in the family

    Notes
    -----
    - Original uncorrected p-values are ALWAYS preserved in 'cluster_p_values'
    - NaN p-values are preserved and excluded from correction
    - With unit="cluster", markers without clusters are skipped
    - With unit="marker", markers without clusters remain in the family with
      p = 1.0, so the family size reflects every hypothesis actually tested

    Examples
    --------
    >>> results = [result1, result2, result3]  # Each has cluster_p_values
    >>> corrected_results = correct_cluster_p_values(
    ...     results, correction_method="fdr_bh", alpha=0.05
    ... )
    >>> # Each result now has 'cluster_p_values_corrected' field
    """
    if unit not in ('marker', 'cluster'):
        raise ValueError(
            f"Unknown correction unit: {unit!r}. Use 'marker' or 'cluster'."
        )

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
                result['correction_unit'] = unit
        return results_list

    if unit == 'marker':
        return _correct_at_marker_level(
            results_list, correction_method, alpha, verbose
        )

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
    rejected, corrected_p, correction_name = _dispatch_correction(
        all_p_values, correction_method, alpha
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
        results_list[result_idx]['correction_unit'] = 'cluster'
        results_list[result_idx]['correction_family_size'] = n_total_tests

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

        p_vals = np.asarray(result['cluster_p_values'], dtype=float)
        unit = result.get('correction_unit', 'cluster')

        # With unit="cluster" a marker without clusters contributes nothing.
        # With unit="marker" it is a tested hypothesis and must stay visible,
        # otherwise the reported family size cannot be reconciled with the CSV.
        if len(p_vals) == 0 and unit != 'marker':
            continue

        # Get correction info
        p_vals_corr = np.asarray(
            result.get('cluster_p_values_corrected', p_vals), dtype=float
        )
        alpha = result.get('correction_alpha', 0.05)
        method = result.get('correction_method', 'none')

        # Count significant clusters
        n_sig_uncorr = int(np.sum(p_vals <= alpha))
        # Use stored rejection array if available, otherwise recompute
        rejected_arr = result.get('cluster_rejected')
        if rejected_arr is not None:
            n_sig_corr = int(np.sum(rejected_arr))
        else:
            n_sig_corr = int(np.sum(p_vals_corr <= alpha))

        # Always report the uncorrected and the corrected p-value side by side,
        # at both the marker level and the best-cluster level.
        row = {
            'marker_name': result.get('marker_name', 'unknown'),
            'marker_type': result.get('marker_type', 'unknown'),
            'n_clusters': len(p_vals),
            'min_cluster_p_uncorrected': float(np.min(p_vals)) if len(p_vals) else np.nan,
            'min_cluster_p_corrected': float(np.min(p_vals_corr)) if len(p_vals_corr) else np.nan,
            'n_sig_uncorrected': n_sig_uncorr,
            'n_sig_corrected': n_sig_corr,
            'marker_p_uncorrected': result.get('marker_p_value', np.nan),
            'marker_p_corrected': result.get('marker_p_value_corrected', np.nan),
            'marker_rejected': result.get('marker_rejected', pd.NA),
            'correction_method': method,
            'correction_unit': unit,
            'correction_family_size': result.get('correction_family_size', pd.NA),
            'correction_alpha': alpha,
        }
        summary_data.append(row)


    summary_df = pd.DataFrame(summary_data)
    
    if output_path:
        summary_df.to_csv(output_path, index=False)
        print(f"✓ Correction summary saved to {output_path}")
    
    return summary_df
