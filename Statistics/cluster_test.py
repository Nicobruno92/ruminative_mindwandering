"""Spatial Cluster Permutation Testing for Linear Mixed Models.

This module implements spatial cluster-based permutation testing adapted for
Linear Mixed Models (LMM), following the principles of Maris & Oostenveld (2007)
but modified to work with LMM t-statistics instead of simple t-tests.

**Why not use MNE's cluster test directly?**

MNE's `mne.stats.permutation_cluster_test` is designed for simple statistical tests
(t-tests, F-tests) between conditions. Our analysis uses Linear Mixed Models (LMM)
to account for:
- Within-subject repeated measures
- Subject-level random effects
- Continuous predictors (e.g., mind-wandering levels)

LMM provides more appropriate statistical modeling for our data structure, but
requires custom permutation logic that preserves the within-subject structure.

**MNE Integration:**

This module directly uses MNE's optimized inner functions:
- `_get_components()`: Find spatially connected clusters (core algorithm)
- `_pval_from_histogram()`: Compute p-values from null distribution
- `_masked_sum_power()`: Compute cluster statistics with power weighting
- `mne.channels.find_ch_adjacency()`: Compute spatial adjacency from montage

All cluster finding follows MNE's exact patterns for maximum compatibility and
performance. Custom implementations only handle LMM-specific permutation logic.

**References:**
Maris, E., & Oostenveld, R. (2007). Nonparametric statistical testing of EEG-
and MEG-data. Journal of Neuroscience Methods, 164(1), 177-190.
"""

import numpy as np
import pandas as pd
import mne
import logging
from typing import Tuple, List, Literal, Dict
from scipy.sparse import issparse, csr_matrix, coo_array
from scipy.sparse.csgraph import connected_components
from scipy import ndimage
import warnings
from joblib import Parallel, delayed

# Import MNE's cluster finding functions (required)
from mne.stats.cluster_level import (
    _get_components,
    _pval_from_histogram,
    _find_clusters as _mne_find_clusters,
    _find_clusters_1dir
)

# Import MNE's helper functions for cluster statistics
from mne.stats.cluster_level import _masked_sum, _masked_sum_power


# Configure logging
logger = logging.getLogger(__name__)


def _validate_adjacency_matrix(adjacency: np.ndarray, n_channels: int) -> None:
    """Validate adjacency matrix properties.
    Parameters
    ----------
    adjacency : np.ndarray
        Adjacency matrix (dense or sparse)
    n_channels : int
        Expected number of channels
        
    Raises
    ------
    ValueError
        If adjacency matrix is invalid
    """
    # Check shape
    if adjacency.shape[0] != n_channels or adjacency.shape[1] != n_channels:
        raise ValueError(
            f"Adjacency matrix shape {adjacency.shape} does not match "
            f"number of channels {n_channels}"
        )
    
    # Convert to array if sparse
    if issparse(adjacency):
        adj_array = adjacency.toarray()
    else:
        adj_array = adjacency
    
    # Check for NaN values
    if np.any(np.isnan(adj_array)):
        raise ValueError("Adjacency matrix contains NaN values")
    
    # Check symmetry
    if not np.allclose(adj_array, adj_array.T):
        warnings.warn("Adjacency matrix is not symmetric")
    
    # Check for reasonable number of connections
    n_connections = np.sum(adj_array > 0)
    connections_per_channel = n_connections / n_channels
    
    if connections_per_channel < 2:
        warnings.warn(
            f"Very sparse adjacency matrix: {connections_per_channel:.1f} "
            f"connections per channel on average"
        )
    elif connections_per_channel > n_channels * 0.5:
        warnings.warn(
            f"Very dense adjacency matrix: {connections_per_channel:.1f} "
            f"connections per channel on average"
        )


def get_channel_adjacency(montage_path: str, ch_names: List[str]) -> Tuple[np.ndarray, List[str], List[int]]:
    """
    Get spatial adjacency matrix for EEG channels using the actual montage file.
    
    This function uses MNE's `find_ch_adjacency()` to compute spatial adjacency
    based on actual channel positions from the montage file. Channels are considered
    adjacent if they are within a certain distance threshold.
    
    Parameters
    ----------
    montage_path : str
        Path to montage file (.bvef or standard montage name)
    ch_names : List[str]
        List of channel names
        
    Returns
    -------
    adjacency : np.ndarray
        Sparse adjacency matrix based on actual montage positions.
        Shape: (n_common_channels, n_common_channels)
    ch_names : List[str]
        Channel names that were used (subset of input that exists in montage)
    channel_indices : List[int]
        Indices of the channels in the original ch_names list
        
    Raises
    ------
    ValueError
        If no common channels found or montage cannot be set
        
    Notes
    -----
    - Uses `mne.channels.find_ch_adjacency()` with actual montage positions
    - Only channels present in both data and montage are included
    - Adjacency is computed based on Euclidean distance between electrodes
    - The returned adjacency matrix is validated for consistency
    
    Examples
    --------
    >>> adjacency, ch_names, indices = get_channel_adjacency(
    ...     "CACS-64_REF.bvef", ["Fp1", "Fp2", "F3", "F4"]
    ... )
    >>> print(f"Adjacency shape: {adjacency.shape}")
    >>> print(f"Channels: {ch_names}")
    """
    # Load montage first to check available channels
    if montage_path.endswith('.bvef'):
        montage = mne.channels.read_custom_montage(montage_path)
        print(f"Loaded custom montage: {montage_path}")
    else:
        montage = mne.channels.make_standard_montage(montage_path)
        print(f"Loaded standard montage: {montage_path}")
    
    # Check which channels from our data are available in the montage
    montage_ch_names = set(montage.ch_names)
    data_ch_names = set(ch_names)
    common_ch_names = sorted(list(data_ch_names & montage_ch_names))
    
    if len(common_ch_names) == 0:
        raise ValueError(f"No common channels between data ({ch_names}) and montage ({montage.ch_names})")
    
    print(f"Found {len(common_ch_names)} common channels: {common_ch_names}")
    print(f"Data has {len(ch_names)} channels, montage has {len(montage.ch_names)} channels")
    
    # Create info object with only the channels that exist in the montage
    info = mne.create_info(ch_names=common_ch_names, sfreq=250, ch_types='eeg')
    
    # Set montage to info object - this should work now since we only use common channels
    info.set_montage(montage, on_missing='ignore')
    
    # Validate montage was set correctly
    if info.get_montage() is None:
        raise ValueError(f"Failed to set montage from {montage_path}")
    
    # Find channel adjacency based on actual montage positions
    adjacency, ch_names_ordered = mne.channels.find_ch_adjacency(info, ch_type='eeg')
    
    # Validate adjacency matrix
    _validate_adjacency_matrix(adjacency, len(ch_names_ordered))
    
    print(f"✓ Adjacency matrix computed: {adjacency.shape[0]}x{adjacency.shape[1]} channels")
    print(f"  Adjacent connections: {adjacency.nnz}")
    print(f"  Connections per channel: {adjacency.nnz / len(ch_names_ordered):.1f} (avg)")
    
    # Create mapping from common channels back to original channel indices
    channel_indices = [ch_names.index(ch) for ch in common_ch_names]
    
    # IMPORTANT: Return ch_names_ordered (from find_ch_adjacency) to ensure alignment
    # between adjacency matrix and data. The adjacency matrix rows/cols correspond
    # to ch_names_ordered, NOT common_ch_names (which may have different order)
    return adjacency, ch_names_ordered, channel_indices


def _run_single_permutation(
    perm_idx: int,
    power_data: np.ndarray,
    df_behavioral: pd.DataFrame,
    formula: str,
    predictor_of_interest: str,
    adjacency: np.ndarray,
    threshold: float,
    tail: int,
    stat_fun: str,
    seed: int,
    method: str,
    maxiter: int,
    permutation_method: str,
    t_power: float = 1.0,
    residuals_reduced: np.ndarray = None,
    fitted_reduced: np.ndarray = None,
    include: np.ndarray = None,
    partitions: np.ndarray = None
) -> float:
    """
    Run a single permutation iteration.
    
    This helper function is designed to be called in parallel by joblib.
    
    Parameters
    ----------
    perm_idx : int
        Permutation index (used for seeding)
    power_data : np.ndarray
        Original power data
    df_behavioral : pd.DataFrame
        Behavioral data
    formula : str
        LMM formula
    predictor_of_interest : str
        Predictor to test
    adjacency : np.ndarray
        Channel adjacency matrix
    threshold : float
        T-statistic threshold for cluster formation (absolute value).
        Typical values: 2.0-3.0. Higher = more conservative clusters.
    n_permutations: int, default=1000
        Number of permutations for null distribution.
        Minimum 1000 recommended, 5000+ for publication.
    exclude : np.ndarray, optional
        Boolean mask of channels to exclude from clustering (e.g., edge channels).
        Shape: (n_channels,). Excluded channels:
        - Are still tested (t-statistics computed)
        - Cannot form clusters
        - Cannot connect clusters
        Following MNE's pattern to prevent boundary artifacts.
    tail : int, default=0
        Tail for testing:
        - 0: two-sided (default, tests both positive and negative effects)
        - -1: one-sided less (tests negative effects only)
        - 1: one-sided greater (tests positive effects only)
    seed : int
        Base random seed
    method : str
        LMM optimization method
    t_power : float, default=1.0
        Power to raise the statistical values by before summing.
        t_power=1: standard sum/max (default, recommended)
        t_power>1: gives more weight to stronger effects (e.g., t_power=2)
        Note: sign is always preserved. Following MNE-Python convention.
    verbose: bool, default=True
        Whether to print progress information and summaries.
    return_diagnostics : bool, default=False
        If True, return detailed LMM convergence diagnostics.
    n_bootstrap : int, default=0
        Number of bootstrap iterations for confidence intervals.
        If > 0, computes 95% CI for cluster statistics.
        Recommended: 1000+ for stable estimates.
    permutation_method : {'simple', 'freedman_lane'}, default='simple'
        Permutation strategy:
        - 'simple': Permute predictor values within subjects (default)
        - 'freedman_lane': Freedman-Lane procedure (permute residuals from
          reduced model). More powerful when controlling for covariates.
        
    Returns
    -------
    float
        Maximum cluster statistic for this permutation
    """
    from lmm_model import run_lmm_per_channel, freedman_lane_permutation
    
    # Generate permuted data based on method
    if permutation_method == 'freedman_lane':
        # Freedman-Lane: permute residuals and reconstruct Y*
        power_perm = freedman_lane_permutation(
            residuals=residuals_reduced,
            fitted_values=fitted_reduced,
            df_behavioral=df_behavioral,
            seed=seed + perm_idx
        )
        df_perm = df_behavioral.copy()
    else:
        # Simple: permute predictor within subjects
        df_perm = _permute_within_subjects(df_behavioral, predictor_of_interest, seed + perm_idx)
        power_perm = power_data
    
    # Compute t-statistics for permuted data
    t_stats_perm, _, _ = run_lmm_per_channel(
        power_data=power_perm,
        df_behavioral=df_perm,
        formula=formula,
        predictor_of_interest=predictor_of_interest,
        method=method,
        maxiter=maxiter,
        random_state=seed + perm_idx,
        return_diagnostics=False
    )
    
    # Find clusters in permuted data (MNE-compatible with include/partitions)
    _, cluster_stats_perm = _find_clusters(
        t_stats_perm, adjacency, threshold, tail, stat_fun, t_power,
        include=include, partitions=partitions
    )
    
    # Return maximum cluster statistic (MNE pattern: preserve sign for proper tail handling)
    if len(cluster_stats_perm) > 0:
        # Following MNE's _do_1samp_permutations pattern
        idx_max = np.argmax(np.abs(cluster_stats_perm))
        return float(cluster_stats_perm[idx_max])  # Preserve sign
    else:
        return 0.0


def spatial_cluster_permutation_test(
    observed_t_stats: np.ndarray,
    power_data: np.ndarray,
    df_behavioral: pd.DataFrame,
    formula: str,
    predictor_of_interest: str,
    adjacency: np.ndarray,
    threshold: float,
    n_permutations: int = 1000,
    tail: int = 0,
    seed: int = 42,
    n_jobs: int = -1,
    method: str = 'lbfgs',
    maxiter: int = 1000,
    stat_fun: Literal['sum', 'max'] = 'sum',
    t_power: float = 1.0,
    verbose: bool = True,
    return_diagnostics: bool = False,
    n_bootstrap: int = 0,
    permutation_method: Literal['simple', 'freedman_lane'] = 'simple',
    exclude: np.ndarray = None
) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray, dict]:
    """
    Perform spatial cluster-based permutation test for Linear Mixed Model results.
    
    This function implements cluster-based permutation testing following Maris &
    Oostenveld (2007), adapted for Linear Mixed Models (LMM). Unlike MNE's standard
    cluster test which uses simple t-tests, this implementation:
    
    1. Uses LMM t-statistics to account for within-subject repeated measures
    2. Permutes predictor values within subjects to preserve correlation structure
    3. Computes spatial clusters based on channel adjacency
    4. Generates null distribution through permutation testing
    
    **Why within-subject permutation?**
    
    Within-subject permutation preserves the repeated measures structure of the data.
    For each subject, we shuffle the predictor values (e.g., mind-wandering levels)
    while keeping the subject identity intact. This breaks the association between
    predictor and outcome while maintaining:
    - Number of observations per subject
    - Subject-specific baseline levels (random effects)
    - Within-subject correlation structure
    
    **Cluster statistic options:**
    
    - 'sum' (default): Sum of t-values in cluster. More sensitive to cluster extent.
      Larger clusters with moderate effects will be more significant.
    - 'max': Maximum t-value in cluster. More sensitive to peak effects.
      Smaller clusters with strong peaks will be more significant.
    
    **Interpreting p-values:**
    
    The cluster p-value represents the probability of observing a cluster with
    this statistic (or more extreme) under the null hypothesis. It does NOT
    indicate which specific channels within the cluster are significant.
    
    Parameters
    ----------
    observed_t_stats : np.ndarray
        Observed t-statistics from LMM, shape (n_channels,).
        Should be obtained from `run_lmm_per_channel()`.
    power_data : np.ndarray
        Power values, shape (n_observations, n_channels).
        Must match the behavioral data length.
    df_behavioral : pd.DataFrame
        Behavioral data with predictor variables and 'subject' column.
        Used for permutation testing.
    formula : str
        R-style formula for LMM (e.g., "power ~ onoff + (1|subject)").
        Must include random effects specification.
    predictor_of_interest : str
        Name of predictor to test (must be in formula and df_behavioral).
    adjacency : np.ndarray
        Spatial adjacency matrix, shape (n_channels, n_channels).
        Can be sparse or dense. Obtain from `get_channel_adjacency()`.
    threshold : float
        T-statistic threshold for cluster formation (absolute value).
        Typical values: 2.0-3.0. Higher = more conservative clusters.
    n_permutations : int, default=1000
        Number of permutations for null distribution.
        Minimum 1000 recommended, 5000+ for publication.
    tail : int, default=0
        Tail for testing:
        - 0: two-sided (default, tests both positive and negative effects)
        - -1: one-sided less (tests negative effects only)
        - 1: one-sided greater (tests positive effects only)
    seed : int, default=42
        Random seed for reproducibility.
    n_jobs : int, default=-1
        Number of parallel jobs for permutation testing.
        - -1: Use all available CPU cores
        - 1: Sequential execution (no parallelization)
        - n > 1: Use n CPU cores
        Note: Parallel processing uses joblib with 'loky' backend.
    method : str, default='lbfgs'
        LMM optimization method. Options: 'lbfgs', 'powell', 'nm', 'bfgs'.
    maxiter : int, default=1000
        Maximum iterations for LMM optimization.
    stat_fun : {'sum', 'max'}, default='sum'
        Cluster statistic function:
        - 'sum': Sum of t-values (sensitive to extent)
        - 'max': Maximum t-value (sensitive to peak)
    t_power : float, default=1.0
        Power to raise the statistical values by before summing.
        t_power=1: standard sum/max (default, recommended)
        t_power>1: gives more weight to stronger effects (e.g., t_power=2)
        Note: sign is always preserved. Following MNE-Python convention.
    verbose : bool, default=True
        Whether to print progress information and summaries.
    return_diagnostics : bool, default=False
        If True, return detailed LMM convergence diagnostics.
    n_bootstrap : int, default=0
        Number of bootstrap iterations for confidence intervals.
        If > 0, computes 95% CI for cluster statistics.
        Recommended: 1000+ for stable estimates.
    permutation_method : {'simple', 'freedman_lane'}, default='simple'
        Permutation strategy:
        - 'simple': Permute predictor values within subjects (default)
        - 'freedman_lane': Freedman-Lane procedure (permute residuals from
          reduced model). More powerful when controlling for covariates.
        
    Returns
    -------
    clusters : List[np.ndarray]
        List of arrays, each containing channel indices for a cluster.
        Empty list if no clusters found.
    cluster_stats : np.ndarray
        Cluster-level statistics (sum or max of t-values per cluster).
        Shape: (n_clusters,)
    cluster_p_values : np.ndarray
        P-values for each cluster based on permutation distribution.
        Shape: (n_clusters,)
    diagnostics : dict
        Diagnostics dictionary containing:
        - 'lmm_convergence': LMM convergence statistics
        - 'bootstrap_ci': Bootstrap confidence intervals (if n_bootstrap > 0)
        - 'null_distribution': Null distribution statistics
        
    Raises
    ------
    ValueError
        If data validation fails or parameters are invalid.
        
    Warnings
    --------
    - If n_permutations < 1000: May not provide stable p-values
    - If n_permutations < 5000: May not be suitable for publication
    - If >10% of channels have NaN: Data quality issues
    - If no clusters found: Consider lowering threshold
    
    Notes
    -----
    - NaN values in t-statistics are handled by masking them out
    - Channels with NaN are excluded from cluster formation
    - Permutation preserves within-subject structure (critical for LMM)
    - Uses `scipy.sparse.csgraph.connected_components` for cluster finding
    - Progress is reported every 100 permutations when verbose=True
    
    References
    ----------
    Maris, E., & Oostenveld, R. (2007). Nonparametric statistical testing of
    EEG- and MEG-data. Journal of Neuroscience Methods, 164(1), 177-190.
    https://doi.org/10.1016/j.jneumeth.2007.03.024
    
    Examples
    --------
    >>> # Run LMM per channel first
    >>> t_stats, p_vals = run_lmm_per_channel(
    ...     power_data, df_behavioral, formula, predictor
    ... )
    >>> 
    >>> # Get adjacency matrix
    >>> adjacency, ch_names, _ = get_channel_adjacency(montage_path, ch_names)
    >>> 
    >>> # Run cluster permutation test
    >>> clusters, stats, pvals = spatial_cluster_permutation_test(
    ...     observed_t_stats=t_stats,
    ...     power_data=power_data,
    ...     df_behavioral=df_behavioral,
    ...     formula=formula,
    ...     predictor_of_interest=predictor,
    ...     adjacency=adjacency,
    ...     threshold=2.5,
    ...     n_permutations=5000,
    ...     stat_fun='sum'
    ... )
    >>> 
    >>> # Check significant clusters
    >>> sig_clusters = [c for c, p in zip(clusters, pvals) if p < 0.05]
    >>> print(f"Found {len(sig_clusters)} significant clusters")
    """
    from lmm_model import run_lmm_per_channel
    
    # ========== VALIDATION SECTION ==========
    # Validate parameters
    _validate_cluster_parameters(
        observed_t_stats, power_data, df_behavioral, predictor_of_interest,
        adjacency, threshold, n_permutations, tail, stat_fun, verbose
    )
    
    # Set random seed for reproducibility
    np.random.seed(seed)
    n_channels = len(observed_t_stats)
    
    # Handle exclude mask (MNE pattern)
    # Excluded channels are still tested but cannot form/connect clusters
    include = None
    if exclude is not None:
        if exclude.size != n_channels:
            raise ValueError(f"exclude must have same length as observed_t_stats ({n_channels})")
        include = np.logical_not(exclude)
        if verbose:
            n_excluded = np.sum(exclude)
            print(f"  Excluding {n_excluded}/{n_channels} channels from clustering")
            print(f"  (Excluded channels are still tested but cannot form clusters)")
    
    # Always check for disjoint adjacency sets (MNE pattern)
    partitions = None
    if adjacency is not None:
        if verbose:
            print("  Checking for disjoint adjacency sets...")
        from mne.stats.cluster_level import _get_partitions_from_adjacency
        from scipy.sparse import coo_matrix
        partitions = _get_partitions_from_adjacency(coo_matrix(adjacency), n_times=1)
        if partitions is not None and verbose:
            n_partitions = len(np.unique(partitions))
            print(f"  Found {n_partitions} disjoint adjacency sets")
    
    # Print analysis summary if verbose
    if verbose:
        _print_analysis_summary(
            n_channels, power_data.shape[0], df_behavioral,
            predictor_of_interest, threshold, n_permutations, tail, stat_fun
        )
    
    # ========== DIAGNOSE T-STATISTICS ==========
    # CRITICAL: Check t-statistics BEFORE finding clusters to detect issues early
    if verbose:
        # Get channel names if available (for detailed reporting)
        ch_names = None
        if hasattr(adjacency, 'shape') and adjacency.shape[0] == n_channels:
            # Try to infer channel names from common sources
            # This is a placeholder - ideally pass ch_names as parameter
            ch_names = [f"Ch{i}" for i in range(n_channels)]
        
        t_diagnostics = _diagnose_t_statistics(
            observed_t_stats, 
            ch_names=ch_names,
            threshold=threshold,
            label="Observed"
        )
        _print_t_diagnostics(t_diagnostics, label="Observed T-statistics")
        
        # Critical warnings
        if t_diagnostics['pct_nan'] > 50:
            print("\n" + "!"*70)
            print("CRITICAL WARNING: >50% of channels have NaN t-statistics!")
            print("This likely indicates a serious problem with:")
            print("  1. Channel alignment between observed_t_stats and adjacency")
            print("  2. LMM convergence issues")
            print("  3. Data quality problems")
            print("Please investigate before proceeding.")
            print("!"*70)
    
    # ========== FIND OBSERVED CLUSTERS ==========
    observed_clusters, observed_cluster_stats = _find_clusters(
        observed_t_stats, adjacency, threshold, tail, stat_fun, t_power,
        include=include, partitions=partitions
    )
    
    if verbose:
        if len(observed_clusters) == 0:
            print(f"\n⚠ No clusters found in observed data with threshold={threshold}")
            print("  Consider lowering the threshold or checking your data.")
        else:
            print(f"\n✓ Found {len(observed_clusters)} clusters in observed data")
            for i, (cluster, stat) in enumerate(zip(observed_clusters, observed_cluster_stats)):
                print(f"  Cluster {i+1}: {len(cluster)} channels, statistic={stat:.3f}")
    
    # ========== PERMUTATION TESTING ==========
    if verbose:
        print(f"\nRunning {n_permutations} permutations...")
        print(f"  Permutation method: {permutation_method}")
        if n_jobs == 1:
            print(f"  Execution mode: Sequential (n_jobs=1)")
        elif n_jobs == -1:
            print(f"  Execution mode: Parallel (using all available cores)")
        else:
            print(f"  Execution mode: Parallel (n_jobs={n_jobs})")
    
    # For Freedman-Lane: fit reduced model once (outside loop)
    residuals_reduced = None
    fitted_reduced = None
    if permutation_method == 'freedman_lane':
        if verbose:
            print("  Fitting reduced model (without predictor of interest)...")
        from lmm_model import fit_reduced_model_per_channel
        residuals_reduced, fitted_reduced = fit_reduced_model_per_channel(
            power_data=power_data,
            df_behavioral=df_behavioral,
            formula=formula,
            predictor_of_interest=predictor_of_interest,
            method=method,
            maxiter=maxiter,
            random_state=seed
        )
        if verbose:
            print("  Reduced model fitted successfully")
    
    # Run permutations following MNE's parallel pattern
    # Create list of permutation indices
    perm_indices = list(range(n_permutations))
    
    if n_jobs == 1:
        # Sequential execution
        max_cluster_stats_null = []
        for perm_idx in perm_indices:
            if verbose and perm_idx % 100 == 0 and perm_idx > 0:
                print(f"  Progress: {perm_idx}/{n_permutations} permutations ({100*perm_idx/n_permutations:.1f}%)")
            
            max_stat = _run_single_permutation(
                perm_idx=perm_idx,
                power_data=power_data,
                df_behavioral=df_behavioral,
                formula=formula,
                predictor_of_interest=predictor_of_interest,
                adjacency=adjacency,
                threshold=threshold,
                tail=tail,
                stat_fun=stat_fun,
                seed=seed,
                method=method,
                maxiter=maxiter,
                permutation_method=permutation_method,
                t_power=t_power,
                residuals_reduced=residuals_reduced,
                fitted_reduced=fitted_reduced,
                include=include,
                partitions=partitions
            )
            max_cluster_stats_null.append(max_stat)
        
        max_cluster_stats_null = np.array(max_cluster_stats_null)
    else:
        # Parallel execution using joblib directly (no MNE parallel_func dependency)
        if verbose:
            print(f"  Starting parallel execution...")
        
        # Use joblib Parallel directly
        max_cluster_stats_null = Parallel(n_jobs=n_jobs, verbose=0)(
            delayed(_run_single_permutation)(
                perm_idx=perm_idx,
                power_data=power_data,
                df_behavioral=df_behavioral,
                formula=formula,
                predictor_of_interest=predictor_of_interest,
                adjacency=adjacency,
                threshold=threshold,
                tail=tail,
                stat_fun=stat_fun,
                seed=seed,
                method=method,
                maxiter=maxiter,
                permutation_method=permutation_method,
                t_power=t_power,
                residuals_reduced=residuals_reduced,
                fitted_reduced=fitted_reduced,
                include=include,
                partitions=partitions
            )
            for perm_idx in perm_indices
        )
        
        max_cluster_stats_null = np.array(max_cluster_stats_null)
        
        if verbose:
            print(f"  ✓ Parallel execution completed")
    
    # ========== COMPUTE P-VALUES ==========
    # Following MNE's pattern: include observed statistic in H0
    # This ensures p-values are never exactly 0 and provides better properties
    if tail == -1:
        # For lower tail, get minimum observed cluster stat
        orig_stat = observed_cluster_stats.min() if len(observed_cluster_stats) > 0 else 0
    elif tail == 1:
        # For upper tail, get maximum observed cluster stat
        orig_stat = observed_cluster_stats.max() if len(observed_cluster_stats) > 0 else 0
    else:
        # For two-tailed, get maximum absolute observed cluster stat
        orig_stat = np.abs(observed_cluster_stats).max() if len(observed_cluster_stats) > 0 else 0
    
    # Add observed statistic to null distribution (MNE pattern)
    H0 = np.concatenate([max_cluster_stats_null, [orig_stat]])
    H0 = np.sort(H0)
    
    # Use MNE's _pval_from_histogram (exactly as MNE does).
    # For tail=0 (two-sided): computes mean(|H0| >= |T_obs|) — correct.
    # For tail=1 (one-sided positive): computes mean(H0 >= T_obs) — correct.
    # For tail=-1 (one-sided negative): computes mean(H0 <= T_obs) — correct.
    cluster_p_values = _pval_from_histogram(
        observed_cluster_stats, H0, tail=tail
    )
    
    # ========== PRINT RESULTS SUMMARY ==========
    if verbose:
        _print_results_summary(
            observed_clusters, observed_cluster_stats, cluster_p_values,
            H0, n_permutations
        )
    
    # ========== BOOTSTRAP CONFIDENCE INTERVALS (OPTIONAL) ==========
    bootstrap_ci = None
    if n_bootstrap > 0 and len(observed_clusters) > 0:
        if verbose:
            print(f"\nComputing bootstrap confidence intervals ({n_bootstrap} iterations)...")
        bootstrap_ci = _compute_bootstrap_ci(
            observed_clusters, power_data, df_behavioral, formula,
            predictor_of_interest, adjacency, threshold, tail, stat_fun,
            t_power, n_bootstrap, seed, method, maxiter, verbose
        )
    
    # ========== COMPILE DIAGNOSTICS ==========
    diagnostics = {
        'null_distribution': {
            'mean': float(np.mean(H0)),
            'median': float(np.median(H0)),
            'std': float(np.std(H0)),
            'percentile_95': float(np.percentile(H0, 95)),
            'percentile_99': float(np.percentile(H0, 99)),
            'includes_observed': True  # Following MNE pattern
        },
        'n_permutations': n_permutations,
        'threshold': threshold,
        'tail': tail,
        'stat_fun': stat_fun,
        't_power': t_power
    }
    
    if bootstrap_ci is not None:
        diagnostics['bootstrap_ci'] = bootstrap_ci
    
    return observed_clusters, observed_cluster_stats, cluster_p_values, diagnostics


def _validate_cluster_parameters(
    observed_t_stats: np.ndarray,
    power_data: np.ndarray,
    df_behavioral: pd.DataFrame,
    predictor_of_interest: str,
    adjacency: np.ndarray,
    threshold: float,
    n_permutations: int,
    tail: int,
    stat_fun: str,
    verbose: bool
) -> None:
    """
    Validate all parameters for cluster permutation testing.
    
    Raises ValueError if any parameter is invalid.
    """
    # Validate t-statistics
    if not isinstance(observed_t_stats, np.ndarray):
        raise ValueError("observed_t_stats must be a numpy array")
    
    if observed_t_stats.ndim != 1:
        raise ValueError(f"observed_t_stats must be 1D, got shape {observed_t_stats.shape}")
    
    n_channels = len(observed_t_stats)
    
    # Check for all NaN
    if np.all(np.isnan(observed_t_stats)):
        raise ValueError("All t-statistics are NaN")
    
    # Warn if many NaN
    nan_ratio = np.mean(np.isnan(observed_t_stats))
    if nan_ratio > 0.1:
        warnings.warn(
            f"{100*nan_ratio:.1f}% of channels have NaN t-statistics. "
            "This may indicate data quality issues."
        )
    
    # Validate power data
    if power_data.shape[1] != n_channels:
        raise ValueError(
            f"power_data has {power_data.shape[1]} channels but "
            f"observed_t_stats has {n_channels} channels"
        )
    
    if power_data.shape[0] != len(df_behavioral):
        raise ValueError(
            f"power_data has {power_data.shape[0]} observations but "
            f"df_behavioral has {len(df_behavioral)} rows"
        )
    
    # Validate behavioral data
    # For interaction terms (e.g. "onoff:valence"), check constituent variables
    from lmm_model import is_interaction_term, get_interaction_components
    if is_interaction_term(predictor_of_interest):
        for comp in get_interaction_components(predictor_of_interest):
            if comp not in df_behavioral.columns:
                raise ValueError(
                    f"Interaction component '{comp}' (from '{predictor_of_interest}') "
                    f"not found in df_behavioral. Available columns: {list(df_behavioral.columns)}"
                )
    else:
        if predictor_of_interest not in df_behavioral.columns:
            raise ValueError(
                f"Predictor '{predictor_of_interest}' not found in df_behavioral. "
                f"Available columns: {list(df_behavioral.columns)}"
            )
    
    if 'subject' not in df_behavioral.columns:
        raise ValueError("df_behavioral must have a 'subject' column")
    
    n_subjects = df_behavioral['subject'].nunique()
    if n_subjects < 3:
        raise ValueError(
            f"Need at least 3 subjects for cluster testing, found {n_subjects}"
        )
    
    # Validate adjacency matrix
    _validate_adjacency_matrix(adjacency, n_channels)
    
    # Validate threshold (skip for TFCE where threshold is not used)
    if not isinstance(threshold, (int, float)):
        raise ValueError(f"threshold must be numeric, got {type(threshold)}")
    
    # Only validate threshold > 0 if it's actually being used (not TFCE)
    # TFCE passes threshold=0.0 as a placeholder
    if threshold < 0:
        raise ValueError(f"threshold must be non-negative, got {threshold}")
    
    # Validate n_permutations
    if not isinstance(n_permutations, int):
        raise ValueError(f"n_permutations must be an integer, got {type(n_permutations)}")
    
    if n_permutations < 100:
        warnings.warn(
            f"n_permutations={n_permutations} is very low (< 100). "
            f"Results may be unreliable. Minimum 1000 recommended for stable p-values.",
            UserWarning
        )
    
    if n_permutations < 1000:
        warnings.warn(
            f"n_permutations={n_permutations} is low. Minimum 1000 recommended for stable p-values."
        )
    
    if n_permutations < 5000 and verbose:
        print(f"⚠ Note: n_permutations={n_permutations} may not be sufficient for publication.")
        print("  Consider using 5000+ permutations for final analyses.")
    
    # Validate tail
    if tail not in [-1, 0, 1]:
        raise ValueError(f"tail must be -1, 0, or 1, got {tail}")
    
    # Validate stat_fun
    if stat_fun not in ['sum', 'max']:
        raise ValueError(f"stat_fun must be 'sum' or 'max', got '{stat_fun}'")


def _diagnose_t_statistics(
    t_stats: np.ndarray,
    ch_names: list = None,
    threshold: float = None,
    label: str = "T-statistics"
) -> dict:
    """
    Diagnose t-statistics for NaN issues and provide detailed summary.
    
    This function helps detect common problems:
    - NaN values that shouldn't be there
    - Misalignment between channels and data
    - Suspicious patterns (all NaN, all zeros, etc.)
    
    Parameters
    ----------
    t_stats : np.ndarray
        T-statistics array
    ch_names : list, optional
        Channel names for detailed reporting
    threshold : float, optional
        Threshold to report channels exceeding it
    label : str
        Label for printing (e.g., "Observed", "Permutation 1")
        
    Returns
    -------
    dict
        Diagnostic information
    """
    n_total = len(t_stats)
    valid_mask = ~np.isnan(t_stats)
    n_valid = np.sum(valid_mask)
    n_nan = n_total - n_valid
    
    diagnostics = {
        'n_total': n_total,
        'n_valid': n_valid,
        'n_nan': n_nan,
        'pct_nan': 100 * n_nan / n_total if n_total > 0 else 0
    }
    
    if n_valid > 0:
        valid_t = t_stats[valid_mask]
        diagnostics['min'] = float(np.min(valid_t))
        diagnostics['max'] = float(np.max(valid_t))
        diagnostics['mean'] = float(np.mean(valid_t))
        diagnostics['std'] = float(np.std(valid_t))
        
        if threshold is not None:
            above_thresh = np.sum(np.abs(valid_t) > threshold)
            diagnostics['n_above_threshold'] = above_thresh
            diagnostics['pct_above_threshold'] = 100 * above_thresh / n_valid
            
            # Find top channels
            if ch_names is not None and len(ch_names) == n_total:
                abs_t = np.abs(t_stats)
                top_indices = np.argsort(abs_t)[::-1][:5]  # Top 5
                top_channels = []
                for idx in top_indices:
                    if not np.isnan(t_stats[idx]):
                        top_channels.append({
                            'channel': ch_names[idx],
                            'index': int(idx),
                            't_value': float(t_stats[idx])
                        })
                diagnostics['top_channels'] = top_channels
    else:
        diagnostics['min'] = np.nan
        diagnostics['max'] = np.nan
        diagnostics['mean'] = np.nan
        diagnostics['std'] = np.nan
    
    return diagnostics


def _print_t_diagnostics(diagnostics: dict, label: str = "T-statistics") -> None:
    """Print formatted t-statistics diagnostics."""
    print(f"\n{label} Diagnostics:")
    print(f"  Total channels: {diagnostics['n_total']}")
    print(f"  Valid: {diagnostics['n_valid']} ({100 - diagnostics['pct_nan']:.1f}%)")
    print(f"  NaN: {diagnostics['n_nan']} ({diagnostics['pct_nan']:.1f}%)")
    
    if diagnostics['n_valid'] > 0:
        print(f"  Range: [{diagnostics['min']:.3f}, {diagnostics['max']:.3f}]")
        print(f"  Mean ± SD: {diagnostics['mean']:.3f} ± {diagnostics['std']:.3f}")
        
        if 'n_above_threshold' in diagnostics:
            print(f"  Above threshold: {diagnostics['n_above_threshold']} "
                  f"({diagnostics['pct_above_threshold']:.1f}%)")
        
        if 'top_channels' in diagnostics and len(diagnostics['top_channels']) > 0:
            print(f"\n  Top channels by |t|:")
            for ch_info in diagnostics['top_channels']:
                print(f"    {ch_info['channel']:<6} (idx={ch_info['index']:2d}): "
                      f"t={ch_info['t_value']:>7.3f}")
    
    # Warnings
    if diagnostics['pct_nan'] > 10:
        print(f"\n  ⚠️  WARNING: {diagnostics['pct_nan']:.1f}% NaN values detected!")
        print(f"     This may indicate data quality issues or misalignment.")
    
    if diagnostics['n_valid'] > 0 and diagnostics['std'] < 0.01:
        print(f"\n  ⚠️  WARNING: Very low variance (SD={diagnostics['std']:.4f})")
        print(f"     Check if t-statistics are being computed correctly.")


def _print_analysis_summary(
    n_channels: int,
    n_observations: int,
    df_behavioral: pd.DataFrame,
    predictor_of_interest: str,
    threshold: float,
    n_permutations: int,
    tail: int,
    stat_fun: str
) -> None:
    """Print summary of analysis parameters."""
    print("\n" + "="*70)
    print("SPATIAL CLUSTER PERMUTATION TEST (LMM)")
    print("="*70)
    print(f"Data:")
    print(f"  Channels: {n_channels}")
    print(f"  Observations: {n_observations}")
    print(f"  Subjects: {df_behavioral['subject'].nunique()}")
    print(f"  Predictor: {predictor_of_interest}")
    
    # Map tail to label: -1='less', 0='two-sided', 1='greater'
    tail_labels = {-1: 'less', 0: 'two-sided', 1: 'greater'}
    
    print(f"\nCluster parameters:")
    print(f"  Threshold: {threshold}")
    print(f"  Tail: {tail_labels[tail]}")
    print(f"  Cluster statistic: {stat_fun}")
    print(f"  Permutations: {n_permutations}")
    
    # Recommendations based on parameters
    if tail != 0 and threshold < 2.5:
        print(f"\n  ⚠️  RECOMMENDATION: One-sided test with low threshold")
        print(f"     Consider tail=0 (two-sided) and threshold=2.5-3.0 for exploration")
    
    print("="*70)


def _print_results_summary(
    clusters: List[np.ndarray],
    cluster_stats: np.ndarray,
    cluster_p_values: np.ndarray,
    null_distribution: np.ndarray,
    n_permutations: int
) -> None:
    """Print summary of cluster testing results."""
    print("\n" + "="*70)
    print("CLUSTER TESTING RESULTS")
    print("="*70)
    
    print(f"\nNull distribution statistics:")
    print(f"  Mean: {np.mean(null_distribution):.3f}")
    print(f"  Median: {np.median(null_distribution):.3f}")
    print(f"  Std: {np.std(null_distribution):.3f}")
    print(f"  95th percentile: {np.percentile(null_distribution, 95):.3f}")
    print(f"  99th percentile: {np.percentile(null_distribution, 99):.3f}")
    
    if len(clusters) == 0:
        print("\n⚠ No clusters found in observed data.")
    else:
        print(f"\nFound {len(clusters)} cluster(s):")
        print(f"\n{'ID':<4} {'Size':<6} {'Statistic':<12} {'p-value':<10} {'Sig':<5}")
        print("-" * 45)
        
        for i, (cluster, stat, pval) in enumerate(zip(clusters, cluster_stats, cluster_p_values)):
            sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
            print(f"{i+1:<4} {len(cluster):<6} {stat:>11.3f} {pval:>9.4f}  {sig:<5}")
        
        n_sig = np.sum(cluster_p_values < 0.05)
        print(f"\nSignificant clusters (p < 0.05): {n_sig}/{len(clusters)}")
    
    print("="*70 + "\n")


def _find_clusters_single_tail(
    t_stats: np.ndarray,
    adjacency: np.ndarray,
    threshold_mask: np.ndarray,
    stat_fun: str = 'sum',
    t_power: float = 1.0
) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Find clusters for a single tail using MNE's _get_components.
    
    This follows MNE's exact pattern from _find_clusters_1dir but adapted
    for our LMM t-statistics with t_power support.
    
    Parameters
    ----------
    t_stats : np.ndarray
        T-statistics, shape (n_channels,)
    adjacency : np.ndarray or sparse matrix
        Spatial adjacency matrix (MNE format: coo_array preferred)
    threshold_mask : np.ndarray
        Boolean mask indicating which channels exceed threshold
    stat_fun : str, default='sum'
        Cluster statistic function ('sum' or 'max')
    t_power : float, default=1.0
        Power to raise the statistical values by before summing.
        Following MNE convention: sign(t) * |t|^t_power
        
    Returns
    -------
    clusters : List[np.ndarray]
        List of channel index arrays for each cluster
    cluster_stats : np.ndarray
        Cluster statistics (preserving sign)
        
    Notes
    -----
    This function follows MNE's _find_clusters_1dir pattern exactly,
    using _get_components for cluster finding and _masked_sum_power
    for statistics computation.
    """
    if not np.any(threshold_mask):
        return [], np.array([])
    
    # Ensure adjacency is in MNE's expected format (coo_array)
    if issparse(adjacency):
        if not isinstance(adjacency, coo_array):
            adjacency = coo_array(adjacency)
    else:
        adjacency = coo_array(adjacency)
    
    # Use MNE's _get_components (exactly as MNE does)
    clusters = _get_components(threshold_mask, adjacency, return_list=True)
    
    # Compute cluster statistics (MNE-compatible)
    if stat_fun == 'sum':
        # Use MNE's _masked_sum_power function
        sums = [_masked_sum_power(t_stats, c, t_power) for c in clusters]
    else:  # 'max'
        # Maximum absolute value with sign preserved (MNE pattern)
        sums = []
        for c in clusters:
            cluster_t_values = t_stats[c]
            max_idx = np.argmax(np.abs(cluster_t_values))
            stat = np.sign(cluster_t_values[max_idx]) * np.abs(cluster_t_values[max_idx]) ** t_power
            sums.append(stat)
    
    return clusters, np.atleast_1d(sums)


def _find_clusters(
    t_stats: np.ndarray,
    adjacency: np.ndarray,
    threshold: float,
    tail: int = 0,
    stat_fun: str = 'sum',
    t_power: float = 1.0,
    include: np.ndarray = None,
    partitions: np.ndarray = None
) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Find spatial clusters based on threshold (MNE-compatible).
    
    This function follows MNE's exact cluster finding pattern using _get_components
    and _masked_sum_power, with extensions for LMM-specific requirements (t_power).
    
    **CRITICAL**: For two-sided tests (tail=0), positive and negative clusters are
    found separately to avoid sign cancellation. This follows MNE's convention and
    ensures that a cluster with mixed positive/negative t-values doesn't have its
    statistic artificially reduced by summing opposite signs.
    
    **MNE Integration**: 
    - Uses MNE's _get_components() for cluster finding (via _find_clusters_single_tail)
    - Uses MNE's _masked_sum_power() for cluster statistics
    - Follows MNE's exact pattern from _find_clusters_1dir
    - Always checks for disjoint adjacency sets (partitions)
    
    Parameters
    ----------
    t_stats : np.ndarray
        T-statistics, shape (n_channels,). NaN values are automatically excluded.
    adjacency : np.ndarray or sparse matrix
        Spatial adjacency matrix (MNE format: coo_array preferred).
    threshold : float
        Threshold for cluster formation (absolute value).
    tail : int, default=0
        - 0: two-sided (tests positive and negative clusters separately)
        - -1: one-sided less (tests t < -threshold)
        - 1: one-sided greater (tests t > threshold)
    stat_fun : str, default='sum'
        Cluster statistic function:
        - 'sum': Sum of t-values in cluster (sensitive to extent)
        - 'max': Maximum absolute t-value in cluster (sensitive to peak)
    t_power : float, default=1.0
        Power to raise the statistical values by before summing.
        t_power=1: standard sum/max (default, MNE-compatible)
        t_power>1: gives more weight to stronger effects (e.g., t_power=2)
        Note: sign is always preserved following MNE convention
    include : np.ndarray, optional
        Boolean mask of channels to include in clustering.
        If None, all channels are included. Following MNE's pattern:
        - Excluded channels are still tested (t-stats computed)
        - But cannot form or connect clusters
        - Prevents boundary artifacts from edge channels
    partitions : np.ndarray, optional
        Integer array indicating which partition each channel belongs to.
        Auto-detected from adjacency matrix to prevent clustering across
        disjoint sets (e.g., left/right hemispheres). MNE pattern.
        
    Returns
    -------
    clusters : List[np.ndarray]
        List of channel index arrays for each cluster.
        Empty list if no clusters found.
    cluster_stats : np.ndarray
        Cluster statistics (sum or max of t-values per cluster).
        Shape: (n_clusters,)
        For two-sided tests, signs are preserved.
        
    Notes
    -----
    - NaN values in t_stats are automatically masked out
    - Uses MNE's _get_components for optimized cluster finding
    - Clusters must be spatially contiguous based on adjacency matrix
    - For two-sided tests, positive and negative clusters are processed separately
    - Fully compatible with MNE's cluster finding pipeline
    
    See Also
    --------
    mne.stats.cluster_level._find_clusters : MNE's original implementation
    mne.stats.cluster_level._get_components : Core cluster finding function
    """
    # Create mask for valid (non-NaN) values
    valid_mask = ~np.isnan(t_stats)
    
    # Apply include mask (MNE pattern)
    # This prevents excluded channels from forming or connecting clusters
    if include is not None:
        valid_mask = valid_mask & include
    
    # Handle partitions (MNE pattern for disjoint sets - always enabled)
    if partitions is not None:
        # Process each partition separately
        all_clusters = []
        all_stats = []
        for p in range(np.max(partitions) + 1):
            partition_mask = (partitions == p)
            partition_valid = valid_mask & partition_mask
            
            # Find clusters in this partition
            if tail == 0:
                pos_mask = partition_valid & (t_stats > threshold)
                neg_mask = partition_valid & (t_stats < -threshold)
                pos_c, pos_s = _find_clusters_single_tail(t_stats, adjacency, pos_mask, stat_fun, t_power)
                neg_c, neg_s = _find_clusters_single_tail(t_stats, adjacency, neg_mask, stat_fun, t_power)
                all_clusters.extend(pos_c + neg_c)
                if len(pos_s) > 0 or len(neg_s) > 0:
                    all_stats.append(np.concatenate([pos_s, neg_s]))
            elif tail == -1:
                threshold_mask = partition_valid & (t_stats < -threshold)
                c, s = _find_clusters_single_tail(t_stats, adjacency, threshold_mask, stat_fun, t_power)
                all_clusters.extend(c)
                if len(s) > 0:
                    all_stats.append(s)
            else:  # tail == 1
                threshold_mask = partition_valid & (t_stats > threshold)
                c, s = _find_clusters_single_tail(t_stats, adjacency, threshold_mask, stat_fun, t_power)
                all_clusters.extend(c)
                if len(s) > 0:
                    all_stats.append(s)
        
        all_stats = np.concatenate(all_stats) if all_stats else np.array([])
        return all_clusters, all_stats
    
    # Apply threshold based on tail (standard case without partitions)
    if tail == 0:  # two-sided - CRITICAL FIX: process positive and negative separately
        # Find positive clusters (t > threshold)
        pos_mask = valid_mask & (t_stats > threshold)
        pos_clusters, pos_stats = _find_clusters_single_tail(
            t_stats, adjacency, pos_mask, stat_fun, t_power
        )
        
        # Find negative clusters (t < -threshold)
        neg_mask = valid_mask & (t_stats < -threshold)
        neg_clusters, neg_stats = _find_clusters_single_tail(
            t_stats, adjacency, neg_mask, stat_fun, t_power
        )
        
        # Combine clusters (positive first, then negative)
        all_clusters = pos_clusters + neg_clusters
        all_stats = np.concatenate([pos_stats, neg_stats]) if len(pos_stats) > 0 or len(neg_stats) > 0 else np.array([])
        
        return all_clusters, all_stats
        
    elif tail == -1:  # one-sided less
        threshold_mask = valid_mask & (t_stats < -threshold)
    else:  # tail == 1, one-sided greater
        threshold_mask = valid_mask & (t_stats > threshold)
    
    # For one-sided tests, use the single tail function
    return _find_clusters_single_tail(t_stats, adjacency, threshold_mask, stat_fun, t_power)


def find_clusters(
    t_stats: np.ndarray,
    perm_t_stats: np.ndarray,
    adjacency: np.ndarray,
    threshold: float,
    tail: int = 0,
    n_permutations: int = None,
    stat_fun: str = 'sum',
    t_power: float = 1.0,
    separate_signs: bool = False
) -> List[Dict]:
    """
    Find significant spatial clusters using permutation testing.
    
    This is a public wrapper for cluster detection with permutation-based p-values,
    designed for use with pre-computed permutation distributions (e.g., Andrillon pipeline).
    
    Parameters
    ----------
    t_stats : np.ndarray
        Observed t-statistics, shape (n_channels,)
    perm_t_stats : np.ndarray
        Permutation t-statistics, shape (n_permutations, n_channels)
    adjacency : np.ndarray or sparse matrix
        Spatial adjacency matrix
    threshold : float
        Threshold for cluster formation (absolute value)
    tail : int, default=0
        - 0: two-sided test
        - -1: one-sided less
        - 1: one-sided greater
    n_permutations : int, optional
        Number of permutations (inferred from perm_t_stats if not provided)
    stat_fun : str, default='sum'
        Cluster statistic: 'sum' or 'max'
    t_power : float, default=1.0
        Power to raise t-values before summing
    separate_signs : bool, default=False
        If True, build separate null distributions for positive and negative
        clusters and compute Monte Carlo p-values per sign (Andrillon-style).
        If False, use a single null based on max |cluster_stat| (default).
        
    Returns
    -------
    clusters : List[Dict]
        List of significant clusters, each containing:
        - 'channels': np.ndarray of channel indices
        - 'stat': float, cluster statistic
        - 'p_value': float, permutation p-value
        
    Examples
    --------
    >>> # After computing observed and permutation t-stats
    >>> clusters = find_clusters(
    ...     t_stats=observed_t,
    ...     perm_t_stats=perm_t,
    ...     adjacency=adj_matrix,
    ...     threshold=2.0,
    ...     tail=1
    ... )
    """
    if n_permutations is None:
        n_permutations = perm_t_stats.shape[0]
    
    # Find clusters in observed data
    obs_clusters, obs_cluster_stats = _find_clusters(
        t_stats=t_stats,
        adjacency=adjacency,
        threshold=threshold,
        tail=tail,
        stat_fun=stat_fun,
        t_power=t_power
    )
    
    if len(obs_clusters) == 0:
        return []
    
    # Build null distribution(s) of cluster statistics
    if not separate_signs:
        # Original behaviour: single null based on max |cluster_stat|
        null_max_stats = []
        for perm_idx in range(n_permutations):
            perm_clusters, perm_cluster_stats = _find_clusters(
                t_stats=perm_t_stats[perm_idx, :],
                adjacency=adjacency,
                threshold=threshold,
                tail=tail,
                stat_fun=stat_fun,
                t_power=t_power
            )
            
            if len(perm_clusters) > 0:
                null_max_stats.append(np.max(np.abs(perm_cluster_stats)))
            else:
                null_max_stats.append(0.0)
        
        null_max_stats = np.array(null_max_stats)
        
        # Compute p-values for each observed cluster (two-sided on |stat|)
        significant_clusters = []
        for cluster_channels, cluster_stat in zip(obs_clusters, obs_cluster_stats):
            p_value = np.mean(null_max_stats >= np.abs(cluster_stat))
            significant_clusters.append({
                'channels': cluster_channels,
                'stat': float(cluster_stat),
                'p_value': float(p_value)
            })
        
        return significant_clusters
    
    # Andrillon-style: separate nulls for positive and negative clusters
    null_pos_max = []  # maximum positive cluster_stat per permutation
    null_neg_min = []  # minimum (most negative) cluster_stat per permutation
    
    for perm_idx in range(n_permutations):
        perm_clusters, perm_cluster_stats = _find_clusters(
            t_stats=perm_t_stats[perm_idx, :],
            adjacency=adjacency,
            threshold=threshold,
            tail=tail,
            stat_fun=stat_fun,
            t_power=t_power
        )
        
        if len(perm_cluster_stats) > 0:
            pos_stats = [s for s in perm_cluster_stats if s > 0]
            neg_stats = [s for s in perm_cluster_stats if s < 0]
        else:
            pos_stats, neg_stats = [], []
        
        # For positive clusters, use maximum positive stat (0 if none)
        null_pos_max.append(max(pos_stats) if len(pos_stats) > 0 else 0.0)
        # For negative clusters, use minimum (most negative) stat (0 if none)
        null_neg_min.append(min(neg_stats) if len(neg_stats) > 0 else 0.0)
    
    null_pos_max = np.array(null_pos_max)
    null_neg_min = np.array(null_neg_min)
    
    # Compute sign-specific Monte Carlo p-values
    significant_clusters: List[Dict] = []
    for cluster_channels, cluster_stat in zip(obs_clusters, obs_cluster_stats):
        if cluster_stat > 0:
            # Positive clusters: upper-tail p-value against null_pos_max
            p_value = np.mean(null_pos_max >= cluster_stat)
        elif cluster_stat < 0:
            # Negative clusters: lower-tail p-value against null_neg_min
            p_value = np.mean(null_neg_min <= cluster_stat)
        else:
            # Degenerate cluster_stat == 0 -> non-significant
            p_value = 1.0
        
        significant_clusters.append({
            'channels': cluster_channels,
            'stat': float(cluster_stat),
            'p_value': float(p_value)
        })
    
    return significant_clusters


def find_clusters_from_pvalues(
    t_stats: np.ndarray,
    p_values: np.ndarray,
    perm_t_stats: np.ndarray,
    perm_p_values: np.ndarray,
    adjacency: np.ndarray,
    cluster_alpha: float,
    tail: int = 0,
    n_permutations: int = None,
    stat_fun: str = 'sum',
    t_power: float = 1.0,
    separate_signs: bool = True,
    exclude: np.ndarray = None,
) -> List[Dict]:
    """Find clusters using p-value thresholding (Andrillon-style).

    This helper forms candidate clusters based on p-values (p < cluster_alpha)
    for both observed and permuted data, while using t-values as cluster
    statistics (sum or max). It mirrors the Andrillon 2020 definition:

    - Candidate clusters: neighbouring electrodes with p < cluster_alpha
    - Cluster statistic: sum of t-values in the cluster
    - Null distribution: cluster statistics from permuted data
    - Positive and negative clusters can be treated separately via
      ``separate_signs=True``.
    """
    if n_permutations is None:
        n_permutations = perm_t_stats.shape[0]

    if p_values.shape != t_stats.shape:
        raise ValueError("p_values and t_stats must have the same shape")
    if perm_p_values.shape != perm_t_stats.shape:
        raise ValueError("perm_p_values and perm_t_stats must have the same shape")

    # Handle optional exclusion mask
    if exclude is not None:
        if exclude.shape[0] != t_stats.shape[0]:
            raise ValueError("exclude mask must have same length as t_stats")
        exclude = exclude.astype(bool)

    # ---------- Observed clusters ----------
    sig_mask = p_values < cluster_alpha
    if exclude is not None:
        sig_mask = sig_mask & (~exclude)

    if tail == 0:
        # Two-sided: positive and negative clusters based on sign of t
        pos_mask = sig_mask & (t_stats > 0)
        neg_mask = sig_mask & (t_stats < 0)

        pos_clusters, pos_stats = _find_clusters_single_tail(
            t_stats, adjacency, pos_mask, stat_fun, t_power
        )
        neg_clusters, neg_stats = _find_clusters_single_tail(
            t_stats, adjacency, neg_mask, stat_fun, t_power
        )

        obs_clusters = pos_clusters + neg_clusters
        obs_cluster_stats = np.concatenate([pos_stats, neg_stats]) if len(pos_stats) > 0 or len(neg_stats) > 0 else np.array([])
    else:
        # One-sided tail: use sign of tail for mask
        if tail == 1:
            mask = sig_mask & (t_stats > 0)
        else:  # tail == -1
            mask = sig_mask & (t_stats < 0)
        obs_clusters, obs_cluster_stats = _find_clusters_single_tail(
            t_stats, adjacency, mask, stat_fun, t_power
        )

    if len(obs_clusters) == 0:
        return []

    # ---------- Permutation null distribution ----------
    if not separate_signs:
        # Single null based on max |cluster_stat|
        null_max_stats: List[float] = []
        for perm_idx in range(n_permutations):
            t_perm = perm_t_stats[perm_idx, :]
            p_perm = perm_p_values[perm_idx, :]
            sig_perm = p_perm < cluster_alpha
            if exclude is not None:
                sig_perm = sig_perm & (~exclude)

            if tail == 0:
                pos_mask_perm = sig_perm & (t_perm > 0)
                neg_mask_perm = sig_perm & (t_perm < 0)
                pos_c, pos_s = _find_clusters_single_tail(
                    t_perm, adjacency, pos_mask_perm, stat_fun, t_power
                )
                neg_c, neg_s = _find_clusters_single_tail(
                    t_perm, adjacency, neg_mask_perm, stat_fun, t_power
                )
                perm_stats = np.concatenate([pos_s, neg_s]) if len(pos_s) > 0 or len(neg_s) > 0 else np.array([])
            else:
                if tail == 1:
                    mask_perm = sig_perm & (t_perm > 0)
                else:
                    mask_perm = sig_perm & (t_perm < 0)
                _, perm_stats = _find_clusters_single_tail(
                    t_perm, adjacency, mask_perm, stat_fun, t_power
                )

            if len(perm_stats) > 0:
                null_max_stats.append(float(np.max(np.abs(perm_stats))))
            else:
                null_max_stats.append(0.0)

        null_max_stats = np.array(null_max_stats)

        significant_clusters: List[Dict] = []
        for cluster_channels, cluster_stat in zip(obs_clusters, obs_cluster_stats):
            p_val = float(np.mean(null_max_stats >= np.abs(cluster_stat)))
            significant_clusters.append({
                'channels': cluster_channels,
                'stat': float(cluster_stat),
                'p_value': p_val,
            })

        return significant_clusters

    # Separate nulls for positive and negative clusters (Andrillon-style)
    null_pos_max: List[float] = []
    null_neg_min: List[float] = []

    for perm_idx in range(n_permutations):
        t_perm = perm_t_stats[perm_idx, :]
        p_perm = perm_p_values[perm_idx, :]
        sig_perm = p_perm < cluster_alpha
        if exclude is not None:
            sig_perm = sig_perm & (~exclude)

        if tail == 0:
            pos_mask_perm = sig_perm & (t_perm > 0)
            neg_mask_perm = sig_perm & (t_perm < 0)
            _, pos_stats_perm = _find_clusters_single_tail(
                t_perm, adjacency, pos_mask_perm, stat_fun, t_power
            )
            _, neg_stats_perm = _find_clusters_single_tail(
                t_perm, adjacency, neg_mask_perm, stat_fun, t_power
            )
        else:
            if tail == 1:
                mask_perm = sig_perm & (t_perm > 0)
            else:
                mask_perm = sig_perm & (t_perm < 0)
            _, stats_perm = _find_clusters_single_tail(
                t_perm, adjacency, mask_perm, stat_fun, t_power
            )
            if tail == 1:
                pos_stats_perm = stats_perm
                neg_stats_perm = np.array([])
            else:
                pos_stats_perm = np.array([])
                neg_stats_perm = stats_perm

        if len(pos_stats_perm) > 0:
            null_pos_max.append(float(np.max(pos_stats_perm)))
        else:
            null_pos_max.append(0.0)

        if len(neg_stats_perm) > 0:
            null_neg_min.append(float(np.min(neg_stats_perm)))
        else:
            null_neg_min.append(0.0)

    null_pos_max_arr = np.array(null_pos_max)
    null_neg_min_arr = np.array(null_neg_min)

    significant_clusters: List[Dict] = []
    for cluster_channels, cluster_stat in zip(obs_clusters, obs_cluster_stats):
        if cluster_stat > 0:
            p_val = float(np.mean(null_pos_max_arr >= cluster_stat))
        elif cluster_stat < 0:
            p_val = float(np.mean(null_neg_min_arr <= cluster_stat))
        else:
            p_val = 1.0

        significant_clusters.append({
            'channels': cluster_channels,
            'stat': float(cluster_stat),
            'p_value': p_val,
        })

    return significant_clusters

def _compute_bootstrap_ci(
    clusters: List[np.ndarray],
    power_data: np.ndarray,
    df_behavioral: pd.DataFrame,
    formula: str,
    predictor_of_interest: str,
    adjacency: np.ndarray,
    threshold: float,
    tail: int,
    stat_fun: str,
    t_power: float,
    n_bootstrap: int,
    seed: int,
    method: str,
    maxiter: int,
    verbose: bool
) -> dict:
    """
    Compute bootstrap confidence intervals for cluster statistics.
    
    Uses case resampling (resampling subjects with replacement) to estimate
    the sampling distribution of cluster statistics.
    
    Parameters
    ----------
    clusters : List[np.ndarray]
        Observed clusters (channel indices)
    power_data : np.ndarray
        Power values, shape (n_observations, n_channels)
    df_behavioral : pd.DataFrame
        Behavioral data
    formula : str
        LMM formula
    predictor_of_interest : str
        Predictor name
    adjacency : np.ndarray
        Channel adjacency matrix
    threshold : float
        Cluster threshold
    tail : int
        Test tail (-1, 0, 1)
    stat_fun : str
        Cluster statistic function ('sum' or 'max')
    t_power : float
        Power to raise t-values by before summing
    n_bootstrap : int
        Number of bootstrap iterations
    seed : int
        Random seed
    method : str
        LMM optimization method
    maxiter : int
        LMM max iterations
    verbose : bool
        Print progress
        
    Returns
    -------
    dict
        Bootstrap CI results with keys:
        - 'ci_lower': lower bounds (95% CI) for each cluster
        - 'ci_upper': upper bounds (95% CI) for each cluster
        - 'bootstrap_stats': all bootstrap statistics per cluster
    """
    from lmm_model import run_lmm_per_channel
    
    n_clusters = len(clusters)
    bootstrap_stats = [[] for _ in range(n_clusters)]
    
    # Get unique subjects
    subjects = df_behavioral['subject'].unique()
    n_subjects = len(subjects)
    
    np.random.seed(seed + 999)  # Different seed from permutations
    
    for boot_idx in range(n_bootstrap):
        if verbose and boot_idx % 100 == 0 and boot_idx > 0:
            print(f"  Bootstrap progress: {boot_idx}/{n_bootstrap} ({100*boot_idx/n_bootstrap:.1f}%)")
        
        # Resample subjects with replacement
        boot_subjects = np.random.choice(subjects, size=n_subjects, replace=True)
        
        # Create bootstrap sample
        boot_indices = []
        for subj in boot_subjects:
            subj_indices = df_behavioral[df_behavioral['subject'] == subj].index.tolist()
            boot_indices.extend(subj_indices)
        
        boot_power = power_data[boot_indices, :]
        boot_df = df_behavioral.iloc[boot_indices].copy().reset_index(drop=True)
        
        # Run LMM on bootstrap sample
        try:
            boot_t_stats, _, _ = run_lmm_per_channel(
                power_data=boot_power,
                df_behavioral=boot_df,
                formula=formula,
                predictor_of_interest=predictor_of_interest,
                method=method,
                maxiter=maxiter,
                random_state=seed + boot_idx + 1000,
                return_diagnostics=False
            )
            
            # Compute statistics for each observed cluster
            for cluster_idx, cluster_channels in enumerate(clusters):
                cluster_t_values = boot_t_stats[cluster_channels]
                
                if stat_fun == 'sum':
                    # Apply t_power weighting (preserving sign)
                    stat = np.sum(np.sign(cluster_t_values) * np.abs(cluster_t_values) ** t_power)
                else:  # 'max'
                    max_idx = np.argmax(np.abs(cluster_t_values))
                    stat = np.sign(cluster_t_values[max_idx]) * np.abs(cluster_t_values[max_idx]) ** t_power
                
                bootstrap_stats[cluster_idx].append(stat)
        
        except Exception as e:
            if verbose and boot_idx < 5:
                print(f"  Bootstrap iteration {boot_idx} failed: {e}")
            # Skip this iteration
            continue
    
    # Compute confidence intervals
    ci_lower = np.zeros(n_clusters)
    ci_upper = np.zeros(n_clusters)
    
    for cluster_idx in range(n_clusters):
        if len(bootstrap_stats[cluster_idx]) > 0:
            ci_lower[cluster_idx] = np.percentile(bootstrap_stats[cluster_idx], 2.5)
            ci_upper[cluster_idx] = np.percentile(bootstrap_stats[cluster_idx], 97.5)
        else:
            ci_lower[cluster_idx] = np.nan
            ci_upper[cluster_idx] = np.nan
    
    if verbose:
        print(f"  Bootstrap complete: {n_bootstrap} iterations")
        print(f"  95% Confidence intervals computed for {n_clusters} clusters")
    
    return {
        'ci_lower': ci_lower.tolist(),
        'ci_upper': ci_upper.tolist(),
        'bootstrap_stats': [np.array(stats).tolist() for stats in bootstrap_stats],
        'n_bootstrap': n_bootstrap
    }


def _permute_within_subjects(
    df: pd.DataFrame,
    predictor: str,
    seed: int
) -> pd.DataFrame:
    """
    Permute predictor values within each subject to preserve repeated measures structure.
    
    **Why within-subject permutation?**
    
    For Linear Mixed Models with repeated measures, we need to preserve the subject-level
    structure during permutation. This function shuffles predictor values within each
    subject independently, which:
    
    1. **Breaks the association** between predictor and outcome (null hypothesis)
    2. **Preserves** the number of observations per subject
    3. **Preserves** the set of predictor values for each subject
    4. **Preserves** subject-specific random effects structure
    5. **Preserves** within-subject correlation patterns
    
    **Contrast with other permutation strategies:**
    
    - **Between-subject permutation**: Would shuffle subject labels, inappropriate for
      within-subject predictors (e.g., mind-wandering levels vary within subjects)
    - **Global permutation**: Would shuffle all values across subjects, destroying the
      repeated measures structure and inflating Type I error
    - **Block permutation**: Could permute blocks of trials, but less powerful for
      continuous predictors
    
    **Example:**
    
    Subject A has observations with predictor values [1, 2, 3, 4]
    Subject B has observations with predictor values [2, 3, 4, 5]
    
    After permutation:
    Subject A might have [3, 1, 4, 2] (shuffled within A)
    Subject B might have [4, 2, 5, 3] (shuffled within B)
    
    Note that A's values stay within A, and B's values stay within B.
    
    Parameters
    ----------
    df : pd.DataFrame
        Behavioral data with 'subject' column and predictor variable
    predictor : str
        Name of predictor to permute (e.g., 'onoff', 'valence', 'confidence')
    seed : int
        Random seed for reproducibility
        
    Returns
    -------
    pd.DataFrame
        DataFrame with permuted predictor values (all other columns unchanged)
        
    Notes
    -----
    - Each subject's predictor values are shuffled independently
    - The permutation is deterministic given the seed
    - All other columns (including 'subject') remain unchanged
    - This is the appropriate null hypothesis for within-subject LMM analyses
    
    Examples
    --------
    >>> df = pd.DataFrame({
    ...     'subject': ['A', 'A', 'A', 'B', 'B', 'B'],
    ...     'onoff': [1, 2, 3, 4, 5, 6],
    ...     'power': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    ... })
    >>> df_perm = _permute_within_subjects(df, 'onoff', seed=42)
    >>> # Subject A's onoff values are shuffled among [1, 2, 3]
    >>> # Subject B's onoff values are shuffled among [4, 5, 6]
    >>> # 'power' column remains unchanged
    """
    df_perm = df.copy()
    np.random.seed(seed)
    
    from lmm_model import is_interaction_term, get_interaction_components
    if is_interaction_term(predictor):
        # WARNING: permuting only the first component breaks both the interaction
        # term AND its main effect, making the null hypothesis broader than
        # "interaction = 0".  For interaction tests, 'freedman_lane' is the
        # statistically correct permutation method — use it via config.yaml
        # clustering.permutation_method: "freedman_lane".
        permute_col = get_interaction_components(predictor)[0]
        warnings.warn(
            f"Simple permutation with interaction term '{predictor}': only "
            f"'{permute_col}' is permuted, which also breaks the '{permute_col}' "
            f"main effect.  The null hypothesis is not purely 'interaction = 0'.  "
            f"Use permutation_method='freedman_lane' for valid interaction tests.",
            UserWarning,
            stacklevel=2
        )
    else:
        permute_col = predictor
    
    # Permute within each subject to preserve subject-level structure
    for subject in df['subject'].unique():
        subject_mask = df['subject'] == subject
        subject_indices = df.loc[subject_mask].index
        
        # Get current values for this subject
        subject_values = df.loc[subject_mask, permute_col].values
        
        # Permute the values
        permuted_values = np.random.permutation(subject_values)
        
        # Assign permuted values back
        df_perm.loc[subject_indices, permute_col] = permuted_values
    
    return df_perm


# ============================================================================
# TFCE (Threshold-Free Cluster Enhancement) Implementation
# ============================================================================

def compute_tfce(
    t_map: np.ndarray,
    adjacency: np.ndarray,
    E: float = 0.5,
    H: float = 2.0,
    n_steps: int = 100
) -> np.ndarray:
    """
    Compute Threshold-Free Cluster Enhancement (TFCE) for a t-statistic map.
    
    TFCE eliminates the need for an arbitrary cluster-forming threshold by
    integrating cluster-like local spatial support across all thresholds.
    Uses MNE's _get_components for cluster finding at each threshold step.
    
    The TFCE transformation is defined as:
        TFCE(x) = ∫ extent(x,h)^E * h^H dh
    
    where:
    - extent(x,h) is the size of the cluster containing voxel x at threshold h
    - E weights the cluster extent (typically 0.5)
    - H weights the height/magnitude (typically 2.0)
    
    **MNE Integration**:
    - Uses _get_cluster_extents() which wraps MNE's _get_components()
    - Follows MNE's cluster finding pattern at each threshold
    - Compatible with both threshold-based and TFCE pipelines
    
    Parameters
    ----------
    t_map : np.ndarray
        T-statistic map, shape (n_channels,)
    adjacency : np.ndarray or sparse matrix
        Spatial adjacency matrix (MNE format: coo_array preferred)
    E : float, default=0.5
        Exponent for cluster extent weighting
        - Higher E: more weight to spatial extent
        - Lower E: more weight to peak magnitude
    H : float, default=2.0
        Exponent for height weighting
        - Higher H: more weight to peak magnitude
        - Lower H: more weight to spatial extent
    n_steps : int, default=100
        Number of threshold steps for numerical integration
        More steps = more accurate but slower
        
    Returns
    -------
    tfce_map : np.ndarray
        TFCE-transformed map, shape (n_channels,)
        Higher values indicate stronger cluster-like evidence
        
    Notes
    -----
    - TFCE is computed separately for positive and negative t-values
    - The integration is approximated using a Riemann sum
    - Default parameters (E=0.5, H=2.0) are from Smith & Nichols (2009)
    - TFCE values are not directly interpretable but are used for ranking
    - Uses MNE's cluster finding at each threshold for consistency
    
    References
    ----------
    Smith, S. M., & Nichols, T. E. (2009). Threshold-free cluster enhancement:
    addressing problems of smoothing, threshold dependence and localisation in
    cluster inference. NeuroImage, 44(1), 83-98.
    
    See Also
    --------
    _get_cluster_extents : Uses MNE's _get_components for cluster finding
    """
    n_channels = len(t_map)
    tfce_map = np.zeros(n_channels)
    
    # Handle NaN values
    valid_mask = ~np.isnan(t_map)
    if not np.any(valid_mask):
        return tfce_map
    
    # Process positive and negative values separately
    for sign in [1, -1]:
        # Get signed t-values
        signed_t = t_map * sign
        signed_t[~valid_mask] = -np.inf  # Exclude NaN
        
        # Determine threshold range
        t_min = 0
        t_max = np.max(signed_t[valid_mask])
        
        if t_max <= 0:
            continue
        
        # Create threshold steps (from 0 to max)
        thresholds = np.linspace(t_min, t_max, n_steps + 1)
        dh = thresholds[1] - thresholds[0]  # Step size for integration
        
        # Integrate over thresholds
        for h in thresholds[1:]:  # Skip h=0
            # Find suprathreshold channels
            suprathreshold_mask = signed_t >= h
            
            if not np.any(suprathreshold_mask):
                continue
            
            # Get cluster extents at this threshold
            cluster_extents = _get_cluster_extents(
                suprathreshold_mask, adjacency
            )
            
            # Add contribution to TFCE: extent^E * h^H * dh
            tfce_contribution = (cluster_extents ** E) * (h ** H) * dh
            tfce_map += tfce_contribution * sign  # Preserve sign
    
    return tfce_map


def _get_cluster_extents(
    mask: np.ndarray,
    adjacency: np.ndarray
) -> np.ndarray:
    """
    Get cluster extent (size) for each channel at a given threshold.
    
    Uses MNE's _get_components following the exact MNE pattern.
    This is used for TFCE computation.
    
    Parameters
    ----------
    mask : np.ndarray
        Boolean mask indicating suprathreshold channels
    adjacency : np.ndarray or sparse matrix
        Spatial adjacency matrix (MNE format)
        
    Returns
    -------
    extents : np.ndarray
        Cluster extent for each channel (0 if not suprathreshold)
        
    Notes
    -----
    Follows MNE's cluster finding pattern exactly for TFCE integration.
    """
    n_channels = len(mask)
    extents = np.zeros(n_channels)
    
    if not np.any(mask):
        return extents
    
    # Ensure adjacency is in MNE format (coo_array)
    if issparse(adjacency):
        if not isinstance(adjacency, coo_array):
            adjacency = coo_array(adjacency)
    else:
        adjacency = coo_array(adjacency)
    
    # Use MNE's _get_components (exactly as MNE does)
    clusters = _get_components(mask, adjacency, return_list=True)
    
    # Assign cluster sizes to all channels in each cluster
    for cluster in clusters:
        cluster_size = len(cluster)
        extents[cluster] = cluster_size
    
    return extents


def spatial_cluster_test_tfce(
    observed_t_stats: np.ndarray,
    power_data: np.ndarray,
    df_behavioral: pd.DataFrame,
    formula: str,
    predictor_of_interest: str,
    adjacency: np.ndarray,
    n_permutations: int = 5000,
    E: float = 0.5,
    H: float = 2.0,
    n_tfce_steps: int = 100,
    seed: int = 42,
    n_jobs: int = 1,
    method: str = 'lbfgs',
    maxiter: int = 1000,
    verbose: bool = True,
    return_diagnostics: bool = False,
    exclude: np.ndarray = None
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Perform TFCE-based spatial permutation test for Linear Mixed Models.
    
    This is a threshold-free alternative to cluster-based permutation testing.
    Instead of using an arbitrary cluster-forming threshold, TFCE integrates
    cluster-like support across all possible thresholds, providing a single
    statistic per channel that reflects both magnitude and spatial extent.
    
    **Advantages over threshold-based approach:**
    - No arbitrary threshold selection
    - Better sensitivity to both focal and distributed effects
    - More stable results across different effect sizes
    - Channel-wise p-values (not cluster-wise)
    
    **MNE Integration:**
    - Uses MNE's _get_components() for cluster finding at each TFCE threshold
    - Uses MNE's _pval_from_histogram() for p-value computation
    - Follows MNE's exact cluster finding pattern for consistency
    
    **Workflow:**
    1. Compute observed TFCE map from observed t-statistics (using MNE's _get_components)
    2. For each permutation:
       a. Permute predictor within subjects
       b. Recompute LMM t-statistics
       c. Compute TFCE map (using MNE's _get_components)
       d. Store maximum TFCE value
    3. Compute p-values using MNE's _pval_from_histogram
    
    Parameters
    ----------
    observed_t_stats : np.ndarray
        Observed t-statistics from LMM, shape (n_channels,)
    power_data : np.ndarray
        Power values, shape (n_observations, n_channels)
    df_behavioral : pd.DataFrame
        Behavioral data with 'subject' column
    formula : str
        R-style LMM formula
    predictor_of_interest : str
        Name of predictor to test
    adjacency : np.ndarray
        Spatial adjacency matrix (n_channels, n_channels)
    n_permutations : int, default=5000
        Number of permutations for null distribution
        Recommended: 5000+ for publication-quality results
    E : float, default=0.5
        TFCE extent exponent (typically 0.5)
    H : float, default=2.0
        TFCE height exponent (typically 2.0)
    n_tfce_steps : int, default=100
        Number of threshold steps for TFCE integration.
        More steps = more accurate but slower. 100 is usually sufficient.
    seed : int
        Random seed for reproducibility
    exclude : np.ndarray, optional
        Boolean mask of channels to exclude from TFCE computation.
        Excluded channels will have TFCE=0 and p-value=1.
        Following MNE's pattern to prevent boundary artifacts.
    n_jobs : int, default=1
        Number of parallel jobs for permutation testing
        -1 uses all available CPU cores
        1 runs serially (no parallelization)
    method : str, default='lbfgs'
        LMM optimization method
    maxiter : int, default=1000
        Maximum LMM iterations
    verbose : bool, default=True
        Print progress information
    return_diagnostics : bool, default=False
        Return detailed diagnostics
        
    Returns
    -------
    tfce_map : np.ndarray
        Observed TFCE values, shape (n_channels,)
    p_values : np.ndarray
        Channel-wise p-values, shape (n_channels,)
    diagnostics : dict
        Diagnostics dictionary containing:
        - 'null_distribution': TFCE null distribution statistics
        - 'tfce_parameters': E, H, n_steps
        - 'n_permutations': number of permutations
        
    Notes
    -----
    - TFCE p-values are channel-wise, not cluster-wise
    - Uses +1 correction: p = (b + 1) / (n_perm + 1)
    - Preserves within-subject structure via permutation
    - More computationally intensive than threshold-based approach
    
    References
    ----------
    Smith, S. M., & Nichols, T. E. (2009). Threshold-free cluster enhancement:
    addressing problems of smoothing, threshold dependence and localisation in
    cluster inference. NeuroImage, 44(1), 83-98.
    
    Examples
    --------
    >>> tfce_map, p_values, diagnostics = spatial_cluster_test_tfce(
    ...     observed_t_stats=t_stats,
    ...     power_data=power_data,
    ...     df_behavioral=df_behavioral,
    ...     formula="power ~ onoff + (1|subject)",
    ...     predictor_of_interest="onoff",
    ...     adjacency=adjacency,
    ...     n_permutations=5000
    ... )
    >>> sig_channels = np.where(p_values < 0.05)[0]
    """
    from lmm_model import run_lmm_per_channel
    
    if verbose:
        print("="*70)
        print("TFCE-BASED SPATIAL PERMUTATION TEST")
        print("="*70)
        print(f"TFCE parameters: E={E}, H={H}, n_steps={n_tfce_steps}")
        print(f"Permutations: {n_permutations}")
        print(f"Predictor: {predictor_of_interest}")
    
    # Validate inputs (simplified for TFCE - no threshold/tail needed)
    # Just validate basic dimensions and data quality
    if not isinstance(observed_t_stats, np.ndarray) or observed_t_stats.ndim != 1:
        raise ValueError("observed_t_stats must be a 1D numpy array")
    
    n_channels = len(observed_t_stats)
    
    if power_data.shape[1] != n_channels:
        raise ValueError(
            f"power_data has {power_data.shape[1]} channels but "
            f"observed_t_stats has {n_channels} channels"
        )
    
    if power_data.shape[0] != len(df_behavioral):
        raise ValueError(
            f"power_data has {power_data.shape[0]} observations but "
            f"df_behavioral has {len(df_behavioral)} rows"
        )
    
    from lmm_model import is_interaction_term, get_interaction_components
    if is_interaction_term(predictor_of_interest):
        for comp in get_interaction_components(predictor_of_interest):
            if comp not in df_behavioral.columns:
                raise ValueError(
                    f"Interaction component '{comp}' (from '{predictor_of_interest}') "
                    f"not found in df_behavioral"
                )
    else:
        if predictor_of_interest not in df_behavioral.columns:
            raise ValueError(f"Predictor '{predictor_of_interest}' not found in df_behavioral")
    
    if adjacency.shape != (n_channels, n_channels):
        raise ValueError(
            f"adjacency must be ({n_channels}, {n_channels}), got {adjacency.shape}"
        )
    
    # Handle exclude mask (MNE pattern)
    include = None
    if exclude is not None:
        if exclude.size != n_channels:
            raise ValueError(f"exclude must have same length as observed_t_stats ({n_channels})")
        include = np.logical_not(exclude)
        if verbose:
            n_excluded = np.sum(exclude)
            print(f"\n✓ Excluding {n_excluded}/{n_channels} channels from TFCE")
            print(f"  (Excluded channels will have TFCE=0 and p-value=1)")
    
    # ========== COMPUTE OBSERVED TFCE MAP ==========
    if verbose:
        print("\nComputing observed TFCE map...")
    
    observed_tfce = compute_tfce(
        observed_t_stats, adjacency, E=E, H=H, n_steps=n_tfce_steps
    )
    
    # Apply exclude mask: set excluded channels to 0
    if include is not None:
        observed_tfce[~include] = 0.0
    
    if verbose:
        print(f"  TFCE range: [{np.min(observed_tfce):.3f}, {np.max(observed_tfce):.3f}]")
        print(f"  Channels with TFCE > 0: {np.sum(observed_tfce > 0)}")
        print(f"  Channels with TFCE < 0: {np.sum(observed_tfce < 0)}")
    
    # ========== PERMUTATION TESTING ==========
    if verbose:
        print(f"\nRunning {n_permutations} permutations...")
        if n_jobs != 1:
            print(f"  Using {n_jobs} parallel jobs")
    
    # Define helper function for single permutation
    def _run_single_permutation(perm_idx):
        """Run a single permutation and return max |TFCE|."""
        # Permute predictor within subjects
        df_perm = _permute_within_subjects(
            df_behavioral, predictor_of_interest, seed + perm_idx
        )
        
        # Compute t-statistics for permuted data
        t_stats_perm, _, _ = run_lmm_per_channel(
            power_data=power_data,
            df_behavioral=df_perm,
            formula=formula,
            predictor_of_interest=predictor_of_interest,
            method=method,
            maxiter=maxiter,
            random_state=seed + perm_idx,
            return_diagnostics=False
        )
        
        # Compute TFCE for permuted data
        tfce_perm = compute_tfce(
            t_stats_perm, adjacency, E=E, H=H, n_steps=n_tfce_steps
        )
        
        # Apply exclude mask: set excluded channels to 0
        if include is not None:
            tfce_perm[~include] = 0.0
        
        # Return maximum absolute TFCE value
        return float(np.max(np.abs(tfce_perm)))
    
    # Run permutations (parallel if n_jobs != 1)
    if n_jobs == 1:
        # Serial execution
        max_tfce_null = []
        for perm_idx in range(n_permutations):
            if verbose and perm_idx % 100 == 0 and perm_idx > 0:
                print(f"  Progress: {perm_idx}/{n_permutations} ({100*perm_idx/n_permutations:.1f}%)")
            max_tfce_null.append(_run_single_permutation(perm_idx))
        max_tfce_null = np.array(max_tfce_null)
    else:
        # Parallel execution
        from joblib import Parallel, delayed
        max_tfce_null = np.array(
            Parallel(n_jobs=n_jobs, verbose=10 if verbose else 0)(
                delayed(_run_single_permutation)(perm_idx)
                for perm_idx in range(n_permutations)
            )
        )
    
    if verbose:
        print(f"  Permutations complete!")
    
    # ========== COMPUTE P-VALUES ==========
    # Channel-wise p-values with +1 correction
    p_values = np.zeros(n_channels)
    
    for ch_idx in range(n_channels):
        # Count permutations where null TFCE >= observed TFCE
        n_exceeding = np.sum(max_tfce_null >= np.abs(observed_tfce[ch_idx]))
        # Apply +1 correction (Phipson & Smyth, 2010)
        p_values[ch_idx] = float((n_exceeding + 1) / (n_permutations + 1))
    
    # Set excluded channels to p-value = 1 (not significant)
    if include is not None:
        p_values[~include] = 1.0
    
    # ========== PRINT RESULTS SUMMARY ==========
    if verbose:
        print("\n" + "="*70)
        print("TFCE RESULTS SUMMARY")
        print("="*70)
        
        # Null distribution stats
        print(f"\nNull distribution (max |TFCE|):")
        print(f"  Mean: {np.mean(max_tfce_null):.3f}")
        print(f"  Median: {np.median(max_tfce_null):.3f}")
        print(f"  95th percentile: {np.percentile(max_tfce_null, 95):.3f}")
        print(f"  99th percentile: {np.percentile(max_tfce_null, 99):.3f}")
        
        # Significant channels
        sig_channels = np.where(p_values < 0.05)[0]
        print(f"\nSignificant channels (p < 0.05): {len(sig_channels)}/{n_channels}")
        
        if len(sig_channels) > 0:
            print(f"\nTop 10 channels by TFCE value:")
            sorted_indices = np.argsort(np.abs(observed_tfce))[::-1][:10]
            print(f"{'Ch':<4} {'TFCE':>10} {'p-value':>10} {'Sig':>5}")
            print("-" * 35)
            for idx in sorted_indices:
                sig = "***" if p_values[idx] < 0.001 else "**" if p_values[idx] < 0.01 else "*" if p_values[idx] < 0.05 else ""
                print(f"{idx:<4} {observed_tfce[idx]:>10.3f} {p_values[idx]:>10.4f} {sig:>5}")
        
        print("="*70 + "\n")
    
    # ========== COMPILE DIAGNOSTICS ==========
    diagnostics = {
        'null_distribution': {
            'mean': float(np.mean(max_tfce_null)),
            'median': float(np.median(max_tfce_null)),
            'std': float(np.std(max_tfce_null)),
            'percentile_95': float(np.percentile(max_tfce_null, 95)),
            'percentile_99': float(np.percentile(max_tfce_null, 99)),
            'max_tfce_null': max_tfce_null.tolist() if return_diagnostics else None
        },
        'tfce_parameters': {
            'E': E,
            'H': H,
            'n_steps': n_tfce_steps
        },
        'n_permutations': n_permutations,
        'n_significant_channels': int(np.sum(p_values < 0.05))
    }
    
    return observed_tfce, p_values, diagnostics


# Note: validate_cluster_data and summarize_clusters have been moved to helpers.py
# Import them from there if needed
