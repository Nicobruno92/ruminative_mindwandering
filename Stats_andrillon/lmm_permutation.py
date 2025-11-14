"""
LMM Permutation Module - Andrillon 2020 Implementation

This module implements the permutation strategy from Andrillon et al. (2020):
- Fit Linear Mixed Models (LMM) per electrode
- Permute predictor labels within subject × task
- Generate null distribution for cluster-based testing

Key difference from Freedman-Lane:
- Andrillon: Permutes predictor labels directly within subject/task
- Freedman-Lane: Permutes residuals from reduced model

Reference:
    Andrillon et al. (2020). Lines 102-115:
    "We then created permuted datasets by permuting the labels of the 
    predictor within each subject, each task and each electrode"
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import statsmodels.formula.api as smf
from statsmodels.regression.mixed_linear_model import MixedLM
import logging
from joblib import Parallel, delayed

logger = logging.getLogger(__name__)


def fit_lmm_with_permutations(
    data: pd.DataFrame,
    formula: str,
    predictor_of_interest: str,
    n_permutations: int = 1000,
    permutation_within: List[str] = ["subject", "task"],
    random_state: Optional[int] = None,
    method: str = "powell",
    maxiter: int = 5000,
    n_jobs: int = -1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit LMM and generate permutations following Andrillon 2020 methodology.
    
    Parameters
    ----------
    data : pd.DataFrame
        Data for a single electrode containing:
        - Dependent variable (e.g., 'power')
        - Predictor of interest
        - Grouping variables (subject, task, etc.)
    formula : str
        LMM formula in statsmodels format
        Example: "power ~ predictor + covariate + (1|subject)"
    predictor_of_interest : str
        Name of the predictor to test (will be permuted)
    n_permutations : int, default=1000
        Number of permutations to generate
    permutation_within : list of str, default=["subject", "task"]
        Variables to permute within (stratified permutation)
        Andrillon: permutes within subject × task × electrode
        Since we run per electrode, we permute within subject × task
    random_state : int, default=42
        Random seed for reproducibility
    method : str, default="powell"
        Optimization method for LMM fitting
    maxiter : int, default=5000
        Maximum iterations for optimization
    n_jobs : int, default=-1
        Number of parallel jobs for permutations (-1 = all CPUs)
        
    Returns
    -------
    real_stats : np.ndarray, shape (1, 3)
        Real model statistics: [beta, t_value, p_value]
    perm_stats : np.ndarray, shape (n_permutations, 4)
        Permuted statistics: [beta, t_value, p_value, perm_id]
        
    Notes
    -----
    Permutation strategy (Andrillon 2020):
    1. For each permutation:
       2. For each unique combination of grouping variables (e.g., subject × task):
          3. Shuffle predictor values within that group
       4. Refit the full model with permuted predictor
       5. Extract t-value for predictor of interest
    
    This differs from Freedman-Lane which:
    1. Fits reduced model (without predictor)
    2. Permutes residuals
    3. Refits full model
    
    Examples
    --------
    >>> data = pd.DataFrame({
    ...     'power': np.random.randn(100),
    ...     'onoff': np.random.randint(0, 100, 100),
    ...     'subject': np.repeat(range(10), 10),
    ...     'task': np.tile([1, 2], 50)
    ... })
    >>> real, perm = fit_lmm_with_permutations(
    ...     data, 
    ...     "power ~ onoff + (1|subject)",
    ...     "onoff",
    ...     n_permutations=100
    ... )
    >>> real.shape
    (1, 3)
    >>> perm.shape
    (100, 4)
    """
    np.random.seed(random_state)
    
    # Validate inputs
    if predictor_of_interest not in data.columns:
        raise ValueError(f"Predictor '{predictor_of_interest}' not found in data")
    
    for var in permutation_within:
        if var not in data.columns:
            raise ValueError(f"Grouping variable '{var}' not found in data")
    
    # Fit real model
    logger.debug(f"Fitting real model with formula: {formula}")
    real_stats = _fit_single_lmm(
        data, formula, predictor_of_interest, method, maxiter
    )
    
    # Generate permutations in parallel
    logger.debug(f"Generating {n_permutations} permutations with {n_jobs} jobs...")
    
    def _run_single_permutation(perm_id):
        """Run a single permutation (for parallel execution)."""
        # Create permuted dataset
        data_perm = _permute_predictor(
            data.copy(), 
            predictor_of_interest, 
            permutation_within
        )
        
        # Fit model with permuted data
        try:
            perm_result = _fit_single_lmm(
                data_perm, formula, predictor_of_interest, method, maxiter
            )
            return np.array([perm_result[0], perm_result[1], perm_result[2], perm_id])
        except Exception as e:
            logger.warning(f"Permutation {perm_id} failed: {e}")
            return np.array([np.nan, np.nan, np.nan, perm_id])
    
    # Run permutations in parallel
    perm_results = Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(_run_single_permutation)(perm_id) 
        for perm_id in range(n_permutations)
    )
    
    # Convert to array
    perm_stats = np.array(perm_results)
    
    return real_stats, perm_stats


def _fit_single_lmm(
    data: pd.DataFrame,
    formula: str,
    predictor_of_interest: str,
    method: str = "powell",
    maxiter: int = 5000,
) -> np.ndarray:
    """
    Fit a single LMM and extract statistics for predictor of interest.
    
    Parameters
    ----------
    data : pd.DataFrame
        Data to fit
    formula : str
        LMM formula
    predictor_of_interest : str
        Predictor to extract statistics for
    method : str
        Optimization method
    maxiter : int
        Maximum iterations
        
    Returns
    -------
    stats : np.ndarray, shape (3,)
        [beta, t_value, p_value] for predictor of interest
    """
    try:
        # Ensure subject is string type (critical for mixedlm)
        data = data.copy()
        data['subject'] = data['subject'].astype(str)
        
        # Fit model
        model = smf.mixedlm(formula, data, groups=data["subject"])
        result = model.fit(method=method, maxiter=maxiter, disp=False)
        
        # Extract statistics for predictor of interest
        coef_names = result.params.index.tolist()
        
        # Find the coefficient corresponding to predictor
        # Handle both simple predictors and interactions
        pred_idx = None
        for i, name in enumerate(coef_names):
            if predictor_of_interest in name and name != "Intercept":
                pred_idx = i
                break
        
        if pred_idx is None:
            raise ValueError(
                f"Predictor '{predictor_of_interest}' not found in model coefficients"
            )
        
        beta = result.params.iloc[pred_idx]
        t_value = result.tvalues.iloc[pred_idx]
        p_value = result.pvalues.iloc[pred_idx]
        
        return np.array([beta, t_value, p_value])
        
    except Exception as e:
        logger.warning(f"Model fitting failed: {e}")
        return np.array([np.nan, np.nan, np.nan])


def _permute_predictor(
    data: pd.DataFrame,
    predictor: str,
    permutation_within: List[str],
) -> pd.DataFrame:
    """
    Permute predictor values within groups (Andrillon 2020 strategy).
    
    Parameters
    ----------
    data : pd.DataFrame
        Original data
    predictor : str
        Name of predictor to permute
    permutation_within : list of str
        Variables defining groups for stratified permutation
        
    Returns
    -------
    data_permuted : pd.DataFrame
        Data with permuted predictor values
        
    Notes
    -----
    Andrillon strategy: "permuting the labels of the predictor within 
    each subject, each task and each electrode"
    
    Since we run per electrode, we permute within subject × task.
    """
    data_permuted = data.copy()
    
    # Ensure subject remains string type after copy (critical for mixedlm)
    if 'subject' in data_permuted.columns:
        data_permuted['subject'] = data_permuted['subject'].astype(str)
    
    # Get unique combinations of grouping variables
    groups = data.groupby(permutation_within).groups
    
    # Permute within each group
    for group_key, indices in groups.items():
        # Get predictor values for this group
        predictor_values = data.loc[indices, predictor].values
        
        # Shuffle
        permuted_values = np.random.permutation(predictor_values)
        
        # Assign back
        data_permuted.loc[indices, predictor] = permuted_values
    
    return data_permuted


def fit_lmm_per_electrode(
    data: pd.DataFrame,
    electrode_column: str,
    formula: str,
    predictor_of_interest: str,
    n_permutations: int = 1000,
    permutation_within: List[str] = ["subject", "task"],
    random_state: int = 42,
    method: str = "powell",
    maxiter: int = 5000,
    n_jobs: int = 1,
) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    """
    Fit LMM for all electrodes with permutations.
    
    Parameters
    ----------
    data : pd.DataFrame
        Full dataset with all electrodes
    electrode_column : str
        Name of column containing electrode IDs
    formula : str
        LMM formula
    predictor_of_interest : str
        Predictor to test
    n_permutations : int
        Number of permutations
    permutation_within : list of str
        Variables to permute within
    random_state : int
        Random seed
    method : str
        Optimization method
    maxiter : int
        Maximum iterations
    n_jobs : int
        Number of parallel jobs (not implemented yet)
        
    Returns
    -------
    real_results : dict
        Dictionary mapping electrode_id -> real_stats array
    perm_results : dict
        Dictionary mapping electrode_id -> perm_stats array
        
    Examples
    --------
    >>> # Assuming data has multiple electrodes
    >>> real, perm = fit_lmm_per_electrode(
    ...     data,
    ...     electrode_column='channel',
    ...     formula='power ~ onoff + (1|subject)',
    ...     predictor_of_interest='onoff',
    ...     n_permutations=100
    ... )
    >>> len(real)  # Number of electrodes
    64
    """
    electrodes = data[electrode_column].unique()
    n_electrodes = len(electrodes)
    
    logger.info(f"Fitting LMM for {n_electrodes} electrodes...")
    
    real_results = {}
    perm_results = {}
    
    for i, electrode_id in enumerate(electrodes):
        logger.info(f"Electrode {i+1}/{n_electrodes} (ID: {electrode_id})")
        
        # Get data for this electrode
        electrode_data = data[data[electrode_column] == electrode_id].copy()
        
        # Fit LMM with permutations
        real_stats, perm_stats = fit_lmm_with_permutations(
            electrode_data,
            formula,
            predictor_of_interest,
            n_permutations,
            permutation_within,
            random_state + i,  # Different seed per electrode
            method,
            maxiter,
        )
        
        real_results[electrode_id] = real_stats
        perm_results[electrode_id] = perm_stats
    
    logger.info("LMM fitting complete!")
    
    return real_results, perm_results


def format_results_for_clustering(
    real_results: Dict[int, np.ndarray],
    perm_results: Dict[int, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Format LMM results for cluster detection.
    
    Parameters
    ----------
    real_results : dict
        Real model results per electrode
    perm_results : dict
        Permuted results per electrode
        
    Returns
    -------
    real_array : np.ndarray, shape (n_electrodes, 4)
        [electrode_id, beta, t_value, p_value]
    perm_array : np.ndarray, shape (n_electrodes * n_permutations, 5)
        [electrode_id, beta, t_value, p_value, perm_id]
        
    Notes
    -----
    This format is compatible with the cluster detection module
    (cluster_detection.py) which expects these specific array structures.
    """
    # Convert real results to array
    real_list = []
    for electrode_id, stats in real_results.items():
        real_list.append([electrode_id, stats[0], stats[1], stats[2]])
    real_array = np.array(real_list)
    
    # Convert permuted results to array
    perm_list = []
    for electrode_id, perm_stats in perm_results.items():
        for perm_row in perm_stats:
            perm_list.append([
                electrode_id, 
                perm_row[0],  # beta
                perm_row[1],  # t_value
                perm_row[2],  # p_value
                perm_row[3],  # perm_id
            ])
    perm_array = np.array(perm_list)
    
    return real_array, perm_array


if __name__ == "__main__":
    # Example usage and testing
    logging.basicConfig(level=logging.INFO)
    
    # Generate synthetic data
    np.random.seed(42)
    n_subjects = 10
    n_tasks = 2
    n_probes_per_task = 30
    
    data_list = []
    for subj in range(n_subjects):
        for task in range(1, n_tasks + 1):
            for probe in range(n_probes_per_task):
                # Simulate effect: higher onoff -> higher power
                onoff = np.random.randint(0, 100)
                power = 10 + 0.05 * onoff + np.random.randn() * 2
                
                data_list.append({
                    'subject': subj,
                    'task': task,
                    'onoff': onoff,
                    'power': power,
                })
    
    data = pd.DataFrame(data_list)
    
    # Test single electrode
    print("\n=== Testing single electrode ===")
    real, perm = fit_lmm_with_permutations(
        data,
        formula="power ~ onoff + (1|subject)",
        predictor_of_interest="onoff",
        n_permutations=100,
        random_state=42,
    )
    
    print(f"Real stats: beta={real[0,0]:.4f}, t={real[0,1]:.4f}, p={real[0,2]:.4f}")
    print(f"Permuted stats shape: {perm.shape}")
    print(f"Mean permuted t-value: {np.nanmean(perm[:,1]):.4f}")
    print(f"Proportion |perm_t| > |real_t|: {np.mean(np.abs(perm[:,1]) > np.abs(real[0,1])):.4f}")
