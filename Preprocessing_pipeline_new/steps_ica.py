"""ICA utilities for the preprocessing pipeline.

- fit_ica_with_fallbacks: tries a sequence of methods until convergence
- auto_select_ica_components: applies heuristic and ICLabel-based selection
"""

import os
from typing import List, Optional

# Aggressive onnxruntime threading control - set ALL possible environment variables
os.environ.setdefault("ORT_DISABLE_THREAD_AFFINITY", "1")
os.environ.setdefault("KMP_AFFINITY", "disabled") 
os.environ.setdefault("ONNX_DISABLE_THREADPOOL_AFFINITY", "1")
os.environ.setdefault("ORT_NUM_THREADS", "1")
os.environ.setdefault("ONNXRUNTIME_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import mne
import numpy as np

# Force onnxruntime to single-threaded mode programmatically
def _configure_onnxruntime_threading():
    """Configure onnxruntime to use single-threaded execution."""
    try:
        import onnxruntime as ort
        # Create session options that force single-threaded execution
        session_options = ort.SessionOptions()
        session_options.intra_op_num_threads = 1
        session_options.inter_op_num_threads = 1
        session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        # Store globally to be used by ICLabel
        _configure_onnxruntime_threading._session_options = session_options
        print("Configured onnxruntime for single-threaded execution")
    except ImportError:
        pass  # onnxruntime not available
    except Exception as e:
        print(f"Warning: Could not configure onnxruntime threading: {e}")

# Configure threading immediately when module loads
_configure_onnxruntime_threading()


def fit_ica_deterministic(
    raw_for_ica: mne.io.BaseRaw,
    method: str,
    n_components: float,
    max_iter: int,
    random_state: int,
    report: Optional[mne.Report] = None,
) -> mne.preprocessing.ICA:
    """Fit ICA with specified method deterministically.
    
    Parameters
    ----------
    raw_for_ica : mne.io.BaseRaw
        Raw data for ICA fitting (should be filtered and prepared)
    method : str
        ICA method to use (e.g., 'infomax', 'picard', 'fastica')
    n_components : float
        Number of components (if < 1, treated as ratio)
    max_iter : int
        Maximum iterations for ICA convergence
    random_state : int
        Random seed for reproducibility
    report : mne.Report, optional
        MNE report object to add ICA results
        
    Returns
    -------
    mne.preprocessing.ICA
        Fitted ICA object
        
    Raises
    ------
    RuntimeError
        If ICA fitting fails with the specified method
    """
    print(f"Fitting ICA with {method} method (deterministic, no fallbacks)...")
    
    # Configure ICA parameters based on method
    fit_params = None
    if method == 'infomax':
        fit_params = dict(extended=True)
    elif method not in ['picard', 'fastica']:
        raise ValueError(f"Unsupported ICA method: {method}. Use 'infomax', 'picard', or 'fastica'.")
    
    try:
        ica = mne.preprocessing.ICA(
            n_components=n_components,
            method=method,
            max_iter=max_iter,
            random_state=random_state,
            fit_params=fit_params,
        )
        ica.fit(raw_for_ica)
        print(f"ICA fitting successful with {method}")
        
        # Add to report
        if report is not None:
            report.add_ica(ica, title=f"ICA ({method})", inst=raw_for_ica)
        
        return ica
        
    except Exception as exc:
        raise RuntimeError(f"ICA fitting failed with method '{method}': {exc}") from exc


def auto_select_ica_components(
    raw_for_ica: mne.io.BaseRaw,
    ica: mne.preprocessing.ICA,
    prob_min: float,
    muscle_threshold: float,
    max_excluded_ratio: float,
) -> tuple[List[int], str]:
    """Automatically select ICA components for exclusion using ICLabel and heuristics.
    
    Parameters
    ----------
    raw_for_ica : mne.io.BaseRaw
        Raw data used for ICA fitting
    ica : mne.preprocessing.ICA
        Fitted ICA object
    prob_min : float
        Minimum probability threshold for ICLabel classification
    muscle_threshold : float
        Threshold for muscle artifact detection
    max_excluded_ratio : float
        Maximum ratio of components that can be excluded
        
    Returns
    -------
    tuple[List[int], str]
        Tuple containing:
        - List of component indices to exclude
        - Method used for combining artifacts ('sum' or 'union')
    """
    if ica is None:
        return [], "none"
    
    to_exclude = []
    
    # Pattern matching methods first
    # Find EOG components
    eog_components, _ = ica.find_bads_eog(
        inst=raw_for_ica,
        ch_name=["VEOG", "HEOG"] if any(ch in raw_for_ica.ch_names for ch in ["VEOG", "HEOG"]) else None,
    )
    print(f"EOG components detected: {eog_components}")
    
    # Find muscle components
    muscle_components, _ = ica.find_bads_muscle(
        raw_for_ica, 
        threshold=muscle_threshold
    )
    print(f"Muscle components detected: {muscle_components}")
    
    # Combine pattern matching results
    pattern_matching_artifacts = list(set(eog_components + muscle_components))
    
    # ICLabel classification
    from mne_icalabel import label_components
    
    print("Running ICLabel classification...")
    
    # Redirect stderr to suppress onnxruntime threading warnings
    import io
    from contextlib import redirect_stderr
    
    # Capture stderr to suppress threading affinity warnings
    f = io.StringIO()
    with redirect_stderr(f):
        # Run ICLabel classification
        labels = label_components(raw_for_ica, ica, method="iclabel")
    
    # Check if there were serious errors (not just warnings)
    stderr_output = f.getvalue()
    if "CRITICAL" in stderr_output or "FATAL" in stderr_output:
        raise RuntimeError(
            f"ICLabel failed with critical errors: {stderr_output}"
        )
    
    label_names = labels['labels']
    label_probs = labels.get('y_pred_proba', None)
    
    print(f"ICLabel classification successful: {label_names}")
    
    # Find components classified as artifacts with high confidence
    iclabel_artifacts = []
    artifact_types = [
        'muscle artifact', 'eye blink', 'heart beat', 'channel noise'
    ]
    
    # If we have probability data, use it for threshold checking
    if label_probs is not None:
        for idx, (label, prob_array) in enumerate(
            zip(label_names, label_probs)
        ):
            if label in artifact_types:
                # Check if probability is above threshold
                # Handle both scalar and array probability values
                if np.isscalar(prob_array):
                    # Single probability value (scalar)
                    if prob_array >= prob_min:
                        iclabel_artifacts.append(idx)
                elif hasattr(prob_array, '__len__') and len(prob_array) > 0:
                    # Array of probability values
                    max_prob = np.max(prob_array)
                    if max_prob >= prob_min:
                        iclabel_artifacts.append(idx)
                else:
                    # If no probability data, use the label as-is
                    iclabel_artifacts.append(idx)
    else:
        # If no probability data, just use labels directly
        for idx, label in enumerate(label_names):
            if label in artifact_types:
                iclabel_artifacts.append(idx)
    
    print(f"ICLabel artifacts: {iclabel_artifacts}")
    
    # Combine pattern matching and ICLabel results
    # First try with sum (concatenation) of both lists
    max_exclude = int(max_excluded_ratio * ica.n_components_)
    
    # Try concatenating both lists first (sum approach)
    sum_artifacts = pattern_matching_artifacts + iclabel_artifacts
    print(
        f"Sum of both methods: {sum_artifacts} (total: {len(sum_artifacts)})"
    )
    
    if len(sum_artifacts) <= max_exclude:
        print("Using sum approach (concatenation of both lists)")
        to_exclude = sum_artifacts
        combination_method = "sum"
    else:
        # If sum exceeds threshold, fall back to union (remove duplicates)
        union_artifacts = list(set(sum_artifacts))
        print(
            f"Sum exceeds threshold, using union: {union_artifacts} "
            f"(total: {len(union_artifacts)})"
        )
        to_exclude = union_artifacts
        combination_method = "union"
    
    print(f"Combined artifacts from both methods: {to_exclude}")
    print(f"Combination method used: {combination_method}")
    
    # Safety check: don't exclude too many components
    to_exclude = list(set(to_exclude))  # Remove duplicates
    max_exclude = int(max_excluded_ratio * ica.n_components_)
    
    # if len(to_exclude) > max_exclude:
    #     print(
    #         f"Too many components selected for exclusion "
    #         f"({len(to_exclude)} > {max_exclude}). "
    #         f"Keeping only the first {max_exclude} components."
    #     )
    #     to_exclude = sorted(to_exclude)[:max_exclude]
    
    print(f"Final components to exclude: {to_exclude}")
    return to_exclude, combination_method
