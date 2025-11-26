"""Independent Component Analysis (ICA) utilities for the preprocessing pipeline.

Public functions
----------------
- fit_ica_deterministic: Fit ICA with a single specified method deterministically.
- auto_select_ica_components: Select artifact ICs using ICLabel + pattern matching
  and a variance-aware threshold.
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
    """Best-effort: configure onnxruntime for single-threaded execution."""
    try:
        import onnxruntime as ort
        session_options = ort.SessionOptions()
        session_options.intra_op_num_threads = 1
        session_options.inter_op_num_threads = 1
        session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        _configure_onnxruntime_threading._session_options = session_options
    except Exception:
        pass  # safe to ignore; environment vars already constrain threading

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
    print(f"Fitting ICA using '{method}' ...")
    
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
        print("ICA fit OK")
        
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
    variance_penalty_factor: float = 0.5,
    max_excluded_variance_ratio: float = 0.40,
    max_threshold_cap: float = 0.99,
    pattern_bonus: float = 0.10,
) -> tuple[List[int], str, dict]:
    """Select ICA components using variance-weighted thresholds + scoring.
    
    Uses unified scoring system:
    - ICLabel: base score = max probability across artifact classes
    - Pattern matching (EOG/Muscle) = bonus to score
    - Compare final score against variance-adjusted threshold
    - Variance budget as ratio: excluded_var / retained_var
    
    Parameters
    ----------
    raw_for_ica : mne.io.BaseRaw
        Raw data used for ICA fitting
    ica : mne.preprocessing.ICA
        Fitted ICA object
    prob_min : float
        Base probability threshold for ICLabel
    muscle_threshold : float
        Base threshold for muscle detection
    variance_penalty_factor : float, optional
        Penalty per unit variance (default: 0.5)
    max_excluded_variance_ratio : float, optional
        Max ratio of excluded/retained variance (default: 0.40)
    max_threshold_cap : float, optional
        Maximum adjusted threshold (default: 0.99)
    pattern_bonus : float, optional
        Score bonus for pattern matching confirmation (default: 0.10)
        
    Returns
    -------
    tuple[List[int], str, dict]
        (excluded_indices, combination_method, detailed_log)
    """
    if ica is None:
        return [], "none", {}
    
    # =======================================================================
    # Calculate ACTUAL variance explained by each ICA component
    # =======================================================================
    # Uses MNE's get_explained_variance_ratio() method
    # Get ALL components at once (more efficient and consistent)
    
    print("\nCalculating ICA variance explained per component...")
    
    # CRITICAL FIX: get_explained_variance_ratio() ALWAYS returns total variance
    # We need to calculate variance for EACH component INDIVIDUALLY in a loop
    variance_per_component = np.zeros(ica.n_components_)
    
    try:
        for idx in range(ica.n_components_):
            try:
                # Get variance for THIS component only
                var_dict = ica.get_explained_variance_ratio(
                    inst=raw_for_ica,
                    components=[idx]  # Single component as list
                )
                # Extract the value (should be a scalar)
                if isinstance(var_dict, dict):
                    if 'eeg' in var_dict:
                        variance_per_component[idx] = float(var_dict['eeg'])
                    elif 'mag' in var_dict:
                        variance_per_component[idx] = float(var_dict['mag'])
                    elif 'grad' in var_dict:
                        variance_per_component[idx] = float(var_dict['grad'])
                    else:
                        first_key = list(var_dict.keys())[0]
                        variance_per_component[idx] = float(var_dict[first_key])
                else:
                    variance_per_component[idx] = float(var_dict)
            except Exception as e:
                print(f"  WARNING: Error calculating variance for IC{idx}: {e}")
                variance_per_component[idx] = 1.0 / ica.n_components_
        
        print(f"  ✓ Calculated variance for {ica.n_components_} components")
        print(f"  DEBUG: variance shape = {variance_per_component.shape}")
        print(f"  DEBUG: variance sum = {np.sum(variance_per_component):.3f}")
        print(f"  DEBUG: variance range = [{np.min(variance_per_component):.3f}, {np.max(variance_per_component):.3f}]")
        
        # Normalize if needed (should sum to ~1.0)
        total_var = np.sum(variance_per_component)
        if total_var > 0:
            variance_per_component = variance_per_component / total_var
        else:
            print(f"  WARNING: Total variance is 0, using equal weights")
            variance_per_component = np.ones(ica.n_components_) / ica.n_components_
            
    except Exception as e:
        print(f"  WARNING: Could not calculate ICA variance: {e}")
        import traceback
        traceback.print_exc()
        print(f"  Using equal weights for all components")
        variance_per_component = np.ones(ica.n_components_) / ica.n_components_
    
    print(f"\n{'='*70}")
    print(f"ICA SELECTION: {ica.n_components_} components")
    print(f"Variance budget: max excluded {max_excluded_variance_ratio:.0%} of total")
    print(f"Base thresholds: ICLabel={prob_min:.2f}, "
          f"Muscle={muscle_threshold:.2f}")
    print(f"Penalty factor: {variance_penalty_factor:.2f}")
    
    # Print variance distribution (as absolute ratios, not percentages of total)
    print(f"\nVariance explained per component (top 10):")
    top_var_indices = np.argsort(variance_per_component)[::-1][:10]
    for idx in top_var_indices:
        # Show as percentage directly (these are already ratios)
        print(f"  IC{idx}: {variance_per_component[idx]*100:.2f}%")
    total_var = np.sum(variance_per_component)
    print(f"  Sum of all {ica.n_components_} components: {total_var*100:.2f}%")
    if total_var > 1.05:  # Allow 5% tolerance
        print(f"  NOTE: Sum > 100% because ICA components are non-orthogonal")
    
    # =======================================================================
    # PARAMETER CHECK
    # =======================================================================
    print(f"\n{'='*70}")
    print("PARAMETER CHECK (values received by function):")
    print(f"  max_excluded_variance_ratio = {max_excluded_variance_ratio}")
    print(f"  prob_min = {prob_min}")
    print(f"  variance_penalty_factor = {variance_penalty_factor}")
    print(f"  max_threshold_cap = {max_threshold_cap}")
    print(f"  pattern_bonus = {pattern_bonus}")
    print(f"{'='*70}\n")
    
    # Initialize logging
    selection_log = {
        'n_components': ica.n_components_,
        'variance_per_component': variance_per_component.tolist(),
        'base_prob_threshold': prob_min,
        'base_muscle_threshold': muscle_threshold,
        'variance_penalty_factor': variance_penalty_factor,
        'max_excluded_variance_ratio': max_excluded_variance_ratio,
        'max_threshold_cap': max_threshold_cap,
        'component_decisions': [],
        'iclabel_classes': None,
    }
    
    # =======================================================================
    # PATTERN MATCHING: EOG and MUSCLE
    # =======================================================================
    print(f"\n{'-'*70}")
    print("Pattern Matching Detection:")
    
    eog_components, eog_scores = ica.find_bads_eog(
        inst=raw_for_ica,
        ch_name=(["VEOG", "HEOG"]
                 if any(ch in raw_for_ica.ch_names
                        for ch in ["VEOG", "HEOG"])
                 else None),
    )
    print(f"  EOG: {eog_components}")
    
    muscle_components, muscle_scores = ica.find_bads_muscle(
        raw_for_ica,
        threshold=muscle_threshold
    )
    print(f"  Muscle (raw): {muscle_components}")
    
    # =======================================================================
    # ICLABEL CLASSIFICATION
    # =======================================================================
    print(f"\n{'-'*70}")
    print("ICLabel Classification:")
    
    from mne_icalabel import label_components
    from mne_icalabel.iclabel import iclabel_label_components
    import io
    from contextlib import redirect_stderr
    
    f = io.StringIO()
    with redirect_stderr(f):
        labels = label_components(raw_for_ica, ica, method="iclabel")
    
    stderr_output = f.getvalue()
    if "CRITICAL" in stderr_output or "FATAL" in stderr_output:
        raise RuntimeError(
            f"ICLabel failed with critical errors: {stderr_output}"
        )
    
    # DEBUG: Print what ICLabel returned
    print(f"\n  DEBUG: ICLabel returned keys = {list(labels.keys())}")
    
    # Read human-readable labels for display
    label_names = labels.get('labels', [])
    
    # Always compute the full probability matrix using iclabel_label_components
    # Shape: (n_components, n_classes) with classes ordered as below
    try:
        y_pred_proba = iclabel_label_components(
            raw_for_ica, ica, inplace=False, backend=None
        )
        y_pred_proba = np.asarray(y_pred_proba)
        print(f"  ✓ ICLabel probabilities: {y_pred_proba.shape}")
    except Exception as e:
        print(f"  ERROR: iclabel_label_components failed: {e}")
        y_pred_proba = None
    
    # ICLabel canonical class order
    classes = ['brain', 'muscle artifact', 'eye blink', 
               'heart beat', 'line noise', 'channel noise', 'other']
    print(f"  DEBUG: classes = {classes}")
    selection_log['iclabel_classes'] = classes
    
    # Print ICLabel classification for each component
    print("\n  Component classifications (top 3 classes per IC):")
    
    for idx in range(min(ica.n_components_, len(label_names))):
        label = label_names[idx]
        # Use the SAME variance values calculated above
        var_comp = (variance_per_component[idx] 
                    if idx < len(variance_per_component) else 0)
        
        # Get top 3 probabilities for this component
        if (y_pred_proba is not None and idx < len(y_pred_proba) and
                len(classes) > 0):
            probs = y_pred_proba[idx]
            # Check if probs is array-like (has len) or scalar
            if hasattr(probs, '__len__') and not isinstance(probs, (str, bytes)):
                # probs is an array
                n_classes = min(len(probs), len(classes))
            else:
                # probs is a scalar (1D y_pred_proba case)
                n_classes = 0
            
            if n_classes >= 3:
                # Get top 3 indices, ensure valid for both probs and classes
                top3_indices = np.argsort(probs[:n_classes])[-3:][::-1]
                # Build list of valid class:prob pairs
                top3_pairs = []
                for i in top3_indices:
                    if i < len(classes) and i < len(probs):
                        top3_pairs.append(f"{classes[i]}:{probs[i]:.2f}")
                
                if top3_pairs:
                    top3_str = ", ".join(top3_pairs)
                    print(f"    IC{idx:2d}: {label:15s} "
                          f"[{top3_str}] var={var_comp*100:.2f}%")
                else:
                    print(f"    IC{idx:2d}: {label:15s} "
                          f"var={var_comp*100:.2f}% (no valid class data)")
            else:
                print(f"    IC{idx:2d}: {label:15s} var={var_comp*100:.2f}% "
                      f"(insufficient class data)")
        else:
            print(f"    IC{idx:2d}: {label:15s} var={var_comp*100:.2f}%")
    
    artifact_types = [
        'muscle artifact', 'eye blink', 'heart beat', 'channel noise'
    ]
    
    # Build mapping from class name to index (do this ONCE outside loop)
    # y_pred_proba already prepared above
    classes = classes or []
    
    # Convert to numpy array and check shape
    if y_pred_proba is not None:
        y_pred_proba = np.asarray(y_pred_proba)
    
    cls2idx = {cls: i for i, cls in enumerate(classes)} if classes else {}
    
    # Get indices of artifact classes
    artifact_indices = [
        cls2idx[art] for art in artifact_types if art in cls2idx
    ]
    
    # DEBUG: Print ICLabel structure for scoring
    print(f"\nDEBUG ICLabel structure for scoring:")
    print(f"  Classes available: {classes}")
    print(f"  Artifact class indices: {artifact_indices}")
    if y_pred_proba is not None:
        print(f"  Probability matrix shape: {y_pred_proba.shape}")
        print(f"  Probability matrix ndim: {y_pred_proba.ndim}")
    
    # =======================================================================
    # UNIFIED SCORING SYSTEM
    # =======================================================================
    print(f"\n{'-'*70}")
    print("Unified Scoring (ICLabel + Pattern Matching Bonuses):")
    print(f"  Pattern bonus: +{pattern_bonus:.0%} per match")
    print("\n  Detailed component scoring:")
    print(f"  {'IC':<4} {'Label':<15} {'Base':<6} {'Bonus':<6} "
          f"{'Final':<6} {'Thresh':<6} {'Decision':<10} {'Var%':<6}")
    print(f"  {'-'*70}")
    
    component_scores = {}
    selected_components = []
    
    for idx in range(ica.n_components_):
        if idx >= len(variance_per_component):
            continue
            
        var_comp = variance_per_component[idx]
        label = label_names[idx] if idx < len(label_names) else None
        
        # ===================================================================
        # Base score = max probability across artifact classes
        # ===================================================================
        base_score = 0.0
        if y_pred_proba is not None and len(artifact_indices) > 0:
            if idx < len(y_pred_proba):
                # Check if y_pred_proba is 2D (correct) or 1D (ICLabel failed)
                if y_pred_proba.ndim == 2:
                    # Normal case: (n_components, n_classes)
                    artifact_probs = y_pred_proba[idx, artifact_indices]
                    base_score = float(np.max(artifact_probs))
                    
                    # DEBUG: Print for first 3 components
                    if idx < 3:
                        print(f"  DEBUG IC{idx}: All probs = {y_pred_proba[idx]}")
                        print(f"  DEBUG IC{idx}: Artifact probs = {artifact_probs}, max = {base_score}")
                elif y_pred_proba.ndim == 1:
                    # ICLabel returned 1D array - can't extract probabilities
                    # Fall back to using the label directly
                    if idx == 0:  # Print once
                        print(f"  INFO: y_pred_proba is 1D, using label-based binary scoring")
                    # Use the label from ICLabel to decide
                    base_score = 1.0 if label in artifact_types else 0.0
        
        # Apply bonuses for pattern matching confirmation
        bonus_score = 0.0
        bonus_sources = []
        
        if idx in eog_components:
            bonus_score += pattern_bonus
            bonus_sources.append("EOG")
        
        if idx in muscle_components:
            bonus_score += pattern_bonus
            bonus_sources.append("Muscle")
        
        # Always add bonus for channel artifact/noise
        if label == 'channel noise':
            bonus_score += pattern_bonus *2
            bonus_sources.append("Channel")
        
        # Final boosted score
        final_score = min(base_score + bonus_score, 1.0)  # Cap at 1.0
        
        # Calculate variance-adjusted threshold with cap
        variance_penalty = min(
            var_comp * variance_penalty_factor,
            max_threshold_cap - prob_min
        )
        adjusted_threshold = prob_min + variance_penalty
        
        # Decision: exclude if final score >= adjusted threshold
        excluded = final_score >= adjusted_threshold
        
        # Print detailed info for this component
        decision_str = "EXCLUDE" if excluded else "keep"
        bonus_str = f"+{bonus_score:.2f}" if bonus_score > 0 else "    "
        sources_str = (",".join(bonus_sources)) if bonus_sources else ""
        
        print(f"  {idx:<4} {label:<15} {base_score:<6.3f} "
              f"{bonus_str:<6} {final_score:<6.3f} "
              f"{adjusted_threshold:<6.3f} {decision_str:<10} "
              f"{var_comp*100:<5.1f}% {sources_str}")
        
        # Derive richer ICLabel information for logging
        iclabel_probs = None
        iclabel_top3 = None
        iclabel_best_class = None
        iclabel_best_prob = None
        if (y_pred_proba is not None and y_pred_proba.ndim == 2
                and idx < len(y_pred_proba) and classes):
            probs_all = y_pred_proba[idx]
            try:
                iclabel_probs = [float(p) for p in probs_all]
                # Top-3 classes by probability (any type)
                top_idx = np.argsort(probs_all)[-3:][::-1]
                iclabel_top3 = [
                    f"{classes[i]}:{probs_all[i]:.2f}"
                    for i in top_idx if i < len(classes)
                ]
                best_idx_all = int(np.argmax(probs_all))
                if best_idx_all < len(classes):
                    iclabel_best_class = classes[best_idx_all]
                    iclabel_best_prob = float(probs_all[best_idx_all])
            except Exception:
                iclabel_probs = None
                iclabel_top3 = None
                iclabel_best_class = None
                iclabel_best_prob = None

        # Store component info
        component_scores[idx] = {
            'variance': float(var_comp),
            'label': label,
            'base_score': float(base_score),
            'bonus': float(bonus_score),
            'bonus_sources': bonus_sources,
            'final_score': float(final_score),
            'adjusted_threshold': float(adjusted_threshold),
            'excluded': excluded,
            'iclabel_probs': iclabel_probs,
            'iclabel_top3': iclabel_top3,
            'iclabel_best_class': iclabel_best_class,
            'iclabel_best_prob': iclabel_best_prob,
        }
        
        if excluded:
            selected_components.append(idx)
        
        # Log decision
        selection_log['component_decisions'].append(component_scores[idx])
    
    print(f"  Components with bonuses: "
          f"{len([c for c in component_scores.values() if c['bonus'] > 0])}")
    print(f"  Selected for exclusion: {len(selected_components)}")
    
    # =======================================================================
    # COMBINE DETECTION METHODS
    # =======================================================================
    print(f"\n{'-'*70}")
    print("Combining Detection Methods:")
    
    # All components that passed the threshold should be excluded
    to_exclude = selected_components
    print(f"  Components selected for exclusion: {len(to_exclude)}")
    
    # =======================================================================
    # VARIANCE BUDGET CHECK (for monitoring/flagging only)
    # =======================================================================
    sum_excluded_variance = sum(
        [variance_per_component[i] for i in to_exclude
         if i < len(variance_per_component)]
    )
    total_variance = np.sum(variance_per_component)
    retained_variance = total_variance - sum_excluded_variance
    
    # Calculate absolute fraction excluded: excluded / total
    if total_variance > 0:
        excluded_fraction = sum_excluded_variance / total_variance
        excluded_pct = excluded_fraction * 100
        retained_pct = retained_variance * 100
        
        print(f"  Excluded variance: {excluded_pct:.1f}% of total")
        
        if excluded_fraction > max_excluded_variance_ratio:
            # FLAG: Variance budget exceeded, but still exclude all
            print(f"  ⚠️  WARNING: Excluding {excluded_pct:.0f}% of total variance")
            print(f"  Budget allows max {max_excluded_variance_ratio:.0%} of total = "
                  f"{max_excluded_variance_ratio * 100:.0f}%")
            print(f"  Flagging for QA review, but excluding all marked components")
            combination_method = "variance_budget_exceeded"
            selection_log['variance_budget_exceeded'] = True
            selection_log['excluded_fraction'] = float(excluded_fraction)
        else:
            print(f"  ✓ Within variance budget: {excluded_fraction:.2f} <= "
                  f"{max_excluded_variance_ratio:.2f}")
            combination_method = "unified_scoring"
    else:
        # Safety check: if we would exclude everything
        if retained_variance <= 0.01:
            print(f"  ⚠️  CRITICAL: Would exclude {sum_excluded_variance:.1%} "
                  f"of variance")
            print(f"  This suggests severe data quality issues or "
                  f"overly aggressive thresholds")
            combination_method = "variance_critical_warning"
            selection_log['variance_critical_warning'] = True
        else:
            combination_method = "unified_scoring"
    
    # =======================================================================
    # SUMMARY STATISTICS
    # =======================================================================
    print(f"\n{'-'*70}")
    print("SELECTION SUMMARY:")
    
    # Count by category
    label_counts = {}
    for idx, info in component_scores.items():
        label = info['label']
        if label not in label_counts:
            label_counts[label] = {'total': 0, 'excluded': 0}
        label_counts[label]['total'] += 1
        if info['excluded']:
            label_counts[label]['excluded'] += 1
    
    print(f"\n  Components by category:")
    for label in sorted(label_counts.keys()):
        counts = label_counts[label]
        print(f"    {label:<15}: {counts['total']:2d} total, "
              f"{counts['excluded']:2d} excluded")
    
    # Pattern matching summary
    n_with_bonuses = len([c for c in component_scores.values() 
                          if c['bonus'] > 0])
    print(f"\n  Pattern bonuses applied: {n_with_bonuses} components")
    if eog_components:
        print(f"    EOG pattern matches: {sorted(eog_components)}")
    if muscle_components:
        print(f"    Muscle pattern matches: {sorted(muscle_components)}")
    
    # Exclusion summary
    n_total_excluded = len([c for c in component_scores.values() 
                            if c.get('excluded', False)])
    
    print(f"\n  Total components excluded: {n_total_excluded}")
    
    # Show if variance budget was exceeded (flagged for QA)
    if selection_log.get('variance_budget_exceeded', False):
        print(f"  ⚠️  Variance budget exceeded")
        print(f"  Excluding too much variance relative to total allowed")
        print(f"  Flagged for manual QA review")
    
    # Final variance calculation
    final_variance = sum([variance_per_component[i] for i in to_exclude
                          if i < len(variance_per_component)])
    final_variance_pct = (
        final_variance * 100
        if np.sum(variance_per_component) > 0 else 0
    )
    
    print(f"\n{'='*70}")
    print(f"FINAL: Excluding {len(to_exclude)}/{ica.n_components_} "
          f"components ({final_variance_pct:.1f}% of total variance)")
    print(f"Excluded ICs: {sorted(to_exclude)}")
    print(f"{'='*70}\n")
    
    # Sanitize values for JSON serialization (remove NaN/Inf, convert numpy types)
    def sanitize_value(val):
        """Convert values for safe JSON serialization."""
        # Handle NaN/Inf floats
        if isinstance(val, (float, np.floating)):
            if np.isnan(val) or np.isinf(val):
                return None
            return float(val)
        # Handle numpy booleans
        if isinstance(val, (bool, np.bool_)):
            return bool(val)
        # Handle numpy integers
        if isinstance(val, (np.integer,)):
            return int(val)
        return val
    
    # Update log - store as ratio for backwards compatibility
    selection_log['final_n_excluded'] = len(to_exclude)
    selection_log['final_excluded_variance_ratio'] = sanitize_value(
        float(final_variance)
    )
    selection_log['final_excluded_variance_pct'] = sanitize_value(
        float(final_variance_pct)
    )
    selection_log['excluded_components'] = sorted(to_exclude)
    selection_log['combination_method'] = combination_method
    
    # Add concise excluded component summary: label[variance%]
    try:
        excluded_components_summary = []
        for i in sorted(to_exclude):
            var_pct = float(variance_per_component[i] * 100.0) if i < len(variance_per_component) else 0.0
            lbl_i = component_scores.get(i, {}).get('label', None)
            lbl_i = str(lbl_i) if lbl_i is not None else 'unknown'
            excluded_components_summary.append(f"{lbl_i}[{var_pct:.0f}%]")
        selection_log['excluded_components_summary'] = excluded_components_summary
    except Exception:
        selection_log['excluded_components_summary'] = []
    
    # Sanitize ALL values in selection_log (including booleans like variance_budget_exceeded)
    for key, val in list(selection_log.items()):
        if key == 'component_decisions':
            # Handle nested dicts in component_decisions
            for comp_decision in val:
                for k, v in comp_decision.items():
                    comp_decision[k] = sanitize_value(v)
        elif key == 'variance_per_component' or key == 'excluded_components':
            # Lists - sanitize each element
            if isinstance(val, list):
                selection_log[key] = [sanitize_value(v) for v in val]
        else:
            # Single values
            selection_log[key] = sanitize_value(val)
    
    return to_exclude, combination_method, selection_log

