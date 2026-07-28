#!/usr/bin/env python
"""
Classification Distribution Analysis — LOSO Mode.

Run multiple LOSO passes with different random seeds to build a distribution
of classification performance. Includes permutation testing.

Project: depressed_mindwandering
"""

import os
import pickle
import time
import warnings
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from utils.data_utils import get_model_results_folder
from utils.ml_utils import run_model_pipeline_cv, run_within_subject_cv, compute_shap_values_for_pipeline
from utils.plotting_utils import (
    plot_confusion_matrix,
    plot_roc_curve,
    plot_feature_importances,
    plot_shap_feature_importance,
    plot_metric_distribution_with_stats,
    empirical_mean_permutation_pvalue,
)

from joblib import Parallel, delayed

warnings.filterwarnings("ignore", message="resource_tracker:.*")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# =============================================================================
# DIRECTORY HELPERS
# =============================================================================

def get_run_dir(dimension_results_path: str, run_idx: int, n_runs: int,
                run_idx_offset: int = 0, total_n_runs: int = None) -> str:
    """
    Return per-run output directory.

    For single-run LOSO with no offset: uses dimension_results_path directly.
    For multi-run or SLURM parallel mode (offset > 0): uses true_runs/run_{n} subdirectory.

    Parameters
    ----------
    run_idx : int
        Local run index within this job (0-based).
    n_runs : int
        Number of runs in this job.
    run_idx_offset : int
        Global offset — the actual run number = run_idx + run_idx_offset.
        Set when running a single run as part of a SLURM array.
    total_n_runs : int, optional
        Total runs across all jobs. When > 1 with offset=0, forces subdirectory use.
    """
    global_idx = run_idx + run_idx_offset
    effective_total = total_n_runs if total_n_runs is not None else n_runs
    if effective_total <= 1 and run_idx_offset == 0:
        run_dir = dimension_results_path
    else:
        run_dir = os.path.join(dimension_results_path, "true_runs", f"run_{global_idx}")
    Path(run_dir).mkdir(parents=True, exist_ok=True)
    return run_dir


def get_permutation_run_dir(dimension_results_path: str, run_idx: int,
                             perm_idx_offset: int = 0) -> str:
    """Return per-permutation output directory (inside permuted_runs/ subfolder).

    Parameters
    ----------
    run_idx : int
        Local permutation index within this job (0-based).
    perm_idx_offset : int
        Global offset — actual perm number = run_idx + perm_idx_offset.
    """
    global_idx = run_idx + perm_idx_offset
    perm_dir = os.path.join(dimension_results_path, "permuted_runs", f"run_{global_idx}")
    Path(perm_dir).mkdir(parents=True, exist_ok=True)
    return perm_dir


def _build_filename_base(model_type: str, n_runs: int, pipeline_label: str = "loso") -> str:
    """
    Build base filename for results files.

    Parameters
    ----------
    model_type : str
        'rf', 'xgb', or 'lr'.
    n_runs : int
        Number of true runs in this job.
    pipeline_label : str
        'loso' or 'ws' — identifies which pipeline produced this file.
        Previously hardcoded to 'loso' even for within-subject output, so
        WS files were misleadingly named e.g. ``rf_loso_03_shap_values.pkl``
        for a within-subject per-subject run (Fix FIND-007).
    """
    if n_runs > 1:
        return f"{model_type}_{pipeline_label}_{n_runs}runs"
    return f"{model_type}_{pipeline_label}"


# =============================================================================
# FEATURE CORRELATION
# =============================================================================

def _compute_feature_correlations(X: pd.DataFrame, y: pd.Series) -> list:
    """
    Compute absolute correlations between each feature and the target.

    Returns
    -------
    list
        Sorted list of (feature_name, abs_correlation) tuples, descending.
    """
    correlations = []
    for col in X.columns:
        x_vals = pd.to_numeric(X[col], errors="coerce")
        valid = x_vals.notna()
        if valid.sum() < 10:
            continue
        corr = np.abs(np.corrcoef(x_vals[valid], y[valid])[0, 1])
        if np.isfinite(corr):
            correlations.append((col, corr))
    correlations.sort(key=lambda x: x[1], reverse=True)
    return correlations


def _compute_lmm_encoding_preview(
    X: pd.DataFrame, y: pd.Series, groups: pd.Series,
    method: str = "encoding", n_jobs: int = -1,
) -> list:
    """
    Fit LMM (encoding or decoding) on the full dataset as a sanity-check preview.

    This is independent of the CV feature-selection step (which refits per fold
    on train-only data); it just lets the user see which features the mixed
    model flags as label-related across the whole dataset.

    Parameters
    ----------
    method : {'encoding', 'decoding'}
        'encoding' → Feature ~ Label + (1|Subject) (Gaussian).
        'decoding' → Label ~ Feature + (1|Subject) (Binomial).

    Returns
    -------
    list
        Sorted list of (feature_name, pvalue) tuples, ascending by p.
    """
    from joblib import Parallel, delayed
    from utils.ml_utils import (
        _fit_single_lmm_feature_encoding,
        _fit_single_lmm_feature_decoding,
    )
    fit_fn = (
        _fit_single_lmm_feature_encoding if method == "encoding"
        else _fit_single_lmm_feature_decoding
    )
    X_arr = X.values
    y_arr = np.asarray(y).ravel()
    groups_arr = np.asarray(groups).ravel()
    pvalues = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(fit_fn)(i, X_arr[:, i], y_arr, groups_arr)
        for i in range(X_arr.shape[1])
    )
    pairs = list(zip(X.columns.tolist(), pvalues))
    pairs.sort(key=lambda p: (p[1] if np.isfinite(p[1]) else 1.0))
    return pairs


# =============================================================================
# RESULT EXTRACTION
# =============================================================================

def _extract_run_summary(
    run_results: pd.DataFrame,
    run_idx: int,
    random_state: int,
    dimension: str,
    n_folds: int,
    feature_names,
    feature_importances: np.ndarray,
    positive_class_name: str,
    negative_class_name: str,
) -> dict:
    """
    Extract a summary dict from a single run's results DataFrame.

    Parameters
    ----------
    run_results : pd.DataFrame
        Single-row DataFrame from run_model_pipeline_cv.
    run_idx : int
        Run index (for tracking).
    random_state : int
        Seed used for this run.
    dimension : str
        Contrast name.
    n_folds : int
        Number of folds (= number of subjects for LOSO).
    feature_names : Index
        Full feature names.
    feature_importances : np.ndarray
        Mean feature importances across folds.
    positive_class_name, negative_class_name : str
        Human-readable class names.

    Returns
    -------
    dict
        Run summary.
    """
    def _get(col, default=None):
        if col in run_results.columns:
            return run_results[col].values[0]
        return default

    return {
        "run_idx": run_idx,
        "random_state": int(random_state),
        "dimension": dimension,
        "n_folds": n_folds,
        "mean_auc": _get("mean_auc"),
        "std_auc": _get("std_auc"),
        "mean_auprc": _get("mean_auprc"),
        "std_auprc": _get("std_auprc"),
        "mean_mcc": _get("mean_mcc"),
        "std_mcc": _get("std_mcc"),
        "mean_balanced_accuracy": _get("mean_balanced_accuracy"),
        "std_balanced_accuracy": _get("std_balanced_accuracy"),
        "mean_precision": _get("mean_precision"),
        "mean_recall": _get("mean_recall"),
        "mean_f1": _get("mean_f1"),
        "fold_aucs": _get("fold_aucs", []),
        "fold_auprcs": _get("fold_auprcs", []),
        "fold_mccs": _get("fold_mccs", []),
        "fold_bal_accs": _get("fold_bal_accs", []),
        "fold_cms": _get("fold_cms", []),
        "fold_tprs": _get("fold_tprs", []),
        "fold_fprs": _get("fold_fprs", []),
        "fold_precisions_curve": _get("fold_precisions_curve", []),
        "fold_recalls_curve": _get("fold_recalls_curve", []),
        "feature_importances": (
            feature_importances.tolist()
            if hasattr(feature_importances, "tolist")
            else list(feature_importances)
        ),
        "loso_subject_metrics": _get("loso_subject_metrics"),
        "fold_details": _get("fold_details", []),
        "positive_class_name": positive_class_name,
        "negative_class_name": negative_class_name,
    }


# =============================================================================
# SAVING
# =============================================================================

def _save_run_results(
    run_results: pd.DataFrame,
    run_dir: str,
    filename_base: str,
    feature_names,
    feature_importances: np.ndarray,
    save_pickle: bool,
) -> None:
    """Save per-run summary CSV and feature importances."""
    run_results.to_csv(os.path.join(run_dir, f"{filename_base}_summary.csv"), index=False)

    pd.DataFrame({
        "feature": feature_names,
        "importance": feature_importances,
    }).to_csv(
        os.path.join(run_dir, f"{filename_base}_feature_importances.csv"), index=False
    )

    if save_pickle:
        with open(os.path.join(run_dir, f"{filename_base}_detailed.pkl"), "wb") as f:
            pickle.dump(run_results.to_dict(), f)


def _save_probabilities(
    run_results: pd.DataFrame,
    df: pd.DataFrame,
    run_dir: str,
    filename_base: str,
    run_idx: int,
    raw_score_col: str = "onoff",
) -> None:
    """
    Save predicted probabilities (per fold and per sample).

    Saves two files:
    - fold_predictions.csv: one row per fold with lists of y_true, y_pred, y_proba
    - sample_predictions.csv: one row per sample with probe metadata
    """
    fold_details = run_results["fold_details"].values[0] if "fold_details" in run_results.columns else []
    if not fold_details:
        return

    per_fold_rows = []
    sample_rows = []

    for fold in fold_details:
        # Fold-level row
        per_fold_rows.append({
            "run_idx": run_idx,
            "fold_idx": fold.get("fold_idx"),
            "subject": fold.get("subject"),
            "y_true": fold.get("y_true"),
            "y_pred": fold.get("y_pred"),
            "y_proba": fold.get("y_proba", []),
            **{f"label_{k}_pct": v for k, v in fold.get("label_percentages", {}).items()},
        })

        # Sample-level rows
        test_indices = fold.get("test_indices", [])
        y_true_list = fold.get("y_true", [])
        y_pred_list = fold.get("y_pred", [])
        y_proba_list = fold.get("y_proba", [])

        for i, idx in enumerate(test_indices):
            if idx < len(df):
                row_data = df.iloc[idx]
                sample_rows.append({
                    "run_idx": run_idx,
                    "fold_idx": fold.get("fold_idx"),
                    "subject": fold.get("subject"),
                    "sample_idx": idx,
                    "task": row_data.get("task", ""),
                    "probe_number": row_data.get("probe_number", ""),
                    raw_score_col: row_data.get(raw_score_col, ""),
                    "confidence": row_data.get("confidence", ""),
                    "y_true": y_true_list[i] if i < len(y_true_list) else None,
                    "y_pred": y_pred_list[i] if i < len(y_pred_list) else None,
                    "y_proba": y_proba_list[i] if i < len(y_proba_list) else None,
                })

    if per_fold_rows:
        pd.DataFrame(per_fold_rows).to_csv(
            os.path.join(run_dir, f"{filename_base}_fold_predictions.csv"), index=False
        )
    if sample_rows:
        pd.DataFrame(sample_rows).to_csv(
            os.path.join(run_dir, f"{filename_base}_sample_predictions.csv"), index=False
        )


def _save_shap_values(
    run_results: pd.DataFrame,
    X: pd.DataFrame,
    feature_names,
    run_dir: str,
    filename_base: str,
    model_type: str,
    groups: pd.Series = None,
    scale_by_participant: str = "none",
    scaler_type: str = "standard",
    y: pd.Series = None,
) -> "tuple[np.ndarray, np.ndarray, np.ndarray] | None":
    """
    Compute and save SHAP values for a single run.

    Only supported for 'rf' and 'xgb' models.

    When ``scale_by_participant='within'`` (LOSO), the pipeline has no scaler
    step — scaling was applied manually before training.  This function
    reconstructs per-subject scaling on the test fold so that the clf receives
    data on the same within-person scale it was trained to expect.

    Returns
    -------
    tuple of (np.ndarray, np.ndarray, np.ndarray) or None
        ``(shap_values, x_test, y_true)`` for this run, or None if computation
        failed.
    """
    if model_type not in ("rf", "xgb", "lr"):
        return None

    cv_splits  = run_results["cv_splits"].values[0]  if "cv_splits"  in run_results.columns else []
    estimators = run_results["estimators"].values[0] if "estimators" in run_results.columns else []

    if not estimators or not cv_splits:
        return None

    from sklearn.preprocessing import StandardScaler, RobustScaler
    ScalerClass = RobustScaler if scaler_type == "robust" else StandardScaler

    fold_shap_list:   list = []
    fold_x_list:      list = []
    fold_y_true_list: list = []
    for (_, test_idx), estimator in zip(cv_splits, estimators):
        X_test = X.iloc[test_idx].copy()

        # LOSO with within-subject scaling: the pipeline has no scaler step,
        # so we apply the same per-subject normalization that was used during
        # training (fit on each subject's own test data — no leakage because
        # the test subject was fully held out from training).
        if scale_by_participant == "within" and groups is not None:
            groups_test  = groups.iloc[test_idx]
            numeric_cols = X_test.select_dtypes(include=[np.number]).columns.tolist()
            for participant in groups_test.unique():
                mask = groups_test == participant
                subj_scaler = ScalerClass()
                X_test.loc[mask, numeric_cols] = subj_scaler.fit_transform(
                    X_test.loc[mask, numeric_cols]
                )

        # fold_x is the actual (scaler-applied, zero-padded) matrix the
        # explainer used — NOT necessarily identical to X_test.values, since
        # the estimator's own `scaler` step (e.g. within-subject pipelines)
        # may transform it further.  Saving fold_x instead of X_test.values
        # keeps the beeswarm color axis correctly paired with fold_shap
        # (Fix FIND-001).
        fold_shap, fold_x, _ = compute_shap_values_for_pipeline(estimator, X_test, feature_names)
        fold_shap_list.append(fold_shap)
        fold_x_list.append(fold_x)
        if y is not None:
            fold_y_true_list.append(y.iloc[test_idx].values)

    shap_values    = np.concatenate(fold_shap_list, axis=0)
    x_test_stacked = np.concatenate(fold_x_list,    axis=0)
    y_true_stacked = (
        np.concatenate(fold_y_true_list, axis=0)
        if fold_y_true_list else None
    )

    pkl_payload = {
        "shap_values":   shap_values,
        "feature_names": list(feature_names),
        "x_test":        x_test_stacked,
    }
    if y_true_stacked is not None:
        pkl_payload["y_true"] = y_true_stacked

    with open(os.path.join(run_dir, f"{filename_base}_shap_values.pkl"), "wb") as f:
        pickle.dump(pkl_payload, f)

    return shap_values, x_test_stacked, y_true_stacked


def _consolidate_sample_predictions(
    true_runs_path: str,
    filename_base: str,
    output_dir: str,
) -> None:
    """
    Merge sample-level predictions across all true_runs and compute per-probe statistics.

    Parameters
    ----------
    true_runs_path : str
        Path to the ``true_runs/`` subfolder containing run_0/, run_1/, etc.
    filename_base : str
        Prefix used to find ``*_sample_predictions.csv`` files inside each run folder.
    output_dir : str
        Directory where consolidated files are written (typically the model_type/ root).

    Saves two files:
    - {filename_base}_all_sample_predictions.csv: raw stack of all runs
    - {filename_base}_consolidated_sample_predictions.csv: averaged per probe
    """
    if not os.path.exists(true_runs_path):
        return

    all_preds = []
    for run_folder in sorted(os.listdir(true_runs_path)):
        sample_file = os.path.join(true_runs_path, run_folder, f"{filename_base}_sample_predictions.csv")
        if os.path.exists(sample_file):
            all_preds.append(pd.read_csv(sample_file))

    if not all_preds:
        return

    all_df = pd.concat(all_preds, ignore_index=True)
    group_cols = [c for c in ["subject", "task", "probe_number"] if c in all_df.columns]

    agg_dict = {"y_true": "first", "y_pred": "mean", "run_idx": "count"}
    if "y_proba" in all_df.columns:
        agg_dict["y_proba"] = ["mean", "std"]
    _standard_cols = {"subject", "task", "probe_number", "sample_idx", "fold_idx",
                      "y_true", "y_pred", "y_proba", "run_idx"}
    for col in all_df.columns:
        if col not in _standard_cols and col not in agg_dict:
            agg_dict[col] = "first"

    summary = all_df.groupby(group_cols).agg(agg_dict).reset_index()
    summary.columns = ["_".join(str(c) for c in col).strip("_") for col in summary.columns]
    summary = summary.rename(columns={
        "y_proba_mean": "proba_mean",
        "y_proba_std": "proba_std",
        "y_pred_mean": "pred_proportion",
        "run_idx_count": "n_runs",
    })
    if "proba_mean" in summary.columns:
        summary["y_pred_avg"] = (summary["proba_mean"] >= 0.5).astype(int)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    all_df.to_csv(
        os.path.join(output_dir, f"{filename_base}_all_sample_predictions.csv"),
        index=False
    )
    summary.to_csv(
        os.path.join(output_dir, f"{filename_base}_consolidated_sample_predictions.csv"),
        index=False
    )
    print(f"Consolidated predictions: {len(summary)} unique probes across {len(all_preds)} runs")


# =============================================================================
# PLOTTING
# =============================================================================

def _generate_plots(
    all_results: list,
    results_df: pd.DataFrame,
    feature_names,
    dimension: str,
    dimension_results_path: str,
    filename_base: str,
    top_n_features_plot: int,
    positive_class_name: str,
    negative_class_name: str,
    loso_subject_df: pd.DataFrame,
    shap_values_all_runs: list,
    X: pd.DataFrame,
) -> None:
    """Generate all visualizations for distribution analysis results."""
    print(f"Generating plots for {dimension}...")

    # Confusion matrix (averaged across all folds)
    cms_list = [cm for r in all_results for cm in r.get("fold_cms", [])]
    if cms_list:
        avg_cm = np.mean(cms_list, axis=0)
        cell_stats = pd.DataFrame({
            "cell": ["TN", "FP", "FN", "TP"],
            "mean": avg_cm.flatten(),
            "std": np.std(cms_list, axis=0).flatten(),
        })
        plot_confusion_matrix(
            avg_cm, negative_class_name, positive_class_name,
            cell_stats, dimension_results_path, filename_base,
        )
        print("  ✓ Confusion matrix")

    # ROC curve (interpolated, averaged)
    all_tprs = [tpr for r in all_results for tpr in r.get("fold_tprs", [])]
    all_fprs = [fpr for r in all_results for fpr in r.get("fold_fprs", [])]
    if all_tprs and all_fprs:
        mean_fpr = np.linspace(0, 1, 100)
        tprs_interp = []
        aucs_interp = []
        for fpr, tpr in zip(all_fprs, all_tprs):
            if len(fpr) > 1:
                interp_tpr = np.interp(mean_fpr, fpr, tpr)
                interp_tpr[0] = 0.0
                tprs_interp.append(interp_tpr)
                aucs_interp.append(np.trapz(interp_tpr, mean_fpr))
        if tprs_interp:
            mean_tpr = np.mean(tprs_interp, axis=0)
            mean_tpr[-1] = 1.0
            plot_roc_curve(
                mean_fpr, mean_tpr, np.std(tprs_interp, axis=0),
                dimension_results_path, filename_base, np.mean(aucs_interp), len(tprs_interp),
            )
            print("  ✓ ROC curve")

    # Feature importances
    fi_cols = [c for c in results_df.columns if c.startswith("importance_")]
    if fi_cols:
        fnames = [c.replace("importance_", "") for c in fi_cols]
        mean_imp = results_df[fi_cols].mean().values
        std_imp = results_df[fi_cols].std().values
        plot_feature_importances(
            fnames, mean_imp, std_imp,
            dimension_results_path, filename_base, top_n=top_n_features_plot,
        )
        print("  ✓ Feature importances")

    # LOSO per-subject barplot
    if loso_subject_df is not None:
        from utils.plotting_utils import plot_loso_subject_metrics
        plot_loso_subject_metrics(loso_subject_df, dimension_results_path, filename_base)
        print("  ✓ LOSO per-subject metrics")

    # SHAP feature-importance bar chart (mean|SHAP| across all true runs).
    # The native shap.plots.beeswarm rendering (the correctly row/column-aligned
    # SHAP-vs-feature-value plot) is produced separately by
    # scripts/generate_pipeline_plots.py, which reads each run's own paired
    # (shap_values, x_test) directly from its saved pkl rather than
    # reconstructing a color axis here (Fix FIND-002: a custom Plotly
    # beeswarm previously lived in this function and paired combined_shap
    # with a freshly re-sliced, unpaired copy of the raw feature matrix,
    # decorrelating color from the SHAP values it was supposed to explain).
    if shap_values_all_runs:
        valid_shap = [s for s in shap_values_all_runs if s is not None]
        if valid_shap:
            combined_shap = np.concatenate(valid_shap, axis=0)
            plot_shap_feature_importance(
                combined_shap, feature_names,
                dimension_results_path, filename_base,
                max_display=top_n_features_plot,
            )
            print("  ✓ SHAP feature-importance plot")

    print(f"  All plots saved to: {dimension_results_path}")


# =============================================================================
# MAIN ANALYSIS — LOSO
# =============================================================================

def run_distribution_analysis(
    dimension: str,
    df: pd.DataFrame,
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    feature_cols: list,
    config: dict,
    positive_class_name: str = "ON-task",
    negative_class_name: str = "OFF-task",
    n_runs: int = 20,
    results_path: str = "results/MW_Classification/",
    model_type: str = "lr",
    class_weight=None,
    scale_pos_weight=None,
    use_smote: bool = False,
    oversampling_method: str = "SMOTE",
    oversampling_scope: str = "global",
    k: int = 20,
    rf_params: dict = None,
    xgb_params: dict = None,
    lr_params: dict = None,
    ocsvm_params: dict = None,
    iforest_params: dict = None,
    oneclass_target: str = "minority",
    top_n_features_plot: int = 20,
    save_pickle: bool = False,
    save_csv: bool = True,
    save_probabilities: bool = True,
    save_plots: bool = True,
    save_shap: bool = False,
    plot_style: str = "seaborn-v0_8",
    verbose: bool = True,
    feature_selection_method: str = "mrmr",
    scaler: str = "standard",
    use_pca: bool = False,
    pca_n_components: int = None,
    pca_type: str = "standard",
    pca_kernel: str = "rbf",
    smote_k_neighbors: int = 5,
    sample_weights=None,
    logger=None,
    cv_n_jobs: int = -1,
    run_idx_offset: int = 0,
    total_n_runs: int = None,
) -> pd.DataFrame:
    """
    Run n_runs LOSO classifications with different random seeds.

    Each run uses the same data split (LOSO = fixed by subject membership) but
    a different random seed for model initialization, feature selection, and
    SMOTE (if enabled). The distribution of metrics across runs quantifies
    variance due to non-deterministic components.

    Parameters
    ----------
    dimension : str
        Contrast name (e.g., "ON_vs_OFF").
    df : pd.DataFrame
        Prepared data with probe-level metadata (subject, task, probe_number, onoff).
    X : pd.DataFrame
        Feature matrix (n_samples × n_features).
    y : pd.Series
        Binary classification target.
    groups : pd.Series
        Subject ID per sample — defines LOSO folds.
    feature_cols : list
        Feature column names.
    config : dict
        Full pipeline configuration.
    positive_class_name, negative_class_name : str
        Human-readable class label names for plots.
    n_runs : int
        Number of LOSO passes (each with a different seed).
    results_path : str
        Root results directory.
    model_type : str
        Classifier type: 'rf', 'xgb', or 'lr'.
    class_weight : str or None
        Class weight for RF/LR classifiers.
    scale_pos_weight : float or 'auto'
        Class weight for XGBoost.
    use_smote : bool
        Whether to apply SMOTE oversampling.
    oversampling_method : str
        SMOTE variant.
    oversampling_scope : str
        'global' (in pipeline) or 'within' (per subject).
    k : int
        Number of features to select.
    rf_params, xgb_params, lr_params : dict
        Model-specific hyperparameter overrides.
    top_n_features_plot : int
        How many top features to show in importance plots.
    save_pickle : bool
        Save full results dict as .pkl.
    save_csv : bool
        Save per-subject and summary CSV files.
    save_probabilities : bool
        Save per-probe predicted probabilities.
    save_plots : bool
        Generate and save visualizations.
    save_shap : bool
        Compute and save SHAP values (RF/XGB only, slow).
    plot_style : str
        Matplotlib style name.
    verbose : bool
        Print detailed progress.
    feature_selection_method : str
        Feature ranking method.
    scaler : str
        Scaler type.
    use_pca, pca_n_components, pca_type, pca_kernel
        PCA configuration.
    sample_weights : np.ndarray, optional
        Per-trial training weights (e.g. confidence-based), aligned to (X, y,
        groups). Forwarded to ``run_model_pipeline_cv``; only the training fold
        is weighted. ``None`` reproduces the unweighted pipeline exactly.
    logger : AnalysisLogger, optional
        Logger for warning capture.

    Returns
    -------
    pd.DataFrame
        Per-subject (LOSO fold) results from the last/only run, with one
        row per subject and columns: mean_auc, mean_balanced_accuracy,
        mean_auprc, mean_mcc, etc.
    """
    n_subjects = len(np.unique(groups))
    # results_path already includes both feature_type (family) and model_type components
    # (built by build_results_path). Do not append model_type again.
    dimension_results_path = results_path
    Path(dimension_results_path).mkdir(parents=True, exist_ok=True)
    # Consolidated outputs go directly in dimension_results_path (no summaries/ subdir).
    summaries_dir = dimension_results_path

    # When total_n_runs is given (SLURM parallel mode), generate all seeds from
    # the global RNG and select the slice for this job — guarantees the same seed
    # for run N regardless of whether jobs run sequentially or in parallel.
    _total = total_n_runs if total_n_runs is not None else n_runs
    _all_states = np.random.default_rng(config.get("random_seed", 42)).integers(
        1, 10000, size=max(_total, run_idx_offset + n_runs)
    )
    random_states = _all_states[run_idx_offset: run_idx_offset + n_runs]

    # Use total_n_runs for consolidated filename so all jobs write comparable files.
    filename_base = _build_filename_base(model_type, _total)

    print(f"\n*** LOSO: {n_runs} run(s) × {n_subjects} subjects ***")
    print(f"Class mapping: 1={positive_class_name}, 0={negative_class_name}")

    # Show top feature-target correlations as a sanity check
    correlations = _compute_feature_correlations(X, y)
    if verbose and correlations:
        print("\nTop 10 feature-target correlations:")
        for col, corr in correlations[:10]:
            print(f"  {col}: {corr:.3f}")
    if not correlations or all(c < 0.1 for _, c in correlations[:10]):
        print("WARNING: No strong feature-target correlations — model may struggle.")

    # LMM preview on the full dataset (sanity check only — CV refits per fold on train-only)
    if verbose and feature_selection_method in ("lmm_encoding", "lmm", "lmm_decoding"):
        lmm_method = "decoding" if feature_selection_method in ("lmm", "lmm_decoding") else "encoding"
        print(f"\nComputing LMM {lmm_method} preview on full dataset ({X.shape[1]} features)...")
        lmm_preview = _compute_lmm_encoding_preview(
            X, y, groups, method=lmm_method,
            n_jobs=config.get("lmm_n_jobs", -1),
        )
        print(f"Top 10 LMM {lmm_method} features (lowest p-values, on full data — not the CV selection):")
        for col, pval in lmm_preview[:10]:
            print(f"  {col}: p={pval:.3e}")
        n_sig = sum(1 for _, p in lmm_preview if p < 0.05)
        print(f"  → {n_sig}/{len(lmm_preview)} features with p<0.05 (uncorrected, full-data preview)")

    feature_names = X.columns
    shap_values_all_runs = []
    all_results = []
    start_time = time.time()
    _dim_contrast_cfg = config.get("label_contrasts", {}).get(dimension, {})
    _label_col = _dim_contrast_cfg.get("column_name") or _dim_contrast_cfg.get("label_source", "onoff")

    for run_idx, random_state in enumerate(
        tqdm(random_states, desc=f"LOSO {model_type} [{dimension}]")
    ):
        run_start = time.time()
        run_dir = get_run_dir(
            dimension_results_path, run_idx, n_runs,
            run_idx_offset=run_idx_offset, total_n_runs=_total,
        )

        ctx = logger.capture_warnings(f"Run {run_idx}") if logger else nullcontext()
        with ctx:
            run_results = run_model_pipeline_cv(
                X=X,
                y=y,
                groups=groups,
                model_type=model_type,
                fixed_random_state=int(random_state),
                class_weight=class_weight,
                scale_pos_weight=scale_pos_weight,
                use_smote=use_smote,
                oversampling_method=oversampling_method,
                oversampling_scope=oversampling_scope,
                k=k,
                rf_params=rf_params,
                xgb_params=xgb_params,
                lr_params=lr_params,
                ocsvm_params=ocsvm_params,
                iforest_params=iforest_params,
                oneclass_target=oneclass_target,
                feature_selection_method=feature_selection_method,
                scaler=scaler,
                scale_by_participant=config.get("scale_by_participant", "none"),
                lmm_n_jobs=config.get("lmm_n_jobs", 1),
                lmm_prefilter_factor=config.get("lmm_prefilter_factor", 0),
                use_pca=use_pca,
                pca_n_components=pca_n_components,
                pca_type=pca_type,
                pca_kernel=pca_kernel,
                smote_k_neighbors=smote_k_neighbors,
                sample_weights=sample_weights,
                cv_n_jobs=cv_n_jobs,
            )

        if run_results.empty:
            print(f"  Skipping run {run_idx}: empty results")
            continue

        feature_importances = run_results["feature_importances"].values[0]

        # SHAP computation (RF/XGB/LR, opt-in)
        if save_shap and model_type in ("rf", "xgb", "lr"):
            shap_result = _save_shap_values(
                run_results, X, feature_names, run_dir, filename_base, model_type,
                groups=groups,
                scale_by_participant=config.get("scale_by_participant", "none"),
                scaler_type=config.get("scaler", "standard"),
                y=y,
            )
            if shap_result is not None:
                shap_values_all_runs.append(shap_result[0])

        # Save per-run files
        if save_csv:
            _save_run_results(
                run_results, run_dir, filename_base,
                feature_names, feature_importances, save_pickle,
            )
            # Per-subject LOSO metrics for this specific run — needed for
            # subject-level distribution plots comparing true vs permuted.
            _loso_sub_raw = (
                run_results["loso_subject_metrics"].values[0]
                if "loso_subject_metrics" in run_results.columns
                else None
            )
            if _loso_sub_raw is not None:
                pd.DataFrame(_loso_sub_raw).to_csv(
                    os.path.join(run_dir, f"{filename_base}_loso_subject_metrics.csv"),
                    index=False,
                )
            _save_probabilities(run_results, df, run_dir, filename_base, run_idx, raw_score_col=_label_col)

        all_results.append(_extract_run_summary(
            run_results, run_idx, int(random_state), dimension, n_subjects,
            feature_names, feature_importances,
            positive_class_name, negative_class_name,
        ))

        print(f"  [Run {run_idx+1}/{n_runs}] {time.time()-run_start:.1f}s "
              f"| AUC={run_results['mean_auc'].values[0]:.3f} "
              f"| MCC={run_results['mean_mcc'].values[0]:.3f}")

    total_time = time.time() - start_time
    print(f"\nTotal: {total_time:.1f}s ({total_time/60:.1f}min)")
    if logger:
        logger.log(f"Done in {total_time:.1f}s")

    if not all_results:
        print("No successful runs.")
        return pd.DataFrame()

    # Consolidate sample-level predictions across true_runs/
    if save_probabilities and n_runs > 1:
        true_runs_path = os.path.join(dimension_results_path, "true_runs")
        _consolidate_sample_predictions(true_runs_path, filename_base, output_dir=dimension_results_path)

    # ── Per-run summary DataFrame (one row per run) ──────────────────────────
    # Primary return value and reference for permutation-test statistics.
    # ALL n_runs passes are included — not just the last one.
    _exclude_run = {
        "fold_aucs", "fold_auprcs", "fold_mccs", "fold_bal_accs",
        "fold_cms", "fold_tprs", "fold_fprs",
        "fold_precisions_curve", "fold_recalls_curve",
        "loso_subject_metrics", "fold_details",
        "feature_importances",  # added as expanded columns below
    }
    results_df = pd.DataFrame([
        {k: v for k, v in r.items() if k not in _exclude_run}
        for r in all_results
    ])
    # Per-run feature importances as individual columns (enables mean ± std across runs)
    for run_row_idx, r in enumerate(all_results):
        fi = r.get("feature_importances", [])
        for i, feat in enumerate(feature_names):
            results_df.loc[run_row_idx, f"importance_{feat}"] = fi[i] if i < len(fi) else np.nan

    if save_csv:
        results_df.to_csv(
            os.path.join(summaries_dir, f"{filename_base}_runs_summary.csv"), index=False
        )

    # ── Per-subject metrics aggregated across ALL runs ────────────────────────
    # Each subject gets one row with metrics averaged over all n_runs passes.
    _all_subject_rows = []
    for r in all_results:
        for sm in (r.get("loso_subject_metrics") or []):
            _all_subject_rows.append({**sm, "run_idx": r["run_idx"]})

    loso_subject_df = None
    if _all_subject_rows:
        _subject_df_all = pd.DataFrame(_all_subject_rows)
        _num_metrics = ["auc", "auprc", "mcc", "balanced_accuracy", "precision", "recall", "f1"]
        _existing = [m for m in _num_metrics if m in _subject_df_all.columns]
        loso_subject_df = (
            _subject_df_all.groupby("subject")[_existing].mean().reset_index()
        )
        if save_csv:
            # Full cross-run detail (one row per run × subject)
            _subject_df_all.to_csv(
                os.path.join(summaries_dir, f"{filename_base}_loso_subject_metrics_all_runs.csv"),
                index=False,
            )
            # Averaged summary (one row per subject)
            loso_subject_df.to_csv(
                os.path.join(summaries_dir, f"{filename_base}_loso_subject_metrics.csv"),
                index=False,
            )
        print(f"\nPer-subject LOSO summary ({len(loso_subject_df)} subjects, "
              f"averaged over {len(all_results)} run(s)):")
        for metric in ["auc", "auprc", "mcc", "balanced_accuracy"]:
            if metric in loso_subject_df.columns:
                vals = loso_subject_df[metric]
                print(f"  {metric.upper()}: {vals.mean():.4f} ± {vals.std():.4f}")

    # ── Fold-level metrics across ALL runs (for subject-level distributions) ──
    _all_fold_rows = []
    for r in all_results:
        fold_aucs  = r.get("fold_aucs",  [])
        fold_aprcs = r.get("fold_auprcs", [])
        fold_mccs  = r.get("fold_mccs",  [])
        fold_bacs  = r.get("fold_bal_accs", [])
        for sm in (r.get("loso_subject_metrics") or []):
            fi = sm.get("fold_idx", 0)
            _all_fold_rows.append({
                "run_idx":            r["run_idx"],
                "subject":            sm.get("subject"),
                "fold_idx":           fi,
                "auc":                fold_aucs[fi]  if fi < len(fold_aucs)  else np.nan,
                "auprc":              fold_aprcs[fi] if fi < len(fold_aprcs) else np.nan,
                "mcc":                fold_mccs[fi]  if fi < len(fold_mccs)  else np.nan,
                "balanced_accuracy":  fold_bacs[fi]  if fi < len(fold_bacs)  else np.nan,
            })
    if _all_fold_rows and save_csv:
        pd.DataFrame(_all_fold_rows).to_csv(
            os.path.join(summaries_dir, f"{filename_base}_fold_metrics_all_runs.csv"),
            index=False,
        )

    if save_pickle:
        with open(os.path.join(dimension_results_path, f"{filename_base}_detailed.pkl"), "wb") as f:
            pickle.dump(all_results, f)

    if save_plots and all_results:
        _generate_plots(
            all_results, results_df, feature_names, dimension,
            dimension_results_path, filename_base, top_n_features_plot,
            positive_class_name, negative_class_name,
            loso_subject_df, shap_values_all_runs, X,
        )

    print(f"\nDone. Results → {dimension_results_path}")
    return results_df, all_results, shap_values_all_runs


# =============================================================================
# PERMUTATION TESTING
# =============================================================================

def run_permutation_distribution_analysis(
    dimension: str,
    df: pd.DataFrame,
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    feature_cols: list,
    config: dict,
    positive_class_name: str = "ON-task",
    negative_class_name: str = "OFF-task",
    n_permutations: int = 100,
    results_path: str = "results/MW_Classification/",
    model_type: str = "lr",
    class_weight=None,
    scale_pos_weight=None,
    use_smote: bool = False,
    oversampling_method: str = "SMOTE",
    oversampling_scope: str = "global",
    k: int = 20,
    rf_params: dict = None,
    xgb_params: dict = None,
    lr_params: dict = None,
    ocsvm_params: dict = None,
    iforest_params: dict = None,
    oneclass_target: str = "minority",
    top_n_features_plot: int = 20,
    save_pickle: bool = False,
    save_csv: bool = True,
    save_probabilities: bool = True,
    save_plots: bool = True,
    save_shap: bool = False,
    plot_style: str = "seaborn-v0_8",
    verbose: bool = True,
    feature_selection_method: str = "mrmr",
    scaler: str = "standard",
    use_pca: bool = False,
    pca_n_components: int = None,
    pca_type: str = "standard",
    pca_kernel: str = "rbf",
    true_auc_list=None,
    true_bal_acc_list=None,
    true_auprc_list=None,
    true_mcc_list=None,
    permutation_scope: str = "global",
    smote_k_neighbors: int = 5,
    sample_weights=None,
    logger=None,
    n_runs: int = 1,
    n_perm_jobs: int = 1,
    cv_n_jobs: int = 1,
    permutation_seed: int = None,
    perm_idx_offset: int = 0,
    total_n_permutations: int = None,
):
    """
    Run permutation test for LOSO classification.

    Labels are shuffled n_permutations times and the full LOSO pipeline is
    re-run each time. The resulting null distribution is compared to the true
    classification metrics to estimate empirical p-values.

    Notes
    -----
    ``permutation_seed`` (when not None) overrides ``config['random_seed']``
    for the RNG that produces both the permuted-label sequences and the
    per-permutation random states. Type I error simulations MUST pass a
    sim-specific seed here; otherwise every simulation receives the same
    permuted-label sequence, which silently couples the simulations and
    breaks the independence assumption behind the empirical FPR estimate.

    Parameters
    ----------
    Same as run_distribution_analysis, plus:
    n_permutations : int
        Number of label permutations.
    true_auc_list, true_bal_acc_list, true_auprc_list, true_mcc_list
        True metric values from the real analysis (for p-value computation).
    permutation_scope : str
        'global': shuffle all labels together.
        'within_subject': shuffle independently per subject (preserves class
        proportions per subject — tighter null distribution).
    sample_weights : np.ndarray, optional
        Per-trial training weights forwarded to every permutation so the null
        uses the same pipeline as the real run. Weights stay attached to trials
        (they index confidence, not labels), so label shuffling leaves them
        correctly aligned to X.

    Returns
    -------
    tuple
        (results_df, perm_summary, all_results, shap_values_all_runs):
        - results_df: one row per permutation with mean metrics
        - perm_summary: dict with empirical p-values
        - all_results: full list of per-permutation dicts
        - shap_values_all_runs: list of SHAP matrices per permutation (if save_shap=True)
    """
    if n_permutations < 1:
        print("No permutation runs requested.")
        return pd.DataFrame(), {}, [], []

    true_auc_list = np.asarray(true_auc_list or [])
    true_bal_acc_list = np.asarray(true_bal_acc_list or [])
    true_auprc_list = np.asarray(true_auprc_list or [])
    true_mcc_list = np.asarray(true_mcc_list or [])

    # Permuted runs live inside results_path (which already includes model_type).
    perm_base_path = os.path.join(results_path, "permuted_runs")
    Path(perm_base_path).mkdir(parents=True, exist_ok=True)

    _total_perms = total_n_permutations if total_n_permutations is not None else n_permutations
    filename_base = f"{model_type}_permutation_{_total_perms}perms"
    feature_names = X.columns
    n_subjects = len(np.unique(groups))
    _perm_seed = permutation_seed if permutation_seed is not None else config.get("random_seed", 42)
    rng = np.random.default_rng(_perm_seed)
    all_results = []
    shap_values_all_runs = []

    start_time = time.time()
    print(f"\n*** LOSO Permutation: {n_permutations} runs × {n_subjects} subjects ***")
    print(f"Permutation scope: {permutation_scope}")

    # Build shared CV kwargs dict for the job helper.
    _loso_cv_kwargs = dict(
        model_type=model_type,
        class_weight=class_weight,
        scale_pos_weight=scale_pos_weight,
        use_smote=use_smote,
        oversampling_method=oversampling_method,
        oversampling_scope=oversampling_scope,
        k=k,
        rf_params=rf_params,
        xgb_params=xgb_params,
        lr_params=lr_params,
        ocsvm_params=ocsvm_params,
        iforest_params=iforest_params,
        oneclass_target=oneclass_target,
        feature_selection_method=feature_selection_method,
        scaler=scaler,
        scale_by_participant=config.get("scale_by_participant", "none"),
        lmm_n_jobs=config.get("lmm_n_jobs", 1),
        lmm_prefilter_factor=config.get("lmm_prefilter_factor", 0),
        use_pca=use_pca,
        pca_n_components=pca_n_components,
        pca_type=pca_type,
        pca_kernel=pca_kernel,
        cv_n_jobs=cv_n_jobs,
        smote_k_neighbors=smote_k_neighbors,
        sample_weights=sample_weights,
    )

    # Pre-generate permuted labels and random seeds deterministically from the RNG.
    # When perm_idx_offset > 0 (SLURM parallel mode), advance the RNG through the
    # first perm_idx_offset permutations so job N gets bit-identical labels to a
    # sequential run that generated all permutations in order.
    _total_perms = total_n_permutations if total_n_permutations is not None else n_permutations
    perm_inputs = []
    for _i in range(perm_idx_offset + n_permutations):
        if permutation_scope in ("within_subject", "within"):
            y_perm = y.groupby(groups, group_keys=False).transform(
                lambda x: rng.permutation(x.values)
            )
        else:
            y_perm = pd.Series(rng.permutation(y.values), index=y.index)
        random_state = int(rng.integers(1, 10000))
        if _i >= perm_idx_offset:
            global_perm_idx = _i
            perm_run_dir = os.path.join(perm_base_path, f"run_{global_perm_idx}")
            Path(perm_run_dir).mkdir(parents=True, exist_ok=True)
            perm_inputs.append((_i, y_perm, random_state, perm_run_dir))

    print(f"  Dispatching {n_permutations} permutation jobs (n_perm_jobs={n_perm_jobs}, cv_n_jobs={cv_n_jobs})")

    _perm_contrast_cfg = config.get("label_contrasts", {}).get(dimension, {})
    _perm_label_col = _perm_contrast_cfg.get("column_name") or _perm_contrast_cfg.get("label_source", "onoff")
    _job_kwargs = dict(
        X=X, groups=groups, df=df, feature_names=feature_names,
        filename_base=filename_base, cv_kwargs=_loso_cv_kwargs,
        save_shap=save_shap, save_csv=save_csv,
        save_probabilities=save_probabilities, dimension=dimension,
        positive_class_name=positive_class_name,
        negative_class_name=negative_class_name, n_subjects=n_subjects,
        raw_score_col=_perm_label_col,
    )

    if n_perm_jobs == 1:
        for run_idx, y_perm, random_state, perm_run_dir in perm_inputs:
            perm_start = time.time()
            result = _run_permutation_loso_job(
                run_idx=run_idx, y_perm=y_perm, random_state=random_state,
                perm_run_dir=perm_run_dir, **_job_kwargs,
            )
            elapsed = time.time() - perm_start
            status = "ok" if result is not None else "empty"
            print(f"  [Perm {run_idx+1}/{n_permutations}] {elapsed:.1f}s | {status}")
            if result is not None:
                all_results.append(result)
    else:
        job_results = Parallel(n_jobs=n_perm_jobs, backend='loky')(
            delayed(_run_permutation_loso_job)(
                run_idx=run_idx, y_perm=y_perm, random_state=random_state,
                perm_run_dir=perm_run_dir, **_job_kwargs,
            )
            for run_idx, y_perm, random_state, perm_run_dir in perm_inputs
        )
        for result in job_results:
            if result is not None:
                all_results.append(result)

    total_time = time.time() - start_time
    print(f"\nPermutation total: {total_time:.1f}s ({total_time/60:.1f}min)")

    if not all_results:
        print("No successful permutation runs.")
        return pd.DataFrame(), {}, [], []

    # Build results DataFrame
    exclude_keys = {
        "fold_aucs", "fold_auprcs", "fold_mccs", "fold_bal_accs",
        "feature_importances", "fold_cms", "fold_tprs", "fold_fprs",
        "fold_precisions_curve", "fold_recalls_curve",
        "loso_subject_metrics", "fold_details",
    }
    results_df = pd.DataFrame([
        {k: v for k, v in r.items() if k not in exclude_keys}
        for r in all_results
    ])
    for i, feat in enumerate(feature_names):
        results_df[f"importance_{feat}"] = [r["feature_importances"][i] for r in all_results]

    # Consolidated permutation summary goes to results_path (already the model_type root).
    perm_consolidated_dir = results_path
    if save_csv:
        results_df.to_csv(
            os.path.join(perm_consolidated_dir, f"{filename_base}_summary.csv"), index=False
        )

    if save_pickle:
        with open(os.path.join(perm_consolidated_dir, f"{filename_base}_detailed.pkl"), "wb") as f:
            pickle.dump(all_results, f)

    # P-value computation
    metrics_to_check = [
        ("AUC", "mean_auc", true_auc_list),
        ("Balanced Accuracy", "mean_balanced_accuracy", true_bal_acc_list),
        ("AUPRC", "mean_auprc", true_auprc_list),
        ("MCC", "mean_mcc", true_mcc_list),
    ]
    perm_summary = {}
    print(f"\nPermutation test results for {dimension}:")

    for metric_name, col_name, true_scores in metrics_to_check:
        if col_name not in results_df.columns or len(true_scores) == 0:
            continue
        perm_scores = results_df[col_name].values
        true_mean = np.mean(true_scores)
        perm_mean = np.mean(perm_scores)
        perm_std = np.std(perm_scores)
        # Add-one (Phipson & Smyth 2010): an unbiased, never-zero permutation
        # p-value. Smallest reportable p is 1/(n_perm+1).
        p_value = (1 + np.sum(perm_scores >= true_mean)) / (1 + len(perm_scores))
        print(f"  {metric_name}: null={perm_mean:.4f}±{perm_std:.4f}, "
              f"true={true_mean:.4f}, p={p_value:.4f}")
        perm_summary[f"perm_{col_name}"] = perm_mean
        perm_summary[f"perm_{col_name}_std"] = perm_std
        perm_summary[f"p_{col_name}"] = p_value

    # Plot permutation distributions
    if save_plots:
        from scipy import stats as scipy_stats
        results_for_plotting = {}
        for metric_name, col_name, true_scores in metrics_to_check:
            if col_name not in results_df.columns or len(true_scores) == 0:
                continue
            perm_scores = results_df[col_name].values
            _, mwu_p = scipy_stats.mannwhitneyu(true_scores, perm_scores, alternative="greater")
            empirical_p = (1 + np.sum(perm_scores >= np.mean(true_scores))) / (1 + len(perm_scores))
            results_for_plotting[metric_name] = {
                "true_values": true_scores,
                "perm_values": perm_scores,
                "p_value": mwu_p,
                "empirical_p": empirical_p,
            }

        if results_for_plotting:
            from utils.plotting_utils import plot_consolidated_permutation_results
            # Plots comparing true vs permuted distributions go to the model_type root.
            plot_consolidated_permutation_results(
                results_dict=results_for_plotting,
                dimension=dimension,
                model_type=model_type,
                save_path=perm_consolidated_dir,
                filename_base=filename_base,
            )
            print("  ✓ Permutation distribution plots")

    return results_df, perm_summary, all_results, shap_values_all_runs


# =============================================================================
# PARALLEL JOB HELPERS — WITHIN-SUBJECT
# Must be module-level (not closures) so joblib loky backend can pickle them.
# =============================================================================

def _run_subject_cv_job(
    subject: str,
    X_sub: pd.DataFrame,
    y_sub: pd.Series,
    y_raw_sub,               # pd.Series | None
    groups_sub,              # pd.Series | None
    df_sub: pd.DataFrame,
    run_idx: int,
    random_state: int,
    min_samples_per_class: int,
    min_minority_ratio: float,
    cv_kwargs: dict,
    save_shap: bool,
    save_probabilities: bool,
    feature_names,           # pd.Index
    shap_save_dir: str,
    shap_filename_prefix: str,
    raw_score_col: str = "onoff",
) -> "dict | None":
    """
    Run within-subject CV for a single subject.

    Parameters
    ----------
    subject : str
        Subject identifier (for output labelling only).
    X_sub, y_sub : pd.DataFrame, pd.Series
        Feature matrix and binary labels for this subject.
    y_raw_sub : pd.Series | None
        Continuous onoff scores for per-fold re-binarization
        (within_subject_median contracts only). None otherwise.
    groups_sub : pd.Series | None
        Task labels for GroupKFold; None for other strategies.
    df_sub : pd.DataFrame
        Full raw rows for this subject (used to extract metadata for
        per-sample prediction records).
    run_idx : int
        Run index — written into every output record.
    random_state : int
        RNG seed for this run (pre-generated by the caller).
    min_samples_per_class, min_minority_ratio : int, float
        Filtering thresholds; returns None if not met.
    cv_kwargs : dict
        All keyword arguments forwarded to ``run_within_subject_cv``
        except ``X``, ``y``, ``groups``, ``fixed_random_state``, and
        ``y_raw`` (which are passed explicitly).
    save_shap : bool
        Whether to compute and save SHAP values.
    save_probabilities : bool
        Whether to build per-fold and per-sample prediction records.
    feature_names : pd.Index
        Feature column names (for SHAP).
    shap_save_dir : str
        Directory where SHAP artefacts are written.
    shap_filename_prefix : str
        Filename prefix for SHAP files (subject label appended).

    Returns
    -------
    dict | None
        Keys: ``sub_metrics``, ``importances``, ``shap_vals``,
        ``fold_predictions``, ``sample_predictions``.
        Returns None if the subject was filtered out or CV returned
        no results.
    """
    counts = y_sub.value_counts()
    if len(counts) < 2 or counts.min() < min_samples_per_class:
        return None
    if counts.min() / len(y_sub) < min_minority_ratio:
        return None

    sub_results = run_within_subject_cv(
        X=X_sub,
        y=y_sub,
        groups=groups_sub,
        fixed_random_state=random_state,
        y_raw=y_raw_sub,
        **cv_kwargs,
    )
    if sub_results.empty:
        return None

    auc = sub_results['mean_auc'].values[0]
    sub_metrics = {
        'run_idx': run_idx,
        'subject': subject,
        'mean_auc': auc,
        'mean_auprc': sub_results['mean_auprc'].values[0],
        'mean_mcc': sub_results['mean_mcc'].values[0],
        'mean_balanced_accuracy': sub_results['mean_balanced_accuracy'].values[0],
        'mean_precision': sub_results['mean_precision'].values[0],
        'mean_recall': sub_results['mean_recall'].values[0],
        'mean_f1': sub_results['mean_f1'].values[0],
        'n_samples': len(y_sub),
        'n_positive': int(y_sub.sum()),
        'n_negative': int(len(y_sub) - y_sub.sum()),
    }
    importances = sub_results['feature_importances'].values[0]

    shap_vals  = None
    shap_x     = None
    shap_y_true = None
    if save_shap and cv_kwargs.get('model_type') in ("rf", "xgb", "lr"):
        shap_result = _save_shap_values(
            sub_results, X_sub, feature_names, shap_save_dir,
            f"{shap_filename_prefix}_{subject}", cv_kwargs['model_type'],
            y=y_sub,
        )
        if shap_result is not None:
            shap_vals, shap_x, shap_y_true = shap_result

    fold_predictions: list = []
    sample_predictions: list = []
    if save_probabilities:
        fold_details = sub_results['fold_details'].values[0]
        for fold in fold_details:
            fold_predictions.append({
                "run_idx": run_idx,
                "subject": subject,
                "fold_idx": fold.get("fold_idx"),
                "y_true": fold.get("y_true"),
                "y_pred": fold.get("y_pred"),
                "y_proba": fold.get("y_proba", []),
                **{f"label_{k}_pct": v for k, v in fold.get("label_percentages", {}).items()},
            })
            test_indices = fold.get("test_indices", [])
            y_true_list = fold.get("y_true", [])
            y_pred_list = fold.get("y_pred", [])
            y_proba_list = fold.get("y_proba", [])
            for i, idx in enumerate(test_indices):
                if idx < len(df_sub):
                    row_data = df_sub.iloc[idx]
                    sample_predictions.append({
                        "run_idx": run_idx,
                        "subject": subject,
                        "fold_idx": fold.get("fold_idx"),
                        "sample_idx": idx,
                        "task": row_data.get("task", ""),
                        "probe_number": row_data.get("probe_number", ""),
                        raw_score_col: row_data.get(raw_score_col, ""),
                        "confidence": row_data.get("confidence", ""),
                        "y_true": y_true_list[i] if i < len(y_true_list) else None,
                        "y_pred": y_pred_list[i] if i < len(y_pred_list) else None,
                        "y_proba": y_proba_list[i] if i < len(y_proba_list) else None,
                    })

    return {
        'sub_metrics':    sub_metrics,
        'importances':    importances,
        'shap_vals':      shap_vals,
        'shap_x':         shap_x,
        'shap_y_true':    shap_y_true,
        'fold_predictions':   fold_predictions,
        'sample_predictions': sample_predictions,
    }


def _run_permutation_ws_job(
    run_idx: int,
    y_perm: pd.Series,
    random_state: int,
    perm_run_dir: str,
    X: pd.DataFrame,
    subjects: pd.Series,
    tasks: pd.Series,
    df: pd.DataFrame,
    feature_names,           # pd.Index
    filename_base: str,
    min_samples_per_class: int,
    min_minority_ratio: float,
    cv_kwargs: dict,
    use_fold_rebinarize: bool,
    label_col: str,
    save_shap: bool,
    save_csv: bool,
    save_probabilities: bool,
    dimension: str,
    y_raw_perm: "pd.Series | None" = None,
    n_subject_jobs: int = 1,
) -> "dict | None":
    """
    Run one full within-subject permutation pass.

    Iterates over all subjects using the pre-shuffled ``y_perm`` labels,
    accumulates per-subject metrics, and optionally saves per-run CSVs.

    Parameters
    ----------
    run_idx : int
        Permutation index.
    y_perm : pd.Series
        Pre-shuffled binary labels (shuffled within-subject by the caller).
    random_state : int
        RNG seed for this permutation (pre-generated by the caller).
    perm_run_dir : str
        Per-permutation output directory (pre-created by the caller).
    X : pd.DataFrame
        Full feature matrix (all subjects).
    subjects : pd.Series
        Subject ID per sample.
    tasks : pd.Series
        Task label per sample (used only for GroupKFold).
    df : pd.DataFrame
        Full raw dataframe (for per-sample prediction metadata).
    feature_names : pd.Index
        Feature column names.
    filename_base : str
        Filename prefix for saved artefacts.
    min_samples_per_class, min_minority_ratio : int, float
        Per-subject filtering thresholds.
    cv_kwargs : dict
        Forwarded to ``run_within_subject_cv`` (same as ``_run_subject_cv_job``).
    use_fold_rebinarize : bool
        Whether to pass continuous ``y_raw`` for per-fold re-binarization.
    label_col : str
        Column in ``df`` holding continuous onoff scores (used when
        ``use_fold_rebinarize`` is True).
    save_shap : bool
        Whether to compute and save SHAP values.
    save_csv : bool
        Whether to write per-permutation CSV summaries.
    save_probabilities : bool
        Whether to write per-sample prediction CSVs.
    dimension : str
        Contrast label (written into summary CSV).

    Returns
    -------
    dict | None
        Keys: ``global_summary``, ``run_subject_results``.
        Returns None if no subjects passed filtering.
    """
    unique_subjects = np.unique(subjects)
    _fs_method = cv_kwargs.get('feature_selection_method', 'mrmr')
    _pass_groups = cv_kwargs.get('cv_strategy') == 'group_kfold' or _fs_method in ('lmm', 'lmm_encoding', 'lmm_decoding')

    # Build per-subject inputs using permuted labels — identical to the true-run
    # subject dispatch, so we can reuse _run_subject_cv_job with n_subject_jobs parallelism.
    subject_inputs = []
    for subject in unique_subjects:
        sub_mask = (subjects == subject)
        if use_fold_rebinarize:
            raw_series = y_raw_perm if y_raw_perm is not None else df[label_col]
            y_raw_sub = raw_series[sub_mask].reset_index(drop=True)
        else:
            y_raw_sub = None
        subject_inputs.append(dict(
            subject=subject,
            X_sub=X[sub_mask].reset_index(drop=True),
            y_sub=y_perm[sub_mask].reset_index(drop=True),
            y_raw_sub=y_raw_sub,
            groups_sub=tasks[sub_mask].reset_index(drop=True) if _pass_groups else None,
            df_sub=df[sub_mask].reset_index(drop=True),
        ))

    job_results = Parallel(n_jobs=n_subject_jobs, backend='loky')(
        delayed(_run_subject_cv_job)(
            subject=inp['subject'],
            X_sub=inp['X_sub'],
            y_sub=inp['y_sub'],
            y_raw_sub=inp['y_raw_sub'],
            groups_sub=inp['groups_sub'],
            df_sub=inp['df_sub'],
            run_idx=run_idx,
            random_state=random_state,
            min_samples_per_class=min_samples_per_class,
            min_minority_ratio=min_minority_ratio,
            cv_kwargs=cv_kwargs,
            save_shap=save_shap,
            save_probabilities=save_probabilities,
            feature_names=feature_names,
            shap_save_dir=perm_run_dir,
            shap_filename_prefix=f"{filename_base}_{run_idx}",
        )
        for inp in subject_inputs
    )

    run_subject_results: list = []
    importances_list: list = []
    run_sample_predictions: list = []

    for result in job_results:
        if result is None:
            continue
        run_subject_results.append(result['sub_metrics'])
        importances_list.append(result['importances'])
        run_sample_predictions.extend(result['sample_predictions'])

    if not run_subject_results:
        return None

    run_df = pd.DataFrame(run_subject_results)
    run_mean_importances = np.mean(importances_list, axis=0)

    if save_csv:
        run_df.to_csv(
            os.path.join(perm_run_dir, f"{filename_base}_ws_subject_metrics.csv"), index=False
        )
        summary_metrics = run_df[[c for c in run_df.columns if c.startswith('mean_')]].mean().to_dict()
        summary_metrics['run_idx'] = run_idx
        summary_metrics['dimension'] = dimension
        pd.DataFrame([summary_metrics]).to_csv(
            os.path.join(perm_run_dir, f"{filename_base}_summary.csv"), index=False
        )
        pd.DataFrame({"feature": feature_names, "importance": run_mean_importances}).to_csv(
            os.path.join(perm_run_dir, f"{filename_base}_feature_importances.csv"), index=False
        )
        if save_probabilities and run_sample_predictions:
            pd.DataFrame(run_sample_predictions).to_csv(
                os.path.join(perm_run_dir, f"{filename_base}_sample_predictions.csv"), index=False
            )

    global_summary = {
        'run_idx': run_idx,
        'mean_auc': run_df['mean_auc'].mean(),
        'mean_auprc': run_df['mean_auprc'].mean(),
        'mean_mcc': run_df['mean_mcc'].mean(),
    }
    return {
        'global_summary': global_summary,
        'run_subject_results': run_subject_results,
    }


def _run_permutation_loso_job(
    run_idx: int,
    y_perm: pd.Series,
    random_state: int,
    perm_run_dir: str,
    X: pd.DataFrame,
    groups: pd.Series,
    df: pd.DataFrame,
    feature_names,           # pd.Index
    filename_base: str,
    cv_kwargs: dict,
    save_shap: bool,
    save_csv: bool,
    save_probabilities: bool,
    dimension: str,
    positive_class_name: str,
    negative_class_name: str,
    n_subjects: int,
    raw_score_col: str = "onoff",
) -> "dict | None":
    """
    Run one full LOSO permutation pass.

    Calls ``run_model_pipeline_cv`` with the pre-shuffled ``y_perm``,
    saves per-permutation artefacts, and returns a summary dict suitable
    for ``_extract_run_summary``.

    Parameters
    ----------
    run_idx : int
        Permutation index.
    y_perm : pd.Series
        Pre-shuffled labels (global or within-subject, done by the caller).
    random_state : int
        RNG seed pre-generated by the caller.
    perm_run_dir : str
        Per-permutation output directory (pre-created by the caller).
    X : pd.DataFrame
        Full feature matrix.
    groups : pd.Series
        Subject ID per sample (used by the LOSO CV splitter).
    df : pd.DataFrame
        Raw dataframe (for per-sample prediction metadata).
    feature_names : pd.Index
        Feature column names.
    filename_base : str
        Filename prefix for saved artefacts.
    cv_kwargs : dict
        All keyword arguments forwarded to ``run_model_pipeline_cv``
        except ``X``, ``y``, ``groups``, and ``fixed_random_state``.
    save_shap : bool
        Whether to compute and save SHAP values.
    save_csv : bool
        Whether to write per-permutation CSV summaries.
    save_probabilities : bool
        Whether to write per-sample prediction CSVs.
    dimension : str
        Contrast label (included in summary dict).
    positive_class_name, negative_class_name : str
        Class labels forwarded to ``_extract_run_summary``.
    n_subjects : int
        Total number of subjects (forwarded to ``_extract_run_summary``).

    Returns
    -------
    dict | None
        Full run-summary dict (output of ``_extract_run_summary``), or
        None if ``run_model_pipeline_cv`` returned an empty DataFrame.
    """
    run_results = run_model_pipeline_cv(
        X=X,
        y=y_perm,
        groups=groups,
        fixed_random_state=random_state,
        **cv_kwargs,
    )
    if run_results.empty:
        return None

    feature_importances = run_results["feature_importances"].values[0]

    if save_shap and cv_kwargs.get('model_type') in ("rf", "xgb", "lr"):
        _save_shap_values(
            run_results, X, feature_names, perm_run_dir,
            f"{filename_base}_{run_idx}", cv_kwargs['model_type'],
            groups=groups,
        )

    if save_csv:
        run_results.to_csv(
            os.path.join(perm_run_dir, f"{filename_base}_summary.csv"), index=False
        )
        pd.DataFrame({
            "feature": feature_names,
            "importance": feature_importances,
        }).to_csv(
            os.path.join(perm_run_dir, f"{filename_base}_feature_importances.csv"), index=False
        )
        _perm_sub_raw = (
            run_results["loso_subject_metrics"].values[0]
            if "loso_subject_metrics" in run_results.columns
            else None
        )
        if _perm_sub_raw is not None:
            pd.DataFrame(_perm_sub_raw).to_csv(
                os.path.join(perm_run_dir, f"{filename_base}_loso_subject_metrics.csv"), index=False
            )

    if save_probabilities:
        _save_probabilities(run_results, df, perm_run_dir, filename_base, run_idx, raw_score_col=raw_score_col)

    return _extract_run_summary(
        run_results, run_idx, random_state, dimension, n_subjects,
        feature_names, feature_importances,
        positive_class_name, negative_class_name,
    )


# =============================================================================
# MAIN ANALYSIS — WITHIN-SUBJECT
# =============================================================================

def run_within_subject_distribution_analysis(
    dimension: str,
    df: pd.DataFrame,
    X: pd.DataFrame,
    y: pd.Series,
    subjects: pd.Series,
    tasks: pd.Series,
    feature_cols: list,
    config: dict,
    positive_class_name: str = "ON-task",
    negative_class_name: str = "OFF-task",
    n_runs: int = 1,
    results_path: str = "results/MW_Classification/",
    model_type: str = "lr",
    class_weight=None,
    scale_pos_weight=None,
    use_smote: bool = False,
    oversampling_method: str = "SMOTE",
    oversampling_scope: str = "within",
    cv_strategy: str = "stratified_kfold",
    cv_folds: int = 5,
    min_samples_per_class: int = 0,
    min_minority_ratio: float = 0.0,
    k: int = 20,
    rf_params: dict = None,
    xgb_params: dict = None,
    lr_params: dict = None,
    ocsvm_params: dict = None,
    iforest_params: dict = None,
    oneclass_target: str = "minority",
    top_n_features_plot: int = 20,
    save_pickle: bool = False,
    save_csv: bool = True,
    save_probabilities: bool = True,
    save_plots: bool = True,
    save_shap: bool = False,
    plot_style: str = "seaborn-v0_8",
    verbose: bool = True,
    feature_selection_method: str = "mrmr",
    scaler: str = "standard",
    use_pca: bool = False,
    pca_n_components: int = None,
    pca_type: str = "standard",
    pca_kernel: str = "rbf",
    n_subject_jobs: int = 1,
    cv_n_jobs: int = 1,
    lmm_n_jobs: int = 1,
    lmm_prefilter_factor: int = 0,
    smote_k_neighbors: int = 5,
    logger=None,
    run_idx_offset: int = 0,
    total_n_runs: int = None,
):
    """
    Run within-subject classification (independently per subject).
    Generates a single comprehensive output matching the LOSO structure,
    where each row represents one subject's CV performance.
    """
    if plot_style and save_plots:
        from utils.plotting_utils import set_plot_style
        set_plot_style(plot_style)

    # results_path already includes both feature_type (family) and model_type components.
    dimension_results_path = results_path
    Path(dimension_results_path).mkdir(parents=True, exist_ok=True)

    filename_base = _build_filename_base(model_type, n_runs, pipeline_label="ws")
    feature_names = X.columns
    unique_subjects = np.unique(subjects)

    # Labels are fixed at dataset-load time (within_subject_median computed once
    # on the full subject data). No per-fold re-binarization.
    _label_contrast_cfg = config.get("label_contrasts", {}).get(dimension, {})
    _split_method = _label_contrast_cfg.get("split_method", "threshold")
    _label_col = _label_contrast_cfg.get("column_name", "onoff")
    _use_fold_rebinarize = False

    # Reproducible seeds: generate all seeds from global RNG then slice for this job.
    _total_runs = total_n_runs if total_n_runs is not None else n_runs
    _all_rng_states = np.random.default_rng(config.get("random_seed", 42)).integers(
        1, 10000, size=max(_total_runs, run_idx_offset + n_runs)
    )

    filename_base_global = _build_filename_base(model_type, _total_runs, pipeline_label="ws")

    # Store results aggregated across runs
    all_runs_subject_metrics = []
    shap_values_all_runs = []

    start_time = time.time()
    if verbose:
        print(f"\n{'='*70}\nWithin-Subject Analysis: {dimension} ({model_type.upper()})\n{'='*70}")
        print(f"Algorithm: {model_type.upper()}")
        print(f"Features: {len(feature_names)}")
        print(f"Subjects config: {len(unique_subjects)} total")
        print(f"CV Strategy: {cv_strategy} ({cv_folds} folds)")
        print(f"Runs: {n_runs}")
        if _use_fold_rebinarize:
            print(f"  [within_subject_median] Per-fold re-binarization ACTIVE (no label leakage)")
        if feature_selection_method == "none":
            print(f"Feature selection : disabled (using all {len(feature_names)} features)")
        else:
            print(f"Feature selection : {feature_selection_method} (k={k}/{len(feature_names)}, refit per fold)")
        print(f"Results Path: {dimension_results_path}")
        print("-" * 70)

    # LMM preview on the full dataset (sanity check only — CV refits per fold on train-only)
    if verbose and feature_selection_method in ("lmm_encoding", "lmm", "lmm_decoding"):
        _lmm_method = "decoding" if feature_selection_method in ("lmm", "lmm_decoding") else "encoding"
        print(f"\nComputing LMM {_lmm_method} preview on full dataset ({X.shape[1]} features, grouped by subject)...")
        _lmm_preview = _compute_lmm_encoding_preview(
            X, y, subjects, method=_lmm_method, n_jobs=lmm_n_jobs,
        )
        print(f"Top 10 LMM {_lmm_method} features (lowest p-values, on full data — not the CV selection):")
        for _col, _pval in _lmm_preview[:10]:
            print(f"  {_col}: p={_pval:.3e}")
        _n_sig = sum(1 for _, p in _lmm_preview if p < 0.05)
        print(f"  → {_n_sig}/{len(_lmm_preview)} features with p<0.05 (uncorrected, full-data preview)")

    # LMM selectors need task labels as groups regardless of cv_strategy.
    # (Within-subject: (1|Task) controls for session effects; tasks provide 4 group levels.)
    _use_task_groups_for_lmm = feature_selection_method in ('lmm', 'lmm_encoding', 'lmm_decoding')

    # Build the cv_kwargs dict once — shared across all parallel subject jobs.
    _cv_kwargs = dict(
        cv_strategy=cv_strategy,
        cv_folds=cv_folds,
        model_type=model_type,
        use_smote=use_smote,
        oversampling_method=oversampling_method,
        oversampling_scope=oversampling_scope,
        class_weight=class_weight,
        scale_pos_weight=scale_pos_weight,
        k=k,
        rf_params=rf_params,
        xgb_params=xgb_params,
        lr_params=lr_params,
        ocsvm_params=ocsvm_params,
        iforest_params=iforest_params,
        oneclass_target=oneclass_target,
        feature_selection_method=feature_selection_method,
        scaler=scaler,
        use_pca=use_pca,
        pca_n_components=pca_n_components,
        pca_type=pca_type,
        pca_kernel=pca_kernel,
        cv_n_jobs=cv_n_jobs,
        lmm_n_jobs=lmm_n_jobs,
        lmm_prefilter_factor=lmm_prefilter_factor,
        smote_k_neighbors=smote_k_neighbors,
    )

    for _local_run_idx in range(n_runs):
        global_run_idx = _local_run_idx + run_idx_offset
        run_idx = global_run_idx  # keep variable name for downstream code
        run_start = time.time()
        random_state = int(_all_rng_states[global_run_idx])
        run_dir = os.path.join(dimension_results_path, "true_runs", f"run_{global_run_idx}")
        Path(run_dir).mkdir(parents=True, exist_ok=True)

        # Prepare per-subject inputs for parallel dispatch.
        subject_inputs = []
        for subject in unique_subjects:
            sub_mask = (subjects == subject)
            # Always pass task labels when LMM is used (groups = task/session for (1|Task) model).
            # For group_kfold, also needed for CV splitting.
            _pass_groups = cv_strategy == 'group_kfold' or _use_task_groups_for_lmm
            subject_inputs.append(dict(
                subject=subject,
                X_sub=X[sub_mask].reset_index(drop=True),
                y_sub=y[sub_mask].reset_index(drop=True),
                y_raw_sub=df[sub_mask][_label_col].reset_index(drop=True) if _use_fold_rebinarize else None,
                groups_sub=tasks[sub_mask].reset_index(drop=True) if _pass_groups else None,
                df_sub=df[sub_mask].reset_index(drop=True),
            ))

        job_results = Parallel(n_jobs=n_subject_jobs, backend='loky')(
            delayed(_run_subject_cv_job)(
                subject=inp['subject'],
                X_sub=inp['X_sub'],
                y_sub=inp['y_sub'],
                y_raw_sub=inp['y_raw_sub'],
                groups_sub=inp['groups_sub'],
                df_sub=inp['df_sub'],
                run_idx=run_idx,
                random_state=random_state,
                min_samples_per_class=min_samples_per_class,
                min_minority_ratio=min_minority_ratio,
                cv_kwargs=_cv_kwargs,
                save_shap=save_shap,
                save_probabilities=save_probabilities,
                feature_names=feature_names,
                shap_save_dir=run_dir,
                shap_filename_prefix=filename_base,
                raw_score_col=_label_col,
            )
            for inp in subject_inputs
        )

        run_subject_results = []
        run_fold_predictions = []
        run_sample_predictions = []
        importances_list = []
        shap_list   = []
        shap_x_list = []
        shap_y_list = []
        valid_subjects = 0
        skipped_subjects = 0

        for inp, result in zip(subject_inputs, job_results):
            if result is None:
                skipped_subjects += 1
                if logger:
                    logger.warning(f"Skipping {inp['subject']}: filtered or empty CV.")
                continue
            valid_subjects += 1
            run_subject_results.append(result['sub_metrics'])
            all_runs_subject_metrics.append(result['sub_metrics'])
            importances_list.append(result['importances'])
            if result['shap_vals'] is not None:
                shap_list.append(result['shap_vals'])
                if result.get('shap_x') is not None:
                    shap_x_list.append(result['shap_x'])
                if result.get('shap_y_true') is not None:
                    shap_y_list.append(result['shap_y_true'])
            run_fold_predictions.extend(result['fold_predictions'])
            run_sample_predictions.extend(result['sample_predictions'])

        # Save run-level outputs
        if not run_subject_results:
            continue
            
        run_df = pd.DataFrame(run_subject_results)
        run_mean_importances = np.mean(importances_list, axis=0)
        
        if save_csv:
            run_df.to_csv(os.path.join(run_dir, f"{filename_base}_ws_subject_metrics.csv"), index=False)
            
            # Aggregate to create a global LOSO-like "summary" CSV for the run
            summary_metrics = run_df[[c for c in run_df.columns if c.startswith('mean_')]].mean().to_dict()
            summary_metrics['run_idx'] = run_idx
            summary_metrics['dimension'] = dimension
            summary_metrics['valid_subjects'] = valid_subjects
            pd.DataFrame([summary_metrics]).to_csv(
                os.path.join(run_dir, f"{filename_base}_summary.csv"), index=False
            )
            
            pd.DataFrame({
                "feature": feature_names,
                "importance": run_mean_importances,
            }).to_csv(os.path.join(run_dir, f"{filename_base}_feature_importances.csv"), index=False)
            
        if save_probabilities and run_sample_predictions:
            pd.DataFrame(run_fold_predictions).to_csv(
                os.path.join(run_dir, f"{filename_base}_fold_predictions.csv"), index=False
            )
            pd.DataFrame(run_sample_predictions).to_csv(
                os.path.join(run_dir, f"{filename_base}_sample_predictions.csv"), index=False
            )
            
        if shap_list:
            run_shap_stacked = np.concatenate(shap_list, axis=0)
            shap_values_all_runs.append(run_shap_stacked)
            stacked_payload = {
                "shap_values":   run_shap_stacked,
                "feature_names": list(feature_names),
            }
            if shap_x_list and len(shap_x_list) == len(shap_list):
                stacked_payload["x_test"] = np.concatenate(shap_x_list, axis=0)
            if shap_y_list and len(shap_y_list) == len(shap_list):
                stacked_payload["y_true"] = np.concatenate(shap_y_list, axis=0)
            with open(os.path.join(run_dir, f"{filename_base}_shap_values_stacked.pkl"), "wb") as f:
                pickle.dump(stacked_payload, f)

        if verbose:
            print(f"  [Run {run_idx+1}/{n_runs}] {time.time()-run_start:.1f}s | "
                  f"Avg AUC={run_df['mean_auc'].mean():.3f} | Valid Subs: {valid_subjects}")

    # Compile entirely overall stats
    if not all_runs_subject_metrics:
        print("No subjects matched criteria for any run.")
        return [], []
        
    all_runs_df = pd.DataFrame(all_runs_subject_metrics)
    summary_df = all_runs_df.groupby('subject').mean(numeric_only=True).reset_index()
    
    if save_csv:
        # Consolidated outputs go directly to dimension_results_path (model_type root).
        summary_df.to_csv(
            os.path.join(dimension_results_path, f"{filename_base}_ws_subject_metrics_averaged.csv"),
            index=False,
        )
        true_runs_path = os.path.join(dimension_results_path, "true_runs")
        _consolidate_sample_predictions(true_runs_path, filename_base, output_dir=dimension_results_path)
        
    # TODO Plotting for within subject specific layout
    # (Typically plotting is bypassed or modified for purely subject-level lists compared to LOSO folds)

    return all_runs_subject_metrics, shap_values_all_runs


# =============================================================================
# PERMUTATION TESTING — WITHIN-SUBJECT
# =============================================================================

def run_within_subject_permutation_analysis(
    dimension: str,
    df: pd.DataFrame,
    X: pd.DataFrame,
    y: pd.Series,
    subjects: pd.Series,
    tasks: pd.Series,
    feature_cols: list,
    config: dict,
    positive_class_name: str = "ON-task",
    negative_class_name: str = "OFF-task",
    n_permutations: int = 100,
    results_path: str = "results/MW_Classification/",
    model_type: str = "lr",
    class_weight=None,
    scale_pos_weight=None,
    use_smote: bool = False,
    oversampling_method: str = "SMOTE",
    oversampling_scope: str = "within",
    cv_strategy: str = "stratified_kfold",
    cv_folds: int = 5,
    min_samples_per_class: int = 0,
    min_minority_ratio: float = 0.0,
    k: int = 20,
    rf_params: dict = None,
    xgb_params: dict = None,
    lr_params: dict = None,
    ocsvm_params: dict = None,
    iforest_params: dict = None,
    oneclass_target: str = "minority",
    top_n_features_plot: int = 20,
    save_pickle: bool = False,
    save_csv: bool = True,
    save_probabilities: bool = True,
    save_plots: bool = True,
    save_shap: bool = False,
    plot_style: str = "seaborn-v0_8",
    verbose: bool = True,
    feature_selection_method: str = "mrmr",
    scaler: str = "standard",
    use_pca: bool = False,
    pca_n_components: int = None,
    pca_type: str = "standard",
    pca_kernel: str = "rbf",
    true_ws_metrics_df=None,  # Passed from the real run to compute specific p-values
    n_perm_jobs: int = 1,
    n_subject_jobs: int = 1,
    cv_n_jobs: int = 1,
    lmm_n_jobs: int = 1,
    lmm_prefilter_factor: int = 0,
    smote_k_neighbors: int = 5,
    logger=None,
    permutation_seed: int = None,
    perm_idx_offset: int = 0,
    total_n_permutations: int = None,
):
    """
    Run permutation testing natively inside the within-subject level.
    The y-labels are locally shuffled intra-subject independently, then evaluated.

    ``permutation_seed`` (when not None) overrides ``config['random_seed']``
    for the RNG used to permute labels and draw per-permutation random states.
    Type I error simulations MUST pass a sim-specific seed here; otherwise
    every simulation receives the same permuted-label sequence, coupling
    simulations and breaking the FPR independence assumption.
    """
    if n_permutations < 1:
        return pd.DataFrame(), {}, [], []

    # Permuted runs live inside results_path (which already includes model_type).
    perm_base_path = os.path.join(results_path, "permuted_runs")
    Path(perm_base_path).mkdir(parents=True, exist_ok=True)

    _total_perms = total_n_permutations if total_n_permutations is not None else n_permutations
    filename_base = f"{model_type}_ws_permutation_{_total_perms}perms"
    feature_names = X.columns
    unique_subjects = np.unique(subjects)
    _perm_seed = permutation_seed if permutation_seed is not None else config.get("random_seed", 42)
    rng = np.random.default_rng(_perm_seed)

    # Detect within_subject_median split to enable per-fold re-binarization.
    _label_contrast_cfg = config.get("label_contrasts", {}).get(dimension, {})
    _split_method = _label_contrast_cfg.get("split_method", "threshold")
    _label_col = _label_contrast_cfg.get("column_name", "onoff")
    _use_fold_rebinarize = (_split_method == "within_subject_median") and (_label_col in df.columns)

    start_time = time.time()
    print(f"\n*** Within-Subject Permutation: {n_permutations} runs ***")

    # Build shared CV kwargs dict (same pattern as distribution analysis).
    _cv_kwargs = dict(
        cv_strategy=cv_strategy,
        cv_folds=cv_folds,
        model_type=model_type,
        use_smote=use_smote,
        oversampling_method=oversampling_method,
        oversampling_scope=oversampling_scope,
        class_weight=class_weight,
        scale_pos_weight=scale_pos_weight,
        k=k,
        rf_params=rf_params,
        xgb_params=xgb_params,
        lr_params=lr_params,
        ocsvm_params=ocsvm_params,
        iforest_params=iforest_params,
        oneclass_target=oneclass_target,
        feature_selection_method=feature_selection_method,
        scaler=scaler,
        use_pca=use_pca,
        pca_n_components=pca_n_components,
        pca_type=pca_type,
        pca_kernel=pca_kernel,
        cv_n_jobs=cv_n_jobs,
        lmm_n_jobs=lmm_n_jobs,
        lmm_prefilter_factor=lmm_prefilter_factor,
        smote_k_neighbors=smote_k_neighbors,
    )

    # Pre-generate permuted labels deterministically. When perm_idx_offset > 0
    # (SLURM parallel mode) advance the RNG through the first perm_idx_offset
    # permutations so job N gets bit-identical labels to a sequential run.
    perm_inputs = []
    for _i in range(perm_idx_offset + n_permutations):
        if _use_fold_rebinarize:
            y_raw_perm = df[_label_col].groupby(subjects, group_keys=False).transform(
                lambda x: rng.permutation(x.values)
            )
            y_perm = y_raw_perm.groupby(subjects, group_keys=False).transform(
                lambda x: (x > x.median()).astype(int)
            )
        else:
            y_raw_perm = None
            y_perm = y.groupby(subjects, group_keys=False).transform(
                lambda x: rng.permutation(x.values)
            )
        random_state = int(rng.integers(1, 10000))
        if _i >= perm_idx_offset:
            global_perm_idx = _i
            perm_run_dir = os.path.join(perm_base_path, f"run_{global_perm_idx}")
            Path(perm_run_dir).mkdir(parents=True, exist_ok=True)
            perm_inputs.append((global_perm_idx, y_perm, y_raw_perm, random_state, perm_run_dir))

    print(f"  Dispatching {n_permutations} permutation jobs (n_perm_jobs={n_perm_jobs})")

    job_results = Parallel(n_jobs=n_perm_jobs, backend='loky')(
        delayed(_run_permutation_ws_job)(
            run_idx=run_idx,
            y_perm=y_perm,
            y_raw_perm=y_raw_perm,
            random_state=random_state,
            perm_run_dir=perm_run_dir,
            X=X,
            subjects=subjects,
            tasks=tasks,
            df=df,
            feature_names=feature_names,
            filename_base=filename_base,
            min_samples_per_class=min_samples_per_class,
            min_minority_ratio=min_minority_ratio,
            cv_kwargs=_cv_kwargs,
            use_fold_rebinarize=_use_fold_rebinarize,
            label_col=_label_col,
            save_shap=save_shap,
            save_csv=save_csv,
            save_probabilities=save_probabilities,
            dimension=dimension,
            n_subject_jobs=n_subject_jobs,
        )
        for run_idx, y_perm, y_raw_perm, random_state, perm_run_dir in perm_inputs
    )

    all_perm_results = []
    all_runs_subject_metrics = []

    for result in job_results:
        if result is None:
            continue
        all_perm_results.append(result['global_summary'])
        all_runs_subject_metrics.extend(result['run_subject_results'])

    if not all_perm_results:
        return pd.DataFrame(), {}, [], []

    perm_df = pd.DataFrame(all_perm_results)
    # Consolidated permutation outputs go to results_path (already the model_type root).
    perm_consolidated_dir = results_path
    if save_csv:
        perm_df.to_csv(os.path.join(perm_consolidated_dir, f"{filename_base}_summary_averaged.csv"), index=False)
        _consolidate_sample_predictions(perm_base_path, filename_base, output_dir=perm_consolidated_dir)

    # Compute p-values against actual real data if provided
    perm_summary = {}
    if true_ws_metrics_df is not None and not true_ws_metrics_df.empty:
        from scipy.stats import ttest_1samp, wilcoxon

        # Group real runs by subject
        true_summary = pd.DataFrame(true_ws_metrics_df).groupby('subject').mean(numeric_only=True)

        # --- Unweighted global mean permutation test ---
        true_mean_auc = true_summary['mean_auc'].mean()
        perm_aucs = perm_df['mean_auc'].values
        n_exceed = int(np.sum(perm_aucs >= true_mean_auc))
        p_val = n_exceed / len(perm_aucs) if n_exceed > 0 else 1.0 / (len(perm_aucs) + 1)

        print(f"WS Permutation Global AUC: Null={perm_aucs.mean():.4f}±{perm_aucs.std():.4f}, "
              f"True={true_mean_auc:.4f}, p={p_val:.4f}")

        perm_summary['p_auc'] = p_val
        perm_summary['true_mean_auc'] = float(true_mean_auc)
        perm_summary['perm_mean_auc'] = float(perm_aucs.mean())

        # --- Weighted global mean (weight each subject by n_samples) ---
        perm_subject_df = pd.DataFrame(all_runs_subject_metrics)
        if 'n_samples' in true_summary.columns and not perm_subject_df.empty and 'n_samples' in perm_subject_df.columns:
            weights = true_summary['n_samples'].values
            true_weighted_auc = float(np.average(true_summary['mean_auc'].values, weights=weights))

            perm_weighted_aucs = (
                perm_subject_df.groupby('run_idx')
                .apply(lambda g: np.average(g['mean_auc'], weights=g['n_samples']))
                .values
            )
            n_exceed_w = int(np.sum(perm_weighted_aucs >= true_weighted_auc))
            p_val_weighted = n_exceed_w / len(perm_weighted_aucs) if n_exceed_w > 0 else 1.0 / (len(perm_weighted_aucs) + 1)

            perm_summary['true_weighted_auc'] = true_weighted_auc
            perm_summary['perm_weighted_auc'] = float(np.mean(perm_weighted_aucs))
            perm_summary['p_auc_weighted'] = p_val_weighted
            print(f"WS Permutation Weighted AUC: Null={np.mean(perm_weighted_aucs):.4f}±{np.std(perm_weighted_aucs):.4f}, "
                  f"True={true_weighted_auc:.4f}, p={p_val_weighted:.4f}")

        # --- Second-level tests: per-subject AUC distribution vs. chance (0.5) ---
        subject_aucs = true_summary['mean_auc'].values
        if len(subject_aucs) >= 5:
            t_stat, p_ttest = ttest_1samp(subject_aucs, popmean=0.5)
            perm_summary['t_stat_vs_chance'] = float(t_stat)
            perm_summary['p_ttest_vs_chance'] = float(p_ttest)
            try:
                _, p_wilcoxon = wilcoxon(subject_aucs - 0.5, alternative='greater')
                perm_summary['p_wilcoxon_vs_chance'] = float(p_wilcoxon)
            except ValueError:
                perm_summary['p_wilcoxon_vs_chance'] = float('nan')
            print(f"WS Second-level (n={len(subject_aucs)} subjects): "
                  f"t-test p={p_ttest:.4f} (t={t_stat:.3f}), "
                  f"Wilcoxon p={perm_summary.get('p_wilcoxon_vs_chance', float('nan')):.4f}")

    return perm_df, perm_summary, all_runs_subject_metrics, []