#!/usr/bin/env python
"""
Standalone Script for Generating Pipeline Plots.

Generates publication-quality matplotlib visualisations for a completed LOSO
or Within-Subject classification run without re-running the classification.

Expected directory structure (LOSO):

    {results_dir}/                  <-- family-level dir (e.g. .../all/)
    └── lr/
        ├── true_runs/
        │   └── run_{n}/
        │       ├── lr_loso_20runs_summary.csv
        │       ├── lr_loso_20runs_loso_subject_metrics.csv
        │       ├── lr_loso_20runs_fold_predictions.csv
        │       ├── lr_loso_20runs_feature_importances.csv
        │       └── lr_loso_20runs_shap_values.pkl
        ├── permuted_runs/
        │   └── run_{n}/
        │       ├── lr_loso_20runs_summary.csv
        │       ├── lr_loso_20runs_loso_subject_metrics.csv
        │       └── lr_loso_20runs_feature_importances.csv
        └── plots/

Expected directory structure (Within-Subject):

    {results_dir}/                  <-- family-level dir (e.g. .../all/)
    └── lr/
        ├── true_runs/
        │   └── run_{n}/
        │       ├── lr_ws_10runs_summary.csv
        │       ├── lr_ws_10runs_ws_subject_metrics.csv   <-- mean_ prefix on metrics
        │       ├── lr_ws_10runs_fold_predictions.csv
        │       ├── lr_ws_10runs_feature_importances.csv
        │       └── lr_ws_10runs_shap_values_stacked.pkl  (if saved)
        ├── permuted_runs/
        │   └── run_{n}/
        │       ├── lr_ws_10runs_summary.csv
        │       ├── lr_ws_10runs_ws_subject_metrics.csv
        │       └── lr_ws_10runs_feature_importances.csv
        └── plots/

Mode (loso/ws) is detected automatically from the files present in true_runs/.

USAGE:
    # LOSO:
    python mw_classification_pipeline/scripts/generate_pipeline_plots.py \
        --results_dir results/MW_Classification/LOSO/ON_vs_OFF_within_median/all

    # Within-Subject:
    python mw_classification_pipeline/scripts/generate_pipeline_plots.py \
        --results_dir results/MW_Classification/WithinSubject/ON_vs_OFF_within_median/all/lr

Project: depressed_mindwandering
"""

import ast
import os
import sys
import argparse
import pickle
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — must be set before pyplot import

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "utils"))

from plotting_utils import (
    generate_all_comparison_plots,
    plot_shap_beeswarm_official,
    set_plot_style,
    plot_probability_vs_raw,
)

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION
# =============================================================================
KNOWN_MODEL_TYPES = {"lr", "rf", "xgb"}


# =============================================================================
# I/O HELPERS
# =============================================================================

def find_files_by_suffix(directory: str, suffix: str) -> list:
    """Return all files (non-recursive) in *directory* ending with *suffix*."""
    if not os.path.isdir(directory):
        return []
    return [
        os.path.join(directory, f)
        for f in sorted(os.listdir(directory))
        if f.endswith(suffix) and os.path.isfile(os.path.join(directory, f))
    ]


def find_file_by_suffix(directory: str, suffix: str):
    """Return the first matching file or None."""
    results = find_files_by_suffix(directory, suffix)
    return results[0] if results else None


def load_pkl(path: str):
    """Load a pickle file; return None if path is None or missing."""
    if not path or not os.path.exists(path):
        return None
    with open(path, "rb") as fh:
        return pickle.load(fh)


def _detect_pipeline_mode(model_dir: str) -> str:
    """
    Detect whether the model directory contains LOSO or Within-Subject results.

    Detection is based on the files inside the first true_runs/run_*/ directory:
    - ``_loso_subject_metrics.csv`` present → 'loso'
    - ``_ws_subject_metrics.csv`` present   → 'ws'
    - Fallback: 'loso' if 'WithinSubject' not in path, else 'ws'.

    Parameters
    ----------
    model_dir : str
        Model root directory (e.g., .../all/lr/).

    Returns
    -------
    str
        ``'loso'`` or ``'ws'``.
    """
    base = os.path.join(model_dir, "true_runs")
    run_dirs = collect_run_dirs_from_base(base)
    for rd in run_dirs:
        if not os.path.isdir(rd):
            continue
        for fname in os.listdir(rd):
            if fname.endswith("_loso_subject_metrics.csv"):
                return "loso"
            if fname.endswith("_ws_subject_metrics.csv"):
                return "ws"
    return "ws" if "WithinSubject" in str(model_dir) else "loso"


def collect_run_dirs_from_base(base_dir: str) -> list:
    """Return sorted list of run_{n}/ subdirectories inside *base_dir*."""
    if not os.path.isdir(base_dir):
        return []
    dirs = []
    for entry in os.listdir(base_dir):
        entry_path = os.path.join(base_dir, entry)
        if os.path.isdir(entry_path) and entry.startswith("run_"):
            dirs.append(entry_path)

    def _run_idx(p):
        try:
            return int(Path(p).name.split("_")[1])
        except (IndexError, ValueError):
            return 0

    return sorted(dirs, key=_run_idx)


# =============================================================================
# RESULT RECONSTRUCTION FROM CSV
# =============================================================================

def _parse_list_col(val) -> np.ndarray:
    """
    Parse a CSV cell that holds a serialised Python list or numpy array.

    Tries ast.literal_eval first, then json.loads. Returns an empty float
    array if parsing fails.
    """
    if isinstance(val, (list, np.ndarray)):
        return np.asarray(val, dtype=float)
    if isinstance(val, str):
        val = val.strip()
        try:
            return np.asarray(ast.literal_eval(val), dtype=float)
        except Exception:
            pass
        try:
            import json
            return np.asarray(json.loads(val), dtype=float)
        except Exception:
            pass
    return np.array([])


def reconstruct_run_result(run_path: str) -> dict:
    """
    Build one all_results-compatible dict from the CSV files in a run dir.

    Reconstructed keys
    ------------------
    mean_* / std_*        : from summary CSV (mean/std per metric)
    loso_subject_metrics  : list of dicts from loso_subject_metrics CSV
    fold_fprs / fold_tprs : lists of arrays reconstructed from fold_predictions
    fold_cms              : list of 2×2 np.arrays from fold_predictions
    feature_importances   : np.ndarray from feature_importances CSV
    _feature_names        : list[str] extracted for the caller (popped before use)
    """
    from sklearn.metrics import roc_curve, confusion_matrix as sk_cm

    result = {}

    # -- Summary metrics -------------------------------------------------------
    sf = find_file_by_suffix(run_path, "_summary.csv")
    if sf:
        df = pd.read_csv(sf)
        if not df.empty:
            for col in df.columns:
                if col.startswith("mean_") or col.startswith("std_"):
                    result[col] = float(
                        pd.to_numeric(df[col].iloc[0], errors="coerce")
                    )

    # -- Per-subject metrics ---------------------------------------------------
    # LOSO saves _loso_subject_metrics.csv (no mean_ prefix).
    # Within-Subject saves _ws_subject_metrics.csv (mean_ prefix on metric cols).
    subj_f = (
        find_file_by_suffix(run_path, "_loso_subject_metrics.csv")
        or find_file_by_suffix(run_path, "_ws_subject_metrics.csv")
    )
    if subj_f:
        sdf = pd.read_csv(subj_f)
        # Strip mean_ prefix so both formats share the same key names (e.g. 'auc')
        sdf.columns = [c.replace("mean_", "", 1) for c in sdf.columns]
        result["loso_subject_metrics"] = sdf.to_dict("records")

    # -- Fold predictions → ROC curves + confusion matrices -------------------
    # Prefer _fold_predictions.csv (lists per fold); fall back to
    # _sample_predictions.csv (one row per sample, used in WS permuted runs).
    fold_fprs, fold_tprs, fold_cms = [], [], []

    preds_f = find_file_by_suffix(run_path, "_fold_predictions.csv")
    if preds_f:
        pdf = pd.read_csv(preds_f)
        for fold_idx in sorted(pdf["fold_idx"].unique()):
            fdata = pdf[pdf["fold_idx"] == fold_idx]
            y_true_parts, y_proba_parts, y_pred_parts = [], [], []
            for _, row in fdata.iterrows():
                y_true_parts.append(_parse_list_col(row.get("y_true", "")))
                y_proba_parts.append(_parse_list_col(row.get("y_proba", "")))
                y_pred_parts.append(_parse_list_col(row.get("y_pred", "")))
            yt  = np.concatenate(y_true_parts)  if y_true_parts  else np.array([])
            yp  = np.concatenate(y_proba_parts) if y_proba_parts else np.array([])
            ypr = np.concatenate(y_pred_parts)  if y_pred_parts  else np.array([])
            if len(yt) < 2 or len(np.unique(yt)) < 2:
                continue
            fpr, tpr, _ = roc_curve(yt, yp)
            fold_fprs.append(fpr.tolist())
            fold_tprs.append(tpr.tolist())
            cm = sk_cm(yt.astype(int), ypr.astype(int))
            fold_cms.append(cm)

    elif find_file_by_suffix(run_path, "_sample_predictions.csv"):
        # WS permuted runs store one row per sample; reconstruct per-(subject,fold).
        sdf = pd.read_csv(find_file_by_suffix(run_path, "_sample_predictions.csv"))
        group_cols = [c for c in ("subject", "fold_idx") if c in sdf.columns]
        for _, grp in sdf.groupby(group_cols):
            yt  = grp["y_true"].values.astype(float)
            yp  = grp["y_proba"].values.astype(float)
            ypr = grp["y_pred"].values.astype(float)
            if len(yt) < 2 or len(np.unique(yt)) < 2:
                continue
            fpr, tpr, _ = roc_curve(yt, yp)
            fold_fprs.append(fpr.tolist())
            fold_tprs.append(tpr.tolist())
            cm = sk_cm(yt.astype(int), ypr.astype(int))
            fold_cms.append(cm)

    result["fold_fprs"] = fold_fprs
    result["fold_tprs"] = fold_tprs
    result["fold_cms"]  = fold_cms

    # -- Feature importances ---------------------------------------------------
    fi_f = find_file_by_suffix(run_path, "_feature_importances.csv")
    if fi_f:
        fi_df = pd.read_csv(fi_f)
        if "importance" in fi_df.columns:
            result["feature_importances"] = fi_df["importance"].values
        if "feature" in fi_df.columns:
            result["_feature_names"] = fi_df["feature"].tolist()

    return result


def load_all_results_from_model_dir(
    model_dir: str, from_perms: bool = False
) -> tuple:
    """
    Load (all_results, shap_runs, feature_names) from disk.

    True runs   → ``model_dir/true_runs/run_{n}/``
    Perm runs   → ``model_dir/permuted_runs/run_{n}/``

    Returns
    -------
    all_results : list of dict
        Compatible with generate_all_comparison_plots.
    shap_runs : list of np.ndarray
        Per-run SHAP value matrices (n_samples × n_features).
    feature_names : list of str
        Feature names extracted from FI CSV or SHAP pkl.
    """
    sub_dir = "permuted_runs" if from_perms else "true_runs"
    base = os.path.join(model_dir, sub_dir)
    run_dirs = collect_run_dirs_from_base(base)

    all_results:   list = []
    shap_runs:     list = []
    feature_names: list = []

    for rd in run_dirs:
        r = reconstruct_run_result(rd)
        if not r:
            continue
        fn = r.pop("_feature_names", [])
        if fn:
            feature_names = fn
        all_results.append(r)

        # SHAP pkl — loaded for both true and permuted runs.
        # LOSO true:  single _shap_values.pkl per run (all test subjects stacked)
        # LOSO perm:  single _shap_values.pkl per run
        # WS true:    one _shap_values.pkl per subject → stacked here
        # WS perm:    one _shap_values.pkl per subject → stacked here
        shap_pkl_single = find_file_by_suffix(rd, "_shap_values_stacked.pkl")
        if not shap_pkl_single:
            # Collect all per-subject pkl files ending with _shap_values.pkl
            all_shap_pkls = find_files_by_suffix(rd, "_shap_values.pkl")
            if len(all_shap_pkls) == 1:
                shap_pkl_single = all_shap_pkls[0]
            elif len(all_shap_pkls) > 1:
                # WS: stack per-subject arrays into one (n_all_samples, n_features)
                shap_arrays, fn_candidate = [], []
                for sp in all_shap_pkls:
                    d = load_pkl(sp)
                    if isinstance(d, dict) and "shap_values" in d:
                        shap_arrays.append(d["shap_values"])
                        if not fn_candidate and "feature_names" in d:
                            fn_candidate = d["feature_names"]
                if shap_arrays:
                    shap_runs.append(np.concatenate(shap_arrays, axis=0))
                    if not feature_names and fn_candidate:
                        feature_names = fn_candidate
                shap_pkl_single = None  # already handled

        if shap_pkl_single:
            data = load_pkl(shap_pkl_single)
            if isinstance(data, dict) and "shap_values" in data:
                shap_runs.append(data["shap_values"])
                if not feature_names and "feature_names" in data:
                    feature_names = data["feature_names"]

    return all_results, shap_runs, feature_names


# =============================================================================
# MODEL DETECTION
# =============================================================================

def detect_model_dirs(results_dir: str) -> list:
    """
    Detect per-model subdirs under *results_dir*.

    If *results_dir* itself contains ``true_runs/`` it is returned directly.
    Otherwise scans for known model-type subdirs (lr/, rf/, xgb/).
    """
    if os.path.isdir(os.path.join(results_dir, "true_runs")):
        return [results_dir]

    model_dirs = []
    for entry in sorted(os.listdir(results_dir)):
        entry_path = os.path.join(results_dir, entry)
        if os.path.isdir(entry_path) and entry.lower() in KNOWN_MODEL_TYPES:
            model_dirs.append(entry_path)
    return model_dirs


# =============================================================================
# PLOT GENERATION PER MODEL
# =============================================================================

def generate_model_plots(
    model_dir: str,
    top_n_features: int,
    positive_class: str,
    negative_class: str,
    dimension_name: str,
) -> None:
    """
    Generate all comparison plots for a single model directory.

    Reads true runs from ``true_runs/`` and permuted runs from
    ``permuted_runs/``, reconstructs the all_results list, then calls
    the full matplotlib comparison plot suite.

    Parameters
    ----------
    model_dir : str
        Path to the model root (e.g., .../all/lr/).
    top_n_features : int
        Number of top features to show in importance / SHAP plots.
    positive_class : str
        Positive-class label for confusion-matrix / ROC plots.
    negative_class : str
        Negative-class label for confusion-matrix / ROC plots.
    dimension_name : str
        Contrast name used in plot titles.
    """
    model_type    = Path(model_dir).name
    plots_dir     = os.path.join(model_dir, "plots")
    Path(plots_dir).mkdir(parents=True, exist_ok=True)

    pipeline_mode = _detect_pipeline_mode(model_dir)
    filename_base = f"{model_type}_{'ws' if pipeline_mode == 'ws' else 'loso'}"

    print(f"\n  [Model: {model_type.upper()} | Mode: {pipeline_mode.upper()}]  {model_dir}")

    # Load data from disk
    true_all_results, true_shap_runs, feature_names = load_all_results_from_model_dir(
        model_dir, from_perms=False
    )
    perm_all_results, perm_shap_runs, _ = load_all_results_from_model_dir(
        model_dir, from_perms=True
    )

    if not true_all_results:
        print("    ! No true run data found in true_runs/ — skipping")
        return

    print(
        f"    -> Loaded {len(true_all_results)} true runs, "
        f"{len(perm_all_results)} perm runs, "
        f"{len(feature_names)} features"
    )

    # ------------------------------------------------------------------
    # Main comparison plot suite (global dist, subject violins,
    # ROC, confusion matrix, feature importance)
    # ------------------------------------------------------------------
    generate_all_comparison_plots(
        true_all_results=true_all_results,
        perm_all_results=perm_all_results,
        feature_names=feature_names,
        save_path=plots_dir,
        filename_base=filename_base,
        dimension=dimension_name,
        positive_class_name=positive_class,
        negative_class_name=negative_class,
        top_n_features=top_n_features,
        true_shap_runs=true_shap_runs if true_shap_runs else None,
        perm_shap_runs=perm_shap_runs if perm_shap_runs else None,
    )

    # ------------------------------------------------------------------
    # SHAP beeswarm (official shap library)
    # ------------------------------------------------------------------
    if true_shap_runs and feature_names:
        print("    -> SHAP beeswarm (official)")
        combined_shap = np.concatenate(true_shap_runs, axis=0)

        # Build X_test for colour coding — load from each run's pkl (not just run_0)
        run_dirs_true = collect_run_dirs_from_base(os.path.join(model_dir, "true_runs"))
        x_arrays = []
        for rd in run_dirs_true:
            # LOSO: one _shap_values.pkl per run; WS: multiple per-subject pkls
            all_shap_pkls = find_files_by_suffix(rd, "_shap_values.pkl")
            stacked_pkl = find_file_by_suffix(rd, "_shap_values_stacked.pkl")
            if stacked_pkl:
                all_shap_pkls = [stacked_pkl]
            if all_shap_pkls:
                x_parts = []
                for sp in all_shap_pkls:
                    d = load_pkl(sp)
                    if isinstance(d, dict) and d.get("x_test") is not None:
                        x_parts.append(np.asarray(d["x_test"]))
                if x_parts:
                    x_arrays.append(np.concatenate(x_parts, axis=0))
        # If any runs had no x_test, fill with zeros of the right shape
        if not x_arrays or len(x_arrays) != len(true_shap_runs):
            x_arrays = [
                np.zeros_like(sv) for sv in true_shap_runs
            ]

        combined_x = np.concatenate(x_arrays, axis=0)
        plot_shap_beeswarm_official(
            shap_values=combined_shap,
            x_test=combined_x,
            feature_names=feature_names,
            save_path=plots_dir,
            filename_base=filename_base,
            max_display=top_n_features,
        )
    else:
        print("    ! SHAP values not available — skipping SHAP beeswarm")

    # Add probability vs raw score plots
    import glob
    matching_files = glob.glob(os.path.join(model_dir, "*_consolidated_sample_predictions.csv"))
    if matching_files:
        consolidated_predictions_file = Path(matching_files[0])
        print(f"    -> Plotting Probability vs Raw Score from {consolidated_predictions_file.name}")
        try:
            df_consolidated = pd.read_csv(consolidated_predictions_file)
            plot_probability_vs_raw(df_consolidated, model_dir, filename_base)
        except Exception as e:
            print(f"    ! Failed to plot probability vs raw score: {e}")
    else:
        print(f"    ! Consolidated predictions not found in {model_dir} — skipping scatter plots")

    print(f"    Plots saved -> {plots_dir}")


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate plots from LOSO pipeline results."
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help=(
            "Family-level dir (e.g., .../all/) "
            "or a single model dir (e.g., .../all/lr/)."
        ),
    )
    parser.add_argument(
        "--top_n_features",
        type=int,
        default=20,
        help="Number of top features shown in importance / SHAP plots.",
    )
    parser.add_argument(
        "--positive_class",
        type=str,
        default="ON-task",
        help="Positive-class label for confusion-matrix / ROC plots.",
    )
    parser.add_argument(
        "--negative_class",
        type=str,
        default="OFF-task",
        help="Negative-class label for confusion-matrix / ROC plots.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    results_dir = os.path.abspath(args.results_dir)

    assert os.path.isdir(results_dir), f"Directory does not exist: {results_dir}"

    # Infer dimension name from path (first segment after WithinSubject/LOSO)
    path_parts = Path(results_dir).parts
    dimension_name = "Unknown"
    for keyword in ("WithinSubject", "LOSO"):
        if keyword in path_parts:
            idx = list(path_parts).index(keyword) + 1
            if idx < len(path_parts):
                dimension_name = path_parts[idx]
            break

    set_plot_style("seaborn-v0_8")

    model_dirs = detect_model_dirs(results_dir)
    assert model_dirs, (
        f"No model directories found under {results_dir}.\n"
        f"Expected subdirs named one of {KNOWN_MODEL_TYPES}, "
        f"or a directory containing true_runs/."
    )

    print(f"Results dir : {results_dir}")
    print(f"Dimension   : {dimension_name}")
    print(f"Models found: {[Path(d).name for d in model_dirs]}")

    for model_dir in model_dirs:
        generate_model_plots(
            model_dir=model_dir,
            top_n_features=args.top_n_features,
            positive_class=args.positive_class,
            negative_class=args.negative_class,
            dimension_name=dimension_name,
        )

    print(f"\nDone. Plots generated for {len(model_dirs)} model(s).")


if __name__ == "__main__":
    main()
