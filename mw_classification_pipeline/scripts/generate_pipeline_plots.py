#!/usr/bin/env python
"""
Standalone Script for Generating Pipeline Plots.

Generates Plotly visualisations for a completed WithinSubject or LOSO
classification run without re-running the classification.

Expected directory structure (WithinSubject):

    {results_dir}/                  <-- family-level dir  (e.g. .../all/)
    ├── lr/
    │   ├── runs/
    │   │   └── run0/
    │   │       ├── lr_loso_ws_subject_metrics.csv
    │   │       ├── lr_loso_summary.csv
    │   │       ├── lr_loso_feature_importances.csv
    │   │       └── lr_loso_shap_values_stacked.pkl  (if saved)
    │   ├── permutations/
    │   │   └── *_summary_averaged.csv          <-- WithinSubject pattern
    │   ├── summaries/
    │   │   └── lr_loso_ws_subject_metrics_averaged.csv
    │   └── plots/                  <-- plots are written here
    ├── rf/
    └── xgb/

Expected directory structure (LOSO):

    {results_dir}/                  <-- family-level dir  (e.g. .../all/)
    └── lr/
        ├── runs/
        │   └── run{n}/
        │       ├── lr_loso_20runs_summary.csv
        │       └── lr_loso_20runs_feature_importances.csv
        ├── permutations/
        │   ├── lr_permutation_100perms_summary.csv   <-- LOSO pattern (all perms, one row each)
        │   └── runs/
        │       └── run{n}/
        ├── summaries/
        │   └── lr_loso_20runs_loso_subject_metrics.csv
        └── plots/                  <-- plots are written here

USAGE:
    # All models under a family dir:
    python mw_classification_pipeline/scripts/generate_pipeline_plots.py \
        --results_dir mw_classification_pipeline/results/MW_Classification/WithinSubject/on_vs_off_within_median/all

    # Single model dir:
    python mw_classification_pipeline/scripts/generate_pipeline_plots.py \
        --results_dir mw_classification_pipeline/results/MW_Classification/WithinSubject/on_vs_off_within_median/all/rf

Project: depressed_mindwandering
"""

import os
import sys
import argparse
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import warnings

# Allow importing utils/plotting_utils directly without loading the full
# utils package (which pulls in ML-only dependencies like tqdm, imblearn, etc.)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "utils"))

from plotting_utils import (
    plot_feature_importances,
    plot_feature_importances_true_vs_perm,
    plot_global_permutation_histogram,
    plot_shap_beeswarm_official,
    plot_true_vs_perm_violins,
    plot_subject_distribution_ridgelines,
    set_plot_style,
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


def collect_run_dataframes(model_dir: str, suffix: str) -> pd.DataFrame:
    """
    Concatenate all CSVs matching *suffix* from all runs/run*/ subdirs.

    Parameters
    ----------
    model_dir : str
        Model root dir (e.g., .../lr/).
    suffix : str
        File suffix to match (e.g., '_ws_subject_metrics.csv').
    """
    runs_dir = os.path.join(model_dir, "runs")
    if not os.path.isdir(runs_dir):
        return pd.DataFrame()

    frames = []
    for run_folder in sorted(os.listdir(runs_dir)):
        run_path = os.path.join(runs_dir, run_folder)
        if not os.path.isdir(run_path):
            continue
        for fpath in find_files_by_suffix(run_path, suffix):
            frames.append(pd.read_csv(fpath))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def collect_feature_importances(model_dir: str) -> tuple:
    """
    Aggregate feature importances across all run subdirs.

    Returns
    -------
    feature_names : list[str]
    mean_imp : np.ndarray
    std_imp : np.ndarray
    """
    runs_dir = os.path.join(model_dir, "runs")
    if not os.path.isdir(runs_dir):
        return [], np.array([]), np.array([])

    all_values: list = []
    feature_names: list = []

    for run_folder in sorted(os.listdir(runs_dir)):
        run_path = os.path.join(runs_dir, run_folder)
        fpath = find_file_by_suffix(run_path, "_feature_importances.csv")
        if fpath:
            df = pd.read_csv(fpath)
            if "feature" in df.columns and "importance" in df.columns:
                feature_names = df["feature"].tolist()
                all_values.append(df["importance"].values)

    if not all_values:
        return [], np.array([]), np.array([])

    stacked = np.vstack(all_values)
    return feature_names, stacked.mean(axis=0), stacked.std(axis=0)


def collect_shap_values(model_dir: str) -> tuple:
    """Collect stacked SHAP arrays and matching X_test from all runs/run*/ subdirs.

    Handles two layouts:
    - LOSO: one ``*_shap_values.pkl`` per run dir (whole-run SHAP matrix).
    - WithinSubject: many ``*_{subject}_shap_values.pkl`` per run dir (one per
      subject).  All subject-level arrays are concatenated within a run dir
      before being added to the per-run list.

    Returns
    -------
    shap_runs : list[np.ndarray]
        Per-run SHAP arrays (rows = samples across all subjects in that run).
    x_runs : list[np.ndarray] or list[None]
        Per-run X_test arrays (None when not saved in older pkl files).
    feature_names_from_shap : list[str] or []
        Feature names extracted from the last loaded pkl.
    """
    runs_dir = os.path.join(model_dir, "runs")
    if not os.path.isdir(runs_dir):
        return [], [], []

    shap_runs: list = []
    x_runs: list = []
    feature_names_from_shap: list = []

    for run_folder in sorted(os.listdir(runs_dir)):
        run_path = os.path.join(runs_dir, run_folder)
        if not os.path.isdir(run_path):
            continue

        # Collect *all* _shap_values.pkl files in this run dir
        all_pkl_files = sorted([
            os.path.join(run_path, f)
            for f in os.listdir(run_path)
            if f.endswith("_shap_values.pkl") or f.endswith("_shap_values_stacked.pkl")
        ])
        if not all_pkl_files:
            continue

        run_shap_list: list = []
        run_x_list: list = []

        for pkl_f in all_pkl_files:
            data = load_pkl(pkl_f)
            if not isinstance(data, dict) or "shap_values" not in data:
                continue
            run_shap_list.append(data["shap_values"])
            x_val = data.get("x_test", None)
            if x_val is not None:
                run_x_list.append(x_val)
            if "feature_names" in data:
                feature_names_from_shap = data["feature_names"]

        if not run_shap_list:
            continue

        # Concatenate subject-level arrays within this run
        combined_shap = np.concatenate(run_shap_list, axis=0)
        shap_runs.append(combined_shap)
        if len(run_x_list) == len(run_shap_list):
            x_runs.append(np.concatenate(run_x_list, axis=0))
        else:
            x_runs.append(None)

    return shap_runs, x_runs, feature_names_from_shap


def collect_feature_importances_from_perm(model_dir: str) -> tuple:
    """Aggregate feature importances from all permutations/runs/run*/ subdirs.

    Returns
    -------
    feature_names : list[str]
    mean_imp : np.ndarray
    std_imp : np.ndarray
    """
    perm_runs_dir = os.path.join(model_dir, "permutations", "runs")
    if not os.path.isdir(perm_runs_dir):
        return [], np.array([]), np.array([])

    all_values: list = []
    feature_names: list = []
    for run_folder in sorted(os.listdir(perm_runs_dir)):
        run_path = os.path.join(perm_runs_dir, run_folder)
        fpath = find_file_by_suffix(run_path, "_feature_importances.csv")
        if fpath:
            df = pd.read_csv(fpath)
            if "feature" in df.columns and "importance" in df.columns:
                feature_names = df["feature"].tolist()
                all_values.append(df["importance"].values)

    if not all_values:
        return [], np.array([]), np.array([])

    stacked = np.vstack(all_values)
    return feature_names, stacked.mean(axis=0), stacked.std(axis=0)


def load_perm_summary(model_dir: str) -> pd.DataFrame:
    """Load permutation summary CSV from permutations/ dir.

    WithinSubject saves ``*_summary_averaged.csv``; LOSO saves ``*_summary.csv``.
    Both are tried in order.
    """
    perm_dir = os.path.join(model_dir, "permutations")
    if not os.path.isdir(perm_dir):
        return pd.DataFrame()
    # Try WS pattern first, then LOSO pattern
    fpath = (
        find_file_by_suffix(perm_dir, "_summary_averaged.csv")
        or find_file_by_suffix(perm_dir, "_summary.csv")
    )
    return pd.read_csv(fpath) if fpath else pd.DataFrame()


def collect_subject_metrics_stacked(
    model_dir: str, from_perms: bool = False
) -> pd.DataFrame:
    """
    Stack per-subject metric rows from all run subdirs.

    Parameters
    ----------
    model_dir : str
        Model root (e.g. ``.../all/lr/``). When *from_perms* is True,
        looks inside ``permutations/`` instead.
    from_perms : bool
        When True, read from ``permutations/runs/`` instead of ``runs/``.

    Returns
    -------
    pd.DataFrame
        Stacked rows with columns ``subject`` + metric columns (bare names,
        ``mean_`` prefix stripped).
    """
    base = os.path.join(model_dir, "permutations") if from_perms else model_dir
    # WithinSubject perm saves *_ws_subject_metrics.csv inside perm run dirs
    df = collect_run_dataframes(base, "_ws_subject_metrics.csv")
    if df.empty:
        df = collect_run_dataframes(base, "_loso_subject_metrics.csv")
    # Fallback: legacy flat file at model_dir level (from old code, last-run only)
    if df.empty and not from_perms:
        flat = find_file_by_suffix(model_dir, "_loso_subject_metrics.csv")
        if flat:
            df = pd.read_csv(flat)
    if not df.empty:
        df = df.rename(columns={c: c.replace("mean_", "", 1) for c in df.columns})
    return df


def detect_model_dirs(results_dir: str) -> list:
    """
    Detect per-model subdirs under *results_dir*.

    If *results_dir* already is a model dir (contains runs/ or summaries/)
    it is returned directly.  Otherwise the function scans for known model
    type subdirs (lr/, rf/, xgb/).
    """
    has_runs = os.path.isdir(os.path.join(results_dir, "runs"))
    has_summaries = os.path.isdir(os.path.join(results_dir, "summaries"))
    if has_runs or has_summaries:
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
    Generate all available plots for a single model directory.

    Parameters
    ----------
    model_dir : str
        Path to the model root (e.g., .../all/rf/).
    top_n_features : int
        Number of top features to show in importance plots.
    positive_class : str
        Positive-class label for confusion-matrix / ROC plots.
    negative_class : str
        Negative-class label for confusion-matrix / ROC plots.
    dimension_name : str
        Contrast name used in plot titles.
    """
    model_type = Path(model_dir).name
    plots_dir = os.path.join(model_dir, "plots")
    Path(plots_dir).mkdir(parents=True, exist_ok=True)

    # Filename prefix that analysis_utils uses when saving (single-run default)
    filename_base = f"{model_type}_loso"

    print(f"\n  [Model: {model_type.upper()}]  {model_dir}")

    # ------------------------------------------------------------------
    # A. Feature Importances (averaged across all runs) + True vs Perm
    # ------------------------------------------------------------------
    feature_names, mean_imp, std_imp = collect_feature_importances(model_dir)
    if feature_names:
        print("    -> Feature Importances")
        plot_feature_importances(
            feature_names, mean_imp, std_imp,
            plots_dir, filename_base,
            top_n=top_n_features,
        )
        # Compare true importances against permuted importances
        perm_fi_names, perm_fi_mean, perm_fi_std = collect_feature_importances_from_perm(model_dir)
        print("    -> Feature Importances True vs Permuted")
        plot_feature_importances_true_vs_perm(
            true_feature_names=feature_names,
            true_mean=mean_imp,
            true_std=std_imp,
            perm_feature_names=perm_fi_names,
            perm_mean=perm_fi_mean,
            perm_std=perm_fi_std,
            save_path=plots_dir,
            filename_base=filename_base,
            top_n=top_n_features,
        )
    else:
        print("    ! Feature importances not found — skipping")

    # ------------------------------------------------------------------
    # B. Subject-Level Distribution Ridgelines  (True vs Permuted)
    # ------------------------------------------------------------------
    true_sub_df  = collect_subject_metrics_stacked(model_dir, from_perms=False)
    perm_sub_df  = collect_subject_metrics_stacked(model_dir, from_perms=True)

    if not true_sub_df.empty:
        print("    -> Subject-Level Ridgelines")
        metrics_to_plot = [
            m for m in ("auc", "balanced_accuracy", "auprc", "mcc")
            if m in true_sub_df.columns
        ]
        for metric in metrics_to_plot:
            plot_subject_distribution_ridgelines(
                true_df=true_sub_df,
                perm_df=perm_sub_df,
                metric=metric,
                save_path=plots_dir,
                filename_base=filename_base,
                dimension=dimension_name,
            )
    else:
        print("    ! Subject metrics not found — skipping subject ridgelines")

    # ------------------------------------------------------------------
    # C. Permutation Distributions
    # ------------------------------------------------------------------
    perm_summary_df = load_perm_summary(model_dir)
    run_summaries_df = collect_run_dataframes(model_dir, "_summary.csv")

    if not perm_summary_df.empty and not run_summaries_df.empty:
        print("    -> True vs Permuted violin plots")
        from scipy import stats as scipy_stats

        metric_map = {
            "mean_auc": "AUC",
            "mean_balanced_accuracy": "Balanced Accuracy",
            "mean_auprc": "AUPRC",
            "mean_mcc": "MCC",
        }
        results_for_plotting = {}
        for col, label in metric_map.items():
            if col in run_summaries_df.columns and col in perm_summary_df.columns:
                true_vals = run_summaries_df[col].dropna().values
                perm_vals = perm_summary_df[col].dropna().values
                if len(true_vals) > 0 and len(perm_vals) > 0:
                    _, mwu_p = scipy_stats.mannwhitneyu(
                        true_vals, perm_vals, alternative="greater"
                    )
                    empirical_p = float(np.mean(perm_vals >= np.mean(true_vals)))
                    results_for_plotting[label] = {
                        "true_values": true_vals,
                        "perm_values": perm_vals,
                        "p_value": mwu_p,
                        "empirical_p": empirical_p,
                    }
        if results_for_plotting:
            print("    -> True vs Permuted violin plots")
            plot_true_vs_perm_violins(
                results_for_plotting,
                dimension_name,
                model_type.upper(),
                plots_dir,
                filename_base,
            )
            print("    -> Permutation null-distribution histograms")
            plot_global_permutation_histogram(
                results_for_plotting,
                dimension_name,
                model_type.upper(),
                plots_dir,
                filename_base,
            )
        else:
            print(
                "    ! No shared metric columns between true and perm found — "
                "skipping permutation plots"
            )
    else:
        print("    ! No permutation data — skipping permutation plots")

    # ------------------------------------------------------------------
    # D. SHAP Visualisations (official shap library beeswarm)
    # ------------------------------------------------------------------
    shap_runs, x_runs, shap_feature_names = collect_shap_values(model_dir)
    # Prefer feature names from the pkl over the FI csv (may be more complete)
    effective_feature_names = shap_feature_names if shap_feature_names else feature_names
    if shap_runs and effective_feature_names:
        print("    -> SHAP beeswarm (official)")
        combined_shap = np.concatenate(shap_runs, axis=0)
        # Concatenate x_test arrays; fill zeros for runs that pre-date the fix
        x_arrays = [
            xr if xr is not None else np.zeros((s.shape[0], s.shape[1]))
            for xr, s in zip(x_runs, shap_runs)
        ]
        combined_x = np.concatenate(x_arrays, axis=0)
        plot_shap_beeswarm_official(
            shap_values=combined_shap,
            x_test=combined_x,
            feature_names=effective_feature_names,
            save_path=plots_dir,
            filename_base=filename_base,
            max_display=top_n_features,
        )
    else:
        print("    ! SHAP values not available — skipping SHAP plots")

    print(f"    Plots saved -> {plots_dir}")


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate plots from WithinSubject pipeline results."
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help=(
            "Family-level dir (e.g., .../on_vs_off_within_median/all/) "
            "or a single model dir (e.g., .../all/rf/)."
        ),
    )
    parser.add_argument(
        "--top_n_features",
        type=int,
        default=20,
        help="Number of top features shown in importance plots.",
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
        f"or a directory that itself contains runs/ or summaries/."
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
