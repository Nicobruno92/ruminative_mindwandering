#!/usr/bin/env python
"""
Type I Error Evaluation — LOSO Pipeline.

Runs N simulations of the full LOSO classification pipeline (true runs +
permutation test) on pseudo-synthetic null data to estimate the empirical
Type I error rate.

For each simulation:
  1. Real features X are replaced by X_synth drawn from a multivariate
     Gaussian that preserves the covariance structure and participant
     sample counts of the original data.
  2. Real labels y are kept unchanged (null condition is established
     entirely by the features being random — no feature-label association).
  3. The full pipeline runs: n_runs true LOSO passes, then n_perm
     permutation passes on the same synthetic data.
  4. The empirical p-value (from the pipeline's own permutation test) is
     saved. Across N simulations, FPR = count(p < alpha) / N.

Under the null hypothesis, p-values should follow U[0,1] and FPR should
equal alpha. Inflation indicates a methodological issue (e.g., data
leakage, SMOTE artifacts, feature selection overfitting noise).

USAGE
-----
  # Single simulation (SLURM array worker):
  python run_type1_error_loso.py \\
      --config config_type1_error.yaml \\
      --contrast ON_vs_OFF_within_median \\
      --family all \\
      --model_type rf \\
      --sim_idx 5

  # Aggregate results after all SLURM jobs complete:
  python run_type1_error_loso.py \\
      --config config_type1_error.yaml \\
      --contrast ON_vs_OFF_within_median \\
      --family all \\
      --model_type rf \\
      --aggregate

  # Local sequential run (small test):
  python run_type1_error_loso.py \\
      --config config_type1_error.yaml \\
      --contrast ON_vs_OFF_within_median \\
      --family all \\
      --model_type rf

Project: depressed_mindwandering
"""

# =============================================================================
# Imports
# =============================================================================

import os
import sys
import argparse
import yaml
import numpy as np
import pandas as pd
import warnings
from pathlib import Path
from datetime import datetime

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="resource_tracker")

# Resolve paths: type1_error/ → loso_pipeline/ → mw_classification_pipeline/
_HERE = Path(__file__).resolve().parent
_LOSO_DIR = _HERE.parent
_PIPELINE_DIR = _LOSO_DIR.parent

sys.path.insert(0, str(_PIPELINE_DIR))
sys.path.insert(0, str(_LOSO_DIR))

from utils.data_utils import (
    load_config,
    prepare_data_for_contrast,
    get_project_root,
)
from utils.analysis_utils import (
    run_distribution_analysis,
    run_permutation_distribution_analysis,
)
from utils.logging_utils import AnalysisLogger
from utils.simulation_utils import (
    generate_synthetic_features,
    compute_type1_error_rate,
    plot_pvalue_calibration,
)

# Reuse helpers from the parent LOSO script
from run_loso_classification import (
    filter_features_by_family,
    extract_model_params,
)


# =============================================================================
# Configuration Helpers
# =============================================================================

def load_merged_config(sim_config_path: str) -> tuple[dict, dict]:
    """
    Load and merge the parent LOSO config with the simulation config.

    The parent config supplies all pipeline parameters (model hyperparams,
    feature selection, SMOTE, etc.). The simulation config overrides only
    n_runs, permutation_runs, and adds the simulation-specific block.

    Parameters
    ----------
    sim_config_path : str
        Path to config_type1_error.yaml.

    Returns
    -------
    tuple[dict, dict]
        (merged_config, sim_block) where merged_config is the full config
        used by the pipeline, and sim_block holds simulation-specific params.
    """
    parent_config_path = _LOSO_DIR / "config.yaml"
    parent_config = load_config(str(parent_config_path))
    sim_full = yaml.safe_load(open(sim_config_path))
    sim_block = sim_full["simulation"]

    # Override compute budget
    merged = dict(parent_config)
    merged["n_runs"] = sim_block["n_runs"]
    merged["permutation_runs"] = sim_block["permutation_runs"]

    return merged, sim_block


def build_type1_results_path(
    config: dict,
    sim_block: dict,
    contrast_name: str,
    family_name: str,
    model_type: str,
) -> str:
    """
    Build the output directory for this contrast/family/model combination.

    Structure:
        {results_root}/{results_subdir}/{contrast}/{family}/{model}/

    Parameters
    ----------
    config : dict
        Merged pipeline configuration.
    sim_block : dict
        Simulation-specific config block.
    contrast_name, family_name, model_type : str
        Identifiers for this run.

    Returns
    -------
    str
        Absolute path to the results directory.
    """
    results_root = config["data_paths"].get("results_root", "results/MW_Classification")
    project_root = get_project_root()
    if not os.path.isabs(results_root):
        results_root = str(project_root / results_root)

    path = os.path.join(
        results_root,
        sim_block["results_subdir"],
        contrast_name,
        family_name,
        model_type,
    )
    os.makedirs(path, exist_ok=True)
    return path


def save_used_config(config: dict, sim_block: dict, output_dir: str) -> None:
    """Save merged config + simulation block for reproducibility."""
    os.makedirs(output_dir, exist_ok=True)
    full = dict(config)
    full["simulation"] = sim_block
    with open(os.path.join(output_dir, "used_config.yaml"), "w") as f:
        yaml.dump(full, f, default_flow_style=False)


# =============================================================================
# Single Simulation
# =============================================================================

def run_single_simulation(
    sim_idx: int,
    X_template: pd.DataFrame,
    y_real: pd.Series,
    groups: pd.Series,
    df_prepared: pd.DataFrame,
    contrast_name: str,
    model_type: str,
    config: dict,
    sim_block: dict,
    type1_root: str,
    model_params: dict,
    logger: AnalysisLogger,
) -> dict:
    """
    Run one Type I error simulation.

    Generates X_synth (random features, covariance-preserving), keeps y_real
    unchanged, then runs the full LOSO pipeline (true runs + permutation test)
    to obtain an empirical p-value.

    Parameters
    ----------
    sim_idx : int
        Simulation index (used for reproducible seed derivation).
    X_template : pd.DataFrame
        Real feature matrix — used only as structural template (shape,
        columns, index, covariance). Never fed to the classifier.
    y_real : pd.Series
        Real binary labels. Passed unchanged to the classifier; the null
        condition is established by X_synth having no label association.
    groups : pd.Series
        Participant IDs (defines LOSO folds).
    df_prepared : pd.DataFrame
        Full prepared DataFrame with probe-level metadata (subject, task,
        onoff, etc.). Required by run_distribution_analysis for saving
        per-probe predictions.
    contrast_name : str
        Label contrast key.
    model_type : str
        Classifier key ('rf', 'xgb', 'lr', 'ocsvm', 'iforest').
    config : dict
        Merged pipeline configuration.
    sim_block : dict
        Simulation-specific config block.
    type1_root : str
        Root results directory for this contrast/family/model combination.
    model_params : dict
        Model-specific hyperparameter dicts (keys: 'rf', 'xgb', 'lr', etc.).
    logger : AnalysisLogger
        Warning capture logger.

    Returns
    -------
    dict
        Per-simulation result row with p-values for all metrics.
    """
    # Derive seed for this simulation reproducibly from global seed + sim_idx
    sim_rng = np.random.default_rng(config["random_seed"] + sim_idx)
    sim_random_state = int(sim_rng.integers(1, 10**9))

    print(f"\n{'='*60}")
    print(f"Simulation {sim_idx} | seed={sim_random_state}")
    print(f"  covariance_scope  : {sim_block['covariance_scope']}")
    print(f"  covariance_method : {sim_block['covariance_method']}")
    print(f"{'='*60}")

    # Generate synthetic features (labels stay real)
    X_synth = generate_synthetic_features(
        X_template,
        groups,
        random_state=sim_random_state,
        covariance_scope=sim_block["covariance_scope"],
        covariance_method=sim_block["covariance_method"],
    )

    sim_results_path = os.path.join(type1_root, f"sim_{sim_idx:04d}")
    os.makedirs(sim_results_path, exist_ok=True)

    n_runs = sim_block["n_runs"]
    permutation_runs = sim_block["permutation_runs"]

    # ── True runs: X_synth (random) + y_real ──────────────────────────────
    results_df, _, _ = run_distribution_analysis(
        dimension=contrast_name,
        df=df_prepared,
        X=X_synth,
        y=y_real,
        groups=groups,
        feature_cols=X_synth.columns.tolist(),
        config=config,
        positive_class_name="ON-task",
        negative_class_name="OFF-task",
        n_runs=n_runs,
        results_path=sim_results_path,
        model_type=model_type,
        use_smote=config.get("use_smote", False),
        oversampling_method=config.get("oversampling_method", "SMOTE"),
        oversampling_scope=config.get("oversampling_scope", "global"),
        class_weight=config.get("class_weight"),
        scale_pos_weight=config.get("scale_pos_weight"),
        k=config.get("k", 20),
        rf_params=model_params["rf"],
        xgb_params=model_params["xgb"],
        lr_params=model_params["lr"],
        ocsvm_params=model_params["ocsvm"],
        iforest_params=model_params["iforest"],
        oneclass_target=config.get("oneclass_target", "minority"),
        top_n_features_plot=config.get("top_n_features_plot", 20),
        save_pickle=False,
        save_csv=config.get("save_csv", True),
        save_probabilities=False,   # skip per-probe probs to save disk space
        save_plots=False,           # no per-simulation plots
        save_shap=False,            # no SHAP per simulation
        verbose=False,
        feature_selection_method=config.get("feature_selection_method", "mrmr"),
        scaler=config.get("scaler", "standard"),
        use_pca=config.get("use_pca", False),
        pca_n_components=config.get("pca_n_components"),
        pca_type=config.get("pca_type", "standard"),
        pca_kernel=config.get("pca_kernel", "rbf"),
        logger=logger,
    )

    # Build null result row (NaN p-values) in case true runs produce no output
    null_row: dict = {
        "sim_idx": sim_idx,
        "contrast": contrast_name,
        "model_type": model_type,
        "covariance_scope": sim_block["covariance_scope"],
        "covariance_method": sim_block["covariance_method"],
        "n_runs": n_runs,
        "permutation_runs": permutation_runs,
        "p_mean_auc": np.nan,
        "p_mean_balanced_accuracy": np.nan,
        "p_mean_auprc": np.nan,
        "p_mean_mcc": np.nan,
        "true_mean_auc": np.nan,
        "perm_mean_auc": np.nan,
    }

    if results_df is None or (hasattr(results_df, "empty") and results_df.empty):
        print(f"  WARNING: Simulation {sim_idx} — true runs produced empty results. Saving NaN.")
        pd.DataFrame([null_row]).to_csv(
            os.path.join(sim_results_path, "type1_pvalues.csv"), index=False
        )
        return null_row

    # Extract true metric distributions from fold-level results
    true_auc_list = results_df["mean_auc"].dropna().tolist() if "mean_auc" in results_df.columns else []
    true_bal_acc_list = (
        results_df["mean_balanced_accuracy"].dropna().tolist()
        if "mean_balanced_accuracy" in results_df.columns
        else []
    )
    true_auprc_list = (
        results_df["mean_auprc"].dropna().tolist()
        if "mean_auprc" in results_df.columns
        else []
    )
    true_mcc_list = (
        results_df["mean_mcc"].dropna().tolist()
        if "mean_mcc" in results_df.columns
        else []
    )

    # ── Permutation test: X_synth + y_real shuffled ────────────────────────
    _, perm_summary, _, _ = run_permutation_distribution_analysis(
        dimension=contrast_name,
        df=df_prepared,
        X=X_synth,
        y=y_real,
        groups=groups,
        feature_cols=X_synth.columns.tolist(),
        config=config,
        positive_class_name="ON-task",
        negative_class_name="OFF-task",
        n_permutations=permutation_runs,
        results_path=sim_results_path,
        model_type=model_type,
        use_smote=config.get("use_smote", False),
        oversampling_method=config.get("oversampling_method", "SMOTE"),
        oversampling_scope=config.get("oversampling_scope", "global"),
        class_weight=config.get("class_weight"),
        scale_pos_weight=config.get("scale_pos_weight"),
        k=config.get("k", 20),
        rf_params=model_params["rf"],
        xgb_params=model_params["xgb"],
        lr_params=model_params["lr"],
        ocsvm_params=model_params["ocsvm"],
        iforest_params=model_params["iforest"],
        oneclass_target=config.get("oneclass_target", "minority"),
        top_n_features_plot=config.get("top_n_features_plot", 20),
        save_pickle=False,
        save_csv=config.get("save_csv", True),
        save_probabilities=False,
        save_plots=False,
        save_shap=False,
        verbose=False,
        feature_selection_method=config.get("feature_selection_method", "mrmr"),
        scaler=config.get("scaler", "standard"),
        use_pca=config.get("use_pca", False),
        pca_n_components=config.get("pca_n_components"),
        pca_type=config.get("pca_type", "standard"),
        pca_kernel=config.get("pca_kernel", "rbf"),
        true_auc_list=true_auc_list,
        true_bal_acc_list=true_bal_acc_list,
        true_auprc_list=true_auprc_list,
        true_mcc_list=true_mcc_list,
        permutation_scope=config.get("permutation_scope", "within"),
        logger=logger,
        n_runs=n_runs,
    )

    sim_result = {
        "sim_idx": sim_idx,
        "contrast": contrast_name,
        "model_type": model_type,
        "covariance_scope": sim_block["covariance_scope"],
        "covariance_method": sim_block["covariance_method"],
        "n_runs": n_runs,
        "permutation_runs": permutation_runs,
        "p_mean_auc": perm_summary.get("p_mean_auc", np.nan),
        "p_mean_balanced_accuracy": perm_summary.get("p_mean_balanced_accuracy", np.nan),
        "p_mean_auprc": perm_summary.get("p_mean_auprc", np.nan),
        "p_mean_mcc": perm_summary.get("p_mean_mcc", np.nan),
        "true_mean_auc": float(np.mean(true_auc_list)) if true_auc_list else np.nan,
        "perm_mean_auc": perm_summary.get("perm_mean_auc", np.nan),
    }

    pd.DataFrame([sim_result]).to_csv(
        os.path.join(sim_results_path, "type1_pvalues.csv"), index=False
    )
    print(
        f"  Sim {sim_idx} done | "
        f"true_AUC={sim_result['true_mean_auc']:.3f} | "
        f"p_AUC={sim_result['p_mean_auc']:.3f}"
    )
    return sim_result


# =============================================================================
# Aggregation
# =============================================================================

def aggregate_simulations(
    type1_root: str,
    sim_block: dict,
    contrast_name: str,
    model_type: str,
) -> None:
    """
    Collect per-simulation p-value CSVs and produce FPR summary + plots.

    Scans ``type1_root`` for all ``sim_*/type1_pvalues.csv`` files,
    concatenates them, computes FPR at configured alpha levels, and
    generates calibration plots.

    Parameters
    ----------
    type1_root : str
        Root results directory for this contrast/family/model combination.
    sim_block : dict
        Simulation-specific config block (for alpha_levels).
    contrast_name, model_type : str
        Identifiers for display/logging.
    """
    sim_dirs = sorted(Path(type1_root).glob("sim_*/type1_pvalues.csv"))

    if not sim_dirs:
        print(f"No simulation results found in {type1_root}")
        return

    dfs = [pd.read_csv(p) for p in sim_dirs]
    all_pvalues = pd.concat(dfs, ignore_index=True)
    all_pvalues.to_csv(os.path.join(type1_root, "type1_pvalues_all.csv"), index=False)

    n_total = len(all_pvalues)
    n_nan = all_pvalues["p_mean_auc"].isna().sum()
    print(f"\nAggregated {n_total} simulations ({n_nan} with NaN p-values skipped).")

    # FPR summary table
    alpha_levels = sim_block.get("alpha_levels", [0.01, 0.05, 0.10])
    p_cols = [c for c in all_pvalues.columns if c.startswith("p_")]
    fpr_df = compute_type1_error_rate(all_pvalues, alpha_levels=alpha_levels, p_metric_cols=p_cols)
    fpr_df.to_csv(os.path.join(type1_root, "type1_error_summary.csv"), index=False)

    print(f"\nType I Error Rate Summary — {contrast_name} / {model_type}")
    print(fpr_df.to_string(index=False))

    # Calibration plots
    plot_pvalue_calibration(
        all_pvalues,
        results_path=type1_root,
        pipeline_label=f"LOSO / {contrast_name} / {model_type}",
        p_metric_cols=p_cols,
        alpha_levels=alpha_levels,
    )
    print(f"\nResults saved → {type1_root}")


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Type I error evaluation for LOSO classification pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config", type=str,
        default=str(_HERE / "config_type1_error.yaml"),
        help="Path to config_type1_error.yaml (default: same directory as this script)",
    )
    parser.add_argument(
        "--contrast", type=str, default=None,
        help="Label contrast name. Overrides parent config 'contrast' list.",
    )
    parser.add_argument(
        "--family", type=str, default=None,
        help="Feature family name. Overrides parent config 'run_families'.",
    )
    parser.add_argument(
        "--model_type", type=str, default=None,
        choices=["rf", "xgb", "lr", "ocsvm", "iforest"],
        help="Model type.",
    )
    parser.add_argument(
        "--sim_idx", type=int, default=None,
        help="Run a single simulation by index (for SLURM array workers).",
    )
    parser.add_argument(
        "--aggregate", action="store_true",
        help="Collect sim_*/type1_pvalues.csv and produce FPR summary + plots.",
    )
    parser.add_argument(
        "--n_simulations", type=int, default=None,
        help="Override n_simulations from config (useful for local testing).",
    )
    return parser.parse_args()


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    """Main entry point."""
    args = parse_args()

    config, sim_block = load_merged_config(args.config)

    # CLI overrides
    if args.n_simulations is not None:
        sim_block["n_simulations"] = args.n_simulations

    contrast_name: str = args.contrast or (
        config["contrast"][0]
        if isinstance(config.get("contrast"), list)
        else config.get("contrast", "ON_vs_OFF_within_median")
    )
    family_name: str = args.family or (
        config.get("run_families", ["all"])[0]
        if isinstance(config.get("run_families"), list)
        else config.get("run_families", "all")
    )
    model_type_raw = args.model_type or config.get("model_type", "rf")
    model_type: str = model_type_raw[0] if isinstance(model_type_raw, list) else model_type_raw

    n_simulations: int = sim_block["n_simulations"]
    verbose = config.get("verbose", True)

    print(f"\n{'='*60}")
    print(f"Type I Error Evaluation — LOSO Pipeline")
    print(f"{'='*60}")
    print(f"Contrast          : {contrast_name}")
    print(f"Family            : {family_name}")
    print(f"Model             : {model_type.upper()}")
    print(f"N simulations     : {n_simulations}")
    print(f"N true runs/sim   : {sim_block['n_runs']}")
    print(f"N perm runs/sim   : {sim_block['permutation_runs']}")
    print(f"Covariance scope  : {sim_block['covariance_scope']}")
    print(f"Covariance method : {sim_block['covariance_method']}")
    print(f"{'='*60}\n")

    # Results directory for this contrast/family/model
    type1_root = build_type1_results_path(config, sim_block, contrast_name, family_name, model_type)

    # ── Aggregate-only mode ────────────────────────────────────────────────
    if args.aggregate:
        aggregate_simulations(type1_root, sim_block, contrast_name, model_type)
        return

    # ── Data loading (template) ────────────────────────────────────────────
    config["current_model"] = contrast_name
    config["results_folder_pattern"] = ""

    df_prepared, X_real, y_real, groups, feature_cols = prepare_data_for_contrast(
        config, contrast_name, verbose=verbose
    )

    if len(X_real) < 50:
        print(f"ERROR: Insufficient data ({len(X_real)} samples). Need at least 50.")
        sys.exit(1)

    X_template = filter_features_by_family(
        X_real, family_name, config.get("feature_families", {"all": None})
    )

    model_params = extract_model_params(config)

    save_used_config(config, sim_block, type1_root)
    logger = AnalysisLogger()

    print(f"Template data: {len(X_template)} samples × {X_template.shape[1]} features")
    print(f"Subjects: {groups.nunique()} | Class balance: {y_real.value_counts().to_dict()}")
    print(f"\nNOTE: X_synth will be generated from MVN fitted to real X.")
    print(f"      y_real is used unchanged — null condition is in the features.\n")

    # ── Single simulation mode (SLURM array) ──────────────────────────────
    if args.sim_idx is not None:
        run_single_simulation(
            sim_idx=args.sim_idx,
            X_template=X_template,
            y_real=y_real,
            groups=groups,
            df_prepared=df_prepared,
            contrast_name=contrast_name,
            model_type=model_type,
            config=config,
            sim_block=sim_block,
            type1_root=type1_root,
            model_params=model_params,
            logger=logger,
        )
        return

    # ── Sequential mode (local testing) ───────────────────────────────────
    print(f"Running {n_simulations} simulations sequentially...")
    all_results = []
    for sim_idx in range(n_simulations):
        result = run_single_simulation(
            sim_idx=sim_idx,
            X_template=X_template,
            y_real=y_real,
            groups=groups,
            df_prepared=df_prepared,
            contrast_name=contrast_name,
            model_type=model_type,
            config=config,
            sim_block=sim_block,
            type1_root=type1_root,
            model_params=model_params,
            logger=logger,
        )
        all_results.append(result)

    aggregate_simulations(type1_root, sim_block, contrast_name, model_type)


if __name__ == "__main__":
    main()
