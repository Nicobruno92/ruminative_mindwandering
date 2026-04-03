#!/usr/bin/env python
"""
Mind-Wandering Classification Pipeline — Main Entry Point.

Leave-One-Subject-Out (LOSO) classification of on-task vs off-task EEG markers.
Supports feature family selection to run subsets of markers independently.

USAGE:
    python run_loso_classification.py --config config.yaml
    python run_loso_classification.py --config config.yaml --contrast ON_vs_OFF_extreme
    python run_loso_classification.py --config config.yaml --family spectral
    python run_loso_classification.py --config config.yaml --dry_run

All run parameters come from config.yaml. CLI flags only override specific values.

Project: depressed_mindwandering
"""

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

# Add parent directory to sys.path so we can import utils.*
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.data_utils import (
    load_config,
    prepare_data_for_contrast,
    get_project_root,
    get_feature_columns,
)
from utils.analysis_utils import (
    run_distribution_analysis,
    run_permutation_distribution_analysis,
)
from utils.logging_utils import AnalysisLogger
from utils.plotting_utils import (
    plot_subject_level_densities,
    plot_shap_comparative_boxplots,
    generate_all_comparison_plots,
)


# =============================================================================
# ARGUMENT PARSING
# =============================================================================

def parse_args():
    """Parse command-line arguments. All defaults come from config.yaml."""
    parser = argparse.ArgumentParser(
        description="Run LOSO MW classification (all parameters from config.yaml)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default settings from config.yaml
  python run_loso_classification.py --config config.yaml

  # Override contrast
  python run_loso_classification.py --config config.yaml --contrast ON_vs_OFF_extreme

  # Run a specific feature family
  python run_loso_classification.py --config config.yaml --family spectral

  # Dry run (check data without classifying)
  python run_loso_classification.py --config config.yaml --dry_run
        """
    )
    parser.add_argument("--config", type=str, required=True,
                        help="Path to configuration YAML file")
    parser.add_argument("--contrast", type=str, default=None,
                        help="Override: label contrast name (must match a key in label_contrasts)")
    parser.add_argument("--family", type=str, default=None,
                        help="Override: feature family name (must match a key in feature_families)")
    parser.add_argument("--model_type", type=str, default=None,
                        choices=["rf", "xgb", "lr", "ocsvm", "iforest"],
                        help="Override: model type (rf, xgb, lr, ocsvm, iforest)")
    parser.add_argument("--dry_run", action="store_true",
                        help="Load data and print info without running classification")
    parser.add_argument("--skip_permutation", action="store_true",
                        help="Skip permutation test")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose output")
    return parser.parse_args()


# =============================================================================
# FEATURE FAMILY FILTERING
# =============================================================================

def filter_features_by_family(
    X: pd.DataFrame,
    family_name: str,
    feature_families: dict,
) -> pd.DataFrame:
    """
    Filter the feature matrix to columns belonging to a feature family.

    Feature families are defined in config as:
        feature_families:
          all: null            # null → no filtering, use all features
          spectral: ["power_", "power_normalized_"]

    Filtering matches column names that START WITH any of the given prefixes.

    Parameters
    ----------
    X : pd.DataFrame
        Full feature matrix.
    family_name : str
        Key in feature_families to select.
    feature_families : dict
        Mapping from family name → list of prefixes (or null for all).

    Returns
    -------
    pd.DataFrame
        Filtered feature matrix.
    """
    if family_name not in feature_families:
        raise ValueError(
            f"Unknown feature family: '{family_name}'. "
            f"Available: {list(feature_families.keys())}"
        )

    prefixes = feature_families[family_name]

    if prefixes is None:
        return X  # No filtering — use all features

    # Columns are formatted as {channel}_{marker}_{band} (e.g. AF3_power_theta).
    # Prefixes describe the marker portion, so we match against the substring
    # after the first separator: `_{prefix}` must appear in the column name.
    # `col.startswith(prefix)` is kept as a fallback for marker-first formats.
    selected_cols = [
        col for col in X.columns
        if any(f"_{prefix}" in col or col.startswith(prefix) for prefix in prefixes)
    ]

    if not selected_cols:
        raise ValueError(
            f"Feature family '{family_name}' with prefixes {prefixes} "
            f"matched 0 columns out of {len(X.columns)}. "
            f"Check that prefix patterns match your actual column names."
        )

    print(f"  Family '{family_name}': {len(selected_cols)} / {len(X.columns)} features selected")
    return X[selected_cols]


# =============================================================================
# CONFIGURATION HELPERS
# =============================================================================

def apply_cli_overrides(config: dict, args) -> dict:
    """
    Apply CLI argument overrides to configuration.

    CLI values always take precedence over config file values.

    Parameters
    ----------
    config : dict
        Loaded configuration.
    args : argparse.Namespace
        Parsed CLI arguments.

    Returns
    -------
    dict
        Updated configuration.
    """
    if args.contrast is not None:
        config["contrast"] = args.contrast
    if args.family is not None:
        config["active_family"] = args.family
    if args.model_type is not None:
        config["model_type"] = args.model_type
    if args.verbose:
        config["verbose"] = True
    return config


def save_used_config(config: dict, output_dir: str):
    """Save the exact configuration used for reproducibility."""
    os.makedirs(output_dir, exist_ok=True)
    config_path = os.path.join(output_dir, "used_config.yaml")
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"Config saved to: {config_path}")


def extract_model_params(config: dict) -> dict:
    """
    Extract model-specific parameter dicts from the flat config.

    Parameters
    ----------
    config : dict
        Full configuration dictionary.

    Returns
    -------
    dict
        Keys 'rf', 'xgb', 'lr', each mapping to a param dict.
    """
    param_specs = {
        'rf': ['n_estimators', 'max_depth', 'min_samples_split',
               'min_samples_leaf', 'max_features', 'bootstrap',
               'n_jobs', 'class_weight'],
        'xgb': ['n_estimators', 'max_depth', 'learning_rate',
                'objective', 'seed', 'n_jobs', 'scale_pos_weight'],
        'lr': ['penalty', 'solver', 'class_weight', 'C',
               'l1_ratio', 'max_iter'],
        'ocsvm': ['nu', 'kernel', 'gamma'],
        'iforest': ['n_estimators', 'contamination'],
    }
    all_params = {}
    for model_key, param_names in param_specs.items():
        params = {p: config[f'{model_key}_{p}']
                  for p in param_names
                  if config.get(f'{model_key}_{p}') is not None}
        all_params[model_key] = params
    return all_params


def build_results_path(config: dict, contrast_name: str, family_name: str, model_type: str) -> str:
    """
    Build the output directory for this run.

    Structure: results_root/LOSO/{contrast_name}/{family_name}/

    Parameters
    ----------
    config : dict
        Configuration dictionary.
    contrast_name : str
        Label contrast key.
    family_name : str
        Feature family key (e.g., 'spectral', 'all').
    model_type : str
        Classifier key.

    Returns
    -------
    str
        Absolute path to the results directory for this run.
    """
    results_root = config["data_paths"].get("results_root", "results/MW_Classification")
    project_root = get_project_root()
    if not os.path.isabs(results_root):
        results_root = str(project_root / results_root)

    loso_path = os.path.join(results_root, "LOSO", contrast_name, family_name)
    os.makedirs(loso_path, exist_ok=True)
    return loso_path


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main entry point."""
    args = parse_args()

    print(f"\nLoading configuration from: {args.config}")
    config = load_config(args.config)
    config = apply_cli_overrides(config, args)

    # Key parameters (all from config, CLI only overrides)
    contrast_name = config.get("contrast", "ON_vs_OFF")
    family_name = config.get("active_family", "all")
    feature_families = config.get("feature_families", {"all": None})
    model_type_raw = config.get("model_type", "lr")
    if isinstance(model_type_raw, list):
        raise ValueError(
            f"config 'model_type' is a list {model_type_raw}. "
            "Pass a single model via --model_type or run via run_cluster.sh which iterates over the list."
        )
    model_type = model_type_raw
    n_runs = config.get("n_runs", 20)
    permutation_runs = config.get("permutation_runs", 100)
    verbose = config.get("verbose", True)

    print(f"\n{'='*60}")
    print(f"Mind-Wandering Classification Pipeline (LOSO)")
    print(f"{'='*60}")
    print(f"Contrast  : {contrast_name}")
    print(f"Family    : {family_name}")
    print(f"Model     : {model_type.upper()}")
    print(f"Runs      : {n_runs}")
    print(f"{'='*60}\n")

    logger = AnalysisLogger()

    # Load full feature matrix
    df_prepared, X, y, groups, feature_cols = prepare_data_for_contrast(
        config, contrast_name, verbose=verbose
    )

    if len(X) < 50:
        print(f"ERROR: Insufficient data ({len(X)} samples). Need at least 50.")
        sys.exit(1)

    # Apply feature family filter
    X = filter_features_by_family(X, family_name, feature_families)
    feature_cols = X.columns.tolist()

    if args.dry_run:
        print("\n[DRY RUN] Data loaded successfully. Exiting.")
        print(f"  Samples   : {len(X)}")
        print(f"  Features  : {len(feature_cols)}")
        print(f"  Class dist: {y.value_counts().to_dict()}")
        print(f"  Subjects  : {sorted(groups.unique())}")
        print(f"  Family    : {family_name} ({len(feature_cols)} features)")
        sys.exit(0)

    # Results directory includes family name
    results_path = build_results_path(config, contrast_name, family_name, model_type)
    save_used_config(config, results_path)
    print(f"Results → {results_path}")

    # For compatibility with analysis_utils.get_model_results_folder
    config["current_model"] = contrast_name
    config["results_folder_pattern"] = ""

    model_params = extract_model_params(config)

    # -------------------------------------------------------------------------
    # Main classification
    # -------------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"Running LOSO classification ({n_runs} runs)...")
    print(f"{'='*60}\n")

    results_df, true_all_results, true_shap_values = run_distribution_analysis(
        dimension=contrast_name,
        df=df_prepared,
        X=X,
        y=y,
        groups=groups,
        feature_cols=feature_cols,
        config=config,
        positive_class_name="ON-task",
        negative_class_name="OFF-task",
        n_runs=n_runs,
        results_path=results_path,
        model_type=model_type,
        use_smote=config.get("use_smote", False),
        oversampling_method=config.get("oversampling_method", "SMOTE"),
        oversampling_scope=config.get("oversampling_scope", "global"),
        class_weight=config.get("class_weight"),
        scale_pos_weight=config.get("scale_pos_weight"),
        k=config.get("k", 20),
        rf_params=model_params['rf'],
        xgb_params=model_params['xgb'],
        lr_params=model_params['lr'],
        ocsvm_params=model_params['ocsvm'],
        iforest_params=model_params['iforest'],
        oneclass_target=config.get("oneclass_target", "minority"),
        top_n_features_plot=config.get("top_n_features_plot", 20),
        save_pickle=config.get("save_pickle", False),
        save_csv=config.get("save_csv", True),
        save_probabilities=config.get("save_probabilities", True),
        save_plots=config.get("save_plots", True),
        save_shap=config.get("save_shap", False),
        plot_style=config.get("plot_style", "seaborn"),
        verbose=verbose,
        feature_selection_method=config.get("feature_selection_method", "mrmr"),
        scaler=config.get("scaler", "standard"),
        use_pca=config.get("use_pca", False),
        pca_n_components=config.get("pca_n_components"),
        pca_type=config.get("pca_type", "standard"),
        pca_kernel=config.get("pca_kernel", "rbf"),
        logger=logger,
    )

    if results_df.empty:
        print("ERROR: No successful runs completed.")
        sys.exit(1)

    # Print summary
    print(f"\n{'='*60}")
    print(f"LOSO Results — {contrast_name} / {family_name}")
    print(f"{'='*60}")
    
    # Default metrics if none specified in config
    display_metrics = config.get("scoring_metrics")
    if not display_metrics:
        display_metrics = ['auc', 'balanced_accuracy', 'auprc', 'mcc']
    
    for metric_name in display_metrics:
        col_name = f"mean_{metric_name}"
        if col_name in results_df.columns:
            col = pd.to_numeric(results_df[col_name], errors='coerce')
            label = metric_name.upper().replace('_', ' ')
            print(f"  {label:18}: {col.mean():.4f} ± {col.std():.4f}")

    # -------------------------------------------------------------------------
    # Permutation test
    # -------------------------------------------------------------------------
    if permutation_runs > 0 and not args.skip_permutation:
        print(f"\n{'='*60}")
        print(f"Running permutation test ({permutation_runs} permutations)...")
        print(f"{'='*60}\n")

        perm_results_df, perm_summary, perm_all_results, perm_shap_values = run_permutation_distribution_analysis(
            dimension=contrast_name,
            df=df_prepared,
            X=X,
            y=y,
            groups=groups,
            feature_cols=feature_cols,
            config=config,
            positive_class_name="ON-task",
            negative_class_name="OFF-task",
            n_permutations=permutation_runs,
            results_path=results_path,
            model_type=model_type,
            use_smote=config.get("use_smote", False),
            oversampling_method=config.get("oversampling_method", "SMOTE"),
            oversampling_scope=config.get("oversampling_scope", "global"),
            class_weight=config.get("class_weight"),
            scale_pos_weight=config.get("scale_pos_weight"),
            k=config.get("k", 20),
            rf_params=model_params['rf'],
            xgb_params=model_params['xgb'],
            lr_params=model_params['lr'],
            ocsvm_params=model_params['ocsvm'],
            iforest_params=model_params['iforest'],
            oneclass_target=config.get("oneclass_target", "minority"),
            top_n_features_plot=config.get("top_n_features_plot", 20),
            save_pickle=config.get("save_pickle", False),
            save_csv=config.get("save_csv", True),
            save_probabilities=config.get("save_probabilities", True),
            save_plots=config.get("save_plots", True),
            save_shap=config.get("save_shap", False),
            plot_style=config.get("plot_style", "seaborn"),
            verbose=verbose,
            feature_selection_method=config.get("feature_selection_method", "mrmr"),
            scaler=config.get("scaler", "standard"),
            use_pca=config.get("use_pca", False),
            pca_n_components=config.get("pca_n_components"),
            pca_type=config.get("pca_type", "standard"),
            pca_kernel=config.get("pca_kernel", "rbf"),
            true_auc_list=results_df['mean_auc'].dropna().tolist(),
            true_bal_acc_list=results_df['mean_balanced_accuracy'].dropna().tolist(),
            true_auprc_list=(results_df['mean_auprc'].dropna().tolist()
                             if 'mean_auprc' in results_df.columns else []),
            true_mcc_list=(results_df['mean_mcc'].dropna().tolist()
                           if 'mean_mcc' in results_df.columns else []),
            permutation_scope=config.get("permutation_scope", "within"),
            logger=logger,
            n_runs=n_runs,
        )
        
        if config.get("save_plots", True):
            print("\nGenerating comparison plots via generate_pipeline_plots.py ...")
            import subprocess
            _plot_script = Path(__file__).resolve().parent.parent / "scripts" / "generate_pipeline_plots.py"
            _conda_env = config.get("slurm", {}).get("conda_env", "ML")
            _miniforge = Path(os.path.expanduser("~")) / "miniforge3"
            _ml_python = _miniforge / "envs" / _conda_env / "bin" / "python"
            _python = str(_ml_python) if _ml_python.exists() else sys.executable
            subprocess.run(
                [_python, str(_plot_script),
                 "--results_dir", results_path,
                 "--top_n_features", str(config.get("top_n_features_plot", 20))],
                check=False,
            )

    print(f"\n{'='*60}")
    print(f"Done. Results → {results_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
