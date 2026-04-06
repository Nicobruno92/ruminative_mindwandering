#!/usr/bin/env python
"""
Mind-Wandering Classification Pipeline — Within-Subject Main Entry Point.

Within-Subject classification of on-task vs off-task EEG markers.
Matches the structure of the LOSO pipeline but processes each subject independently.

USAGE:
    python run_within_subject_classification.py --config config.yaml
    python run_within_subject_classification.py --config config.yaml --contrast ON_vs_OFF_extreme
    python run_within_subject_classification.py --config config.yaml --family spectral
    python run_within_subject_classification.py --config config.yaml --dry_run

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

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="resource_tracker")

# Add parent directory to sys.path so we can import utils.*
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.data_utils import (
    load_config,
    prepare_data_for_contrast,
    get_project_root,
)
from utils.analysis_utils import (
    run_within_subject_distribution_analysis,
    run_within_subject_permutation_analysis,
)
from utils.logging_utils import AnalysisLogger
from utils.plotting_utils import (
    plot_subject_level_densities,
    plot_shap_comparative_boxplots,
    generate_all_comparison_plots,
)

# Reuse LOSO's filter_features_by_family function by importing or replicating
def filter_features_by_family(X: pd.DataFrame, family_name: str, feature_families: dict) -> pd.DataFrame:
    """Filter the feature matrix to columns belonging to a feature family."""
    if family_name not in feature_families:
        raise ValueError(f"Unknown feature family: '{family_name}'. "
                         f"Available: {list(feature_families.keys())}")

    prefixes = feature_families[family_name]
    if prefixes is None:
        return X

    # Columns are formatted as {channel}_{marker}_{band} (e.g. AF3_power_theta).
    # Prefixes describe the marker portion, so we match against the substring
    # after the first separator: `_{prefix}` must appear in the column name.
    # `col.startswith(prefix)` is kept as a fallback for marker-first formats.
    selected_cols = [col for col in X.columns if any(f"_{p}" in col or col.startswith(p) for p in prefixes)]
    if not selected_cols:
        raise ValueError(f"Family '{family_name}' matched 0 columns out of {len(X.columns)}.")

    print(f"  Family '{family_name}': {len(selected_cols)} / {len(X.columns)} features selected")
    return X[selected_cols]


def apply_cli_overrides(config: dict, args) -> dict:
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
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "used_config.yaml"), 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def extract_model_params(config: dict) -> dict:
    """Extract model-specific parameter dicts from the flat config."""
    # We parse from classifiers inside config for within-subject
    cls_config = config.get('classifiers', {})

    # Defaults in case not defined in yaml block
    rf_params = cls_config.get('rf', {})
    xgb_params = cls_config.get('xgb', {})
    lr_params = cls_config.get('lr', {})
    ocsvm_params = cls_config.get('ocsvm', {})
    iforest_params = cls_config.get('iforest', {})

    return {
        'rf': rf_params,
        'xgb': xgb_params,
        'lr': lr_params,
        'ocsvm': ocsvm_params,
        'iforest': iforest_params,
    }


def build_results_path(config: dict, contrast_name: str, family_name: str, model_type: str) -> str:
    results_root = config.get("project", {}).get("results_dir", "results/MW_Classification")
    project_root = get_project_root()
    if not os.path.isabs(results_root):
        results_root = str(project_root / results_root)

    # Note the explicit 'WithinSubject' folder injected here natively
    ws_path = os.path.join(results_root, "WithinSubject", contrast_name, family_name)
    os.makedirs(ws_path, exist_ok=True)
    return ws_path


def parse_args():
    parser = argparse.ArgumentParser(description="Run Within-Subject MW classification")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--contrast", type=str, default=None, help="Override label contrast")
    parser.add_argument("--family", type=str, default=None, help="Override feature family")
    parser.add_argument("--model_type", type=str, default=None, help="Override model type")
    parser.add_argument("--dry_run", action="store_true", help="Print info without running")
    parser.add_argument("--skip_permutation", action="store_true", help="Skip permutation test")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"\nLoading configuration from: {args.config}")
    config = load_config(args.config)
    config = apply_cli_overrides(config, args)

    contrast_name = config.get("contrast", "on_vs_off")
    family_name = config.get("active_family", "all")
    feature_families = config.get("feature_families", {"all": None})
    model_type = config.get("model_type", "lr")
    n_runs = config.get("n_runs", 1)
    
    outputs_cfg = config.get("outputs", {})
    permutation_runs = config.get("n_permutations", 0) if outputs_cfg.get("run_permutations") else 0
    verbose = config.get("verbose", True)
    
    cv_cfg = config.get("cv", {})
    prep_cfg = config.get("preprocessing", {})
    fs_cfg = config.get("feature_selection", {})
    os_cfg = config.get("oversampling", {})

    print(f"\n{'='*60}")
    print(f"Mind-Wandering Classification Pipeline (WITHIN-SUBJECT)")
    print(f"{'='*60}")
    print(f"Contrast  : {contrast_name}")
    print(f"Family    : {family_name}")
    print(f"Model     : {model_type.upper()}")
    print(f"Runs      : {n_runs}")
    print(f"{'='*60}\n")

    logger = AnalysisLogger()

    # Reuse core prepare_data_for_contrast. 
    # Notice that config for WithinSubject doesn't use `data_paths.root_dir` identically 
    # to LOSO. We map it to make `prepare_data_for_contrast` happy.
    if 'data_paths' not in config and 'project' in config:
        config['data_paths'] = {
            'root_dir': config['project'].get('root_dir'),
            'data_file': config['project'].get('data_file')
        }

    df_prepared, X, y, groups, feature_cols = prepare_data_for_contrast(
        config, contrast_name, verbose=verbose
    )

    X = filter_features_by_family(X, family_name, feature_families)
    feature_cols = X.columns.tolist()

    if args.dry_run:
        print("\n[DRY RUN] Data loaded successfully.")
        print(f"  Samples   : {len(X)}")
        print(f"  Features  : {len(feature_cols)}")
        print(f"  Class dist: {y.value_counts().to_dict()}")
        print(f"  Subjects  : {sorted(groups.unique())}")
        sys.exit(0)

    results_path = build_results_path(config, contrast_name, family_name, model_type)
    save_used_config(config, results_path)

    # Mappings for analysis utils
    config["current_model"] = contrast_name
    config["results_folder_pattern"] = ""
    
    model_params = extract_model_params(config)

    # In within-subject, df_prepared["task"] holds the tasks which map to groups if using group_kfold
    tasks = df_prepared["task"] if "task" in df_prepared.columns else groups

    oneclass_target = config.get("classifiers", {}).get("oneclass_target", "minority")

    true_ws_metrics_list, true_shap_values = run_within_subject_distribution_analysis(
        dimension=contrast_name,
        df=df_prepared,
        X=X,
        y=y,
        subjects=groups,
        tasks=tasks,
        feature_cols=feature_cols,
        config=config,
        positive_class_name=config.get("label_contrasts", {}).get(contrast_name, {}).get("positive_class_name", "ON"),
        negative_class_name=config.get("label_contrasts", {}).get(contrast_name, {}).get("negative_class_name", "OFF"),
        n_runs=n_runs,
        results_path=results_path,
        model_type=model_type,
        use_smote=os_cfg.get("use_smote", False),
        oversampling_method=os_cfg.get("method", "SMOTE"),
        oversampling_scope=os_cfg.get("scope", "within"),
        cv_strategy=cv_cfg.get("strategy", "stratified_kfold"),
        cv_folds=cv_cfg.get("folds", 5),
        min_samples_per_class=cv_cfg.get("min_samples_per_class", 0),
        min_minority_ratio=config.get("min_minority_ratio", 0.0),
        k=fs_cfg.get("k", 20),
        rf_params=model_params['rf'],
        xgb_params=model_params['xgb'],
        lr_params=model_params['lr'],
        ocsvm_params=model_params['ocsvm'],
        iforest_params=model_params['iforest'],
        oneclass_target=oneclass_target,
        top_n_features_plot=config.get("top_n_features_plot", 20),
        save_pickle=outputs_cfg.get("save_pickle", False),
        save_csv=True,
        save_probabilities=outputs_cfg.get("save_probabilities", True),
        save_plots=outputs_cfg.get("save_plots", False),
        save_shap=outputs_cfg.get("save_shap", False),
        verbose=verbose,
        feature_selection_method=fs_cfg.get("method", "mrmr"),
        scaler=prep_cfg.get("scaler", "standard"),
        use_pca=prep_cfg.get("pca", {}).get("use_pca", False),
        pca_n_components=prep_cfg.get("pca", {}).get("n_components"),
        pca_type=prep_cfg.get("pca", {}).get("pca_type", "standard"),
        pca_kernel=prep_cfg.get("pca", {}).get("pca_kernel", "rbf"),
        logger=logger,
    )

    if not true_ws_metrics_list:
        print("Done. No applicable subjects to run.")
        sys.exit(0)

    print(f"\n{'='*60}")
    print(f"WITHIN-SUBJECT Results — {contrast_name} / {family_name}")
    print(f"{'='*60}")
    
    true_ws_metrics_df = pd.DataFrame(true_ws_metrics_list)
    display_metrics = outputs_cfg.get("scoring_metrics", ['auc', 'balanced_accuracy', 'f1'])
    summary = true_ws_metrics_df.groupby('subject').mean(numeric_only=True)
    
    print("\nGlobal Averages across all subjects:")
    for metric_name in display_metrics:
        col = f"mean_{metric_name}"
        if col in summary.columns:
            val = summary[col].mean()
            std = summary[col].std()
            print(f"  {metric_name.upper():18}: {val:.4f} ± {std:.4f}")

    if permutation_runs > 0 and not args.skip_permutation:
        print(f"\n{'='*60}")
        print(f"Running permutation test ({permutation_runs} permutations)...")
        print(f"{'='*60}\n")
        
        perm_df, perm_summary, perm_all_results, perm_shap_values = run_within_subject_permutation_analysis(
            dimension=contrast_name,
            df=df_prepared,
            X=X,
            y=y,
            subjects=groups,
            tasks=tasks,
            feature_cols=feature_cols,
            config=config,
            positive_class_name=config.get("label_contrasts", {}).get(contrast_name, {}).get("positive_class_name", "ON"),
            negative_class_name=config.get("label_contrasts", {}).get(contrast_name, {}).get("negative_class_name", "OFF"),
            n_permutations=permutation_runs,
            results_path=results_path,
            model_type=model_type,
            use_smote=os_cfg.get("use_smote", False),
            oversampling_method=os_cfg.get("method", "SMOTE"),
            oversampling_scope=os_cfg.get("scope", "within"),
            cv_strategy=cv_cfg.get("strategy", "stratified_kfold"),
            cv_folds=cv_cfg.get("folds", 5),
            min_samples_per_class=cv_cfg.get("min_samples_per_class", 0),
            min_minority_ratio=config.get("min_minority_ratio", 0.0),
            k=fs_cfg.get("k", 20),
            rf_params=model_params['rf'],
            xgb_params=model_params['xgb'],
            lr_params=model_params['lr'],
            ocsvm_params=model_params['ocsvm'],
            iforest_params=model_params['iforest'],
            oneclass_target=oneclass_target,
            top_n_features_plot=config.get("top_n_features_plot", 20),
            save_pickle=outputs_cfg.get("save_pickle", False),
            save_csv=True,
            save_probabilities=outputs_cfg.get("save_probabilities", True),
            save_plots=outputs_cfg.get("save_plots", False),
            save_shap=outputs_cfg.get("save_shap", False),
            verbose=verbose,
            feature_selection_method=fs_cfg.get("method", "mrmr"),
            scaler=prep_cfg.get("scaler", "standard"),
            true_ws_metrics_df=true_ws_metrics_df,
            logger=logger,
        )
        
        if outputs_cfg.get("save_plots", True):
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
