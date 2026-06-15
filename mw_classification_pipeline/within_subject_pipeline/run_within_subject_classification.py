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
    plot_auc_vs_onoff_dispersion,
    plot_auc_vs_class_imbalance,
)

def resolve_family(family_name: str, feature_families: dict) -> tuple:
    """
    Extract epoch_types and prefixes from a feature family config entry.

    Parameters
    ----------
    family_name : str
        Key in feature_families.
    feature_families : dict
        Full feature_families config block.

    Returns
    -------
    tuple[list[str], list[str] | None]
        (epoch_types, prefixes) where prefixes=None means keep all columns.
    """
    if family_name not in feature_families:
        raise ValueError(f"Unknown feature family: '{family_name}'. "
                         f"Available: {list(feature_families.keys())}")
    cfg = feature_families[family_name]
    if not isinstance(cfg, dict):
        raise ValueError(
            f"feature_families.{family_name} must be a dict with 'epoch_types' and "
            f"'prefixes' keys. Got: {cfg!r}"
        )
    return cfg.get("epoch_types", ["state"]), cfg.get("prefixes")


def filter_features_by_family(X: pd.DataFrame, family_name: str, prefixes: list) -> pd.DataFrame:
    """
    Filter the feature matrix to columns matching the given prefixes.

    Uses word-boundary logic: prefix ``P1`` matches ``Pz_P1`` and
    ``Pz_P1_latency`` but NOT ``Pz_P10``.
    """
    import re

    if prefixes is None:
        return X

    def _matches(col: str, prefix: str) -> bool:
        p = re.escape(prefix)
        return bool(re.search(rf"_{p}(_|$)", col)) or bool(re.match(rf"{p}(_|$)", col))

    selected_cols = [col for col in X.columns if any(_matches(col, p) for p in prefixes)]
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
    ws_path = os.path.join(results_root, "WithinSubject", contrast_name, family_name, model_type)
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
    parser.add_argument("--run_idx", type=int, default=None,
                        help="SLURM array mode: run only this true-run index (0-based). Skips perms.")
    parser.add_argument("--perm_idx", type=int, default=None,
                        help="SLURM array mode: run only this permutation index (0-based). Skips true runs.")
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"\nLoading configuration from: {args.config}")
    config = load_config(args.config)
    config = apply_cli_overrides(config, args)

    contrast_name = config.get("contrast", "on_vs_off")
    family_name = config.get("active_family", "all")
    feature_families = config.get("feature_families", {})
    model_type = config.get("model_type", "lr")
    n_runs = config.get("n_runs", 1)

    outputs_cfg = config.get("outputs", {})
    permutation_runs = config.get("n_permutations", 0) if outputs_cfg.get("run_permutations") else 0
    verbose = config.get("verbose", True)

    cv_cfg = config.get("cv", {})
    prep_cfg = config.get("preprocessing", {})
    fs_cfg = config.get("feature_selection", {})
    os_cfg = config.get("oversampling", {})
    par_cfg = config.get("parallelism", {})
    n_subject_jobs = par_cfg.get("n_subject_jobs", 1)
    n_perm_jobs = par_cfg.get("n_perm_jobs", 1)
    cv_n_jobs = par_cfg.get("cv_n_jobs", 1)
    lmm_n_jobs = fs_cfg.get("lmm_n_jobs", 1)
    lmm_prefilter_factor = fs_cfg.get("lmm_prefilter_factor", 0)
    smote_k_neighbors = os_cfg.get("k_neighbors", 5)

    # ── SLURM parallel mode ────────────────────────────────────────────────────
    run_idx_offset = 0
    perm_idx_offset = 0
    total_n_runs_for_seeds = n_runs
    total_n_perms_for_seeds = permutation_runs

    if args.run_idx is not None and args.perm_idx is not None:
        raise ValueError("Cannot specify both --run_idx and --perm_idx.")

    if args.run_idx is not None:
        if args.run_idx >= n_runs:
            raise ValueError(f"--run_idx {args.run_idx} >= n_runs={n_runs}")
        run_idx_offset = args.run_idx
        n_runs = 1
        permutation_runs = 0
        print(f"[SLURM mode] True run {args.run_idx} of {total_n_runs_for_seeds}")

    if args.perm_idx is not None:
        if args.perm_idx >= permutation_runs:
            raise ValueError(f"--perm_idx {args.perm_idx} >= n_permutations={permutation_runs}")
        perm_idx_offset = args.perm_idx
        n_runs = 0
        permutation_runs = 1
        print(f"[SLURM mode] Permutation {args.perm_idx} of {total_n_perms_for_seeds}")
    # ──────────────────────────────────────────────────────────────────────────

    # Resolve family → inject epoch_types so the loader knows what to load
    family_epoch_types, family_prefixes = resolve_family(family_name, feature_families)
    config["epoch_types"] = family_epoch_types

    print(f"\n{'='*60}")
    print(f"Mind-Wandering Classification Pipeline (WITHIN-SUBJECT)")
    print(f"{'='*60}")
    print(f"Contrast    : {contrast_name}")
    print(f"Family      : {family_name}")
    print(f"Epoch types : {family_epoch_types}")
    print(f"Model       : {model_type.upper()}")
    print(f"Runs        : {n_runs}")
    print(f"{'='*60}\n")

    logger = AnalysisLogger()

    if 'data_paths' not in config and 'project' in config:
        config['data_paths'] = {
            'root_dir': config['project'].get('root_dir'),
            'data_file': config['project'].get('data_file')
        }

    # ── Data cache (fast path) ─────────────────────────────────────────────────
    # If precompute_data_cache.py was run beforehand, load the compact pickle
    # (~4 MB, < 1 s) instead of re-reading 784k rows of CSVs (~5 min).
    import pickle as _pickle
    from utils.data_utils import get_project_root as _get_root
    _results_root = config.get("project", {}).get("results_dir", "results/MW_Classification")
    if not os.path.isabs(_results_root):
        _results_root = str(_get_root() / _results_root)
    _cache_path = os.path.join(_results_root, "data_cache",
                               f"{contrast_name}__{family_name}.pkl")

    if os.path.exists(_cache_path):
        print(f"Loading from cache: {_cache_path}")
        _t0 = __import__('time').time()
        with open(_cache_path, "rb") as _f:
            _cached = _pickle.load(_f)
        df_prepared = _cached["df"]
        X = _cached["X"]
        y = _cached["y"]
        groups = _cached["groups"]
        feature_cols = _cached["feature_cols"]
        # Apply suffix exclusions declared in config (e.g. ["_std"]) so that
        # the cache path and the slow path produce identical feature spaces.
        _excl = config.get("exclude_column_suffixes", [])
        if _excl:
            _keep = [c for c in feature_cols if not any(c.endswith(s) for s in _excl)]
            X = X[_keep]
            feature_cols = _keep
        print(f"  Cache loaded in {__import__('time').time() - _t0:.1f}s "
              f"({len(X)} samples × {len(feature_cols)} features)")
    else:
        # Slow path: load from raw CSV files
        df_prepared, X, y, groups, feature_cols = prepare_data_for_contrast(
            config, contrast_name, verbose=verbose
        )
        X = filter_features_by_family(X, family_name, family_prefixes)
        feature_cols = X.columns.tolist()
    # ──────────────────────────────────────────────────────────────────────────

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

    true_ws_metrics_list = []
    true_shap_values = []
    true_ws_metrics_df = pd.DataFrame()

    if n_runs > 0:
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
            n_subject_jobs=n_subject_jobs,
            cv_n_jobs=cv_n_jobs,
            lmm_n_jobs=lmm_n_jobs,
            lmm_prefilter_factor=lmm_prefilter_factor,
            smote_k_neighbors=smote_k_neighbors,
            logger=logger,
            run_idx_offset=run_idx_offset,
            total_n_runs=total_n_runs_for_seeds,
        )

    if n_runs > 0 and not true_ws_metrics_list:
        print("Done. No applicable subjects to run.")
        sys.exit(0)

    if n_runs > 0 and true_ws_metrics_list:
        # AUC vs on/off scale dispersion scatter plot
        _ws_subject_auc_df = (
            pd.DataFrame([{"subject": m["subject"], "auc": m["mean_auc"]}
                          for m in true_ws_metrics_list])
            .groupby("subject")["auc"]
            .mean()
            .reset_index()
        )
        _dim_path = results_path
        _fname_base = f"{model_type}_loso_{total_n_runs_for_seeds}runs" if total_n_runs_for_seeds > 1 else f"{model_type}_loso"
        _label_col = config.get("label_contrasts", {}).get(contrast_name, {}).get("column_name", "onoff")
        plot_auc_vs_onoff_dispersion(
            _ws_subject_auc_df, df_prepared, _dim_path, _fname_base, "WithinSubject",
            label_col=_label_col,
        )
        plot_auc_vs_class_imbalance(
            _ws_subject_auc_df, df_prepared, _dim_path, _fname_base, "WithinSubject",
            label_col=_label_col,
        )

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
            n_perm_jobs=n_perm_jobs,
            n_subject_jobs=n_subject_jobs,
            cv_n_jobs=cv_n_jobs,
            lmm_n_jobs=lmm_n_jobs,
            lmm_prefilter_factor=lmm_prefilter_factor,
            smote_k_neighbors=smote_k_neighbors,
            logger=logger,
            perm_idx_offset=perm_idx_offset,
            total_n_permutations=total_n_perms_for_seeds,
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
