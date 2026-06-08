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
from utils.ml_utils import compute_within_subject_confidence_weights
from utils.logging_utils import AnalysisLogger
from utils.plotting_utils import (
    plot_subject_level_densities,
    plot_shap_comparative_boxplots,
    generate_all_comparison_plots,
    plot_auc_vs_onoff_dispersion,
    plot_auc_vs_class_imbalance,
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
    parser.add_argument("--run_idx", type=int, default=None,
                        help="SLURM array mode: run only this true-run index (0-based). "
                             "Skips permutations. Use merge_loso_results.py to aggregate.")
    parser.add_argument("--perm_idx", type=int, default=None,
                        help="SLURM array mode: run only this permutation index (0-based). "
                             "Skips true runs. Use merge_loso_results.py to aggregate.")
    return parser.parse_args()


# =============================================================================
# FEATURE FAMILY FILTERING
# =============================================================================

def resolve_family(family_name: str, feature_families: dict) -> tuple:
    """
    Extract epoch_types and prefixes from a feature family config entry.

    Each family entry is a dict with keys ``epoch_types`` and ``prefixes``.
    Returns a (epoch_types_list, prefixes_or_None) tuple.

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
        raise ValueError(
            f"Unknown feature family: '{family_name}'. "
            f"Available: {list(feature_families.keys())}"
        )
    cfg = feature_families[family_name]
    if not isinstance(cfg, dict):
        raise ValueError(
            f"feature_families.{family_name} must be a dict with 'epoch_types' and "
            f"'prefixes' keys. Got: {cfg!r}"
        )
    epoch_types = cfg.get("epoch_types", ["state"])
    prefixes = cfg.get("prefixes")
    return epoch_types, prefixes


def filter_features_by_family(
    X: pd.DataFrame,
    family_name: str,
    prefixes: list,
) -> pd.DataFrame:
    """
    Filter the feature matrix to columns matching the given prefixes.

    Matching uses word-boundary logic so that prefix ``P1`` matches
    ``Pz_P1`` and ``Pz_P1_latency`` but NOT ``Pz_P10``.

    Rule: a column matches prefix ``p`` if:
      - ``_{p}`` appears at the end of the column name, OR
      - ``_{p}_`` appears anywhere in the column name, OR
      - the column starts with ``{p}_`` or equals ``{p}``

    Parameters
    ----------
    X : pd.DataFrame
        Full feature matrix.
    family_name : str
        Family name (used only for logging/error messages).
    prefixes : list[str] or None
        Marker component names to keep. None means keep all.

    Returns
    -------
    pd.DataFrame
        Filtered feature matrix.
    """
    import re

    if prefixes is None:
        return X

    def _matches(col: str, prefix: str) -> bool:
        p = re.escape(prefix)
        return bool(re.search(rf"_{p}(_|$)", col)) or bool(re.match(rf"{p}(_|$)", col))

    selected_cols = [col for col in X.columns if any(_matches(col, p) for p in prefixes)]

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

    Structure: results_root/LOSO/{contrast_name}/{family_name}/{model_type}/

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

    loso_path = os.path.join(results_root, "LOSO", contrast_name, family_name, model_type)
    os.makedirs(loso_path, exist_ok=True)
    return loso_path


def build_confidence_sample_weights(
    config: dict,
    contrast_name: str,
    df_prepared: "pd.DataFrame",
    groups: "pd.Series",
):
    """
    Build per-trial training sample weights from probe confidence, if enabled.

    Reads the optional ``confidence_weight`` block of the active contrast. When
    absent or ``enabled: false`` this returns ``None`` and the pipeline behaves
    exactly as before. When enabled, confidence is normalised within subject (see
    :func:`compute_within_subject_confidence_weights`) into weights aligned to
    ``df_prepared`` / ``groups`` (positional order, matching X / y).

    Confidence is a metacognitive probe rating used here as a proxy for label
    reliability: more-confident trials weigh more in the classifier loss. The
    rationale is strongest for the ON/OFF contrasts (confidence pertains chiefly
    to the on/off judgement); using it on content dimensions is exploratory.

    Parameters
    ----------
    config : dict
        Full pipeline configuration.
    contrast_name : str
        Active label contrast key.
    df_prepared : pd.DataFrame
        Prepared probe-level data; must contain a ``confidence`` column.
    groups : pd.Series
        Subject ID per sample, positionally aligned to ``df_prepared``.

    Returns
    -------
    np.ndarray or None
        Per-trial weights in ``[w_min, 1]``, or ``None`` if weighting is off.
    """
    cw_cfg = (
        config.get("label_contrasts", {})
        .get(contrast_name, {})
        .get("confidence_weight")
    )
    if not cw_cfg or not cw_cfg.get("enabled", False):
        return None

    if "confidence" not in df_prepared.columns:
        raise ValueError(
            f"confidence_weight is enabled for contrast '{contrast_name}' but the "
            f"prepared data has no 'confidence' column."
        )

    confidence = df_prepared["confidence"].to_numpy(dtype=float)
    if np.isnan(confidence).any():
        raise ValueError(
            f"confidence_weight is enabled for contrast '{contrast_name}' but the "
            f"'confidence' column contains NaNs ({int(np.isnan(confidence).sum())} "
            f"of {len(confidence)}). Resolve missing confidence before weighting."
        )

    return compute_within_subject_confidence_weights(
        confidence,
        groups.to_numpy(),
        w_min=cw_cfg.get("w_min", 0.1),
        normalization=cw_cfg.get("normalization", "within_subject"),
    )


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
    feature_families = config.get("feature_families", {})
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
    par_cfg = config.get("parallelism", {})
    n_perm_jobs = par_cfg.get("n_perm_jobs", 1)
    perm_cv_n_jobs = par_cfg.get("perm_cv_n_jobs", -1)
    true_cv_n_jobs = par_cfg.get("true_cv_n_jobs", -1)

    # ── SLURM parallel mode ────────────────────────────────────────────────────
    # When --run_idx or --perm_idx are provided, this job runs exactly ONE pass
    # and saves to the per-job subdirectory. The merge_loso_results.py script
    # aggregates all jobs after they complete.
    run_idx_offset = 0
    perm_idx_offset = 0
    total_n_runs_for_seeds = n_runs
    total_n_perms_for_seeds = permutation_runs

    if args.run_idx is not None and args.perm_idx is not None:
        raise ValueError("Cannot specify both --run_idx and --perm_idx in the same job.")

    if args.run_idx is not None:
        if args.run_idx >= n_runs:
            raise ValueError(f"--run_idx {args.run_idx} is out of range (n_runs={n_runs})")
        run_idx_offset = args.run_idx
        n_runs = 1
        permutation_runs = 0  # perms are submitted as separate jobs
        print(f"[SLURM mode] True run {args.run_idx} of {total_n_runs_for_seeds}")

    if args.perm_idx is not None:
        if args.perm_idx >= permutation_runs:
            raise ValueError(f"--perm_idx {args.perm_idx} is out of range (permutation_runs={permutation_runs})")
        perm_idx_offset = args.perm_idx
        n_runs = 0           # true runs are submitted as separate jobs
        permutation_runs = 1
        print(f"[SLURM mode] Permutation {args.perm_idx} of {total_n_perms_for_seeds}")
    # ──────────────────────────────────────────────────────────────────────────

    # Resolve family → inject epoch_types into config so the loader knows what to load
    family_epoch_types, family_prefixes = resolve_family(family_name, feature_families)
    config["epoch_types"] = family_epoch_types

    print(f"\n{'='*60}")
    print(f"Mind-Wandering Classification Pipeline (LOSO)")
    print(f"{'='*60}")
    print(f"Contrast    : {contrast_name}")
    print(f"Family      : {family_name}")
    print(f"Epoch types : {family_epoch_types}")
    print(f"Model       : {model_type.upper()}")
    print(f"Runs        : {n_runs}")
    print(f"{'='*60}\n")

    logger = AnalysisLogger()

    # Load only the epoch types declared by the selected family
    df_prepared, X, y, groups, feature_cols = prepare_data_for_contrast(
        config, contrast_name, verbose=verbose
    )

    if len(X) < 50:
        print(f"ERROR: Insufficient data ({len(X)} samples). Need at least 50.")
        sys.exit(1)

    # Filter columns to the family's declared prefixes
    X = filter_features_by_family(X, family_name, family_prefixes)
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
    # Dedicated, human-readable record of which subjects entered the analysis and
    # which were excluded (and why). Written per classification for transparency.
    provenance = config.get("_data_provenance")
    if provenance is not None:
        with open(os.path.join(results_path, "subject_exclusions.yaml"), "w") as f:
            yaml.dump(provenance, f, default_flow_style=False, sort_keys=False)
        print(
            f"Subjects: {provenance['n_subjects_final']}/{provenance['n_subjects_requested']} "
            f"kept | excluded: {len(provenance['excluded_no_data_or_all_neutral'])} no-data, "
            f"{len(provenance['excluded_min_samples'])} low-count, "
            f"{len(provenance['excluded_min_minority_ratio'])} imbalanced"
        )
    print(f"Results → {results_path}")

    # For compatibility with analysis_utils.get_model_results_folder
    config["current_model"] = contrast_name
    config["results_folder_pattern"] = ""

    model_params = extract_model_params(config)

    # -------------------------------------------------------------------------
    # Main classification
    # -------------------------------------------------------------------------
    smote_k_neighbors = config.get("smote_k_neighbors", 5)
    _fs_method = config.get("feature_selection_method", "mrmr")
    _fs_k = config.get("k", 20)
    if _fs_method == "none":
        _fs_msg = f"Feature selection : disabled (using all {len(feature_cols)} features)"
    else:
        _fs_msg = f"Feature selection : {_fs_method} (k={_fs_k}/{len(feature_cols)}, refit per fold)"
    # Optional confidence-based sample weighting (None unless enabled per contrast)
    sample_weights = build_confidence_sample_weights(
        config, contrast_name, df_prepared, groups
    )

    results_df = None

    if n_runs > 0:
        print(f"\n{'='*60}")
        print(f"Running LOSO classification ({n_runs} runs)...")
        print(_fs_msg)
        if sample_weights is not None:
            print(
                f"Confidence weighting : ENABLED "
                f"(within-subject, w_min="
                f"{config['label_contrasts'][contrast_name]['confidence_weight'].get('w_min', 0.1)})"
            )
        print(f"{'='*60}\n")

    if n_runs > 0:
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
        plot_style=config.get("plot_style", "seaborn-v0_8"),
        verbose=verbose,
        feature_selection_method=config.get("feature_selection_method", "mrmr"),
        scaler=config.get("scaler", "standard"),
        use_pca=config.get("use_pca", False),
        pca_n_components=config.get("pca_n_components"),
        pca_type=config.get("pca_type", "standard"),
        pca_kernel=config.get("pca_kernel", "rbf"),
        smote_k_neighbors=smote_k_neighbors,
        sample_weights=sample_weights,
        logger=logger,
        cv_n_jobs=true_cv_n_jobs,
        run_idx_offset=run_idx_offset,
        total_n_runs=total_n_runs_for_seeds,
    )

        if results_df.empty:
            print("ERROR: No successful runs completed.")
            sys.exit(1)

        # AUC vs on/off scale dispersion scatter plot
        _subject_rows = []
        for r in true_all_results:
            for sm in (r.get("loso_subject_metrics") or []):
                _subject_rows.append({"subject": sm["subject"], "auc": sm["auc"]})
        if _subject_rows:
            _loso_subject_auc_df = (
                pd.DataFrame(_subject_rows)
                .groupby("subject")["auc"]
                .mean()
                .reset_index()
            )
            _dim_path = results_path
            _fname_base = f"{model_type}_loso_{total_n_runs_for_seeds}runs" if total_n_runs_for_seeds > 1 else f"{model_type}_loso"
            _contrast_cfg = config.get("label_contrasts", {}).get(contrast_name, {})
            _label_col = _contrast_cfg.get("column_name") or _contrast_cfg.get("label_source", "onoff")
            plot_auc_vs_onoff_dispersion(
                _loso_subject_auc_df, df_prepared, _dim_path, _fname_base, "LOSO",
                label_col=_label_col,
            )
            plot_auc_vs_class_imbalance(
                _loso_subject_auc_df, df_prepared, _dim_path, _fname_base, "LOSO",
                label_col=_label_col,
            )

        # Print summary
        print(f"\n{'='*60}")
        print(f"LOSO Results — {contrast_name} / {family_name}")
        print(f"{'='*60}")

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
    # In SLURM perm-only mode (--perm_idx), results_df is None because true
    # runs are submitted as separate jobs. True metrics are passed as empty
    # lists; merge_loso_results.py computes p-values after all jobs complete.
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
            plot_style=config.get("plot_style", "seaborn-v0_8"),
            verbose=verbose,
            feature_selection_method=config.get("feature_selection_method", "mrmr"),
            scaler=config.get("scaler", "standard"),
            use_pca=config.get("use_pca", False),
            pca_n_components=config.get("pca_n_components"),
            pca_type=config.get("pca_type", "standard"),
            pca_kernel=config.get("pca_kernel", "rbf"),
            true_auc_list=(results_df['mean_auc'].dropna().tolist()
                           if results_df is not None and 'mean_auc' in results_df.columns else []),
            true_bal_acc_list=(results_df['mean_balanced_accuracy'].dropna().tolist()
                               if results_df is not None and 'mean_balanced_accuracy' in results_df.columns else []),
            true_auprc_list=(results_df['mean_auprc'].dropna().tolist()
                             if results_df is not None and 'mean_auprc' in results_df.columns else []),
            true_mcc_list=(results_df['mean_mcc'].dropna().tolist()
                           if results_df is not None and 'mean_mcc' in results_df.columns else []),
            permutation_scope=config.get("permutation_scope", "within"),
            smote_k_neighbors=smote_k_neighbors,
            sample_weights=sample_weights,
            logger=logger,
            n_runs=n_runs,
            n_perm_jobs=n_perm_jobs,
            cv_n_jobs=perm_cv_n_jobs,
            perm_idx_offset=perm_idx_offset,
            total_n_permutations=total_n_perms_for_seeds,
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
