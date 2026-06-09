#!/usr/bin/env python
"""
LOSO spatial decoding driver (max-statistic FWER design).

Per-electrode searchlight using the *existing* LOSO engine restricted to each
channel's columns. Two modes, matching the two SLURM arrays:

  TRUE mode (default):      compute the n_runs-averaged true AUC per channel.
      one task per (contrast, channel) via --channel, or all channels in-process.

  PERMUTATION mode (--perm_idx P): draw ONE within-subject label shuffle (seed=P)
      and score ALL channels with n_runs=1 under that shuffle. The same shuffle is
      applied across channels so the per-permutation max-over-channels AUC is a valid
      family-wise null draw. One task per (contrast, permutation index).

merge_spatial_results.py then combines the true AUCs and the permutation max-null into
per-channel FWER p-values + topomaps.

USAGE:
    # true AUC for one electrode (SLURM true array)
    python run_loso_spatial_decoding.py --config config.yaml \
        --contrast ON_vs_OFF_within_median --channel Pz
    # one permutation across all electrodes (SLURM perm array)
    python run_loso_spatial_decoding.py --config config.yaml \
        --contrast ON_vs_OFF_within_median --perm_idx 0
"""

import os
import sys
import argparse
import tempfile
import yaml
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from utils.data_utils import load_config, get_project_root
from utils.analysis_utils import run_distribution_analysis
from utils.spatial_decoding_utils import (
    parse_channels_from_columns,
    select_channel_columns,
    load_or_prepare_data,
    permute_labels_within_subject,
)


_PARAM_SPECS = {
    "rf": ["n_estimators", "max_depth", "min_samples_split", "min_samples_leaf",
           "max_features", "bootstrap", "n_jobs", "class_weight"],
    "xgb": ["n_estimators", "max_depth", "learning_rate", "objective", "seed",
            "n_jobs", "scale_pos_weight"],
    "lr": ["penalty", "solver", "class_weight", "C", "l1_ratio", "max_iter"],
    "ocsvm": ["nu", "kernel", "gamma"],
    "iforest": ["n_estimators", "contamination"],
}


def extract_model_params(config: dict) -> dict:
    """Extract model-specific param dicts from the flat LOSO config (e.g. rf_n_estimators)."""
    out = {}
    for model_key, names in _PARAM_SPECS.items():
        out[model_key] = {p: config[f"{model_key}_{p}"] for p in names
                          if config.get(f"{model_key}_{p}") is not None}
    return out


def resolve_model_type(config: dict) -> str:
    mt = config.get("model_type", "rf")
    return mt[0] if isinstance(mt, list) else mt


def build_results_path(config: dict, contrast: str, family: str, model: str) -> str:
    """results_root/SpatialDecoding/LOSO/{contrast}/{family}/{model}/"""
    results_root = config["data_paths"].get("results_root", "results/MW_Classification")
    if not os.path.isabs(results_root):
        results_root = str(get_project_root() / results_root)
    path = os.path.join(results_root, "SpatialDecoding", "LOSO", contrast, family, model)
    os.makedirs(path, exist_ok=True)
    return path


def _score_channel(X_ch, y, df, groups, contrast, config, model_type, model_params, n_runs):
    """One searchlight cell: mean AUC over n_runs of LOSO for a single channel."""
    with tempfile.TemporaryDirectory(prefix="spatial_") as scratch:
        results_df, _all, _shap = run_distribution_analysis(
            dimension=contrast, df=df, X=X_ch, y=y, groups=groups,
            feature_cols=X_ch.columns.tolist(), config=config,
            n_runs=n_runs, results_path=scratch, model_type=model_type,
            use_smote=config.get("use_smote", False),
            oversampling_method=config.get("oversampling_method", "SMOTE"),
            oversampling_scope=config.get("oversampling_scope", "within"),
            k=config.get("k", 10),
            rf_params=model_params["rf"], xgb_params=model_params["xgb"],
            lr_params=model_params["lr"], ocsvm_params=model_params["ocsvm"],
            iforest_params=model_params["iforest"],
            feature_selection_method=config.get("feature_selection_method", "mrmr"),
            scaler=config.get("scaler", "standard"),
            smote_k_neighbors=config.get("smote_k_neighbors", 5),
            save_pickle=False, save_csv=False, save_probabilities=False,
            save_plots=False, save_shap=False, verbose=False,
            cv_n_jobs=config.get("parallelism", {}).get("true_cv_n_jobs", 1),
        )
    aucs = results_df["mean_auc"].dropna().tolist()
    if not aucs:
        return float("nan"), float("nan"), float("nan")
    # auc_single = first run (run_idx 0), matched to the single-pass permutation statistic.
    return float(np.mean(aucs)), float(np.std(aucs)), float(aucs[0])


def parse_args():
    ap = argparse.ArgumentParser(description="LOSO spatial decoding (max-stat FWER)")
    ap.add_argument("--config", required=True)
    ap.add_argument("--contrast", required=True)
    ap.add_argument("--channel", default=None,
                    help="TRUE mode: single channel shard. None = all channels.")
    ap.add_argument("--perm_idx", type=int, default=None,
                    help="PERMUTATION mode: run one shuffle (seed=perm_idx) across all channels.")
    return ap.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    config["contrast"] = args.contrast
    family = config.get("active_family") or "all"
    config["active_family"] = family
    config["epoch_types"] = config["feature_families"][family]["epoch_types"]
    model_type = resolve_model_type(config)
    model_params = extract_model_params(config)
    n_runs = config.get("n_runs", 20)

    results_root = config["data_paths"].get("results_root", "results/MW_Classification")
    if not os.path.isabs(results_root):
        results_root = str(get_project_root() / results_root)
    cache_dir = os.path.join(results_root, "data_cache")
    prefixes = config["feature_families"][family].get("prefixes")

    mode = "PERM" if args.perm_idx is not None else "TRUE"
    print(f"\n{'='*60}\nLOSO spatial decoding [{mode}] — {args.contrast} / {family} / {model_type}\n{'='*60}")
    df, X, y, groups, _cols = load_or_prepare_data(
        config, args.contrast, family, prefixes, cache_dir, verbose=False
    )
    all_channels = parse_channels_from_columns(X.columns.tolist())
    sd = config["spatial_decoding"]
    if sd.get("channels", "all") != "all":
        all_channels = [c for c in all_channels if c in sd["channels"]]

    results_path = build_results_path(config, args.contrast, family, model_type)
    with open(os.path.join(results_path, "used_config.yaml"), "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"Samples={len(X)} | channels={len(all_channels)}")

    if mode == "PERM":
        # One within-subject shuffle applied to every channel -> one max-null draw.
        y_perm = permute_labels_within_subject(y, groups, seed=args.perm_idx)
        rows = []
        for i, ch in enumerate(all_channels):
            X_ch = select_channel_columns(X, ch)
            auc, _, _ = _score_channel(X_ch, y_perm, df, groups, args.contrast,
                                       config, model_type, model_params, n_runs=1)
            rows.append({"perm_idx": args.perm_idx, "channel": ch, "auc": auc})
            print(f"  [perm {args.perm_idx}] [{i+1}/{len(all_channels)}] {ch}: auc={auc:.4f}")
        import pandas as pd
        perm_dir = os.path.join(results_path, "perms")
        os.makedirs(perm_dir, exist_ok=True)
        out = os.path.join(perm_dir, f"perm-{args.perm_idx}.csv")
        pd.DataFrame(rows).to_csv(out, index=False)
        max_auc = float(np.nanmax([r["auc"] for r in rows])) if rows else float("nan")
        print(f"Perm {args.perm_idx} max-over-channels AUC = {max_auc:.4f}\nWritten: {out}")
        return

    # TRUE mode
    channels = [args.channel] if args.channel else all_channels
    import pandas as pd
    rows = []
    for i, ch in enumerate(channels):
        X_ch = select_channel_columns(X, ch)
        mean_auc, std_auc, auc_single = _score_channel(X_ch, y, df, groups, args.contrast,
                                                       config, model_type, model_params, n_runs)
        rows.append({"channel": ch, "n_features": X_ch.shape[1],
                     "mean_auc": mean_auc, "std_auc": std_auc, "auc_single": auc_single})
        print(f"  [true] [{i+1}/{len(channels)}] {ch}: n_feat={X_ch.shape[1]} "
              f"mean_auc={mean_auc:.4f} auc_single={auc_single:.4f}")
    true_dir = os.path.join(results_path, "true")
    os.makedirs(true_dir, exist_ok=True)
    if args.channel is not None:
        out = os.path.join(true_dir, f"channel-{args.channel}.csv")
    else:
        out = os.path.join(true_dir, "true_per_channel_auc.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"True AUC written: {out}")


if __name__ == "__main__":
    main()
