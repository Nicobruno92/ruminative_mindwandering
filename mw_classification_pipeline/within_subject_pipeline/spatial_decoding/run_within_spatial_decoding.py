#!/usr/bin/env python
"""
Within-Subject spatial decoding driver (max-statistic FWER design).

Per-electrode searchlight using the *existing* within-subject engine restricted to each
channel's columns. The per-channel statistic is the GROUP-MEAN AUC across subjects.

  TRUE mode (default):       group-mean AUC per channel (n_runs averaged).
  PERMUTATION mode (--perm_idx P): one within-subject label shuffle (seed=P) scored
      across ALL channels (n_runs=1); the per-permutation max-over-channels group-mean
      AUC is a family-wise null draw.

USAGE:
    python run_within_spatial_decoding.py --config config.yaml \
        --contrast on_vs_off_within_median --channel Pz
    python run_within_spatial_decoding.py --config config.yaml \
        --contrast on_vs_off_within_median --perm_idx 0
"""

import os
import sys
import argparse
import tempfile
import yaml
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from utils.data_utils import load_config, get_project_root
from utils.analysis_utils import run_within_subject_distribution_analysis
from utils.spatial_decoding_utils import (
    parse_channels_from_columns,
    select_channel_columns,
    load_or_prepare_data,
    permute_labels_within_subject,
)


def build_results_path(config: dict, contrast: str, family: str, model: str) -> str:
    """results_dir/SpatialDecoding/WithinSubject/{contrast}/{family}/{model}/"""
    results_root = config.get("project", {}).get("results_dir", "results/MW_Classification")
    if not os.path.isabs(results_root):
        results_root = str(get_project_root() / results_root)
    path = os.path.join(results_root, "SpatialDecoding", "WithinSubject", contrast, family, model)
    os.makedirs(path, exist_ok=True)
    return path


def make_common(config: dict, model: str) -> dict:
    """Shared kwargs for the within-subject engine, drawn from the config blocks."""
    fs = config.get("feature_selection", {})
    cv = config.get("cv", {})
    osc = config.get("oversampling", {})
    prep = config.get("preprocessing", {})
    cls = config.get("classifiers", {})
    par = config.get("parallelism", {})
    return dict(
        config=config, model_type=model,
        use_smote=osc.get("use_smote", False), oversampling_method=osc.get("method", "SMOTE"),
        oversampling_scope=osc.get("scope", "within"),
        cv_strategy=cv.get("strategy", "stratified_kfold"), cv_folds=cv.get("folds", 5),
        min_samples_per_class=cv.get("min_samples_per_class", 0),
        min_minority_ratio=config.get("min_minority_ratio", 0.0),
        k=fs.get("k", 10), rf_params=cls.get("rf", {}),
        feature_selection_method=fs.get("method", "mrmr"),
        scaler=prep.get("scaler", "standard"),
        smote_k_neighbors=osc.get("k_neighbors", 5),
        n_subject_jobs=par.get("n_subject_jobs", 1), cv_n_jobs=par.get("cv_n_jobs", 1),
        lmm_n_jobs=fs.get("lmm_n_jobs", 1), lmm_prefilter_factor=fs.get("lmm_prefilter_factor", 0),
        save_pickle=False, save_csv=False, save_probabilities=False,
        save_plots=False, save_shap=False, verbose=False,
    )


def _group_mean_auc(X_ch, y, df, subjects, tasks, contrast, common, n_runs):
    """
    Group-mean AUC across subjects for one channel.

    Returns (mean_auc, std_auc, auc_single):
      mean_auc   — group mean of each subject's run-averaged AUC (stable; for the map).
      auc_single — group mean of each subject's FIRST-run AUC (matched to the single-pass
                   permutation statistic; used for the FWER test).
    """
    with tempfile.TemporaryDirectory(prefix="spatial_") as scratch:
        metrics, _shap = run_within_subject_distribution_analysis(
            dimension=contrast, df=df, X=X_ch, y=y, subjects=subjects, tasks=tasks,
            feature_cols=X_ch.columns.tolist(), n_runs=n_runs, results_path=scratch, **common,
        )
    subj_df = pd.DataFrame(metrics)
    if subj_df.empty:
        return float("nan"), float("nan"), float("nan")
    per_subject = subj_df.groupby("subject")["mean_auc"].mean()
    mean_auc, std_auc = float(per_subject.mean()), float(per_subject.std())
    if "run_idx" in subj_df.columns:
        first = subj_df.loc[subj_df["run_idx"] == subj_df["run_idx"].min()]
        auc_single = float(first.groupby("subject")["mean_auc"].mean().mean())
    else:
        auc_single = mean_auc
    return mean_auc, std_auc, auc_single


def parse_args():
    ap = argparse.ArgumentParser(description="Within-Subject spatial decoding (max-stat FWER)")
    ap.add_argument("--config", required=True)
    ap.add_argument("--contrast", required=True)
    ap.add_argument("--channel", default=None, help="TRUE mode: single channel shard. None = all.")
    ap.add_argument("--perm_idx", type=int, default=None,
                    help="PERMUTATION mode: one shuffle (seed=perm_idx) across all channels.")
    return ap.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    config["contrast"] = args.contrast
    family = config.get("active_family") or "all"
    config["active_family"] = family
    config["epoch_types"] = config["feature_families"][family]["epoch_types"]
    model = config.get("model_type", "rf")
    n_runs = config.get("n_runs", 20)
    common = make_common(config, model)

    results_root = config.get("project", {}).get("results_dir", "results/MW_Classification")
    if not os.path.isabs(results_root):
        results_root = str(get_project_root() / results_root)
    cache_dir = os.path.join(results_root, "data_cache")
    prefixes = config["feature_families"][family].get("prefixes")

    mode = "PERM" if args.perm_idx is not None else "TRUE"
    print(f"\n{'='*60}\nWS spatial decoding [{mode}] — {args.contrast} / {family} / {model}\n{'='*60}")
    df, X, y, groups, _cols = load_or_prepare_data(
        config, args.contrast, family, prefixes, cache_dir, verbose=False
    )
    tasks = df["task"] if "task" in df.columns else groups
    all_channels = parse_channels_from_columns(X.columns.tolist())
    sd = config["spatial_decoding"]
    if sd.get("channels", "all") != "all":
        all_channels = [c for c in all_channels if c in sd["channels"]]

    results_path = build_results_path(config, args.contrast, family, model)
    with open(os.path.join(results_path, "used_config.yaml"), "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"Samples={len(X)} | channels={len(all_channels)}")

    if mode == "PERM":
        y_perm = permute_labels_within_subject(y, groups, seed=args.perm_idx)
        rows = []
        for i, ch in enumerate(all_channels):
            X_ch = select_channel_columns(X, ch)
            auc, _, _ = _group_mean_auc(X_ch, y_perm, df, groups, tasks, args.contrast, common, n_runs=1)
            rows.append({"perm_idx": args.perm_idx, "channel": ch, "auc": auc})
            print(f"  [perm {args.perm_idx}] [{i+1}/{len(all_channels)}] {ch}: group_auc={auc:.4f}")
        perm_dir = os.path.join(results_path, "perms")
        os.makedirs(perm_dir, exist_ok=True)
        out = os.path.join(perm_dir, f"perm-{args.perm_idx}.csv")
        pd.DataFrame(rows).to_csv(out, index=False)
        max_auc = float(np.nanmax([r["auc"] for r in rows])) if rows else float("nan")
        print(f"Perm {args.perm_idx} max-over-channels group AUC = {max_auc:.4f}\nWritten: {out}")
        return

    channels = [args.channel] if args.channel else all_channels
    rows = []
    for i, ch in enumerate(channels):
        X_ch = select_channel_columns(X, ch)
        mean_auc, std_auc, auc_single = _group_mean_auc(X_ch, y, df, groups, tasks,
                                                        args.contrast, common, n_runs)
        rows.append({"channel": ch, "n_features": X_ch.shape[1],
                     "mean_auc": mean_auc, "std_auc": std_auc, "auc_single": auc_single})
        print(f"  [true] [{i+1}/{len(channels)}] {ch}: n_feat={X_ch.shape[1]} "
              f"group_auc={mean_auc:.4f} auc_single={auc_single:.4f}")
    true_dir = os.path.join(results_path, "true")
    os.makedirs(true_dir, exist_ok=True)
    out = (os.path.join(true_dir, f"channel-{args.channel}.csv") if args.channel
           else os.path.join(true_dir, "true_per_channel_auc.csv"))
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"True group AUC written: {out}")


if __name__ == "__main__":
    main()
