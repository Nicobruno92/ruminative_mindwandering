#!/usr/bin/env python
"""
Post-hoc SHAP extraction for the FWER-significant electrode cluster of a
spatial-decoding searchlight run.

The searchlight (``run_loso_spatial_decoding.py`` / ``run_within_spatial_decoding.py``)
never computes SHAP by default: fitting an explainer for every one of the ~64
electrodes, for every TRUE run AND every permutation draw, would be far too slow.
Once ``merge_spatial_results.py`` has produced ``per_channel_metrics.csv`` with the
FWER-significant electrodes flagged (``sig`` column), this script re-fits ONLY
those electrodes' models — same contrast, family, model, and n_runs as the
original run, read straight from its ``used_config.yaml`` — with SHAP enabled, so
per-marker importances become available for exactly the cluster that survived
correction.

Requires the ``ML`` conda environment (sklearn / SHAP / joblib).

USAGE:
    python scripts/extract_cluster_shap.py --results_dir \\
        results/MW_Classification/SpatialDecoding/LOSO/ON_vs_OFF_within_median/all/rf

Outputs (under ``{results_dir}/shap_cluster/``):
    {channel}/true_runs/run_*/..._shap_values.pkl   — one pickle per run (LOSO)
                                                        or per run×subject (WS),
                                                        written by the existing
                                                        ``_save_shap_values`` helper.
    cluster_shap_summary.csv                         — one row per (channel, marker):
                                                        mean(|SHAP|) averaged over
                                                        every pickle for that channel.
"""

from __future__ import annotations

import os
import pickle
import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.data_utils import get_project_root
from utils.spatial_decoding_utils import select_channel_columns, load_or_prepare_data


# =============================================================================
# CONFIG / RESULTS-DIR RESOLUTION
# =============================================================================

def detect_pipeline(results_dir: str) -> str:
    """
    Infer 'loso' or 'within_subject' from the results-dir path.

    Both drivers write to a fixed convention documented in their own
    ``build_results_path``: ``{results_root}/SpatialDecoding/{LOSO|WithinSubject}/...``.

    Raises
    ------
    ValueError
        If neither path segment is present.
    """
    parts = Path(results_dir).resolve().parts
    if "LOSO" in parts:
        return "loso"
    if "WithinSubject" in parts:
        return "within_subject"
    raise ValueError(
        f"Could not infer pipeline from '{results_dir}' — expected a "
        "'.../SpatialDecoding/LOSO/...' or '.../SpatialDecoding/WithinSubject/...' path."
    )


def load_used_config(results_dir: str) -> dict:
    """Load the exact config the searchlight TRUE-mode run used (written by its own main())."""
    cfg_path = os.path.join(results_dir, "used_config.yaml")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(
            f"{cfg_path} not found — the TRUE-mode searchlight run must complete "
            "(and write used_config.yaml) before extracting cluster SHAP."
        )
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def resolve_results_root(config: dict, pipeline: str) -> str:
    """Mirror each driver's own results-root resolution (data_paths vs project block)."""
    if pipeline == "loso":
        root = config["data_paths"].get("results_root", "results/MW_Classification")
    else:
        root = config.get("project", {}).get("results_dir", "results/MW_Classification")
    if not os.path.isabs(root):
        root = str(get_project_root() / root)
    return root


def significant_channels(results_dir: str) -> list[str]:
    """Read per_channel_metrics.csv (written by merge_spatial_results.py) for the FWER-significant channels."""
    metrics_path = os.path.join(results_dir, "per_channel_metrics.csv")
    if not os.path.exists(metrics_path):
        raise FileNotFoundError(
            f"{metrics_path} not found — run merge_spatial_results.py for this "
            "contrast/family/model first."
        )
    metrics = pd.read_csv(metrics_path)
    return sorted(metrics.loc[metrics["sig"], "channel"].tolist())


# =============================================================================
# SHAP AGGREGATION
# =============================================================================

def aggregate_channel_shap(shap_dir: str) -> "tuple[pd.DataFrame, int] | tuple[None, int]":
    """
    Average |SHAP| per feature across every pickle written under ``shap_dir``.

    Each pickle (one per run for LOSO, one per run×subject for Within-Subject —
    both written by ``utils.analysis_utils._save_shap_values``) already carries
    its own ``feature_names``, so alignment does not depend on directory layout.

    Returns
    -------
    (pd.DataFrame or None, int)
        DataFrame with columns ``feature``, ``mean_abs_shap`` (None if no
        pickle was found), and the number of pickles aggregated.
    """
    files = sorted(Path(shap_dir).rglob("*_shap_values.pkl"))
    if not files:
        return None, 0

    feature_names = None
    abs_means = []
    for f in files:
        with open(f, "rb") as fh:
            payload = pickle.load(fh)
        if feature_names is None:
            feature_names = list(payload["feature_names"])
        abs_means.append(np.abs(payload["shap_values"]).mean(axis=0))

    mean_abs_shap = np.mean(abs_means, axis=0)
    df = pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs_shap})
    return df, len(files)


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results_dir", required=True,
                    help="…/SpatialDecoding/{LOSO|WithinSubject}/{contrast}/{family}/{model}")
    ap.add_argument("--n_runs", type=int, default=None,
                    help="Override n_runs (default: same as the searchlight run, from used_config.yaml).")
    args = ap.parse_args()

    pipeline = detect_pipeline(args.results_dir)
    channels = significant_channels(args.results_dir)
    if not channels:
        print(f"No FWER-significant channels in {args.results_dir}/per_channel_metrics.csv — nothing to do.")
        return

    config = load_used_config(args.results_dir)
    family = config["active_family"]
    prefixes = config["feature_families"][family].get("prefixes")
    n_runs = args.n_runs if args.n_runs is not None else config.get("n_runs", 20)

    results_root = resolve_results_root(config, pipeline)
    cache_dir = os.path.join(results_root, "data_cache")
    df, X, y, groups, _cols = load_or_prepare_data(
        config, config["contrast"], family, prefixes, cache_dir, verbose=False
    )

    shap_root = os.path.join(args.results_dir, "shap_cluster")
    print(f"{pipeline} / {config['contrast']} / {family}: {len(channels)} significant channel(s): {channels}")

    summary_rows = []
    if pipeline == "loso":
        from loso_pipeline.spatial_decoding.run_loso_spatial_decoding import (
            _score_channel, extract_model_params, resolve_model_type,
        )
        model_type = resolve_model_type(config)
        model_params = extract_model_params(config)
        for ch in channels:
            X_ch = select_channel_columns(X, ch)
            shap_dir = os.path.join(shap_root, ch)
            mean_auc, std_auc, auc_single = _score_channel(
                X_ch, y, df, groups, config["contrast"], config, model_type, model_params, n_runs,
                save_shap=True, shap_dir=shap_dir,
            )
            print(f"  {ch}: n_feat={X_ch.shape[1]} mean_auc={mean_auc:.4f} -> {shap_dir}")
            summary_rows.append((ch, shap_dir))
    else:
        from within_subject_pipeline.spatial_decoding.run_within_spatial_decoding import (
            _group_mean_auc, make_common,
        )
        model = config.get("model_type", "rf")
        common = make_common(config, model)
        tasks = df["task"] if "task" in df.columns else groups
        for ch in channels:
            X_ch = select_channel_columns(X, ch)
            shap_dir = os.path.join(shap_root, ch)
            mean_auc, std_auc, auc_single = _group_mean_auc(
                X_ch, y, df, groups, tasks, config["contrast"], common, n_runs,
                save_shap=True, shap_dir=shap_dir,
            )
            print(f"  {ch}: n_feat={X_ch.shape[1]} group_auc={mean_auc:.4f} -> {shap_dir}")
            summary_rows.append((ch, shap_dir))

    # ── Aggregate mean(|SHAP|) per marker, per significant channel ──────────
    summary_frames = []
    for ch, shap_dir in summary_rows:
        agg, n_files = aggregate_channel_shap(shap_dir)
        if agg is None:
            print(f"  WARNING: no SHAP pickle found for {ch} under {shap_dir} — skipped in summary.")
            continue
        agg["channel"] = ch
        agg["marker"] = agg["feature"].str.removeprefix(f"{ch}_")
        agg["n_files_aggregated"] = n_files
        summary_frames.append(agg[["channel", "marker", "mean_abs_shap", "n_files_aggregated"]])

    if summary_frames:
        summary = pd.concat(summary_frames, ignore_index=True)
        summary = summary.sort_values(["channel", "mean_abs_shap"], ascending=[True, False])
        os.makedirs(shap_root, exist_ok=True)
        out_csv = os.path.join(shap_root, "cluster_shap_summary.csv")
        summary.to_csv(out_csv, index=False)
        print(f"\nSummary written: {out_csv} ({len(summary)} channel×marker rows)")
    else:
        print("\nNo SHAP pickles produced — summary CSV not written.")


if __name__ == "__main__":
    main()
