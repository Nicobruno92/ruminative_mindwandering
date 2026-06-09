#!/usr/bin/env python
"""Merge within-subject spatial-decoding outputs into FWER metrics + topomaps.

Reads the true per-channel group-mean AUCs (true/) and the permutation per-channel
group AUCs (perms/), forms the max-over-channels null per permutation, computes
per-channel family-wise (FWER) p-values via the max-statistic test, and renders the
AUC topomaps.
"""

import os
import sys
import argparse
import glob
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from utils.spatial_decoding_utils import maxstat_pvalues, plot_channel_topomap


def load_true_aucs(results_dir: str):
    true_files = sorted(glob.glob(os.path.join(results_dir, "true", "*.csv")))
    if not true_files:
        raise FileNotFoundError(f"No true/*.csv in {results_dir}")
    df = pd.concat([pd.read_csv(f) for f in true_files], ignore_index=True)
    df = df.drop_duplicates(subset="channel", keep="last")
    return dict(zip(df["channel"], df["mean_auc"])), df


def build_max_null(results_dir: str) -> np.ndarray:
    perm_files = sorted(glob.glob(os.path.join(results_dir, "perms", "perm-*.csv")))
    if not perm_files:
        raise FileNotFoundError(f"No perms/perm-*.csv in {results_dir}")
    return np.asarray(
        [float(np.nanmax(pd.read_csv(f)["auc"].to_numpy(dtype=float))) for f in perm_files],
        dtype=float,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", required=True,
                    help="…/SpatialDecoding/WithinSubject/{contrast}/{family}/{model}")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--montage", default="standard_1020")
    args = ap.parse_args()

    true_auc, true_df = load_true_aucs(args.results_dir)
    max_null = build_max_null(args.results_dir)
    print(f"Channels: {len(true_auc)} | permutations: {len(max_null)} | "
          f"FWER threshold (q{1-args.alpha:.2f}): {np.quantile(max_null, 1-args.alpha):.4f}")

    metrics = maxstat_pvalues(true_auc, max_null, alpha=args.alpha)
    metrics = metrics.merge(true_df[["channel", "n_features", "std_auc"]],
                            on="channel", how="left")
    metrics = metrics.sort_values("channel").reset_index(drop=True)
    metrics.to_csv(os.path.join(args.results_dir, "per_channel_metrics.csv"), index=False)

    plot_channel_topomap(metrics, "mean_auc",
                         os.path.join(args.results_dir, "topomap_auc.png"),
                         montage=args.montage, title="Within-subject decoding AUC")
    plot_channel_topomap(metrics, "mean_auc",
                         os.path.join(args.results_dir, "topomap_sig.png"),
                         montage=args.montage, mask_col="sig",
                         title="AUC (FWER-significant electrodes marked)")
    n_sig = int(metrics["sig"].sum())
    print(f"Merged {len(metrics)} channels ({n_sig} FWER-significant) → "
          f"{args.results_dir}/per_channel_metrics.csv + topomaps")


if __name__ == "__main__":
    main()
