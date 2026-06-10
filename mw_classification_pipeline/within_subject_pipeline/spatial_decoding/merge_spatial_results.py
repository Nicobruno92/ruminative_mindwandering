#!/usr/bin/env python
"""Merge within-subject spatial-decoding outputs into FWER metrics + topomaps.

Thin wrapper around utils.spatial_decoding_utils.merge_spatial_maxstat. The per-channel
statistic is the group-mean AUC across subjects; significance is the max-statistic
(FWER) permutation test over the 64 electrodes.
"""

import os
import sys
import argparse
import yaml
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from utils.spatial_decoding_utils import merge_spatial_maxstat, build_max_null_from_perms, DEFAULT_MONTAGE


def topomap_opts(results_dir: str):
    """Read topomap cmap/vlim from the run's used_config.yaml (viridis / [0.5,1.0] default)."""
    cmap, vlim = "viridis", [0.5, None]
    cfg_path = os.path.join(results_dir, "used_config.yaml")
    if os.path.exists(cfg_path):
        tm = ((yaml.safe_load(open(cfg_path)) or {}).get("spatial_decoding", {}) or {}).get("topomap", {}) or {}
        cmap = tm.get("cmap", cmap)
        vlim = tm.get("vlim", vlim)
    return cmap, (tuple(vlim) if vlim else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", required=True,
                    help="…/SpatialDecoding/WithinSubject/{contrast}/{family}/{model}")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--montage", default=DEFAULT_MONTAGE)
    args = ap.parse_args()

    max_null = build_max_null_from_perms(args.results_dir)
    print(f"Permutations: {len(max_null)} | "
          f"FWER threshold (q{1-args.alpha:.2f}): {np.quantile(max_null, 1-args.alpha):.4f}")
    cmap, vlim = topomap_opts(args.results_dir)
    metrics = merge_spatial_maxstat(args.results_dir, alpha=args.alpha,
                                    montage=args.montage, title="Within-subject decoding AUC",
                                    cmap=cmap, vlim=vlim)
    n_sig = int(metrics["sig"].sum())
    print(f"Merged {len(metrics)} channels ({n_sig} FWER-significant) → "
          f"{args.results_dir}/per_channel_metrics.csv + topomaps")


if __name__ == "__main__":
    main()
