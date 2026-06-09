#!/usr/bin/env python
"""Merge LOSO spatial-decoding outputs into per-channel FWER metrics + topomaps.

Thin wrapper around utils.spatial_decoding_utils.merge_spatial_maxstat: reads the true
per-channel AUCs (true/) and the permutation per-channel AUCs (perms/), forms the
max-over-channels null, computes per-channel family-wise (FWER) p-values, and renders
the AUC topomaps.
"""

import sys
import argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from utils.spatial_decoding_utils import merge_spatial_maxstat, build_max_null_from_perms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", required=True,
                    help="…/SpatialDecoding/LOSO/{contrast}/{family}/{model}")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--montage", default="standard_1020")
    args = ap.parse_args()

    max_null = build_max_null_from_perms(args.results_dir)
    print(f"Permutations: {len(max_null)} | "
          f"FWER threshold (q{1-args.alpha:.2f}): {np.quantile(max_null, 1-args.alpha):.4f}")
    metrics = merge_spatial_maxstat(args.results_dir, alpha=args.alpha,
                                    montage=args.montage, title="Spatial decoding AUC")
    n_sig = int(metrics["sig"].sum())
    print(f"Merged {len(metrics)} channels ({n_sig} FWER-significant) → "
          f"{args.results_dir}/per_channel_metrics.csv + topomaps")


if __name__ == "__main__":
    main()
