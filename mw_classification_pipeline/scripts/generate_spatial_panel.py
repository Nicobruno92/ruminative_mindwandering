#!/usr/bin/env python
"""Build the combined 5-dimension topomap panel from per-dimension metrics."""
import os
import sys
import argparse
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.spatial_decoding_utils import plot_combined_topomap_panel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pipeline_dir", required=True,
                    help="…/SpatialDecoding/{LOSO|WithinSubject}")
    ap.add_argument("--family", default="all")
    ap.add_argument("--model", default="rf")
    ap.add_argument("--value_col", default="mean_auc")
    ap.add_argument("--montage", default="standard_1020")
    args = ap.parse_args()

    per_dim = {}
    for contrast_dir in sorted(Path(args.pipeline_dir).iterdir()):
        if not contrast_dir.is_dir() or contrast_dir.name == "combined":
            continue
        metrics_csv = contrast_dir / args.family / args.model / "per_channel_metrics.csv"
        if metrics_csv.exists():
            per_dim[contrast_dir.name] = pd.read_csv(metrics_csv)

    if not per_dim:
        raise FileNotFoundError(f"No per_channel_metrics.csv under {args.pipeline_dir}")

    out = os.path.join(args.pipeline_dir, "combined", "topomap_panel_auc.png")
    plot_combined_topomap_panel(per_dim, value_col=args.value_col, mask_col="sig",
                                out_path=out, montage=args.montage)
    print(f"Panel written: {out}  ({len(per_dim)} dimensions)")


if __name__ == "__main__":
    main()
