#!/usr/bin/env python
"""
Regenerate all spatial-decoding topomaps as SVG from existing per_channel_metrics.csv files.

Does NOT re-run any classification. Reads the already-computed CSVs and calls
the same plotting functions used during the pipeline, changing only the output
extension to .svg.

Covers:
  SpatialDecoding/{LOSO,WithinSubject}/{contrast}/all/rf/
    → topomap_auc.svg
    → topomap_sig.svg
  SpatialDecoding/{LOSO,WithinSubject}/combined/
    → topomap_panel_auc.svg
"""
from __future__ import annotations

import sys
import argparse
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "mw_classification_pipeline"))

from utils.spatial_decoding_utils import (
    plot_channel_topomap,
    plot_combined_topomap_panel,
    DEFAULT_MONTAGE,
)

SPATIAL_ROOT = (
    REPO_ROOT
    / "mw_classification_pipeline"
    / "results"
    / "MW_Classification"
    / "SpatialDecoding"
)

# Dimension-label → human-readable title used in topomap titles.
DIM_TITLE = {
    "ON_vs_OFF_within_median":   "On vs Off (onoff)",
    "on_vs_off_within_median":   "On vs Off (onoff)",
    "valence_within_median":     "Valence",
    "selfother_within_median":   "Self–Other",
    "time_within_median":        "Time",
    "confidence_within_median":  "Confidence",
}


def _plot_single(metrics_csv: Path, montage: str, vlim: tuple) -> None:
    metrics = pd.read_csv(metrics_csv)
    results_dir = metrics_csv.parent

    # Infer a human-readable title from the contrast directory name
    contrast_dir = results_dir.parent.parent.name
    title = DIM_TITLE.get(contrast_dir, contrast_dir)

    auc_svg = results_dir / "topomap_auc.svg"
    sig_svg = results_dir / "topomap_sig.svg"

    plot_channel_topomap(
        metrics, "mean_auc", str(auc_svg),
        montage=montage, title=title, vlim=vlim,
    )
    plot_channel_topomap(
        metrics, "mean_auc", str(sig_svg),
        montage=montage, mask_col="sig", vlim=vlim,
        title=f"{title} (FWER-significant electrodes marked)",
    )
    print(f"  ✓ {auc_svg.relative_to(SPATIAL_ROOT)}")
    print(f"  ✓ {sig_svg.relative_to(SPATIAL_ROOT)}")


def _plot_panel(pipeline_dir: Path, family: str, model: str,
                montage: str, vlim: tuple) -> None:
    per_dim: dict[str, pd.DataFrame] = {}
    for contrast_dir in sorted(pipeline_dir.iterdir()):
        if not contrast_dir.is_dir() or contrast_dir.name == "combined":
            continue
        csv = contrast_dir / family / model / "per_channel_metrics.csv"
        if csv.exists():
            per_dim[contrast_dir.name] = pd.read_csv(csv)

    if not per_dim:
        print(f"  ! No metrics found under {pipeline_dir.name}, skipping panel.")
        return

    out = pipeline_dir / "combined" / "topomap_panel_auc.svg"
    plot_combined_topomap_panel(
        per_dim, value_col="mean_auc", mask_col="sig",
        out_path=str(out), montage=montage, vlim=vlim,
    )
    print(f"  ✓ {out.relative_to(SPATIAL_ROOT)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--family", default="all")
    ap.add_argument("--model", default="rf")
    ap.add_argument("--montage", default=DEFAULT_MONTAGE,
                    help="MNE montage name or path to .bvef (default: CACS-64_REF.bvef)")
    ap.add_argument("--vmin", type=float, default=0.5)
    ap.add_argument("--vmax", type=float, default=None)
    ap.add_argument("--pipeline", choices=["LOSO", "WithinSubject", "both"],
                    default="both")
    args = ap.parse_args()

    vlim = (args.vmin, args.vmax)
    pipelines = (
        ["LOSO", "WithinSubject"] if args.pipeline == "both"
        else [args.pipeline]
    )

    for pipeline_name in pipelines:
        pipeline_dir = SPATIAL_ROOT / pipeline_name
        if not pipeline_dir.exists():
            print(f"Pipeline dir not found, skipping: {pipeline_dir}")
            continue

        print(f"\n=== {pipeline_name} ===")

        # Per-contrast individual topomaps
        for csv in sorted(pipeline_dir.rglob(f"{args.family}/{args.model}/per_channel_metrics.csv")):
            if "combined" in csv.parts:
                continue
            print(f"  Processing {csv.parent.relative_to(pipeline_dir)} …")
            _plot_single(csv, args.montage, vlim)

        # Combined panel
        print(f"  Building combined panel …")
        _plot_panel(pipeline_dir, args.family, args.model, args.montage, vlim)

    print("\nDone.")


if __name__ == "__main__":
    main()
