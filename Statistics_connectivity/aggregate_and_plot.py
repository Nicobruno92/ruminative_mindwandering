#!/usr/bin/env python3
"""
Aggregate connectivity results from parallel band jobs and generate figures.

This script runs after all band-specific jobs complete. It:
1. Loads results from each band
2. Generates multi-band comparison figures
3. Creates summary tables
"""

import argparse
import gc
from pathlib import Path
from typing import Dict

import pandas as pd
import yaml

from plot_connectivity import (
    plot_multi_band_grid,
    plot_multi_band_circle_grid,
    create_summary_table,
)


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_band_results(
    output_dir: Path,
    bands: list,
) -> Dict[str, pd.DataFrame]:
    """
    Load results from individual band CSV files.

    Parameters
    ----------
    output_dir : Path
        Directory containing band result CSVs.
    bands : list
        List of band names to load.

    Returns
    -------
    Dict[str, pd.DataFrame]
        {band: results_df} mapping.
    """
    all_results = {}
    
    for band in bands:
        csv_path = output_dir / f"lmm_results_{band}.csv"
        if not csv_path.exists():
            print(f"  WARNING: {csv_path} not found, skipping {band}")
            continue
        
        print(f"  Loading {band} results from {csv_path}")
        all_results[band] = pd.read_csv(csv_path)
    
    return all_results


def _sig_col_for_method(method: str) -> str:
    """Map correction method name to significance column."""
    mapping = {
        "fdr": "significant_fdr",
        "nbs": "significant_nbs",
        "max_t": "significant_perm",
    }
    return mapping.get(method, "significant_fdr")


def main(config_path: str = "Statistics_connectivity/config.yaml"):
    """
    Aggregate results and generate multi-band figures.

    Parameters
    ----------
    config_path : str
        Path to configuration YAML file.
    """
    print("=" * 70)
    print("  CONNECTIVITY RESULTS AGGREGATION")
    print("=" * 70)

    # Load config
    config = load_config(config_path)
    project = config.get("project", {})
    output_base = Path(project.get("output_path", "results/connectivity_lmm"))
    bands = config.get("bands", ["theta", "alpha", "beta", "gamma"])
    analysis_level = config.get("analysis_level", "both")
    out_cfg = config.get("output", {})

    print(f"\n  Output: {output_base}")
    print(f"  Analysis level: {analysis_level}")
    print(f"  Bands: {bands}")

    # Determine significance columns per level
    roi_correction = config.get("correction", {}).get("roi", "fdr")
    ch_correction = config.get("correction", {}).get("channel", "fdr")
    roi_sig_col = _sig_col_for_method(roi_correction)
    ch_sig_col = _sig_col_for_method(ch_correction)

    # ── ROI-level aggregation ────────────────────────────────────────────
    if analysis_level in ("roi", "both"):
        print("\n" + "=" * 70)
        print("  ROI-LEVEL AGGREGATION")
        print("=" * 70)
        
        roi_output = output_base / "roi_level"
        roi_results = load_band_results(roi_output, bands)
        
        if roi_results:
            # Multi-band matrix grid
            if out_cfg.get("save_figures", True):
                print("\n  Generating ROI multi-band matrix...")
                fig = plot_multi_band_grid(
                    roi_results,
                    sig_col=roi_sig_col,
                    suptitle=f"ROI Connectivity: {roi_correction.upper()} correction",
                )
                fig_path = roi_output / f"roi_multiband_grid.{out_cfg.get('fig_format', 'png')}"
                fig.savefig(fig_path, dpi=out_cfg.get("fig_dpi", 300), bbox_inches="tight")
                print(f"  Saved: {fig_path}")
                del fig
                gc.collect()
                
                # Multi-band circle grid
                print("\n  Generating ROI multi-band circle plot...")
                fig = plot_multi_band_circle_grid(
                    roi_results,
                    level="roi",
                    sig_col=roi_sig_col,
                    suptitle=f"ROI Connectivity Networks: {roi_correction.upper()}",
                )
                fig_path = roi_output / f"roi_multiband_circle.{out_cfg.get('fig_format', 'png')}"
                fig.savefig(fig_path, dpi=out_cfg.get("fig_dpi", 300), bbox_inches="tight")
                print(f"  Saved: {fig_path}")
                del fig
                gc.collect()
            
            # Summary table
            if out_cfg.get("save_csv", True):
                print("\n  Creating ROI summary table...")
                summary = create_summary_table(
                    roi_results,
                    save_path=roi_output / "roi_summary_all_bands.csv",
                )
                print(f"  Summary shape: {summary.shape}")

    # ── Channel-level aggregation ────────────────────────────────────────
    if analysis_level in ("channel", "both"):
        print("\n" + "=" * 70)
        print("  CHANNEL-LEVEL AGGREGATION")
        print("=" * 70)
        
        channel_output = output_base / "channel_level"
        channel_results = load_band_results(channel_output, bands)
        
        if channel_results:
            # Multi-band circle grid (skip matrix - too large)
            if out_cfg.get("save_figures", True):
                print("\n  Generating channel multi-band circle plot...")
                fig = plot_multi_band_circle_grid(
                    channel_results,
                    level="channel",
                    sig_col=ch_sig_col,
                    suptitle=f"Channel Connectivity Networks: {ch_correction.upper()}",
                )
                fig_path = channel_output / f"channel_multiband_circle.{out_cfg.get('fig_format', 'png')}"
                fig.savefig(fig_path, dpi=out_cfg.get("fig_dpi", 300), bbox_inches="tight")
                print(f"  Saved: {fig_path}")
                del fig
                gc.collect()
            
            # Summary table
            if out_cfg.get("save_csv", True):
                print("\n  Creating channel summary table...")
                summary = create_summary_table(
                    channel_results,
                    save_path=channel_output / "channel_summary_all_bands.csv",
                )
                print(f"  Summary shape: {summary.shape}")

    print("\n" + "=" * 70)
    print("  AGGREGATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Aggregate connectivity results and generate figures"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="Statistics_connectivity/config.yaml",
        help="Path to configuration YAML file",
    )
    args = parser.parse_args()
    main(config_path=args.config)
