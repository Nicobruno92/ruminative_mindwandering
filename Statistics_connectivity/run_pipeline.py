"""
Main pipeline for connectivity LMM statistics.

Orchestrates the analysis:
1. Load connectivity data from aggregated CSVs
2. Optionally aggregate to ROI pairs
3. Run per-connection LMM for each band
4. Apply FDR correction
5. Generate contrast matrix figures
6. Save results
"""

import argparse
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional

from reader import (
    load_connectivity_data,
    aggregate_to_roi_pairs,
    prepare_connectivity_for_lmm,
    get_roi_pair_labels,
)
from lmm_connectivity import (
    run_lmm_per_connection,
    apply_fdr_correction,
)
from plot_connectivity import (
    plot_contrast_matrix,
    plot_multi_band_grid,
    plot_channel_contrast_matrix,
    create_summary_table,
)


# =============================================================================
# CONFIGURATION
# =============================================================================

def load_config(config_path: str) -> Dict:
    """
    Load YAML configuration file.

    Parameters
    ----------
    config_path : str
        Path to config.yaml.

    Returns
    -------
    Dict
        Configuration dictionary.
    """
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# =============================================================================
# SINGLE-LEVEL PIPELINE
# =============================================================================

def run_analysis_level(
    df: pd.DataFrame,
    level_name: str,
    bands: List[str],
    config: Dict,
    output_dir: Path,
) -> Dict[str, pd.DataFrame]:
    """
    Run the full LMM pipeline for one analysis level (ROI or channel).

    Parameters
    ----------
    df : pd.DataFrame
        Long-format connectivity data.
    level_name : str
        "roi" or "channel".
    bands : List[str]
        Frequency bands to analyze.
    config : Dict
        Pipeline configuration.
    output_dir : Path
        Output directory for this level.

    Returns
    -------
    Dict[str, pd.DataFrame]
        {band: results_df} with FDR-corrected results.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    lmm_cfg = config.get("lmm", {})
    fdr_cfg = config.get("fdr", {})
    project = config.get("project", {})

    all_results = {}

    for band in bands:
        print(f"\n{'='*60}")
        print(f"  [{level_name.upper()}] Band: {band}")
        print(f"{'='*60}")

        # Prepare data
        epoch_types = config.get("epoch_types", None)
        # Combine epoch types if multiple
        epoch_type_filter = None
        if epoch_types and len(epoch_types) == 1:
            epoch_type_filter = epoch_types[0]

        df_wide, connection_ids = prepare_connectivity_for_lmm(
            df=df,
            band=band,
            epoch_type=epoch_type_filter,
            onoff_max_value=project.get("onoff_max_value"),
            min_predictor_variability=project.get("min_predictor_variability"),
            min_minority_ratio=project.get("min_minority_ratio"),
        )

        if df_wide.empty:
            print(f"  [SKIP] No data for band={band}")
            continue

        # Run LMM
        results = run_lmm_per_connection(
            df_wide=df_wide,
            connection_ids=connection_ids,
            formula=lmm_cfg.get("formula", "power ~ onoff + (1|subject)"),
            predictor_of_interest=lmm_cfg.get("predictor_of_interest", "onoff"),
            method=lmm_cfg.get("method", "powell"),
            maxiter=lmm_cfg.get("maxiter", 500),
            random_state=lmm_cfg.get("random_state", 42),
        )

        # FDR correction
        results = apply_fdr_correction(
            results,
            alpha=fdr_cfg.get("alpha", 0.05),
            method=fdr_cfg.get("method", "fdr_bh"),
        )

        # Save per-band results
        band_csv = output_dir / f"lmm_results_{band}.csv"
        results.to_csv(band_csv, index=False)
        print(f"  Saved: {band_csv}")

        all_results[band] = results

    return all_results


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main(config_path: str = "Statistics_connectivity/config.yaml"):
    """
    Run the complete connectivity statistics pipeline.

    Parameters
    ----------
    config_path : str
        Path to configuration YAML file.
    """
    print("=" * 70)
    print("  CONNECTIVITY STATISTICS PIPELINE")
    print("=" * 70)

    # Load config
    config = load_config(config_path)
    project = config.get("project", {})
    output_base = Path(project.get("output_path", "results/connectivity_lmm"))
    features_root = project["features_root"]
    rois = config.get("rois", {})
    bands = config.get("bands", ["theta", "alpha", "beta", "gamma"])
    analysis_level = config.get("analysis_level", "both")
    out_cfg = config.get("output", {})

    print(f"\n  Features root: {features_root}")
    print(f"  Output: {output_base}")
    print(f"  Analysis level: {analysis_level}")
    print(f"  Bands: {bands}")

    # ── Step 1: Load connectivity data ────────────────────────────────────
    print("\n" + "=" * 70)
    print("  STEP 1: Loading connectivity data")
    print("=" * 70)

    raw_df = load_connectivity_data(
        features_root=features_root,
        subjects=project.get("subjects"),
        tasks=project.get("tasks"),
        epoch_types=config.get("epoch_types"),
        bands=bands,
        verbose=True,
    )

    # ── Step 2: ROI-level analysis ────────────────────────────────────────
    if analysis_level in ("roi", "both"):
        print("\n" + "=" * 70)
        print("  STEP 2a: ROI-level analysis")
        print("=" * 70)

        roi_df = aggregate_to_roi_pairs(raw_df, rois, verbose=True)
        roi_output = output_base / "roi_level"

        roi_results = run_analysis_level(
            df=roi_df,
            level_name="roi",
            bands=bands,
            config=config,
            output_dir=roi_output,
        )

        # Generate ROI figures
        if out_cfg.get("save_figures", True) and roi_results:
            print("\n  Generating ROI-level figures...")

            # Multi-band grid
            fig_path = roi_output / "connectivity_matrix_grid.png"
            plot_multi_band_grid(
                all_results=roi_results,
                suptitle="wSMI Connectivity: Effect of Mind-Wandering (onoff)",
                save_path=str(fig_path),
                dpi=out_cfg.get("fig_dpi", 300),
            )
            plt.close("all")

            # Individual band figures
            for band, results in roi_results.items():
                fig_path = roi_output / f"connectivity_matrix_{band}.png"
                fig = plot_contrast_matrix(
                    results_df=results,
                    title=f"wSMI {band} – LMM t-statistic (onoff)",
                )
                fig.savefig(fig_path, dpi=out_cfg.get("fig_dpi", 300), bbox_inches="tight")
                plt.close(fig)

            # Summary table
            summary_path = roi_output / "summary_all_bands.csv"
            create_summary_table(roi_results, save_path=str(summary_path))

    # ── Step 3: Channel-level analysis ────────────────────────────────────
    if analysis_level in ("channel", "both"):
        print("\n" + "=" * 70)
        print("  STEP 2b: Channel-level analysis")
        print("=" * 70)

        channel_output = output_base / "channel_level"

        channel_results = run_analysis_level(
            df=raw_df,
            level_name="channel",
            bands=bands,
            config=config,
            output_dir=channel_output,
        )

        # Generate channel-level figures
        if out_cfg.get("save_figures", True) and channel_results:
            print("\n  Generating channel-level figures...")

            for band, results in channel_results.items():
                fig_path = channel_output / f"channel_matrix_{band}.png"
                plot_channel_contrast_matrix(
                    results_df=results,
                    title=f"wSMI {band} – Channel-level LMM t-statistic",
                    save_path=str(fig_path),
                    dpi=out_cfg.get("fig_dpi", 300),
                )
                plt.close("all")

            # Summary table
            summary_path = channel_output / "summary_all_bands.csv"
            create_summary_table(channel_results, save_path=str(summary_path))

    # ── Done ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  PIPELINE COMPLETE")
    print(f"  Results saved to: {output_base}")
    print("=" * 70)


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Connectivity statistics pipeline (LMM + FDR)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="Statistics_connectivity/config.yaml",
        help="Path to configuration YAML file",
    )
    args = parser.parse_args()

    main(config_path=args.config)
