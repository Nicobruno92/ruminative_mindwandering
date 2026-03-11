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
import gc
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
    apply_nbs_correction,
    apply_max_t_permutation,
)
from plot_connectivity import (
    plot_contrast_matrix,
    plot_multi_band_grid,
    plot_channel_contrast_matrix,
    plot_roi_connectivity_circle,
    plot_channel_connectivity_circle,
    plot_multi_band_circle_grid,
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

    Correction strategy:
    - channel level: NBS (Network-Based Statistic) permutation, the
      graph analogue of cluster-based permutation testing.
    - roi level: FDR (Benjamini-Hochberg), appropriate for the small
      number of tests (~28 ROI pairs).

    The correction method can be overridden via config["correction"]["channel"]
    and config["correction"]["roi"].

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
        {band: results_df} with corrected significance results.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    lmm_cfg = config.get("lmm", {})
    fdr_cfg = config.get("fdr", {})
    correction_cfg = config.get("correction", {})
    nbs_cfg = config.get("nbs", {})
    project = config.get("project", {})

    # Determine correction method for this level
    default_method = "nbs" if level_name == "channel" else "fdr"
    correction_method = correction_cfg.get(level_name, default_method)

    all_results = {}

    for band in bands:
        print(f"\n{'='*60}")
        print(f"  [{level_name.upper()}] Band: {band}")
        print(f"  Correction: {correction_method}")
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
        formula = lmm_cfg.get("formula", "power ~ onoff + (1|subject)")
        predictor = lmm_cfg.get("predictor_of_interest", "onoff")
        results = run_lmm_per_connection(
            df_wide=df_wide,
            connection_ids=connection_ids,
            formula=formula,
            predictor_of_interest=predictor,
            method=lmm_cfg.get("method", "powell"),
            maxiter=lmm_cfg.get("maxiter", 500),
            random_state=lmm_cfg.get("random_state", 42),
        )

        # Always apply FDR as baseline reference
        results = apply_fdr_correction(
            results,
            alpha=fdr_cfg.get("alpha", 0.05),
            method=fdr_cfg.get("method", "fdr_bh"),
        )

        # Apply additional correction based on level
        # Get n_jobs from config or environment (SLURM_CPUS_PER_TASK)
        import os
        n_jobs = nbs_cfg.get("n_jobs", int(os.environ.get("SLURM_CPUS_PER_TASK", 1)))
        
        if correction_method == "nbs":
            results = apply_nbs_correction(
                results_df=results,
                df_wide=df_wide,
                connection_ids=connection_ids,
                formula=formula,
                predictor_of_interest=predictor,
                method=lmm_cfg.get("method", "powell"),
                maxiter=lmm_cfg.get("maxiter", 500),
                primary_threshold=nbs_cfg.get("primary_threshold", 3.0),
                n_permutations=nbs_cfg.get("n_permutations", 5000),
                alpha=nbs_cfg.get("alpha", 0.05),
                random_state=lmm_cfg.get("random_state", 42),
                n_jobs=n_jobs,
            )
        elif correction_method == "max_t":
            results = apply_max_t_permutation(
                results_df=results,
                df_wide=df_wide,
                connection_ids=connection_ids,
                formula=formula,
                predictor_of_interest=predictor,
                method=lmm_cfg.get("method", "powell"),
                maxiter=lmm_cfg.get("maxiter", 500),
                n_permutations=nbs_cfg.get("n_permutations", 5000),
                alpha=nbs_cfg.get("alpha", 0.05),
                random_state=lmm_cfg.get("random_state", 42),
                n_jobs=n_jobs,
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

    roi_output = output_base / "roi_level"
    channel_output = output_base / "channel_level"
    if analysis_level in ("roi", "both"):
        roi_output.mkdir(parents=True, exist_ok=True)
    if analysis_level in ("channel", "both"):
        channel_output.mkdir(parents=True, exist_ok=True)

    roi_results: Dict[str, pd.DataFrame] = {}
    channel_results: Dict[str, pd.DataFrame] = {}

    # ── Steps 1-3: Per-band load → analyse → free ─────────────────────────
    # Processing one band at a time keeps peak memory at ~1/n_bands of the
    # full dataset instead of loading all bands simultaneously.
    for band in bands:
        print("\n" + "=" * 70)
        print(f"  Processing band: {band}")
        print("=" * 70)

        # ── Load only this band's connectivity data ───────────────────────
        print(f"  Loading {band} connectivity data...")
        band_df = load_connectivity_data(
            features_root=features_root,
            subjects=project.get("subjects"),
            tasks=project.get("tasks"),
            epoch_types=config.get("epoch_types"),
            bands=[band],
            verbose=True,
        )

        # ── ROI-level analysis for this band ─────────────────────────────
        if analysis_level in ("roi", "both"):
            roi_df = aggregate_to_roi_pairs(band_df, rois, verbose=False)
            band_roi_results = run_analysis_level(
                df=roi_df,
                level_name="roi",
                bands=[band],
                config=config,
                output_dir=roi_output,
            )
            roi_results.update(band_roi_results)
            del roi_df

        # ── Channel-level analysis for this band ─────────────────────────
        if analysis_level in ("channel", "both"):
            band_channel_results = run_analysis_level(
                df=band_df,
                level_name="channel",
                bands=[band],
                config=config,
                output_dir=channel_output,
            )
            channel_results.update(band_channel_results)

        # Free this band's data before loading the next
        del band_df
        gc.collect()

    # ── Figures (after all bands collected) ───────────────────────────────
    if out_cfg.get("save_figures", True):

        # Determine which significance column to use for plots per level
        roi_correction = config.get("correction", {}).get("roi", "fdr")
        ch_correction = config.get("correction", {}).get("channel", "nbs")
        roi_sig_col = _sig_col_for_method(roi_correction)
        ch_sig_col = _sig_col_for_method(ch_correction)

        if roi_results:
            print("\n  Generating ROI-level figures...")

            # Matrix grid
            fig_path = roi_output / "connectivity_matrix_grid.png"
            plot_multi_band_grid(
                all_results=roi_results,
                sig_col=roi_sig_col,
                suptitle="wSMI Connectivity: Effect of Mind-Wandering (onoff)",
                save_path=str(fig_path),
                dpi=out_cfg.get("fig_dpi", 300),
            )
            plt.close("all")

            # Per-band matrices
            for band, results in roi_results.items():
                fig_path = roi_output / f"connectivity_matrix_{band}.png"
                fig = plot_contrast_matrix(
                    results_df=results,
                    sig_col=roi_sig_col,
                    title=f"wSMI {band} – LMM t-statistic (onoff)",
                )
                fig.savefig(fig_path, dpi=out_cfg.get("fig_dpi", 300), bbox_inches="tight")
                plt.close(fig)

            # Circle plots (ROI)
            fig_path = roi_output / "connectivity_circle_grid.png"
            plot_multi_band_circle_grid(
                all_results=roi_results,
                level="roi",
                sig_col=roi_sig_col,
                suptitle="wSMI Connectivity Circles: MW Effect (onoff)",
                save_path=str(fig_path),
                dpi=out_cfg.get("fig_dpi", 300),
            )
            plt.close("all")
            for band, results in roi_results.items():
                fig_path = roi_output / f"connectivity_circle_{band}.png"
                plot_roi_connectivity_circle(
                    results_df=results,
                    sig_col=roi_sig_col,
                    title=f"wSMI {band} – ROI Connectivity Circle",
                    save_path=str(fig_path),
                    dpi=out_cfg.get("fig_dpi", 300),
                )
                plt.close("all")

            summary_path = roi_output / "summary_all_bands.csv"
            create_summary_table(roi_results, save_path=str(summary_path))

        if channel_results:
            print("\n  Generating channel-level figures...")
            for band, results in channel_results.items():
                # Matrix plot
                fig_path = channel_output / f"channel_matrix_{band}.png"
                plot_channel_contrast_matrix(
                    results_df=results,
                    sig_col=ch_sig_col,
                    title=f"wSMI {band} – Channel LMM t-stat ({ch_correction})",
                    save_path=str(fig_path),
                    dpi=out_cfg.get("fig_dpi", 300),
                )
                plt.close("all")

                # Circle plot (channel)
                fig_path = channel_output / f"channel_circle_{band}.png"
                plot_channel_connectivity_circle(
                    results_df=results,
                    rois=rois,
                    sig_col=ch_sig_col,
                    title=f"wSMI {band} – Channel Connectivity ({ch_correction})",
                    save_path=str(fig_path),
                    dpi=out_cfg.get("fig_dpi", 300),
                )
                plt.close("all")

            # Circle grid across bands
            fig_path = channel_output / "channel_circle_grid.png"
            plot_multi_band_circle_grid(
                all_results=channel_results,
                level="channel",
                rois=rois,
                sig_col=ch_sig_col,
                suptitle=f"Channel Connectivity Circles ({ch_correction})",
                save_path=str(fig_path),
                dpi=out_cfg.get("fig_dpi", 300),
            )
            plt.close("all")

            summary_path = channel_output / "summary_all_bands.csv"
            create_summary_table(channel_results, save_path=str(summary_path))

    # ── Done ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  PIPELINE COMPLETE")
    print(f"  Results saved to: {output_base}")
    print("=" * 70)


def _sig_col_for_method(method: str) -> str:
    """
    Map correction method name to its significance column.

    Parameters
    ----------
    method : str
        Correction method: "fdr", "nbs", or "max_t".

    Returns
    -------
    str
        Column name for significance boolean.
    """
    mapping = {
        "fdr": "significant_fdr",
        "nbs": "significant_nbs",
        "max_t": "significant_perm",
    }
    return mapping.get(method, "significant_fdr")


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
