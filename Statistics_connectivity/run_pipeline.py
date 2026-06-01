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
import os
import shutil
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional

from reader import (
    load_connectivity_data,
    aggregate_to_roi_pairs,
    apply_qa_filter,
    merge_pca_results,
    normalize_wsmi_by_subject,
    normalize_predictors,
    prepare_connectivity_for_lmm,
    get_roi_pair_labels,
)
from lmm_connectivity import (
    run_lmm_per_connection,
    apply_fdr_correction,
    apply_nbs_correction,
    apply_max_t_permutation,
    apply_family_wise_correction,
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
    
    # Get epoch types to analyze separately
    epoch_types = config.get("epoch_types", None)
    if not epoch_types:
        # If no epoch types specified, analyze all data together
        epoch_types = [None]

    for band in bands:
        for epoch_type in epoch_types:
            epoch_label = epoch_type if epoch_type else "all"
            print(f"\n{'='*60}")
            print(f"  [{level_name.upper()}] Band: {band} | Epoch: {epoch_label}")
            print(f"  Correction: {correction_method}")
            print(f"{'='*60}")

            df_wide, connection_ids = prepare_connectivity_for_lmm(
                df=df,
                band=band,
                epoch_type=epoch_type,
                onoff_max_value=project.get("onoff_max_value"),
                min_predictor_variability=project.get("min_predictor_variability"),
                min_minority_ratio=project.get("min_minority_ratio"),
            )

            if df_wide.empty:
                print(f"  [SKIP] No data for band={band}, epoch={epoch_label}")
                continue

            # Run LMM for each predictor
            formula = lmm_cfg.get("formula", "power ~ onoff + (1|subject)")
            predictors = lmm_cfg.get("predictor_of_interest", ["onoff"])
            if isinstance(predictors, str):
                predictors = [predictors]
                
            for predictor in predictors:
                print(f"  [{level_name.upper()}] Predictor: {predictor}")
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
                n_jobs = nbs_cfg.get("n_jobs", int(os.environ.get("SLURM_CPUS_PER_TASK", 1)))
                
                # Permutation method (simple vs freedman_lane). Freedman-Lane
                # preserves the covariance structure between predictor and
                # nuisance covariates under H0; recommended when covariates
                # are correlated with the predictor of interest.
                permutation_method = nbs_cfg.get("permutation_method", "simple")

                # Path for per-permutation convergence diagnostics CSV. We
                # build it here (not inside the correction function) so the
                # filename matches the band/predictor naming used elsewhere.
                epoch_suffix_pre = (
                    f"_{epoch_type.replace('_connectivity', '')}"
                    if epoch_type else ""
                )
                band_key_pre = f"{band}{epoch_suffix_pre}"
                perm_diag_csv = str(
                    output_dir
                    / f"perm_diagnostics_{band_key_pre}_{predictor}.csv"
                )

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
                        permutation_method=permutation_method,
                        perm_diagnostics_csv=perm_diag_csv,
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
                        permutation_method=permutation_method,
                        perm_diagnostics_csv=perm_diag_csv,
                    )

                # Save per-band-epoch-predictor results
                epoch_suffix = f"_{epoch_type.replace('_connectivity', '')}" if epoch_type else ""
                band_key = f"{band}{epoch_suffix}"
                band_csv = output_dir / f"lmm_results_{band_key}_{predictor}.csv"
                results.to_csv(band_csv, index=False)
                print(f"  Saved: {band_csv}")

                if predictor not in all_results:
                    all_results[predictor] = {}
                all_results[predictor][band_key] = results

    return all_results


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main(config_path: str = "Statistics_connectivity/config.yaml", single_band: str = None):
    """
    Run the complete connectivity statistics pipeline.

    Parameters
    ----------
    config_path : str
        Path to configuration YAML file.
    single_band : str, optional
        If provided, only analyze this frequency band (for parallel execution).
        Otherwise, analyze all bands sequentially.
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

    # Override bands if single_band specified (for parallel execution)
    if single_band:
        if single_band not in bands:
            raise ValueError(f"Band '{single_band}' not in config bands: {bands}")
        bands = [single_band]
        print(f"\n  PARALLEL MODE: Processing only {single_band} band")

    analysis_level = config.get("analysis_level", "both")
    out_cfg = config.get("output", {})
    preproc_cfg = config.get("preprocessing", {})

    print(f"\n  Features root: {features_root}")
    print(f"  Output: {output_base}")
    print(f"  Analysis level: {analysis_level}")
    print(f"  Bands: {bands}")

    roi_output = output_base / "roi_level"
    channel_output = output_base / "channel_level"
    output_base.mkdir(parents=True, exist_ok=True)
    if analysis_level in ("roi", "both"):
        roi_output.mkdir(parents=True, exist_ok=True)
    if analysis_level in ("channel", "both"):
        channel_output.mkdir(parents=True, exist_ok=True)

    # Snapshot the resolved config alongside the results so re-runs and
    # downstream consumers (figures, tables) can be traced back to the exact
    # parameters that produced them. ``shutil.copy2`` preserves mtime — useful
    # when comparing config edits across runs.
    snapshot_path = output_base / "used_config.yaml"
    shutil.copy2(config_path, snapshot_path)
    print(f"  Config snapshot: {snapshot_path}")

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

        # ── QA filter ─────────────────────────────────────────────────────
        # Drop (subject, task, epoch_type) tuples that failed preprocessing QA.
        if project.get("exclude_failed_qa", False) and project.get("qa_summary_path"):
            band_df = apply_qa_filter(
                df=band_df,
                qa_summary_path=project["qa_summary_path"],
                epoch_types=config.get("epoch_types"),
                verbose=True,
            )

        # ── Merge PCA components ──────────────────────────────────────────
        # Brings PC1/PC2/PC3 from probe-level PCA into the connectivity df so
        # they can be referenced from the LMM formula.
        pca_path = project.get("pca_results_path")
        if pca_path and Path(pca_path).exists():
            band_df = merge_pca_results(
                df=band_df,
                pca_results_path=pca_path,
                verbose=True,
            )

        # ── Per-subject wSMI normalization ────────────────────────────────
        if preproc_cfg.get("normalize_by_subject", False):
            band_df = normalize_wsmi_by_subject(
                df=band_df,
                method=preproc_cfg.get("normalization_method", "zscore"),
                channel_wise=preproc_cfg.get("channel_wise", False),
                subject_column=preproc_cfg.get("subject_column", "subject"),
                verbose=True,
            )

        # ── Predictor z-scoring ───────────────────────────────────────────
        # Z-scored continuous predictors give comparable β coefficients across
        # covariates with different units; binary predictors (onoff) are
        # auto-skipped inside normalize_predictors.
        pred_norm_cfg = preproc_cfg.get("predictor_normalization", {})
        if pred_norm_cfg.get("enabled", False):
            band_df = normalize_predictors(
                df=band_df,
                predictors=pred_norm_cfg.get("predictors", []),
                method=pred_norm_cfg.get("method", "zscore"),
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
            for p, p_res in band_roi_results.items():
                if p not in roi_results:
                    roi_results[p] = {}
                roi_results[p].update(p_res)
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
            for p, p_res in band_channel_results.items():
                if p not in channel_results:
                    channel_results[p] = {}
                channel_results[p].update(p_res)

        # Free this band's data before loading the next
        del band_df
        gc.collect()

    # ── Family-wise correction across (band × predictor) tests ───────────
    # Each per-(band, predictor) NBS / FDR test controls FWER only within
    # itself; running multiple tests inflates the joint α. Bonferroni at the
    # family level prevents that.  When enabled, the per-test CSVs are
    # rewritten with the new p_value_*_family columns and downstream plots
    # use the family-wise sig column.
    fwc_cfg = config.get("family_wise_correction", {})
    fwc_enabled = fwc_cfg.get("enabled", False)
    fwc_method = fwc_cfg.get("method", "bonferroni")
    fwc_alpha = fwc_cfg.get("alpha", 0.05)

    roi_correction = config.get("correction", {}).get("roi", "fdr")
    ch_correction = config.get("correction", {}).get("channel", "nbs")
    roi_base_p = _p_col_for_method(roi_correction)
    ch_base_p = _p_col_for_method(ch_correction)

    if fwc_enabled and fwc_method != "none":
        if roi_results:
            roi_results, roi_fwc_cols, n_roi_tests = apply_family_wise_correction(
                roi_results, base_p_col=roi_base_p, alpha=fwc_alpha, method=fwc_method,
            )
            print(
                f"  Family-wise [{fwc_method}] across {n_roi_tests} ROI test(s) "
                f"on {roi_base_p} → {roi_fwc_cols['sig']}"
            )
        else:
            roi_fwc_cols = {"p": roi_base_p, "sig": ""}

        if channel_results:
            channel_results, ch_fwc_cols, n_ch_tests = apply_family_wise_correction(
                channel_results, base_p_col=ch_base_p, alpha=fwc_alpha, method=fwc_method,
            )
            print(
                f"  Family-wise [{fwc_method}] across {n_ch_tests} channel test(s) "
                f"on {ch_base_p} → {ch_fwc_cols['sig']}"
            )
        else:
            ch_fwc_cols = {"p": ch_base_p, "sig": ""}

        # Rewrite per-test CSVs so they include the family-wise columns.
        for predictor, by_band in roi_results.items():
            for band_key, df in by_band.items():
                df.to_csv(roi_output / f"lmm_results_{band_key}_{predictor}.csv",
                          index=False)
        for predictor, by_band in channel_results.items():
            for band_key, df in by_band.items():
                df.to_csv(channel_output / f"lmm_results_{band_key}_{predictor}.csv",
                          index=False)
    else:
        roi_fwc_cols = {"p": roi_base_p, "sig": ""}
        ch_fwc_cols = {"p": ch_base_p, "sig": ""}

    # ── Figures (after all bands collected) ───────────────────────────────
    if out_cfg.get("save_figures", True):

        # Determine which significance column to use for plots per level.
        # If family-wise correction was applied, prefer that; otherwise the
        # per-test column for the configured correction method.
        roi_sig_col = roi_fwc_cols["sig"] or _sig_col_for_method(roi_correction)
        ch_sig_col = ch_fwc_cols["sig"] or _sig_col_for_method(ch_correction)

        if roi_results:
            print("\n  Generating ROI-level figures...")

            for predictor, p_roi_results in roi_results.items():
                if not p_roi_results:
                    continue
                    
                # Matrix grid
                fig_path = roi_output / f"connectivity_matrix_grid_{predictor}.png"
                plot_multi_band_grid(
                    all_results=p_roi_results,
                    sig_col=roi_sig_col,
                    suptitle=f"wSMI Connectivity: Effect of {predictor}",
                    save_path=str(fig_path),
                    dpi=out_cfg.get("fig_dpi", 300),
                )
                plt.close("all")

                # Per-band matrices
                for band, results in p_roi_results.items():
                    fig_path = roi_output / f"connectivity_matrix_{band}_{predictor}.png"
                    fig = plot_contrast_matrix(
                        results_df=results,
                        sig_col=roi_sig_col,
                        title=f"wSMI {band} – LMM t-statistic ({predictor})",
                    )
                    fig.savefig(fig_path, dpi=out_cfg.get("fig_dpi", 300), bbox_inches="tight")
                    plt.close(fig)

                # Circle plots (ROI)
                fig_path = roi_output / f"connectivity_circle_grid_{predictor}.png"
                plot_multi_band_circle_grid(
                    all_results=p_roi_results,
                    level="roi",
                    sig_col=roi_sig_col,
                    suptitle=f"wSMI Connectivity Circles: Effect of {predictor}",
                    save_path=str(fig_path),
                    dpi=out_cfg.get("fig_dpi", 300),
                )
                plt.close("all")
                
                for band, results in p_roi_results.items():
                    fig_path = roi_output / f"connectivity_circle_{band}_{predictor}.png"
                    plot_roi_connectivity_circle(
                        results_df=results,
                        sig_col=roi_sig_col,
                        title=f"wSMI {band} – ROI Connectivity Circle ({predictor})",
                        save_path=str(fig_path),
                        dpi=out_cfg.get("fig_dpi", 300),
                    )
                    plt.close("all")

                summary_path = roi_output / f"summary_all_bands_{predictor}.csv"
                create_summary_table(p_roi_results, save_path=str(summary_path))

        if channel_results:
            print("\n  Generating channel-level figures...")
            for predictor, p_channel_results in channel_results.items():
                if not p_channel_results:
                    continue
                    
                for band, results in p_channel_results.items():
                    # Matrix plot — pass `rois` so channels are grouped by ROI
                    # block (separator lines + topographic order within block).
                    fig_path = channel_output / f"channel_matrix_{band}_{predictor}.png"
                    plot_channel_contrast_matrix(
                        results_df=results,
                        sig_col=ch_sig_col,
                        title=f"wSMI {band} – Channel LMM t-stat ({ch_correction}, {predictor})",
                        save_path=str(fig_path),
                        dpi=out_cfg.get("fig_dpi", 300),
                        rois=rois,
                    )
                    plt.close("all")

                    # Circle plot (channel)
                    fig_path = channel_output / f"channel_circle_{band}_{predictor}.png"
                    plot_channel_connectivity_circle(
                        results_df=results,
                        rois=rois,
                        sig_col=ch_sig_col,
                        title=f"wSMI {band} – Channel Connectivity ({ch_correction}, {predictor})",
                        save_path=str(fig_path),
                        dpi=out_cfg.get("fig_dpi", 300),
                    )
                    plt.close("all")

                # Circle grid across bands
                fig_path = channel_output / f"channel_circle_grid_{predictor}.png"
                plot_multi_band_circle_grid(
                    all_results=p_channel_results,
                    level="channel",
                    rois=rois,
                    sig_col=ch_sig_col,
                    suptitle=f"Channel Connectivity Circles ({ch_correction}, {predictor})",
                    save_path=str(fig_path),
                    dpi=out_cfg.get("fig_dpi", 300),
                )
                plt.close("all")

                summary_path = channel_output / f"summary_all_bands_{predictor}.csv"
                create_summary_table(p_channel_results, save_path=str(summary_path))

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


def _p_col_for_method(method: str) -> str:
    """
    Map correction method name to its p-value column (used as the input to
    the family-wise Bonferroni multiplier).
    """
    mapping = {
        "fdr": "p_value_fdr",
        "nbs": "p_value_nbs",
        "max_t": "p_value_perm",
    }
    return mapping.get(method, "p_value_fdr")


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
    parser.add_argument(
        "--band",
        type=str,
        default=None,
        help="Run analysis for a single band (for parallel execution)",
    )
    args = parser.parse_args()
    main(config_path=args.config, single_band=args.band)
