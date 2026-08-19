"""
LMM assumption diagnostics for the Andrillon cluster-permutation pipeline.

The pipeline already computes per-channel residual diagnostics inside
``run_lmm_per_channel``, but they were only ever printed to the job log and
buried in ``results.pkl`` — no report read them, so no assumption was ever
actually inspected. This module refits the *observed* model for every
(predictor, marker) combination — no permutations, so it is cheap — and turns
those diagnostics into CSVs and a summary figure.

What is reported and why
------------------------
Normality is reported descriptively (skewness, excess kurtosis, outlier
fraction), not as a test. With n ~ 1700 observations per channel, Shapiro-Wilk
rejects on deviations far too small to matter for a permutation-based cluster
test — it returned "100% of channels violate normality" for essentially every
marker, which is unactionable. What does matter for a cluster test is whether
the t-statistic is being driven by a handful of extreme probes, which is what
the outlier fraction and excess kurtosis measure.

Note on interpretation: the cluster p-values come from the permutation
distribution, not from the t reference distribution, so non-normal residuals do
not invalidate the type-I error rate. The costs are lost power and unstable,
outlier-driven t-statistics. Heteroscedasticity is the more consequential
finding, because Freedman-Lane permutes residuals within subject and therefore
assumes they are exchangeable.

Usage
-----
    python Stats_andrillon/lmm_assumption_diagnostics.py \\
        --config Stats_andrillon/config_andrillon.yaml

    # restrict to one predictor and/or one marker
    python Stats_andrillon/lmm_assumption_diagnostics.py \\
        --config Stats_andrillon/config_andrillon.yaml \\
        --predictor onoff --marker sleep/psd_relative_gamma
"""

# =============================================================================
# IMPORTS
# =============================================================================

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402

_STATS_DIR = Path(__file__).resolve().parent.parent / "Statistics"
sys.path.insert(0, str(_STATS_DIR))

from helpers import get_model_folder_name  # noqa: E402
from lmm_model import (  # noqa: E402
    RESIDUAL_EXKURT_FLAG,
    RESIDUAL_OUTLIER_SD,
    RESIDUAL_SKEW_FLAG,
    run_lmm_per_channel,
)

from andrillon_pipeline import (  # noqa: E402
    _resolve_formula_for_predictor,
    get_marker_list,
    load_marker_matrix,
)

# =============================================================================
# CONFIGURATION
# =============================================================================

_REPO_ROOT = Path(__file__).resolve().parents[1]
PALETTE = yaml.safe_load(open(_REPO_ROOT / "color_palette.yaml"))

DIAGNOSTICS_SUBDIR = "assumption_diagnostics"
CHANNEL_CSV_SUFFIX = "_channel_diagnostics.csv"
SUMMARY_CSV_NAME = "assumption_summary.csv"
SUMMARY_FIG_NAME = "assumption_summary.png"

# Figure geometry. Kept here rather than inline so the figure can be resized
# without hunting through plotting code.
FIG_WIDTH_PER_PANEL = 4.2
FIG_HEIGHT_PER_MARKER = 0.28
FIG_MIN_HEIGHT = 4.0
FIG_DPI = 300

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def resolve_predictors(config: Dict, predictor_override: Optional[str]) -> List[str]:
    """
    Resolve which predictors of interest to run diagnostics for.

    Parameters
    ----------
    config : dict
        Pipeline configuration.
    predictor_override : str or None
        Single predictor passed on the CLI. When ``None``, every predictor
        declared in ``lmm.predictor_of_interest`` is used.

    Returns
    -------
    list of str
        Predictor names.
    """
    if predictor_override is not None:
        return [predictor_override]

    declared = config["lmm"]["predictor_of_interest"]
    if isinstance(declared, str):
        return [declared]
    return list(declared)


def marker_folder_name(marker_name: str) -> str:
    """
    Convert ``"<epoch_type>/<marker>"`` into the on-disk folder name.

    Mirrors the naming used by :func:`andrillon_pipeline.run_marker_analysis`
    so diagnostics land next to the cluster results of the same marker.
    """
    if "/" in marker_name:
        marker_type, base = marker_name.split("/", 1)
        return f"{marker_type}_{base.replace('/', '_').replace(' ', '_')}"
    return marker_name.replace("/", "_").replace(" ", "_")


def diagnose_marker(marker_name: str, config: Dict, predictor: str) -> Dict:
    """
    Fit the observed LMM for one marker and collect its assumption diagnostics.

    No permutations are run — this is the observed fit only, which is what the
    residual diagnostics describe.

    Parameters
    ----------
    marker_name : str
        Marker identifier, ``"<epoch_type>/<marker>"``.
    config : dict
        Pipeline configuration.
    predictor : str
        Predictor of interest, which determines the formula and the subject
        variability filter applied during loading.

    Returns
    -------
    dict
        Two keys: ``"summary"`` (one record for the marker) and ``"per_channel"``
        (a :class:`pandas.DataFrame`, one row per channel).
    """
    formula = _resolve_formula_for_predictor(config, predictor)

    power_data, df_behavioral, channels = load_marker_matrix(
        marker_name=marker_name,
        config=config,
        predictor=predictor,
    )

    t_stats, p_values, diag = run_lmm_per_channel(
        power_data=power_data,
        df_behavioral=df_behavioral,
        formula=formula,
        predictor_of_interest=predictor,
        method=config["lmm"]["method"],
        maxiter=config["lmm"]["maxiter"],
        random_state=config["andrillon_clustering"]["seed"],
        return_diagnostics=True,
    )

    # The per-channel diagnostic lists only receive an entry for channels that
    # produced a fit, so they are shorter than `channels` whenever a channel
    # failed or was skipped. Name the rows by the channels that succeeded.
    fitted_channels = [
        ch for idx, ch in enumerate(channels) if np.isfinite(t_stats[idx])
    ]
    n_fitted = len(diag["residual_skew"])
    if len(fitted_channels) != n_fitted:
        raise RuntimeError(
            f"{marker_name} / {predictor}: {n_fitted} diagnostic records but "
            f"{len(fitted_channels)} channels with finite t-statistics. "
            "Cannot align diagnostics to channel names."
        )

    per_channel = pd.DataFrame({
        "predictor": predictor,
        "marker": marker_name,
        "channel": fitted_channels,
        "conditional_r2": diag["conditional_r2"],
        "residual_variance": diag["residual_variance"],
        "residual_skew": diag["residual_skew"],
        "residual_exkurt": diag["residual_exkurt"],
        "pct_residual_outliers": diag["pct_residual_outliers"],
        "breusch_pagan_p": diag["breusch_pagan_p"],
    })

    summary = {
        "predictor": predictor,
        "marker": marker_name,
        "n_observations": int(power_data.shape[0]),
        "n_subjects": int(df_behavioral["subject"].nunique()),
        "n_channels": len(channels),
        "n_converged": diag["n_converged"],
        "n_warnings": diag["n_warnings"],
        "n_failed": diag["n_failed"],
        "convergence_rate": diag["convergence_rate"],
        "warning_rate": diag.get("warning_rate", np.nan),
        "conditional_r2_mean": diag.get("conditional_r2_mean", np.nan),
        "residual_skew_mean": diag.get("residual_skew_mean", np.nan),
        "residual_abs_skew_max": diag.get("residual_abs_skew_max", np.nan),
        "pct_high_skew": diag.get("pct_high_skew", np.nan),
        "residual_exkurt_mean": diag.get("residual_exkurt_mean", np.nan),
        "residual_exkurt_max": diag.get("residual_exkurt_max", np.nan),
        "pct_high_kurtosis": diag.get("pct_high_kurtosis", np.nan),
        "pct_residual_outliers_mean": diag.get("pct_residual_outliers_mean", np.nan),
        "pct_residual_outliers_max": diag.get("pct_residual_outliers_max", np.nan),
        "pct_heteroscedasticity": diag.get("pct_heteroscedasticity", np.nan),
        "skew_flag_threshold": RESIDUAL_SKEW_FLAG,
        "exkurt_flag_threshold": RESIDUAL_EXKURT_FLAG,
        "outlier_sd_threshold": RESIDUAL_OUTLIER_SD,
        "formula": formula,
    }

    logger.info(
        "%s / %s: skew %+.2f, exkurt %+.2f, outliers %.2f%%, hetero %.0f%% of channels",
        marker_name, predictor,
        summary["residual_skew_mean"],
        summary["residual_exkurt_mean"],
        summary["pct_residual_outliers_mean"],
        summary["pct_heteroscedasticity"],
    )

    return {"summary": summary, "per_channel": per_channel}


def plot_assumption_summary(
    summary_df: pd.DataFrame,
    output_path: Path,
    predictor: str,
) -> None:
    """
    Draw a three-panel overview of residual shape across markers.

    Panels: mean residual skewness, mean excess kurtosis, and mean outlier
    percentage. Bars past the flag threshold are drawn solid; bars inside it are
    drawn hollow, following the project convention that fill — never hue —
    encodes "this one warrants attention".

    Parameters
    ----------
    summary_df : pd.DataFrame
        One row per marker, as produced by :func:`diagnose_marker`.
    output_path : Path
        Destination PNG path.
    predictor : str
        Predictor of interest, used in the figure title.
    """
    ordered = summary_df.sort_values("residual_exkurt_mean", ascending=True)
    markers = ordered["marker"].tolist()
    y = np.arange(len(markers))

    accent = PALETTE["palette"]["blue"]
    panels = [
        ("residual_skew_mean", "Mean residual skewness", RESIDUAL_SKEW_FLAG, True),
        ("residual_exkurt_mean", "Mean excess kurtosis", RESIDUAL_EXKURT_FLAG, False),
        ("pct_residual_outliers_mean",
         f"% obs. with |resid| > {RESIDUAL_OUTLIER_SD:g} SD", None, False),
    ]

    height = max(FIG_MIN_HEIGHT, FIG_HEIGHT_PER_MARKER * len(markers) + 1.6)
    fig, axes = plt.subplots(
        1, len(panels),
        figsize=(FIG_WIDTH_PER_PANEL * len(panels), height),
        sharey=True,
    )

    for ax, (column, label, threshold, use_abs) in zip(axes, panels):
        values = ordered[column].to_numpy(dtype=float)
        if threshold is None:
            flagged = np.zeros(values.shape, dtype=bool)
        elif use_abs:
            flagged = np.abs(values) > threshold
        else:
            flagged = values > threshold

        ax.barh(
            y, values,
            color=[accent if f else "none" for f in flagged],
            edgecolor=accent,
            linewidth=1.1,
            height=0.7,
        )
        if threshold is not None:
            ax.axvline(threshold, color=PALETTE["palette"]["gray"],
                       linestyle="--", linewidth=1.0)
            if use_abs:
                ax.axvline(-threshold, color=PALETTE["palette"]["gray"],
                           linestyle="--", linewidth=1.0)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel(label)
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].set_yticks(y)
    axes[0].set_yticklabels(markers, fontsize=8)
    fig.suptitle(
        f"LMM residual diagnostics — predictor: {predictor}\n"
        "(solid = past flag threshold; dashed line = threshold)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# MAIN
# =============================================================================


def run_assumption_diagnostics(
    config_path: str,
    predictor_override: Optional[str] = None,
    marker_override: Optional[str] = None,
    n_jobs_override: Optional[int] = None,
) -> pd.DataFrame:
    """
    Run assumption diagnostics across predictors × markers and write reports.

    Parameters
    ----------
    config_path : str
        Path to ``config_andrillon.yaml``.
    predictor_override : str, optional
        Restrict to a single predictor.
    marker_override : str, optional
        Restrict to a single marker.
    n_jobs_override : int, optional
        Override ``andrillon_clustering.n_jobs``. That value is sized for the
        SLURM allocation; an interactive session usually has far fewer cores,
        and loky reads the node's physical core count rather than the cgroup
        limit, so oversubscription is easy to hit.

    Returns
    -------
    pd.DataFrame
        The concatenated summary table (also written to disk).
    """
    config = yaml.safe_load(open(config_path))
    output_root = Path(config["project"]["output_path"])
    n_jobs = (
        n_jobs_override if n_jobs_override is not None
        else config["andrillon_clustering"].get("n_jobs", 1)
    )

    predictors = resolve_predictors(config, predictor_override)
    markers = [marker_override] if marker_override else get_marker_list(config)
    logger.info(
        "Running assumption diagnostics for %d predictor(s) × %d marker(s)",
        len(predictors), len(markers),
    )

    all_summaries: List[pd.DataFrame] = []

    for predictor in predictors:
        model_folder = get_model_folder_name(
            _resolve_formula_for_predictor(config, predictor), predictor
        )
        diag_dir = output_root / model_folder / DIAGNOSTICS_SUBDIR
        diag_dir.mkdir(parents=True, exist_ok=True)

        results = Parallel(n_jobs=n_jobs, backend="loky", verbose=5)(
            delayed(diagnose_marker)(marker, config, predictor)
            for marker in markers
        )

        for marker, result in zip(markers, results):
            channel_csv = diag_dir / f"{marker_folder_name(marker)}{CHANNEL_CSV_SUFFIX}"
            result["per_channel"].to_csv(channel_csv, index=False)

        summary_df = pd.DataFrame([r["summary"] for r in results])
        summary_df.to_csv(diag_dir / SUMMARY_CSV_NAME, index=False)
        plot_assumption_summary(
            summary_df, diag_dir / SUMMARY_FIG_NAME, predictor
        )
        logger.info("Wrote diagnostics for predictor '%s' to %s", predictor, diag_dir)

        all_summaries.append(summary_df)

    combined = pd.concat(all_summaries, ignore_index=True)
    combined_path = output_root / SUMMARY_CSV_NAME
    combined.to_csv(combined_path, index=False)
    logger.info("Wrote combined summary across predictors to %s", combined_path)

    return combined


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LMM assumption diagnostics for the Andrillon pipeline"
    )
    parser.add_argument("--config", type=str, required=True,
                        help="Path to config_andrillon.yaml")
    parser.add_argument("--predictor", type=str, default=None,
                        help="Restrict to a single predictor of interest")
    parser.add_argument("--marker", type=str, default=None,
                        help="Restrict to a single marker, e.g. sleep/psd_relative_gamma")
    parser.add_argument("--n-jobs", type=int, default=None,
                        help="Override andrillon_clustering.n_jobs (sized for SLURM)")

    args = parser.parse_args()
    run_assumption_diagnostics(
        args.config, args.predictor, args.marker, args.n_jobs
    )
