"""Residual diagnostics for the objective-marker models.

This module exists because the original pipeline validated no distributional
assumption at all. It provides Gaussian-track diagnostics (QQ, residuals vs
fitted, skew/kurtosis, Breusch-Pagan) and binomial-track binned residual plots
(Gelman & Hill), which are the standard substitute for DHARMa's
simulation-based residuals -- DHARMa is not installed and cannot be installed
because the conda channel is blocked by the institutional proxy.

With n ~ 2460, Shapiro-Wilk rejects on trivial deviations, so normality is
reported as descriptive summaries plus plots rather than a pass/fail test.
"""

# =============================================================================
# IMPORTS
# =============================================================================

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import statsmodels.api as sm  # noqa: E402
from scipy import stats  # noqa: E402
from statsmodels.stats.diagnostic import het_breuschpagan  # noqa: E402

# =============================================================================
# CONFIGURATION
# =============================================================================

FIG_DPI = 300
SE_BAND_MULTIPLIER = 2.0

# =============================================================================
# DIAGNOSTICS
# =============================================================================


def binned_residuals(
    fitted: np.ndarray, residuals: np.ndarray, n_bins: int
) -> pd.DataFrame:
    """Gelman-Hill binned residual summary.

    Raw residuals from a binomial model are uninformative when plotted against
    fitted values because the response is discrete. Averaging within bins of
    the fitted value reveals systematic misfit that the raw plot hides.

    Parameters
    ----------
    fitted : np.ndarray
        Fitted values on the response scale.
    residuals : np.ndarray
        Residuals (Pearson or response scale).
    n_bins : int
        Number of equal-count bins.

    Returns
    -------
    pd.DataFrame
        Columns: bin, x_mean, y_mean, se, n, outside_band. ``outside_band`` is
        True where the bin mean falls outside +/- 2 SE, which under a
        well-specified model should happen for about 5% of bins.
    """
    df = pd.DataFrame({
        "fitted": np.asarray(fitted, dtype=float),
        "resid": np.asarray(residuals, dtype=float),
    })
    df["bin"] = pd.qcut(df["fitted"].rank(method="first"), n_bins, labels=False)

    agg = (
        df.groupby("bin")
        .agg(x_mean=("fitted", "mean"), y_mean=("resid", "mean"),
             sd=("resid", "std"), n=("resid", "size"))
        .reset_index()
    )
    agg["se"] = agg["sd"] / np.sqrt(agg["n"].clip(lower=1))
    agg["outside_band"] = agg["y_mean"].abs() > SE_BAND_MULTIPLIER * agg["se"]
    return agg.drop(columns=["sd"])


def gaussian_residual_diagnostics(
    residuals: np.ndarray, fitted: np.ndarray
) -> dict:
    """Descriptive normality and homoscedasticity summaries.

    Parameters
    ----------
    residuals : np.ndarray
        Model residuals.
    fitted : np.ndarray
        Fitted values, used as the Breusch-Pagan regressor.

    Returns
    -------
    dict
        Keys: skew, kurtosis, breusch_pagan_stat, breusch_pagan_p, n.
    """
    residuals = np.asarray(residuals, dtype=float)
    fitted = np.asarray(fitted, dtype=float)
    exog = sm.add_constant(fitted)
    bp_stat, bp_p, _, _ = het_breuschpagan(residuals, exog)
    return {
        "skew": float(stats.skew(residuals)),
        "kurtosis": float(stats.kurtosis(residuals)),
        "breusch_pagan_stat": float(bp_stat),
        "breusch_pagan_p": float(bp_p),
        "n": int(len(residuals)),
    }


# =============================================================================
# PLOTTING
# =============================================================================


def save_diagnostic_plots(
    fitted: np.ndarray,
    residuals: np.ndarray,
    label: str,
    output_dir: Path,
    n_bins: int,
) -> Path:
    """Write a three-panel diagnostic figure: QQ, residuals vs fitted, binned.

    Parameters
    ----------
    fitted : np.ndarray
        Fitted values.
    residuals : np.ndarray
        Residuals.
    label : str
        Figure title and filename stem, e.g. ``"glmm_commission_rate"``.
    output_dir : Path
        Destination directory; created if absent.
    n_bins : int
        Bin count for the binned-residual panel.

    Returns
    -------
    Path
        Path to the saved PNG.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)

    stats.probplot(residuals, dist="norm", plot=axes[0])
    axes[0].set_title("Normal Q-Q")

    axes[1].scatter(fitted, residuals, s=6, alpha=0.25, linewidths=0)
    axes[1].axhline(0, color="black", lw=1.2, ls="--")
    axes[1].set_xlabel("Fitted")
    axes[1].set_ylabel("Residual")
    axes[1].set_title("Residuals vs Fitted")

    binned = binned_residuals(fitted, residuals, n_bins)
    axes[2].errorbar(binned["x_mean"], binned["y_mean"],
                     yerr=SE_BAND_MULTIPLIER * binned["se"],
                     fmt="o", markersize=5, capsize=3)
    axes[2].axhline(0, color="black", lw=1.2, ls="--")
    axes[2].set_xlabel("Mean fitted value in bin")
    axes[2].set_ylabel("Mean residual")
    axes[2].set_title(
        f"Binned residuals ({int(binned['outside_band'].sum())}/"
        f"{len(binned)} outside ±2 SE)"
    )

    fig.suptitle(label, fontsize=13, fontweight="bold")
    out_path = output_dir / f"{label}_diagnostics.png"
    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")
    return out_path
