#!/usr/bin/env python
"""
Simulation Utilities — Type I Error Evaluation.

Provides functions to:
  1. Generate pseudo-synthetic null feature matrices (preserving covariance
     structure and participant membership, with no feature-label association).
  2. Compute empirical Type I error rates from N simulation p-values.
  3. Plot p-value calibration diagnostics (histogram, QQ-plot, FPR curve).

The null condition is established solely by making the features random:
labels (y_real) are passed unchanged to the true pipeline run. The
permutation test inside each simulation then shuffles y_real to build the
null distribution — identical to the real pipeline procedure.

Project: depressed_mindwandering
"""

# =============================================================================
# Imports
# =============================================================================

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

from sklearn.covariance import LedoitWolf
from statsmodels.stats.proportion import proportion_confint
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# =============================================================================
# Synthetic Data Generation
# =============================================================================

def generate_synthetic_features(
    X: pd.DataFrame,
    groups: pd.Series,
    random_state: int,
    covariance_scope: str = "per_participant",
    covariance_method: str = "ledoit_wolf",
) -> pd.DataFrame:
    """
    Generate a synthetic feature matrix preserving covariance structure.

    Samples new feature values from a multivariate Gaussian fitted to the
    real data. The generated X_synth has the same statistical structure
    (mean, covariance) as the original data but contains no information
    about any label. Labels must NOT be modified by this function —
    the caller passes y_real unchanged to the classification pipeline.

    Parameters
    ----------
    X : pd.DataFrame
        Original feature matrix (n_samples × n_features).
    groups : pd.Series
        Participant ID per sample, same index as X.
    random_state : int
        RNG seed for full reproducibility.
    covariance_scope : str
        'per_participant' : fit a separate covariance matrix per subject.
            Preserves individual neural signatures. Requires LedoitWolf
            when n_features > n_samples_per_subject (the typical case).
        'global' : fit one covariance matrix across all subjects. Less
            realistic but stable with very few samples per subject.
        'diag' : diagonal covariance (independent features). Degenerate
            baseline — use to verify that the pipeline does not inflate
            FPR even under maximum null conditions.
    covariance_method : str
        'ledoit_wolf' (default) : Ledoit-Wolf shrinkage estimator.
            Recommended for EEG feature matrices where n_features >> n_samples.
        'full' : unregularized sample covariance. May be singular for
            wide matrices; use only when n_samples >> n_features per group.
        'diag' : diagonal covariance (overrides covariance_scope='diag').

    Returns
    -------
    pd.DataFrame
        Synthetic feature matrix with the same index and columns as X.

    Notes
    -----
    Ledoit-Wolf shrinkage produces a well-conditioned positive-definite
    estimator suitable for the typical EEG scenario (300–800 features,
    ~100 samples per participant). ``sklearn.covariance.LedoitWolf``
    is used for the computation.
    """
    rng = np.random.default_rng(random_state)
    X_values = X.values.astype(float)
    synth_values = np.empty_like(X_values)

    if covariance_scope == "per_participant":
        unique_groups = groups.unique()
        for subj in unique_groups:
            mask = (groups == subj).values
            X_sub = X_values[mask]
            n_sub, n_feat = X_sub.shape
            means = X_sub.mean(axis=0)

            if covariance_method == "diag":
                stds = X_sub.std(axis=0)
                stds[stds == 0] = 1.0
                synth_values[mask] = rng.normal(size=(n_sub, n_feat)) * stds + means

            elif covariance_method == "full":
                cov = np.cov(X_sub, rowvar=False)
                synth_values[mask] = rng.multivariate_normal(means, cov, size=n_sub)

            else:  # ledoit_wolf (default)
                lw = LedoitWolf()
                lw.fit(X_sub)
                synth_values[mask] = rng.multivariate_normal(
                    means, lw.covariance_, size=n_sub
                )

    elif covariance_scope == "global":
        n_total, n_feat = X_values.shape
        means = X_values.mean(axis=0)

        if covariance_method == "diag":
            stds = X_values.std(axis=0)
            stds[stds == 0] = 1.0
            synth_values = rng.normal(size=X_values.shape) * stds + means

        elif covariance_method == "full":
            cov = np.cov(X_values, rowvar=False)
            synth_values = rng.multivariate_normal(means, cov, size=n_total)

        else:  # ledoit_wolf
            lw = LedoitWolf()
            lw.fit(X_values)
            synth_values = rng.multivariate_normal(
                means, lw.covariance_, size=n_total
            )

    elif covariance_scope == "diag":
        means = X_values.mean(axis=0)
        stds = X_values.std(axis=0)
        stds[stds == 0] = 1.0
        synth_values = rng.normal(size=X_values.shape) * stds + means

    else:
        raise ValueError(
            f"Unknown covariance_scope: '{covariance_scope}'. "
            f"Expected one of: 'per_participant', 'global', 'diag'."
        )

    return pd.DataFrame(synth_values, index=X.index, columns=X.columns)


# =============================================================================
# Type I Error Rate Computation
# =============================================================================

def compute_type1_error_rate(
    p_values_df: pd.DataFrame,
    alpha_levels: list[float] = (0.01, 0.05, 0.10),
    p_metric_cols: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Compute empirical Type I error rates at multiple alpha levels.

    Under the null hypothesis, p-values should follow U[0,1], so FPR
    should equal alpha. Deviations indicate miscalibration.

    Parameters
    ----------
    p_values_df : pd.DataFrame
        One row per simulation. Must contain columns starting with 'p_'
        (e.g., 'p_mean_auc', 'p_mean_balanced_accuracy').
    alpha_levels : list of float
        Significance thresholds at which to report FPR.
    p_metric_cols : list of str, optional
        Which p-value columns to use. Defaults to all columns starting
        with 'p_'.

    Returns
    -------
    pd.DataFrame
        Columns: alpha, metric, n_simulations, n_significant, fpr,
        ci_lower, ci_upper (Wilson 95% CI on the FPR proportion).
    """
    if p_metric_cols is None:
        p_metric_cols = [c for c in p_values_df.columns if c.startswith("p_")]

    rows = []
    for alpha in alpha_levels:
        for col in p_metric_cols:
            p_vals = p_values_df[col].dropna()
            n_valid = len(p_vals)
            if n_valid == 0:
                continue
            n_significant = int((p_vals < alpha).sum())
            fpr = n_significant / n_valid
            ci_low, ci_high = proportion_confint(
                n_significant, n_valid, alpha=0.05, method="wilson"
            )
            rows.append({
                "alpha": alpha,
                "metric": col.replace("p_", ""),
                "n_simulations": n_valid,
                "n_significant": n_significant,
                "fpr": round(fpr, 4),
                "ci_lower": round(float(ci_low), 4),
                "ci_upper": round(float(ci_high), 4),
            })

    return pd.DataFrame(rows)


# =============================================================================
# P-Value Calibration Plots
# =============================================================================

def plot_pvalue_calibration(
    p_values_df: pd.DataFrame,
    results_path: str,
    pipeline_label: str,
    p_metric_cols: Optional[list[str]] = None,
    alpha_levels: list[float] = (0.01, 0.05, 0.10),
) -> None:
    """
    Generate p-value calibration diagnostic plots and save as HTML.

    Three plots are produced:
    1. Histogram of p-values per metric (should be flat/uniform under H0).
    2. QQ-plot: empirical p-value quantiles vs. uniform U[0,1] quantiles.
       Points above the diagonal indicate anti-conservatism (inflated FPR).
       Points below indicate conservatism.
    3. FPR vs alpha curve: empirical CDF of p-values overlaid on the
       ideal y=x diagonal.

    Parameters
    ----------
    p_values_df : pd.DataFrame
        One row per simulation. Columns starting with 'p_' are used.
    results_path : str
        Directory where plots are saved (a 'plots/' subdirectory is created).
    pipeline_label : str
        Short label for the pipeline (e.g., 'LOSO', 'WithinSubject').
    p_metric_cols : list of str, optional
        Which columns to plot. Defaults to all 'p_*' columns.
    alpha_levels : list of float
        Alpha levels shown as vertical lines on the histogram.
    """
    if p_metric_cols is None:
        p_metric_cols = [c for c in p_values_df.columns if c.startswith("p_")]

    plots_dir = Path(results_path) / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    n_metrics = len(p_metric_cols)
    if n_metrics == 0:
        return

    # ── Plot 1: p-value histograms ──────────────────────────────────────────
    fig_hist = make_subplots(
        rows=1, cols=n_metrics,
        subplot_titles=[c.replace("p_", "") for c in p_metric_cols],
    )
    for col_idx, col in enumerate(p_metric_cols, start=1):
        vals = p_values_df[col].dropna().values
        fig_hist.add_trace(
            go.Histogram(
                x=vals,
                nbinsx=20,
                name=col,
                marker_color="steelblue",
                opacity=0.75,
            ),
            row=1, col=col_idx,
        )
        # Expected uniform density line (count-scaled)
        n_bins = 20
        n_vals = len(vals)
        expected_count = n_vals / n_bins
        fig_hist.add_hline(
            y=expected_count,
            line_dash="dash",
            line_color="red",
            annotation_text="U[0,1] expected",
            row=1, col=col_idx,
        )
    fig_hist.update_layout(
        title=f"{pipeline_label} — Type I Error: p-value distributions (N={len(p_values_df)} sims)",
        showlegend=False,
        height=400,
    )
    fig_hist.write_html(str(plots_dir / "pvalue_histogram.html"))

    # ── Plot 2: QQ-plot vs. Uniform ─────────────────────────────────────────
    fig_qq = go.Figure()
    for col in p_metric_cols:
        vals = np.sort(p_values_df[col].dropna().values)
        n = len(vals)
        uniform_quantiles = (np.arange(1, n + 1) - 0.5) / n
        fig_qq.add_trace(go.Scatter(
            x=uniform_quantiles,
            y=vals,
            mode="markers",
            name=col.replace("p_", ""),
            marker=dict(size=6, opacity=0.6),
        ))
    # Ideal diagonal
    fig_qq.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode="lines",
        line=dict(color="red", dash="dash"),
        name="y = x (ideal)",
    ))
    fig_qq.update_layout(
        title=f"{pipeline_label} — QQ-plot: empirical p-values vs. U[0,1]",
        xaxis_title="Theoretical uniform quantile",
        yaxis_title="Empirical p-value quantile",
        height=500,
    )
    fig_qq.write_html(str(plots_dir / "pvalue_qqplot.html"))

    # ── Plot 3: FPR vs alpha curve ───────────────────────────────────────────
    alpha_grid = np.linspace(0, 1, 101)
    fig_fpr = go.Figure()
    for col in p_metric_cols:
        vals = p_values_df[col].dropna().values
        fprs = [float((vals < a).mean()) for a in alpha_grid]
        fig_fpr.add_trace(go.Scatter(
            x=alpha_grid,
            y=fprs,
            mode="lines",
            name=col.replace("p_", ""),
        ))
    # Ideal diagonal
    fig_fpr.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode="lines",
        line=dict(color="red", dash="dash"),
        name="ideal (FPR = α)",
    ))
    fig_fpr.update_layout(
        title=f"{pipeline_label} — Empirical FPR vs. alpha (N={len(p_values_df)} sims)",
        xaxis_title="Alpha level",
        yaxis_title="Empirical FPR",
        height=500,
    )
    fig_fpr.write_html(str(plots_dir / "fpr_vs_alpha.html"))

    print(f"  Calibration plots saved → {plots_dir}")
