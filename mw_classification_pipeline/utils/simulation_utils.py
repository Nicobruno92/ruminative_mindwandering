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

from scipy import stats as scipy_stats
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

def _verdict(fpr: float, alpha: float, ci_lower: float, ci_upper: float) -> tuple[str, str]:
    """Return a calibration verdict ("OK" / "INFLATED" / "CONSERVATIVE") and a color."""
    if ci_lower > alpha:
        return "INFLATED", "#d62728"  # red
    if ci_upper < alpha:
        return "CONSERVATIVE", "#1f77b4"  # blue
    return "OK", "#2ca02c"  # green


def plot_pvalue_calibration(
    p_values_df: pd.DataFrame,
    results_path: str,
    pipeline_label: str,
    p_metric_cols: Optional[list[str]] = None,
    alpha_levels: list[float] = (0.01, 0.05, 0.10),
) -> None:
    """
    Generate p-value calibration diagnostic plots and save them as HTML.

    Four plots are produced, each labelled to be readable on its own:

    1. ``fpr_summary_bars.html`` — bar chart of empirical FPR at each
       alpha level, with Wilson 95% CI error bars and a horizontal line
       at the ideal FPR=α. Colour-coded verdict per metric (green=OK,
       red=inflated, blue=conservative). **This is the headline plot.**
    2. ``pvalue_histogram.html`` — histogram of p-values per metric.
       Under H0 the distribution should be flat (uniform). Annotated
       with the expected count per bin, the α=0.05 boundary, and the
       Kolmogorov–Smirnov test against U[0,1].
    3. ``pvalue_qqplot.html`` — empirical vs. theoretical uniform
       quantiles. Includes a 95% confidence envelope from Beta order
       statistics, so deviation from y=x can be judged visually.
    4. ``fpr_vs_alpha.html`` — empirical CDF of p-values overlaid on the
       y=x diagonal, with α=0.01/0.05/0.10 marked.

    Parameters
    ----------
    p_values_df : pd.DataFrame
        One row per simulation. Columns starting with ``"p_"`` are used.
    results_path : str
        Directory where plots are saved (a ``plots/`` subdir is created).
    pipeline_label : str
        Short label for the pipeline (e.g. ``'LOSO / contrast / model'``).
    p_metric_cols : list of str, optional
        Which columns to plot. Defaults to all ``p_*`` columns.
    alpha_levels : list of float
        Alpha levels for the bar plot, vertical lines and FPR table.
    """
    if p_metric_cols is None:
        p_metric_cols = [c for c in p_values_df.columns if c.startswith("p_")]

    plots_dir = Path(results_path) / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    n_metrics = len(p_metric_cols)
    if n_metrics == 0:
        return

    n_total_sims = len(p_values_df)
    metric_clean = [c.replace("p_", "").replace("mean_", "") for c in p_metric_cols]

    # =========================================================================
    # Plot 1 — FPR summary: single panel with grouped bars (headline plot)
    # =========================================================================
    # x-axis = metric; one bar per alpha level (grouped); error bars = Wilson
    # 95% CI; horizontal dashed lines = ideal FPR for each alpha.
    alpha_palette = {
        0.01: "#9ecae1",  # light blue
        0.05: "#4292c6",  # blue
        0.10: "#08519c",  # dark blue
    }
    alpha_line_colors = {0.01: "#999999", 0.05: "#d62728", 0.10: "#7f7f7f"}

    fig_bars = go.Figure()

    # Pre-compute FPR table for all (alpha, metric) combos
    fpr_table = {}
    for alpha in alpha_levels:
        row_data = []
        for col, label in zip(p_metric_cols, metric_clean):
            vals = p_values_df[col].dropna().values
            n = len(vals)
            if n == 0:
                row_data.append(None)
                continue
            n_sig = int((vals < alpha).sum())
            fpr = n_sig / n
            ci_lo, ci_hi = proportion_confint(n_sig, n, alpha=0.05, method="wilson")
            verdict, _ = _verdict(fpr, alpha, float(ci_lo), float(ci_hi))
            row_data.append({
                "label": label, "fpr": fpr, "n": n, "n_sig": n_sig,
                "ci_lo": float(ci_lo), "ci_hi": float(ci_hi), "verdict": verdict,
            })
        fpr_table[alpha] = row_data

    # Add one trace per alpha so plotly groups them automatically
    for alpha in alpha_levels:
        rows = [r for r in fpr_table[alpha] if r is not None]
        x_labels = [r["label"] for r in rows]
        fprs = [r["fpr"] for r in rows]
        err_plus = [r["ci_hi"] - r["fpr"] for r in rows]
        err_minus = [r["fpr"] - r["ci_lo"] for r in rows]
        hover = [
            f"<b>{r['label']}</b><br>"
            f"α = {alpha:g} (ideal FPR)<br>"
            f"Empirical FPR = {r['fpr']:.3f} ({r['n_sig']}/{r['n']})<br>"
            f"Wilson 95% CI = [{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]<br>"
            f"Verdict: <b>{r['verdict']}</b>"
            for r in rows
        ]
        fig_bars.add_trace(go.Bar(
            x=x_labels, y=fprs,
            name=f"α = {alpha:g}",
            marker_color=alpha_palette.get(alpha, "#4292c6"),
            error_y=dict(
                type="data", symmetric=False,
                array=err_plus, arrayminus=err_minus,
                color="black", thickness=1.2, width=5,
            ),
            text=[f"{f:.1%}" for f in fprs], textposition="outside",
            textfont=dict(size=11),
            hovertext=hover, hoverinfo="text",
        ))

    # Reference horizontal lines at each alpha (the "ideal" FPR)
    for alpha in alpha_levels:
        fig_bars.add_hline(
            y=alpha, line_dash="dash",
            line_color=alpha_line_colors.get(alpha, "red"),
            annotation_text=f"ideal FPR for α={alpha:g}",
            annotation_position="right",
            annotation_font_color=alpha_line_colors.get(alpha, "red"),
            annotation_font_size=11,
        )

    # Y-axis range: enough headroom to see the highest CI
    all_ci_hi = [
        r["ci_hi"] for rows in fpr_table.values() for r in rows if r is not None
    ]
    y_max = max(max(all_ci_hi) if all_ci_hi else 0.15, max(alpha_levels)) * 1.25

    fig_bars.update_layout(
        title=dict(
            text=(
                f"<b>Type I error rate — {pipeline_label}</b><br>"
                f"<span style='font-size:12px;color:#555'>"
                f"For each metric, what fraction of {n_total_sims} null simulations "
                f"were declared significant at α=0.01, 0.05 and 0.10.<br>"
                f"<b>Bar height</b> = empirical FPR · "
                f"<b>error bars</b> = Wilson 95% CI · "
                f"<b>dashed lines</b> = ideal FPR (=α). "
                f"Calibrated ⇔ bar tops sit on the dashed lines."
                f"</span>"
            ),
            x=0.02, xanchor="left",
        ),
        xaxis=dict(title="<b>Metric</b>", tickfont=dict(size=12)),
        yaxis=dict(
            title="<b>Empirical false positive rate</b><br>"
                  "<span style='font-size:11px;color:#555'>"
                  "fraction of null sims with p &lt; α</span>",
            range=[0, y_max], tickformat=".0%",
        ),
        barmode="group", bargap=0.30, bargroupgap=0.05,
        plot_bgcolor="white",
        legend=dict(
            title="<b>Significance threshold</b>",
            yanchor="top", y=0.98, xanchor="right", x=0.99,
            bgcolor="rgba(255,255,255,0.9)", bordercolor="lightgray", borderwidth=1,
        ),
        height=560,
        margin=dict(t=130, b=60, l=80, r=140),
    )
    fig_bars.write_html(str(plots_dir / "fpr_summary_bars.html"))

    # =========================================================================
    # Plot 2 — p-value histograms with KS test annotation (2 columns grid)
    # =========================================================================
    n_cols = 2 if n_metrics > 2 else n_metrics
    n_rows = (n_metrics + n_cols - 1) // n_cols
    fig_hist = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=[f"<b>{m}</b>" for m in metric_clean],
        horizontal_spacing=0.10, vertical_spacing=0.18,
    )
    n_bins = 20
    for idx, (col, label) in enumerate(zip(p_metric_cols, metric_clean)):
        row_i = idx // n_cols + 1
        col_i = idx % n_cols + 1
        vals = p_values_df[col].dropna().values
        n_vals = len(vals)
        if n_vals == 0:
            continue
        ks_stat, ks_p = scipy_stats.kstest(vals, "uniform")
        n_sig_05 = int((vals < 0.05).sum())
        fpr_05 = n_sig_05 / n_vals
        ci_lo_05, ci_hi_05 = proportion_confint(n_sig_05, n_vals, alpha=0.05, method="wilson")
        verdict_05, color_05 = _verdict(fpr_05, 0.05, float(ci_lo_05), float(ci_hi_05))

        fig_hist.add_trace(
            go.Histogram(
                x=vals, nbinsx=n_bins,
                xbins=dict(start=0, end=1, size=1.0 / n_bins),
                marker_color="steelblue", opacity=0.85,
                showlegend=False,
            ),
            row=row_i, col=col_i,
        )
        expected_count = n_vals / n_bins
        fig_hist.add_hline(
            y=expected_count, line_dash="dash", line_color="red",
            row=row_i, col=col_i,
        )
        fig_hist.add_vline(
            x=0.05, line_dash="dot", line_color="orange",
            row=row_i, col=col_i,
        )
        # Subplot index for axis ref: 1, 2, 3, ... → "", "2", "3", ...
        sub_idx = idx + 1
        ax_suffix = "" if sub_idx == 1 else str(sub_idx)
        fig_hist.add_annotation(
            xref=f"x{ax_suffix} domain", yref=f"y{ax_suffix} domain",
            x=0.98, y=0.98, xanchor="right", yanchor="top",
            text=(
                f"FPR(α=0.05) = <b>{fpr_05:.1%}</b> "
                f"<span style='color:{color_05}'>[{verdict_05}]</span><br>"
                f"95% CI [{ci_lo_05:.1%}, {ci_hi_05:.1%}]<br>"
                f"KS vs U[0,1]: p = {ks_p:.3f}"
            ),
            showarrow=False, align="right",
            bgcolor="rgba(255,255,255,0.92)",
            bordercolor="lightgray", borderwidth=1,
            font=dict(size=10),
        )
        fig_hist.update_xaxes(title="p-value", range=[0, 1], row=row_i, col=col_i)
        fig_hist.update_yaxes(title="count" if col_i == 1 else None, row=row_i, col=col_i)

    fig_hist.update_layout(
        title=dict(
            text=(
                f"<b>p-value distribution under H0 — {pipeline_label}</b><br>"
                f"<span style='font-size:12px;color:#555'>"
                f"N = {n_total_sims} null simulations. Under H0 the histogram "
                f"should be flat (uniform). Excess mass near 0 ⇒ inflated FPR.<br>"
                f"<span style='color:#d62728'>━ ━</span> red dashed = expected "
                f"uniform count per bin · "
                f"<span style='color:orange'>┄ ┄</span> orange dotted = "
                f"α = 0.05 boundary"
                f"</span>"
            ),
            x=0.02, xanchor="left",
        ),
        height=320 * n_rows + 130, plot_bgcolor="white",
        margin=dict(t=150, b=60, l=70, r=40),
    )
    fig_hist.write_html(str(plots_dir / "pvalue_histogram.html"))

    # =========================================================================
    # Plot 3 — QQ plot with 95% pointwise envelope from Beta order statistics
    # =========================================================================
    fig_qq = go.Figure()
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    for idx, (col, label) in enumerate(zip(p_metric_cols, metric_clean)):
        vals = np.sort(p_values_df[col].dropna().values)
        n = len(vals)
        if n == 0:
            continue
        ranks = np.arange(1, n + 1)
        uniform_quantiles = (ranks - 0.5) / n
        # 95% pointwise envelope: U_(k) ~ Beta(k, n-k+1)
        ci_lo = scipy_stats.beta.ppf(0.025, ranks, n - ranks + 1)
        ci_hi = scipy_stats.beta.ppf(0.975, ranks, n - ranks + 1)
        color = palette[idx % len(palette)]
        # Envelope (only labelled for the first metric, then reused legend)
        fig_qq.add_trace(go.Scatter(
            x=np.concatenate([uniform_quantiles, uniform_quantiles[::-1]]),
            y=np.concatenate([ci_hi, ci_lo[::-1]]),
            fill="toself", fillcolor="rgba(150,150,150,0.18)",
            line=dict(color="rgba(0,0,0,0)"),
            name="95% envelope (under H0)",
            showlegend=(idx == 0), hoverinfo="skip",
        ))
        ks_stat, ks_p = scipy_stats.kstest(vals, "uniform")
        fig_qq.add_trace(go.Scatter(
            x=uniform_quantiles, y=vals,
            mode="markers",
            name=f"{label} (KS p={ks_p:.3f})",
            marker=dict(size=7, color=color, opacity=0.8),
        ))
    fig_qq.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines",
        line=dict(color="red", dash="dash"),
        name="y = x (perfect calibration)",
    ))
    fig_qq.update_layout(
        title=(
            f"<b>QQ-plot of p-values vs. U[0,1] — {pipeline_label}</b><br>"
            f"<span style='font-size:13px;color:gray'>"
            f"Each point = one of {n_total_sims} sims. "
            f"On the dashed line ⇒ calibrated. "
            f"Above the line ⇒ p-values too small ⇒ <b>inflated FPR</b>. "
            f"Below ⇒ conservative. Grey band = 95% region expected under H0."
            f"</span>"
        ),
        xaxis=dict(title="Theoretical uniform quantile", range=[0, 1]),
        yaxis=dict(title="Empirical p-value", range=[0, 1]),
        height=560, plot_bgcolor="white",
        legend=dict(yanchor="top", y=0.98, xanchor="left", x=0.02,
                    bgcolor="rgba(255,255,255,0.85)", bordercolor="lightgray"),
    )
    fig_qq.write_html(str(plots_dir / "pvalue_qqplot.html"))

    # =========================================================================
    # Plot 4 — FPR vs alpha curve, with good/bad zones and binomial 95% band
    # =========================================================================
    x_max = 0.30
    alpha_grid = np.linspace(0, x_max, 301)
    fig_fpr = go.Figure()

    # Zone shading: ABOVE diagonal = inflated (bad, light red);
    #               BELOW diagonal = conservative (safe, light green).
    fig_fpr.add_trace(go.Scatter(
        x=[0, x_max, x_max], y=[0, x_max, 1.0], fill="toself",
        fillcolor="rgba(214,39,40,0.07)", line=dict(color="rgba(0,0,0,0)"),
        hoverinfo="skip", showlegend=True,
        name="ABOVE diagonal: inflated FPR (BAD)",
    ))
    fig_fpr.add_trace(go.Scatter(
        x=[0, x_max, x_max, 0], y=[0, 0, x_max, 0], fill="toself",
        fillcolor="rgba(44,160,44,0.07)", line=dict(color="rgba(0,0,0,0)"),
        hoverinfo="skip", showlegend=True,
        name="BELOW diagonal: conservative (safe)",
    ))

    # 95% binomial confidence band around the diagonal: at each α, the
    # empirical FPR over N sims has Wilson 95% CI [lo(α), hi(α)]. If the
    # curve stays inside this band, deviations from y=x are consistent with
    # finite-sample noise — i.e., the test is calibrated.
    n = max((p_values_df[c].dropna().shape[0] for c in p_metric_cols), default=0)
    if n > 0:
        ci_lo_band, ci_hi_band = [], []
        for a in alpha_grid:
            k_expected = a * n  # expected count under perfect calibration
            # Use Wilson CI for the proportion = α (round to nearest int count for ci)
            k_int = int(round(k_expected))
            lo, hi = proportion_confint(k_int, n, alpha=0.05, method="wilson")
            ci_lo_band.append(float(lo))
            ci_hi_band.append(float(hi))
        fig_fpr.add_trace(go.Scatter(
            x=np.concatenate([alpha_grid, alpha_grid[::-1]]),
            y=np.concatenate([ci_hi_band, ci_lo_band[::-1]]),
            fill="toself", fillcolor="rgba(120,120,120,0.18)",
            line=dict(color="rgba(0,0,0,0)"),
            name=f"95% expected band (N={n} sims)",
            hoverinfo="skip",
        ))

    # Diagonal y = x (perfect calibration)
    fig_fpr.add_trace(go.Scatter(
        x=[0, x_max], y=[0, x_max], mode="lines",
        line=dict(color="black", dash="dash", width=2),
        name="ideal (FPR = α)",
    ))

    # Empirical curves per metric
    for idx, (col, label) in enumerate(zip(p_metric_cols, metric_clean)):
        vals = p_values_df[col].dropna().values
        if len(vals) == 0:
            continue
        fprs = [float((vals < a).mean()) for a in alpha_grid]
        fig_fpr.add_trace(go.Scatter(
            x=alpha_grid, y=fprs, mode="lines",
            name=label, line=dict(width=2.5, color=palette[idx % len(palette)]),
            hovertemplate="α=%{x:.3f}<br>FPR=%{y:.3f}<extra>" + label + "</extra>",
        ))

    # Vertical reference lines at the alpha levels of interest
    for a in alpha_levels:
        if a > x_max:
            continue
        fig_fpr.add_vline(
            x=a, line_dash="dot", line_color="gray", opacity=0.6,
            annotation_text=f"α={a:g}", annotation_position="top",
            annotation_font_size=11,
        )

    fig_fpr.update_layout(
        title=dict(
            text=(
                f"<b>Empirical FPR vs. nominal α — {pipeline_label}</b><br>"
                f"<span style='font-size:12px;color:#555'>"
                f"For each α (x-axis), the line shows the fraction of {n_total_sims} null "
                f"sims declared significant.<br>"
                f"<b>On the dashed diagonal</b> = calibrated · "
                f"<b style='color:#d62728'>above</b> = too many false positives (BAD) · "
                f"<b style='color:#2ca02c'>below</b> = fewer false positives than promised (safe).<br>"
                f"Grey band = 95% range expected by sampling noise alone — "
                f"if the line stays inside it, the test is calibrated."
                f"</span>"
            ),
            x=0.02, xanchor="left",
        ),
        xaxis=dict(title="<b>Nominal α</b> (significance threshold you choose)",
                   range=[0, x_max]),
        yaxis=dict(title="<b>Empirical FPR</b> (fraction of null sims with p < α)",
                   range=[0, max(0.4, x_max + 0.1)]),
        height=580, plot_bgcolor="white",
        margin=dict(t=160, b=70, l=80, r=40),
        legend=dict(yanchor="top", y=0.98, xanchor="left", x=0.02,
                    bgcolor="rgba(255,255,255,0.92)",
                    bordercolor="lightgray", borderwidth=1,
                    font=dict(size=11)),
    )
    fig_fpr.write_html(str(plots_dir / "fpr_vs_alpha.html"))

    print(f"  Calibration plots saved → {plots_dir}")
    print(f"    headline:    {plots_dir / 'fpr_summary_bars.html'}")
    print(f"    histogram:   {plots_dir / 'pvalue_histogram.html'}")
    print(f"    qq-plot:     {plots_dir / 'pvalue_qqplot.html'}")
    print(f"    fpr-vs-α:    {plots_dir / 'fpr_vs_alpha.html'}")
