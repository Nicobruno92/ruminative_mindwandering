#!/usr/bin/env python
"""
LOSO: AUC vs median position — exploratory analysis.

Two figures relating LOSO decodability (per-subject AUC) to within-subject
median ratings on the 5 phenomenology dimensions (onoff, valence, selfother,
time, confidence):

  Fig A (dimension_auc_vs_median_variability):
      One point per dimension. x = SD across subjects of each subject's
      within-subject median rating. y = mean LOSO AUC for that dimension.

  Fig B (auc_vs_median_distance_from_50_faceted):
      One panel per dimension, one point per subject. x = |subject's median
      rating - 50| (distance from scale midpoint). y = subject's LOSO AUC.

Status: EXPLORATORY (n=5 dimensions for Fig A; no confirmatory claims).

Usage (from project root):
    /path/to/miniforge3/envs/ML/bin/python \
        mw_classification_pipeline/scripts/plot_auc_vs_median_position.py
"""

# =============================================================================
# Imports
# =============================================================================

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml
from scipy import stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.plotting_utils import COLORS  # noqa: E402


# =============================================================================
# Configuration
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOSO_RESULTS_ROOT = (
    PROJECT_ROOT / "mw_classification_pipeline" / "results" / "MW_Classification" / "LOSO"
)
PROBE_DATA_PATH = (
    PROJECT_ROOT / "results" / "Behavior" / "probe_data" / "probe_level_aggregated_data.csv"
)
OUTPUT_DIR = LOSO_RESULTS_ROOT / "median_position_analysis"

SCALE_MIDPOINT = 50.0
CHANCE_AUC = 0.5
MIN_POINTS_FOR_FIT = 3

# (contrast directory name, display label, marker color) — color scheme
# matches scripts/generate_combined_classification_figure.py
DIMENSIONS: List[Dict[str, str]] = [
    {"contrast": "ON_vs_OFF_within_median", "label": "On/Off-Task", "color": "#DE237B"},
    {"contrast": "valence_within_median", "label": "Valence", "color": "#7B4FBA"},
    {"contrast": "selfother_within_median", "label": "Self/Other", "color": "#E67E22"},
    {"contrast": "confidence_within_median", "label": "Confidence", "color": "#27AE60"},
    {"contrast": "time_within_median", "label": "Time", "color": "#2980B9"},
]


# =============================================================================
# Core data-shaping functions
# =============================================================================

def compute_pearson_fit(x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """
    Pearson correlation and OLS line fit between two 1-D arrays.

    Parameters
    ----------
    x, y : np.ndarray
        Equal-length arrays of paired observations.

    Returns
    -------
    Dict[str, float]
        Keys 'r', 'p', 'slope', 'intercept', 'n'. If fewer than
        `MIN_POINTS_FOR_FIT` finite pairs are available, 'r', 'p', 'slope'
        and 'intercept' are NaN.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    n_valid = int(valid.sum())

    if n_valid < MIN_POINTS_FOR_FIT:
        return {"r": np.nan, "p": np.nan, "slope": np.nan, "intercept": np.nan, "n": n_valid}

    r_val, p_val = stats.pearsonr(x[valid], y[valid])
    slope, intercept, _, _, _ = stats.linregress(x[valid], y[valid])
    return {
        "r": float(r_val),
        "p": float(p_val),
        "slope": float(slope),
        "intercept": float(intercept),
        "n": n_valid,
    }


def compute_subject_medians(
    probe_df: pd.DataFrame, label_source: str, subjects_final: List[str]
) -> pd.DataFrame:
    """
    Per-subject median rating for one dimension, restricted to a subject set.

    Parameters
    ----------
    probe_df : pd.DataFrame
        Probe-level data with a 'subject' column (zero-padded strings) and
        a `label_source` column.
    label_source : str
        Name of the dimension column (e.g. 'onoff', 'valence').
    subjects_final : List[str]
        Zero-padded subject IDs to keep — the LOSO subject set for this
        dimension.

    Returns
    -------
    pd.DataFrame
        Columns: 'subject', 'subject_median'.
    """
    medians = (
        probe_df.groupby("subject")[label_source]
        .median()
        .reset_index()
        .rename(columns={label_source: "subject_median"})
    )
    return medians[medians["subject"].isin(subjects_final)].reset_index(drop=True)


def build_fig_a_row(
    dimension_label: str, subject_medians: pd.DataFrame, subject_aucs: pd.DataFrame
) -> Dict[str, object]:
    """
    One Fig A summary row: across-subject SD of medians vs mean LOSO AUC.

    Parameters
    ----------
    dimension_label : str
        Display name of the dimension (e.g. 'On/Off-Task').
    subject_medians : pd.DataFrame
        Output of `compute_subject_medians` — columns 'subject', 'subject_median'.
    subject_aucs : pd.DataFrame
        Output of `load_subject_aucs` — columns 'subject', 'auc'.

    Returns
    -------
    Dict[str, object]
        Keys: 'dimension', 'median_sd', 'mean_auc', 'n_subjects'.
    """
    return {
        "dimension": dimension_label,
        "median_sd": float(subject_medians["subject_median"].std()),
        "mean_auc": float(subject_aucs["auc"].mean()),
        "n_subjects": int(len(subject_aucs)),
    }


def build_fig_b_rows(
    dimension_label: str, subject_medians: pd.DataFrame, subject_aucs: pd.DataFrame
) -> pd.DataFrame:
    """
    Per-subject Fig B rows: distance of the subject's median from the scale
    midpoint vs that subject's LOSO AUC.

    Parameters
    ----------
    dimension_label : str
        Display name of the dimension (e.g. 'On/Off-Task').
    subject_medians : pd.DataFrame
        Output of `compute_subject_medians`.
    subject_aucs : pd.DataFrame
        Output of `load_subject_aucs`.

    Returns
    -------
    pd.DataFrame
        Columns: 'dimension', 'subject', 'subject_median', 'dist_from_50', 'auc'.
    """
    merged = pd.merge(subject_medians, subject_aucs, on="subject", how="inner")
    merged["dist_from_50"] = (merged["subject_median"] - SCALE_MIDPOINT).abs()
    merged["dimension"] = dimension_label
    return merged[["dimension", "subject", "subject_median", "dist_from_50", "auc"]]


# =============================================================================
# Data loaders
# =============================================================================

def load_dimension_metadata(rf_dir: Path) -> Tuple[List[str], str]:
    """
    Read the LOSO `used_config.yaml` for one dimension.

    Parameters
    ----------
    rf_dir : Path
        Directory containing `used_config.yaml` and the subject-metrics CSV
        (e.g. `.../LOSO/ON_vs_OFF_within_median/all/rf`).

    Returns
    -------
    Tuple[List[str], str]
        (subjects_final as zero-padded strings, label_source column name).
    """
    config = yaml.safe_load((rf_dir / "used_config.yaml").read_text())
    contrast = config["contrast"]
    subjects_final = [str(s).zfill(2) for s in config["_data_provenance"]["subjects_final"]]
    label_source = config["label_contrasts"][contrast]["label_source"]
    return subjects_final, label_source


def load_subject_aucs(rf_dir: Path, subjects_final: List[str]) -> pd.DataFrame:
    """
    Load per-subject LOSO AUC, restricted to `subjects_final`.

    Parameters
    ----------
    rf_dir : Path
        Directory containing `rf_loso_100runs_loso_subject_metrics.csv`.
    subjects_final : List[str]
        Zero-padded subject IDs to keep.

    Returns
    -------
    pd.DataFrame
        Columns: 'subject' (zero-padded string), 'auc'.
    """
    df = pd.read_csv(rf_dir / "rf_loso_100runs_loso_subject_metrics.csv", dtype={"subject": str})
    df["subject"] = df["subject"].str.zfill(2)
    df = df[df["subject"].isin(subjects_final)][["subject", "auc"]]
    return df.reset_index(drop=True)


# =============================================================================
# Fig A: dimension-level scatter
# =============================================================================

def plot_fig_a(fig_a_df: pd.DataFrame, output_dir: Path) -> None:
    """
    Scatter of dimension-level median variability vs mean LOSO AUC.

    One point per dimension (n=5 in the full run). Exploratory — no
    correlation statistic is computed given the small n.

    Parameters
    ----------
    fig_a_df : pd.DataFrame
        Columns: 'dimension', 'median_sd', 'mean_auc', 'n_subjects'.
    output_dir : Path
        Directory to write `dimension_auc_vs_median_variability.{png,pdf,html}`.
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=fig_a_df["median_sd"],
        y=fig_a_df["mean_auc"],
        mode="markers+text",
        text=fig_a_df["dimension"],
        textposition="top center",
        textfont=dict(size=11),
        marker=dict(color=COLORS[0], size=14, opacity=0.85, line=dict(color="white", width=1)),
        hovertemplate="%{text}<br>Median SD: %{x:.2f}<br>Mean AUC: %{y:.3f}<extra></extra>",
    ))

    fig.add_hline(
        y=CHANCE_AUC, line_dash="dash", line_color="gray", opacity=0.5,
        annotation_text="Chance (0.5)", annotation_position="bottom right",
    )

    fig.update_layout(
        template="plotly_white",
        title=dict(
            text=(
                "<b>LOSO: Dimension Decodability vs Across-Subject Median Variability</b>"
                f"<br><sup>Exploratory — n = {len(fig_a_df)} dimensions, "
                "descriptive only (no correlation statistic)</sup>"
            ),
            font=dict(size=15),
        ),
        xaxis_title="SD across subjects of within-subject median rating",
        yaxis_title="Mean LOSO AUC",
        width=750,
        height=600,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "dimension_auc_vs_median_variability"
    fig.write_image(f"{out_path}.png", scale=2)
    fig.write_image(f"{out_path}.pdf")
    fig.write_html(f"{out_path}.html")


# =============================================================================
# Fig B: subject-level faceted scatter
# =============================================================================

def _subplot_domain_ref(index: int, axis: str) -> str:
    """Return the plotly domain ref ('x domain', 'x2 domain', ...) for subplot `index` (0-based)."""
    suffix = "" if index == 0 else str(index + 1)
    return f"{axis}{suffix} domain"


def plot_fig_b(
    fig_b_df: pd.DataFrame,
    dimension_colors: Dict[str, str],
    dimension_order: List[str],
    output_dir: Path,
) -> None:
    """
    Faceted scatter of |subject median - 50| vs subject LOSO AUC, one panel
    per dimension, in a 2x3 grid (5 panels + 1 empty cell).

    Parameters
    ----------
    fig_b_df : pd.DataFrame
        Columns: 'dimension', 'subject', 'subject_median', 'dist_from_50', 'auc'.
    dimension_colors : Dict[str, str]
        Maps dimension display label to a hex marker color.
    dimension_order : List[str]
        Display labels in the order panels should appear (length 5).
    output_dir : Path
        Directory to write `auc_vs_median_distance_from_50_faceted.{png,pdf,html}`.
    """
    fig = make_subplots(rows=2, cols=3, subplot_titles=dimension_order)

    y_min = max(0.0, float(fig_b_df["auc"].min()) - 0.05)
    y_max = min(1.0, float(fig_b_df["auc"].max()) + 0.05)

    for i, dimension in enumerate(dimension_order):
        row = i // 3 + 1
        col = i % 3 + 1
        sub = fig_b_df[fig_b_df["dimension"] == dimension]
        x = sub["dist_from_50"].to_numpy(dtype=float)
        y = sub["auc"].to_numpy(dtype=float)
        fit = compute_pearson_fit(x, y)

        fig.add_trace(go.Scatter(
            x=x, y=y, mode="markers+text",
            text=sub["subject"], textposition="top center", textfont=dict(size=8),
            marker=dict(
                color=dimension_colors[dimension], size=8, opacity=0.85,
                line=dict(color="white", width=1),
            ),
            showlegend=False,
            hovertemplate="Subject: %{text}<br>|median-50|: %{x:.1f}<br>AUC: %{y:.3f}<extra></extra>",
        ), row=row, col=col)

        if np.isfinite(fit["slope"]):
            x_line = np.array([x.min(), x.max()])
            y_line = fit["slope"] * x_line + fit["intercept"]
            fig.add_trace(go.Scatter(
                x=x_line, y=y_line, mode="lines",
                line=dict(color="black", width=2, dash="dash"), showlegend=False,
            ), row=row, col=col)

        fig.add_hline(
            y=CHANCE_AUC, line_dash="dot", line_color="gray", opacity=0.5, row=row, col=col
        )

        annotation_text = (
            f"r = {fit['r']:.2f}, p = {fit['p']:.3f}" if np.isfinite(fit["r"]) else "n < 3"
        )
        fig.add_annotation(
            text=annotation_text,
            xref=_subplot_domain_ref(i, "x"), yref=_subplot_domain_ref(i, "y"),
            x=0.98, y=0.02, xanchor="right", yanchor="bottom",
            showarrow=False, font=dict(size=10),
        )

    fig.update_yaxes(range=[y_min, y_max], title_text="LOSO AUC")
    fig.update_xaxes(title_text="|subject median − 50|")

    fig.update_layout(
        template="plotly_white",
        title=dict(
            text="<b>LOSO: Subject AUC vs Distance of Median from Scale Midpoint (50)</b>",
            font=dict(size=15),
        ),
        height=700,
        width=1100,
        showlegend=False,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "auc_vs_median_distance_from_50_faceted"
    fig.write_image(f"{out_path}.png", scale=2)
    fig.write_image(f"{out_path}.pdf")
    fig.write_html(f"{out_path}.html")


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    """Build Fig A and Fig B and write data CSVs + provenance to OUTPUT_DIR."""
    probe_df = pd.read_csv(PROBE_DATA_PATH, dtype={"subject": str})
    probe_df["subject"] = probe_df["subject"].astype(str).str.zfill(2)

    fig_a_rows: List[Dict[str, object]] = []
    fig_b_frames: List[pd.DataFrame] = []
    dimension_colors: Dict[str, str] = {}
    dimension_order: List[str] = []
    provenance: Dict[str, Dict[str, object]] = {}

    for dim in DIMENSIONS:
        rf_dir = LOSO_RESULTS_ROOT / dim["contrast"] / "all" / "rf"
        subjects_final, label_source = load_dimension_metadata(rf_dir)
        subject_aucs = load_subject_aucs(rf_dir, subjects_final)
        subject_medians = compute_subject_medians(probe_df, label_source, subjects_final)

        fig_a_rows.append(build_fig_a_row(dim["label"], subject_medians, subject_aucs))
        fig_b_frames.append(build_fig_b_rows(dim["label"], subject_medians, subject_aucs))

        dimension_colors[dim["label"]] = dim["color"]
        dimension_order.append(dim["label"])
        provenance[dim["label"]] = {
            "contrast": dim["contrast"],
            "label_source": label_source,
            "n_subjects": int(len(subject_aucs)),
            "subjects_final": subjects_final,
        }

    fig_a_df = pd.DataFrame(fig_a_rows)
    fig_b_df = pd.concat(fig_b_frames, ignore_index=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig_a_df.to_csv(OUTPUT_DIR / "dimension_auc_vs_median_variability_data.csv", index=False)
    fig_b_df.to_csv(OUTPUT_DIR / "auc_vs_median_distance_from_50_data.csv", index=False)

    plot_fig_a(fig_a_df, OUTPUT_DIR)
    plot_fig_b(fig_b_df, dimension_colors, dimension_order, OUTPUT_DIR)

    used_config = {
        "analysis": "loso_median_position_vs_decodability",
        "status": "exploratory",
        "data_sources": {
            "probe_data": str(PROBE_DATA_PATH.relative_to(PROJECT_ROOT)),
            "loso_results_root": str(LOSO_RESULTS_ROOT.relative_to(PROJECT_ROOT)),
        },
        "scale_midpoint": SCALE_MIDPOINT,
        "dimensions": provenance,
    }
    (OUTPUT_DIR / "used_config.yaml").write_text(yaml.safe_dump(used_config, sort_keys=False))

    print(f"Saved Fig A + Fig B + data CSVs + used_config.yaml -> {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
