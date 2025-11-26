#!/usr/bin/env python3
"""Raincloud plots for demographics and psychometric contrasts.

This script reads the CyberSART metadata file and the pairwise group
comparison results, and creates grids of raincloud plots (using
:mod:`ptitprince`) for each continuous contrast.

An asterisk is added on subplots where the corresponding test is
significant (``p_value < ALPHA``). Figures are saved under
``results/demographics``.

Run this script from the ``plots`` environment where ``ptitprince`` and
its dependencies are installed.
"""

from __future__ import annotations

from math import ceil
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ptitprince as pt
import seaborn as sns


# =============================================================================
# CONFIGURATION - Modify these variables as needed
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
METADATA_FILE: Path = PROJECT_ROOT / "metadata_CyberSART.xlsx"
STATS_FILE: Path = PROJECT_ROOT / "results" / "demographics" / "group_comparison_demographics_psychometrics.csv"
PLOTS_DIR: Path = PROJECT_ROOT / "results" / "demographics" / "raincloud_plots"

# Name of sheet in metadata workbook. Use ``None`` to read the first sheet.
METADATA_SHEET_NAME: Optional[str] = None

# Column holding group information (numeric codes, e.g. 1 and 2)
GROUP_COLUMN: str = "group"

# Mapping from numeric group codes to human-readable labels
GROUP_LABELS: Dict[int, str] = {
    1: "Controls",
    2: "Risk of depression",
}

# Significance threshold
ALPHA: float = 0.05

# Layout of the raincloud grid
N_COLS: int = 3
FIG_WIDTH_PER_COL: float = 4.0
FIG_HEIGHT_PER_ROW: float = 4.0

# Palette for rainclouds
PALETTE: Sequence[str] = sns.color_palette("Set2", 2)


# =============================================================================
# DATA LOADING
# =============================================================================

def load_metadata(file_path: Path, sheet_name: Optional[str]) -> pd.DataFrame:
    """Load the metadata Excel file.

    Parameters
    ----------
    file_path : Path
        Path to the metadata Excel file.
    sheet_name : str or None
        Sheet name to read. If ``None``, the first sheet is used.

    Returns
    -------
    pd.DataFrame
        Metadata table.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {file_path}")

    if sheet_name is None:
        df = pd.read_excel(file_path)
    else:
        try:
            df = pd.read_excel(file_path, sheet_name=sheet_name)
        except ValueError:
            df = pd.read_excel(file_path)

    return df


def load_stats(file_path: Path) -> pd.DataFrame:
    """Load the group comparison summary CSV.

    Parameters
    ----------
    file_path : Path
        Path to the CSV created by
        ``group_comparison_demographics_psychometrics.py``.

    Returns
    -------
    pd.DataFrame
        Summary statistics and test results for each variable.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Stats file not found: {file_path}")
    return pd.read_csv(file_path)


# =============================================================================
# PLOTTING HELPERS
# =============================================================================

def build_long_format(
    df_meta: pd.DataFrame,
    variable: str,
    group_col: str,
) -> pd.DataFrame:
    """Create a long-format DataFrame for raincloud plots.

    Parameters
    ----------
    df_meta : pd.DataFrame
        Metadata table.
    variable : str
        Name of the continuous variable to plot.
    group_col : str
        Column with group codes.

    Returns
    -------
    pd.DataFrame
        Table with columns ``group``, ``group_label``, and ``value``.
    """
    required_cols = [group_col, variable]
    missing_cols = [c for c in required_cols if c not in df_meta.columns]
    if missing_cols:
        raise ValueError(
            f"Missing columns in metadata for plotting {variable}: {missing_cols}"
        )

    sub = df_meta[required_cols].dropna(subset=required_cols).copy()

    # Ensure numeric group codes where possible
    sub[group_col] = pd.to_numeric(sub[group_col], errors="coerce")

    sub = sub.dropna(subset=[group_col])
    sub[group_col] = sub[group_col].astype(int)

    sub["group_label"] = sub[group_col].map(GROUP_LABELS)
    sub["group_label"] = sub["group_label"].fillna(sub[group_col].astype(str))
    sub = sub.rename(columns={variable: "value", group_col: "group"})

    return sub


def plot_raincloud_on_axis(
    ax: plt.Axes,
    data: pd.DataFrame,
    variable: str,
    p_value: float,
) -> None:
    """Draw a single raincloud plot on a given axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis on which to draw the raincloud.
    data : pd.DataFrame
        Long-format table with ``group_label`` and ``value``.
    variable : str
        Variable name (used for axis label and title).
    p_value : float
        P-value for the corresponding contrast.
    """
    if data.empty:
        ax.set_visible(False)
        return

    pt.RainCloud(
        x="group_label",
        y="value",
        data=data,
        ax=ax,
        palette=PALETTE,
        width_viol=0.7,
        width_box=0.2,
        point_size=2,
        move=0.2,
        orient="v",
    )

    ax.set_xlabel("")
    ax.set_ylabel(variable)

    title = f"{variable} (p={p_value:.3f})" if np.isfinite(p_value) else variable
    ax.set_title(title)

    # Add asterisk for significance
    if np.isfinite(p_value) and p_value < ALPHA:
        y_min, y_max = ax.get_ylim()
        y_range = y_max - y_min if y_max > y_min else 1.0

        new_y_max = y_max + 0.15 * y_range
        ax.set_ylim(y_min, new_y_max)

        line_y = y_max + 0.05 * y_range
        star_y = line_y + 0.03 * y_range

        xticks = ax.get_xticks()
        if len(xticks) >= 2:
            x1, x2 = float(xticks[0]), float(xticks[1])
        else:
            x1, x2 = 0.0, 1.0

        ax.plot([x1, x2], [line_y, line_y], color="black", linewidth=1.5)
        ax.text(
            (x1 + x2) / 2.0,
            star_y,
            "*",
            ha="center",
            va="bottom",
            fontsize=20,
            fontweight="bold",
        )


def plot_grid_for_domain(
    df_meta: pd.DataFrame,
    df_stats: pd.DataFrame,
    domain: str,
    output_dir: Path,
) -> Optional[Path]:
    """Create a grid of raincloud plots for a given domain.

    Parameters
    ----------
    df_meta : pd.DataFrame
        Metadata table.
    df_stats : pd.DataFrame
        Summary statistics table.
    domain : str
        Domain name (e.g. ``"demographics"`` or ``"psychometrics"``).
    output_dir : Path
        Directory where the PNG figure will be saved.

    Returns
    -------
    Path or None
        Path to the saved figure, or ``None`` if no variables were plotted.
    """
    df_dom = df_stats[(df_stats["domain"] == domain) & (df_stats["type"] == "continuous")]

    variables: List[str] = df_dom["variable"].tolist()
    if not variables:
        print(f"No continuous variables found for domain '{domain}'. Skipping.")
        return None

    n_vars = len(variables)
    n_cols = max(1, N_COLS)
    n_rows = ceil(n_vars / n_cols)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(FIG_WIDTH_PER_COL * n_cols, FIG_HEIGHT_PER_ROW * n_rows),
        squeeze=False,
    )

    for idx, variable in enumerate(variables):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]

        p_value = float(df_dom.loc[df_dom["variable"] == variable, "p_value"].iloc[0])

        try:
            data_long = build_long_format(df_meta, variable, GROUP_COLUMN)
        except ValueError as exc:
            ax.set_visible(False)
            print(f"Skipping {variable}: {exc}")
            continue

        plot_raincloud_on_axis(ax, data_long, variable, p_value)

    # Hide any unused axes
    for idx in range(n_vars, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].set_visible(False)

    fig.suptitle(f"Raincloud plots: {domain}", fontsize=16)
    plt.tight_layout(rect=(0, 0, 1, 0.97))

    output_dir.mkdir(parents=True, exist_ok=True)
    out_png = output_dir / f"raincloud_grid_{domain}.png"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved raincloud grid for domain '{domain}' to: {out_png}")
    return out_png


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    """Generate raincloud grids for all domains.

    Notes
    -----
    - Loads metadata and statistics tables.
    - For each domain (demographics, psychometrics), creates a grid of
      raincloud plots for all continuous variables.
    - Marks plots with an asterisk where the test is significant.
    - Saves PNG files under ``PLOTS_DIR``.
    """
    print("=== RAINCLOUD PLOTS FOR DEMOGRAPHICS & PSYCHOMETRICS ===")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Metadata file: {METADATA_FILE}")
    print(f"Stats file   : {STATS_FILE}")
    print(f"Output plots : {PLOTS_DIR}")

    df_meta = load_metadata(METADATA_FILE, METADATA_SHEET_NAME)
    df_stats = load_stats(STATS_FILE)

    # Ensure group column exists
    if GROUP_COLUMN not in df_meta.columns:
        raise ValueError(f"Group column '{GROUP_COLUMN}' not found in metadata.")

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    for domain in ["demographics", "psychometrics"]:
        plot_grid_for_domain(df_meta, df_stats, domain, PLOTS_DIR)

    print("\nAll raincloud grids generated.")


if __name__ == "__main__":
    main()
