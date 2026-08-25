#!/usr/bin/env python3
"""Pairwise group comparisons for demographics and psychometric measures.

This script reads the CyberSART metadata file, performs pairwise comparisons
between the two groups on demographic and psychometric variables, and saves a
summary table into ``results/demographics``.

The script is designed to run as-is by simply executing it. All configuration
parameters are defined in the CONFIGURATION section below.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats


# =============================================================================
# CONFIGURATION - Modify these variables as needed
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
METADATA_FILE: Path = PROJECT_ROOT / "metadata_CyberSART.xlsx"
RESULTS_DIR: Path = PROJECT_ROOT / "results" / "demographics"
# If the metadata workbook has a specific sheet name, set it here.
# Use ``None`` to read the first sheet.
METADATA_SHEET_NAME: Optional[str] = None
GROUP_COLUMN: str = "group"

# Variables to compare (only those present in the file will be used)
DEMOGRAPHIC_VARS: List[str] = [
    "age",
    "gender",
    "year_birth",
]

PSYCHOMETRIC_VARS: List[str] = [
    "bdi",
    "a_rsq",
    "rrs_d",
    "rrs_r",
    "rrs_b",
    "rrs_tot",
    "mwq",
    "sris",
    "fne",
    "self-esteem",
    "ctq_ae",
    "ctq_ap",
    "ctq_as",
    "ctq_ne",
    "ctq_np",
    "ctq_tot",
    "qf_3",
    "qd_4",
    "qf_7b",
]

OUTPUT_CSV_NAME: str = "group_comparison_demographics_psychometrics.csv"
FEMALE_CATEGORY: str = "F"


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def load_metadata(file_path: Path, sheet_name: Optional[str]) -> pd.DataFrame:
    """Load the metadata Excel file.

    Parameters
    ----------
    file_path : Path
        Path to the ``metadata_CyberSART.xlsx`` file.
    sheet_name : str or None
        Name of the worksheet to read. If ``None``, the first sheet is used.

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
            # Fallback: sheet not found, use first sheet
            df = pd.read_excel(file_path)

    return df


def get_groups(df: pd.DataFrame, group_col: str) -> Tuple[int, int]:
    """Return the two distinct group codes used for comparisons.

    Parameters
    ----------
    df : pd.DataFrame
        Metadata DataFrame.
    group_col : str
        Name of the group column.

    Returns
    -------
    tuple of int
        The two group codes, sorted.

    Raises
    ------
    ValueError
        If fewer or more than two non-missing group levels are present.
    """
    if group_col not in df.columns:
        raise ValueError(f"Group column '{group_col}' not found in metadata.")

    groups = (
        df[group_col]
        .dropna()
        .unique()
        .tolist()
    )

    if len(groups) != 2:
        raise ValueError(
            f"Expected exactly 2 groups in column '{group_col}', found {len(groups)}: {groups}"
        )

    groups_sorted = sorted(groups)
    return int(groups_sorted[0]), int(groups_sorted[1])


def welch_df(x: np.ndarray, y: np.ndarray) -> float:
    """Compute the degrees of freedom for Welch's t-test.

    Parameters
    ----------
    x, y : np.ndarray
        1D arrays containing the two groups.

    Returns
    -------
    float
        Approximate Welch degrees of freedom.
    """
    n1 = x.size
    n2 = y.size
    v1 = np.var(x, ddof=1)
    v2 = np.var(y, ddof=1)

    if n1 < 2 or n2 < 2 or v1 == 0 or v2 == 0:
        return np.nan

    num = (v1 / n1 + v2 / n2) ** 2
    den = (v1 ** 2) / (n1 ** 2 * (n1 - 1)) + (v2 ** 2) / (n2 ** 2 * (n2 - 1))
    return float(num / den) if den > 0 else np.nan


def cohen_d(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Cohen's d for two independent groups.

    Parameters
    ----------
    x, y : np.ndarray
        Arrays containing data for each group.

    Returns
    -------
    float
        Cohen's d effect size. ``np.nan`` if not defined.
    """
    n1 = x.size
    n2 = y.size
    if n1 < 2 or n2 < 2:
        return np.nan

    mean1 = float(np.mean(x))
    mean2 = float(np.mean(y))
    v1 = np.var(x, ddof=1)
    v2 = np.var(y, ddof=1)

    pooled_var = ((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2)
    if pooled_var <= 0:
        return np.nan

    return (mean1 - mean2) / np.sqrt(pooled_var)


def compare_continuous_variable(
    df: pd.DataFrame,
    var: str,
    group_col: str,
    group1: int,
    group2: int,
    domain: str,
) -> Dict[str, float]:
    """Run a Welch t-test for a continuous variable between two groups.

    Parameters
    ----------
    df : pd.DataFrame
        Metadata DataFrame.
    var : str
        Name of the continuous variable to compare.
    group_col : str
        Name of the group column.
    group1, group2 : int
        Codes of the two groups.
    domain : str
        Variable domain (e.g. ``"demographics"`` or ``"psychometrics"``).

    Returns
    -------
    dict
        Summary statistics and test results for the variable.
    """
    sub = df[[group_col, var]].dropna(subset=[group_col, var])

    x = sub.loc[sub[group_col] == group1, var].astype(float).to_numpy()
    y = sub.loc[sub[group_col] == group2, var].astype(float).to_numpy()

    result: Dict[str, float] = {
        "variable": var,
        "domain": domain,
        "type": "continuous",
        "group1": group1,
        "group2": group2,
        "n_group1": int(x.size),
        "n_group2": int(y.size),
        "mean_group1": float(np.mean(x)) if x.size > 0 else np.nan,
        "mean_group2": float(np.mean(y)) if y.size > 0 else np.nan,
        "sd_group1": float(np.std(x, ddof=1)) if x.size > 1 else np.nan,
        "sd_group2": float(np.std(y, ddof=1)) if y.size > 1 else np.nan,
        "test": "welch_t",
        "statistic": np.nan,
        "df": np.nan,
        "p_value": np.nan,
        "effect_size_d": np.nan,
        "note": "",
    }

    # Require at least 2 observations per group for a reliable test
    if x.size < 2 or y.size < 2:
        result["note"] = "Not enough observations in one or both groups for t-test."
        return result

    t_stat, p_val = stats.ttest_ind(x, y, equal_var=False, nan_policy="omit")
    df_welch = welch_df(x, y)
    d = cohen_d(x, y)

    result["statistic"] = float(t_stat)
    result["df"] = df_welch
    result["p_value"] = float(p_val)
    result["effect_size_d"] = float(d) if not np.isnan(d) else np.nan
    return result


def compare_categorical_variable(
    df: pd.DataFrame,
    var: str,
    group_col: str,
    group1: int,
    group2: int,
    domain: str,
) -> Dict[str, float]:
    """Run a chi-square test of independence for a categorical variable.

    Parameters
    ----------
    df : pd.DataFrame
        Metadata DataFrame.
    var : str
        Name of the categorical variable (e.g. ``"gender"``).
    group_col : str
        Name of the group column.
    domain : str
        Variable domain.

    Returns
    -------
    dict
        Summary statistics and chi-square results for the variable.
    """
    sub = df[[group_col, var]].dropna(subset=[group_col, var])

    contingency = pd.crosstab(sub[group_col], sub[var])

    mask_g1 = sub[group_col] == group1
    mask_g2 = sub[group_col] == group2

    n_g1 = int(mask_g1.sum())
    n_g2 = int(mask_g2.sum())

    # Default proportions (only meaningful for gender)
    prop_female_g1 = np.nan
    prop_female_g2 = np.nan

    if var == "gender" and n_g1 > 0 and n_g2 > 0:
        gender_series = sub[var].astype(str).str.upper()
        is_female = gender_series == FEMALE_CATEGORY.upper()
        prop_female_g1 = float(is_female[mask_g1].mean()) if mask_g1.any() else np.nan
        prop_female_g2 = float(is_female[mask_g2].mean()) if mask_g2.any() else np.nan
    result: Dict[str, float] = {
        "variable": var,
        "domain": domain,
        "type": "categorical",
        "group1": group1,
        "group2": group2,
        "n_group1": n_g1,
        "n_group2": n_g2,
        "mean_group1": np.nan,
        "mean_group2": np.nan,
        "sd_group1": np.nan,
        "sd_group2": np.nan,
        "test": "chi2_independence",
        "statistic": np.nan,
        "df": np.nan,
        "p_value": np.nan,
        "effect_size_d": np.nan,
        "prop_female_group1": prop_female_g1,
        "prop_female_group2": prop_female_g2,
        "note": "Categorical variable; counts are summarized in the contingency table.",
    }

    if contingency.size == 0 or contingency.shape[0] < 2 or contingency.shape[1] < 2:
        result["note"] = (
            "Not enough distinct categories or groups to compute chi-square test."
        )
        return result

    chi2, p_val, dof, _ = stats.chi2_contingency(contingency)
    result["statistic"] = float(chi2)
    result["df"] = int(dof)
    result["p_value"] = float(p_val)
    return result


def run_group_comparisons() -> pd.DataFrame:
    """Run all group comparisons and return the summary table.

    Returns
    -------
    pd.DataFrame
        One row per variable with summary statistics and test results.
    """
    df_meta = load_metadata(METADATA_FILE, METADATA_SHEET_NAME)

    # Keep only subjects with a defined group
    df_meta = df_meta.copy()
    df_meta = df_meta[~df_meta[GROUP_COLUMN].isna()]

    group1, group2 = get_groups(df_meta, GROUP_COLUMN)

    # Restrict to variables that are present in the DataFrame
    existing_demographics = [v for v in DEMOGRAPHIC_VARS if v in df_meta.columns]
    existing_psychometrics = [v for v in PSYCHOMETRIC_VARS if v in df_meta.columns]

    results: List[Dict[str, float]] = []

    # Demographic variables
    for var in existing_demographics:
        if df_meta[var].dtype == "object" or df_meta[var].dtype.name == "category":
            res = compare_categorical_variable(
                df_meta,
                var,
                GROUP_COLUMN,
                group1,
                group2,
                "demographics",
            )
        else:
            res = compare_continuous_variable(
                df_meta,
                var,
                GROUP_COLUMN,
                group1,
                group2,
                "demographics",
            )
        results.append(res)

    # Psychometric variables (assumed continuous)
    for var in existing_psychometrics:
        res = compare_continuous_variable(
            df_meta,
            var,
            GROUP_COLUMN,
            group1,
            group2,
            "psychometrics",
        )
        results.append(res)

    df_results = pd.DataFrame(results)

    # Order columns for readability
    column_order: Sequence[str] = [
        "variable",
        "domain",
        "type",
        "group1",
        "group2",
        "n_group1",
        "n_group2",
        "mean_group1",
        "sd_group1",
        "mean_group2",
        "sd_group2",
        "test",
        "statistic",
        "df",
        "p_value",
        "effect_size_d",
        "prop_female_group1",
        "prop_female_group2",
        "note",
    ]
    df_results = df_results[column_order]
    return df_results


def main() -> None:
    """Main execution function.

    Notes
    -----
    - Loads the metadata file.
    - Runs all pairwise comparisons between groups.
    - Saves the results to ``results/demographics``.
    """
    print("=== GROUP COMPARISON: DEMOGRAPHICS & PSYCHOMETRICS ===")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Metadata file: {METADATA_FILE}")

    results_dir = RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)
    output_csv = results_dir / OUTPUT_CSV_NAME

    df_results = run_group_comparisons()

    df_results.to_csv(output_csv, index=False)
    print(f"\nResults saved to: {output_csv}")
    print("\nFirst rows of the results table:\n")
    print(df_results.head().to_string(index=False))


if __name__ == "__main__":
    main()
