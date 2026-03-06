"""
LMM module for connectivity statistics pipeline.

Runs per-connection linear mixed models on wSMI data and applies
FDR correction across connections.
"""

import numpy as np
import pandas as pd
import warnings
from typing import Dict, List, Optional, Tuple
from scipy import stats as sp_stats
from statsmodels.formula.api import mixedlm
from statsmodels.stats.multitest import multipletests


# =============================================================================
# PER-CONNECTION LMM
# =============================================================================

def run_lmm_per_connection(
    df_wide: pd.DataFrame,
    connection_ids: List[str],
    formula: str = "power ~ onoff + (1|subject)",
    predictor_of_interest: str = "onoff",
    method: str = "powell",
    maxiter: int = 500,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Run LMM for each connection independently.

    For each connection_id column, fits an LMM and extracts the t-statistic
    and p-value for the predictor of interest.

    Parameters
    ----------
    df_wide : pd.DataFrame
        Wide-format data from prepare_connectivity_for_lmm().
        Contains metadata columns + one column per connection_id.
    connection_ids : List[str]
        Column names representing connections.
    formula : str
        R-style LMM formula. "power" is the dependent variable placeholder.
    predictor_of_interest : str
        Which fixed effect to extract statistics for.
    method : str
        Optimization method for statsmodels MixedLM.
    maxiter : int
        Maximum iterations for optimization.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Results table with columns:
        connection_id, t_statistic, p_value, coefficient, std_error,
        n_observations, n_subjects, converged
    """
    # Parse formula to extract random effects
    fixed_formula, re_formula = _parse_random_effects(formula)

    results = []
    n_total = len(connection_ids)

    for i, conn_id in enumerate(connection_ids):
        if (i + 1) % 50 == 0 or i == 0:
            print(f"    LMM progress: {i + 1}/{n_total}")

        # Prepare data for this connection
        df_conn = df_wide[["subject", "task", "probe_number", conn_id]].copy()
        # Add predictor columns from the wide df
        for col in _extract_formula_variables(formula):
            if col in df_wide.columns and col != "power":
                df_conn[col] = df_wide[col].values
        df_conn = df_conn.rename(columns={conn_id: "power"})
        df_conn = df_conn.dropna(subset=["power"])

        if len(df_conn) < 10:
            results.append(_empty_result(conn_id, reason="insufficient_data"))
            continue

        n_subjects = df_conn["subject"].nunique()
        if n_subjects < 3:
            results.append(_empty_result(conn_id, reason="too_few_subjects"))
            continue

        # Fit LMM
        result = _fit_single_lmm(
            df_conn, fixed_formula, re_formula,
            predictor_of_interest, method, maxiter,
        )
        result["connection_id"] = conn_id
        result["n_observations"] = len(df_conn)
        result["n_subjects"] = n_subjects
        results.append(result)

    results_df = pd.DataFrame(results)
    n_converged = results_df["converged"].sum()
    print(f"    Done: {n_converged}/{n_total} converged")

    return results_df


def _fit_single_lmm(
    df: pd.DataFrame,
    fixed_formula: str,
    re_formula: Optional[str],
    predictor: str,
    method: str,
    maxiter: int,
) -> Dict:
    """
    Fit a single LMM and return statistics.

    Parameters
    ----------
    df : pd.DataFrame
        Data for one connection with 'power' as dependent variable.
    fixed_formula : str
        Fixed effects formula (e.g., "power ~ onoff").
    re_formula : Optional[str]
        Random effects formula (e.g., "1 + onoff" for random slopes).
    predictor : str
        Predictor to extract stats for.
    method : str
        Optimization method.
    maxiter : int
        Max iterations.

    Returns
    -------
    Dict
        Result dictionary with t_statistic, p_value, coefficient, etc.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")

        md = mixedlm(
            fixed_formula,
            data=df,
            groups=df["subject"],
            re_formula=("~" + re_formula) if re_formula else None,
        )

        fit = md.fit(method=method, maxiter=maxiter, reml=True)

        # Check convergence
        converged = fit.converged

        # Extract stats for predictor of interest
        if predictor in fit.tvalues.index:
            t_stat = float(fit.tvalues[predictor])
            p_val = float(fit.pvalues[predictor])
            coef = float(fit.fe_params[predictor])
            se = float(fit.bse[predictor])
        else:
            # Predictor not found in results
            return {
                "t_statistic": np.nan,
                "p_value": np.nan,
                "coefficient": np.nan,
                "std_error": np.nan,
                "converged": False,
            }

    return {
        "t_statistic": t_stat,
        "p_value": p_val,
        "coefficient": coef,
        "std_error": se,
        "converged": converged,
    }


def _empty_result(conn_id: str, reason: str = "unknown") -> Dict:
    """Create an empty result dict for a failed connection."""
    return {
        "connection_id": conn_id,
        "t_statistic": np.nan,
        "p_value": np.nan,
        "coefficient": np.nan,
        "std_error": np.nan,
        "n_observations": 0,
        "n_subjects": 0,
        "converged": False,
        "reason": reason,
    }


# =============================================================================
# FDR CORRECTION
# =============================================================================

def apply_fdr_correction(
    results_df: pd.DataFrame,
    alpha: float = 0.05,
    method: str = "fdr_bh",
) -> pd.DataFrame:
    """
    Apply FDR correction to LMM p-values.

    Parameters
    ----------
    results_df : pd.DataFrame
        Results from run_lmm_per_connection(). Must have 'p_value' column.
    alpha : float
        Significance threshold.
    method : str
        Correction method ('fdr_bh', 'fdr_by', 'bonferroni').

    Returns
    -------
    pd.DataFrame
        Input DataFrame with added columns:
        p_value_fdr, significant_fdr
    """
    df = results_df.copy()

    # Only correct converged models with valid p-values
    valid_mask = df["converged"] & df["p_value"].notna()
    valid_pvals = df.loc[valid_mask, "p_value"].values

    if len(valid_pvals) == 0:
        df["p_value_fdr"] = np.nan
        df["significant_fdr"] = False
        return df

    # Apply correction
    rejected, pvals_corrected, _, _ = multipletests(
        valid_pvals, alpha=alpha, method=method,
    )

    df["p_value_fdr"] = np.nan
    df["significant_fdr"] = False
    df.loc[valid_mask, "p_value_fdr"] = pvals_corrected
    df.loc[valid_mask, "significant_fdr"] = rejected

    n_sig = rejected.sum()
    n_total = len(valid_pvals)
    print(
        f"  FDR ({method}): {n_sig}/{n_total} significant "
        f"at alpha={alpha}"
    )

    return df


# =============================================================================
# HELPERS
# =============================================================================

def _parse_random_effects(formula: str) -> Tuple[str, Optional[str]]:
    """
    Parse random effects from formula.

    Parameters
    ----------
    formula : str
        Full formula (e.g., "power ~ onoff + (1 + onoff|subject)").

    Returns
    -------
    Tuple[str, Optional[str]]
        (fixed_formula, re_formula or None)
    """
    import re

    # Find random effects: (...)
    re_match = re.search(r"\(([^)]+)\)", formula)

    if re_match:
        re_content = re_match.group(1)
        # Remove random effects from formula
        fixed_formula = formula[:re_match.start()].strip()
        # Clean trailing +
        fixed_formula = re.sub(r"\+\s*$", "", fixed_formula).strip()

        # Parse: "1|subject" or "1 + onoff|subject"
        if "|" in re_content:
            re_part = re_content.split("|")[0].strip()
            if re_part == "1":
                return fixed_formula, None
            else:
                return fixed_formula, re_part
    return formula, None


def _extract_formula_variables(formula: str) -> List[str]:
    """
    Extract all variable names from formula.

    Parameters
    ----------
    formula : str
        R-style formula.

    Returns
    -------
    List[str]
        List of variable names (excluding 'power', 'subject', operators).
    """
    import re

    # Remove random effects
    clean = re.sub(r"\([^)]+\)", "", formula)
    # Remove response variable
    if "~" in clean:
        clean = clean.split("~")[1]
    # Split by operators
    tokens = re.split(r"[~+*:\s]+", clean)
    # Filter
    exclude = {"", "power", "subject", "1"}
    return [t.strip() for t in tokens if t.strip() not in exclude]
