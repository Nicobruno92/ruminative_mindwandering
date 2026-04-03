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
# CONSTANTS
# =============================================================================

PAIR_SEPARATOR = "-"

# Minimum predictor standard deviation required to attempt LMM fitting.
# Below this threshold the predictor is effectively constant, making the
# design matrix singular.  Connections failing this check are skipped and
# contribute 0 (no edge) to the NBS null graph.
_MIN_PREDICTOR_STD = 1e-6


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
    # Parse formula once — shared across all connections
    fixed_formula, re_formula = _parse_random_effects(formula)
    formula_vars = _extract_formula_variables(formula)
    # Base metadata columns to carry into every df_conn
    meta_cols = ["subject", "task", "probe_number"]
    predictor_cols = [col for col in formula_vars if col in df_wide.columns]

    results = []
    n_total = len(connection_ids)

    for i, conn_id in enumerate(connection_ids):
        if (i + 1) % 50 == 0 or i == 0:
            print(f"    LMM progress: {i + 1}/{n_total}")

        # Build a tidy dataframe for this connection:
        # metadata + connectivity value + all formula predictors
        cols_needed = meta_cols + [conn_id] + predictor_cols
        df_conn = (
            df_wide[cols_needed]
            .rename(columns={conn_id: "power"})
            .dropna(subset=["power"])
            .reset_index(drop=True)  # clean integer index after dropna
        )

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


def _predictor_has_variability(df_conn: pd.DataFrame, predictor: str) -> bool:
    """
    Return True if the predictor has sufficient variance for LMM fitting.

    Parameters
    ----------
    df_conn : pd.DataFrame
        Data for a single connection (already dropna'd).
    predictor : str
        Column name of the predictor to validate.

    Returns
    -------
    bool
        False if std ≤ _MIN_PREDICTOR_STD (constant predictor → singular design).
    """
    return df_conn[predictor].std() > _MIN_PREDICTOR_STD


def _run_nbs_permutation(
    perm_seed: int,
    df_wide: pd.DataFrame,
    connection_ids: List[str],
    predictor_of_interest: str,
    formula_vars: List[str],
    fixed_formula: str,
    re_formula: Optional[str],
    nodes: set,
    primary_threshold: float,
    method: str,
    maxiter: int,
) -> Tuple[int, int]:
    """
    Execute one NBS permutation and return the largest null component size.

    Permutation scheme — within-subject restricted permutation
    ----------------------------------------------------------
    For each subject independently, the values of `predictor_of_interest`
    are shuffled across that subject's probe observations.  All other
    covariates remain at their exact observed values.

    This is the correct scheme for a repeated-measures LMM with nuisance
    covariates (Nichols & Holmes, 2002, HBM), because:

    - Between-subject variance structure (LMM random intercepts) is preserved.
    - Every covariate other than the predictor under test retains its
      exact observed value for each probe.  Nuisance regressors are *not*
      permuted, so the null model is a conditional test of
      predictor_of_interest given all other covariates.
    - Only the within-subject relationship between predictor_of_interest
      and neural connectivity is broken, which is the null hypothesis.

    The maximum connected-component size of the null t-statistic graph is
    returned (Zalesky et al., 2010 NeuroImage).

    Parameters
    ----------
    perm_seed : int
        RNG seed for this permutation (derived from the master seed).
    df_wide : pd.DataFrame
        Wide-format data (one row per probe, one column per connection).
    connection_ids : List[str]
        Column names of the connectivity values to test.
    predictor_of_interest : str
        The only column that is permuted; all others stay fixed.
    formula_vars : List[str]
        Pre-extracted formula variable names (avoids repeated parsing in workers).
    fixed_formula : str
        Pre-parsed fixed-effects formula.
    re_formula : Optional[str]
        Pre-parsed random-effects formula.
    nodes : set
        Channel/ROI node names shared across all permutations.
    primary_threshold : float
        |t| cluster-forming threshold.
    method : str
        LMM optimisation method.
    maxiter : int
        Maximum LMM iterations.

    Returns
    -------
    Tuple[int, int]
        (max_pos_component_size, max_neg_component_size).
        Each is 0 if no supra-threshold edges exist for that direction.
        Returning separate values per direction lets ``apply_nbs_correction``
        maintain direction-specific null distributions, avoiding inflation
        of the positive-effect null by negative-effect permutation components
        and vice versa (see Zalesky et al., 2010, NeuroImage, Supplement).
    """
    perm_rng = np.random.RandomState(perm_seed)

    # ── Permute ONLY the predictor of interest, within each subject ───────
    # All other covariates keep their observed values row-for-row.
    df_perm = df_wide.copy()
    for subj in df_perm["subject"].unique():
        mask = df_perm["subject"] == subj
        vals = df_perm.loc[mask, predictor_of_interest].values.copy()
        perm_rng.shuffle(vals)
        df_perm.loc[mask, predictor_of_interest] = vals

    # ── Re-fit LMM for every connection with permuted predictor ───────────
    perm_t: Dict[str, float] = {}
    for conn_id in connection_ids:
        cols_needed = ["subject", conn_id] + [
            col for col in formula_vars if col in df_perm.columns
        ]
        df_conn = (
            df_perm[cols_needed]
            .rename(columns={conn_id: "power"})
            .dropna(subset=["power"])
            .reset_index(drop=True)  # clean integer index after dropna
        )

        # Guard 1: too few observations
        if len(df_conn) < 10:
            continue
        # Guard 2: too few subjects with data for this connection
        if df_conn["subject"].nunique() < 3:
            continue
        # Guard 3: permuted predictor is constant — design matrix would be singular
        if not _predictor_has_variability(df_conn, predictor_of_interest):
            continue

        result = _fit_single_lmm(
            df_conn, fixed_formula, re_formula,
            predictor_of_interest, method, maxiter,
        )

        # Only converged models with valid t-statistics enter the null graph
        if result["converged"] and not np.isnan(result["t_statistic"]):
            perm_t[conn_id] = result["t_statistic"]

    # ── Maximum component size per direction in the null graph ────────────
    perm_pos_comps, perm_neg_comps = _find_supra_threshold_components(
        perm_t, nodes, primary_threshold
    )
    max_pos = max(len(c) for c in perm_pos_comps) if perm_pos_comps else 0
    max_neg = max(len(c) for c in perm_neg_comps) if perm_neg_comps else 0
    return max_pos, max_neg


def apply_nbs_correction(
    results_df: pd.DataFrame,
    df_wide: pd.DataFrame,
    connection_ids: List[str],
    formula: str = "power ~ onoff + (1|subject)",
    predictor_of_interest: str = "onoff",
    method: str = "powell",
    maxiter: int = 500,
    primary_threshold: float = 3.0,
    n_permutations: int = 5000,
    alpha: float = 0.05,
    random_state: int = 42,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """
    Apply Network-Based Statistic (NBS) correction for channel-level connectivity.

    NBS (Zalesky et al., 2010, NeuroImage) tests whether any connected subgraph
    of the channel-pair network has suprathreshold t-statistics with a
    probability unlikely under the null hypothesis, using the maximum
    connected-component size as the test statistic.

    Algorithm
    ---------
    1. Threshold the observed t-statistic graph at |t| >= primary_threshold.
       Positive and negative effects are separated (two directed graphs).
    2. Identify connected components; record the number of edges in each.
    3. Run n_permutations within-subject restricted permutations to build
       the null distribution of the maximum component size.
    4. p-value for component C = (#{null >= |C|} + 1) / (n_permutations + 1)
       (Phipson & Smyth, 2010 continuity correction).
    5. Mark components as significant if p < alpha.

    Multi-covariate conditional testing
    ------------------------------------
    When the model formula contains nuisance covariates
    (e.g., power ~ onoff + valence + time + (1|subject)),
    only `predictor_of_interest` is permuted in each null draw.
    All other covariates retain their exact observed values for every probe.
    This yields a valid conditional test of predictor_of_interest given the
    nuisance regressors, equivalent to a restricted permutation procedure
    (Nichols & Holmes, 2002, HBM).

    Parameters
    ----------
    results_df : pd.DataFrame
        Results from run_lmm_per_connection with observed t-statistics.
    df_wide : pd.DataFrame
        Wide-format data used for permutation re-fitting.
    connection_ids : List[str]
        Connection column names.
    formula : str
        LMM formula.
    predictor_of_interest : str
        Predictor to permute and extract stats for.
    method : str
        LMM optimization method.
    maxiter : int
        Max iterations for LMM.
    primary_threshold : float
        Absolute t-value threshold for forming supra-threshold edges.
    n_permutations : int
        Number of permutations for null distribution.
    alpha : float
        Significance level.
    random_state : int
        Random seed.
    n_jobs : int
        Number of parallel jobs for permutation testing.
        -1 uses all available CPUs. Default=1 (no parallelization).

    Returns
    -------
    pd.DataFrame
        Input DataFrame with added columns:
        p_value_nbs, significant_nbs, nbs_component_id
    """
    from joblib import Parallel, delayed

    df = results_df.copy()

    # All permutation seeds derived from the master seed (fully reproducible)
    rng = np.random.RandomState(random_state)
    perm_seeds = rng.randint(0, 2**31 - 1, size=n_permutations).tolist()

    # Parse formula once — reused by every permutation worker
    fixed_formula, re_formula = _parse_random_effects(formula)

    # ── Observed graph ───────────────────────────────────────────────────
    nodes = _extract_unique_nodes(connection_ids)
    obs_t = dict(zip(df["connection_id"], df["t_statistic"]))

    obs_pos_comps, obs_neg_comps = _find_supra_threshold_components(
        obs_t, nodes, primary_threshold
    )
    # Tag each component with its direction so we use the correct null
    obs_components: List[Tuple[List[str], str]] = (
        [(c, "+") for c in obs_pos_comps] + [(c, "-") for c in obs_neg_comps]
    )

    if not obs_components:
        print(
            f"  NBS [{predictor_of_interest}]: No supra-threshold components "
            f"(|t| >= {primary_threshold}). Consider lowering primary_threshold."
        )
        df["p_value_nbs"] = np.nan
        df["significant_nbs"] = False
        df["nbs_component_id"] = -1
        return df

    obs_sizes = [len(c) for c, _ in obs_components]
    print(
        f"  NBS [{predictor_of_interest}]: "
        f"{len(obs_pos_comps)} positive / {len(obs_neg_comps)} negative "
        f"component(s), sizes = {obs_sizes}"
    )
    print(
        f"  NBS: Running {n_permutations} within-subject permutations "
        f"({n_jobs} parallel jobs) ..."
    )

    # ── Null distribution (direction-specific) ───────────────────────────
    # Pre-compute formula_vars once; passed to every worker to avoid
    # repeated string parsing inside 2000 parallel jobs.
    formula_vars = _extract_formula_variables(formula)

    # Each job returns (max_pos_size, max_neg_size).
    # Keeping the two directions separate prevents inflation of the positive
    # null by negative-direction permutation components and vice versa,
    # improving power while still controlling the per-direction FWER
    # (Zalesky et al., 2010, NeuroImage, Supplement S1).
    null_results = Parallel(n_jobs=n_jobs, verbose=0, backend="loky")(
        delayed(_run_nbs_permutation)(
            seed, df_wide, connection_ids, predictor_of_interest,
            formula_vars, fixed_formula, re_formula,
            nodes, primary_threshold, method, maxiter,
        )
        for seed in perm_seeds
    )
    null_max_pos = np.array([r[0] for r in null_results], dtype=np.intp)
    null_max_neg = np.array([r[1] for r in null_results], dtype=np.intp)

    print(
        f"  NBS: Null (positive) — "
        f"mean={null_max_pos.mean():.1f}, "
        f"95th pct={np.percentile(null_max_pos, 95):.0f}, "
        f"max={null_max_pos.max()}"
    )
    print(
        f"  NBS: Null (negative) — "
        f"mean={null_max_neg.mean():.1f}, "
        f"95th pct={np.percentile(null_max_neg, 95):.0f}, "
        f"max={null_max_neg.max()}"
    )

    # ── Per-component p-values ───────────────────────────────────────────
    # p-value with Phipson & Smyth (2010) continuity correction:
    # p = (# null >= observed + 1) / (n_permutations + 1)
    # Positive observed components are tested against the positive-direction
    # null; negative against the negative-direction null.
    df["p_value_nbs"] = np.nan
    df["significant_nbs"] = False
    df["nbs_component_id"] = -1

    for comp_idx, (comp_edges, direction) in enumerate(obs_components):
        comp_size = len(comp_edges)
        null_dist = null_max_pos if direction == "+" else null_max_neg
        p_val = (np.sum(null_dist >= comp_size) + 1) / (n_permutations + 1)

        for edge in comp_edges:
            mask = df["connection_id"] == edge
            df.loc[mask, "p_value_nbs"] = p_val
            df.loc[mask, "significant_nbs"] = p_val < alpha
            df.loc[mask, "nbs_component_id"] = comp_idx

    n_sig = df["significant_nbs"].sum()
    print(f"  NBS: {n_sig} edges in significant component(s) (alpha={alpha})")

    return df


def _run_max_t_permutation(
    perm_seed: int,
    df_wide: pd.DataFrame,
    connection_ids: List[str],
    predictor_of_interest: str,
    formula_vars: List[str],
    fixed_formula: str,
    re_formula: Optional[str],
    method: str,
    maxiter: int,
) -> float:
    """
    Execute one max-t permutation and return the maximum absolute null t-value.

    Uses the same within-subject restricted permutation scheme as NBS
    (see _run_nbs_permutation for the full rationale).

    Parameters
    ----------
    perm_seed : int
        RNG seed for this permutation.
    df_wide : pd.DataFrame
        Wide-format data.
    connection_ids : List[str]
        Connection column names.
    predictor_of_interest : str
        The only permuted column.
    formula_vars : List[str]
        Pre-extracted formula variable names.
    fixed_formula : str
        Pre-parsed fixed-effects formula.
    re_formula : Optional[str]
        Pre-parsed random-effects formula.
    method : str
        LMM optimisation method.
    maxiter : int
        Maximum LMM iterations.

    Returns
    -------
    float
        Maximum |t| across all connections in this permutation (0.0 if none).
    """
    perm_rng = np.random.RandomState(perm_seed)

    # Permute ONLY the predictor of interest within each subject
    df_perm = df_wide.copy()
    for subj in df_perm["subject"].unique():
        mask = df_perm["subject"] == subj
        vals = df_perm.loc[mask, predictor_of_interest].values.copy()
        perm_rng.shuffle(vals)
        df_perm.loc[mask, predictor_of_interest] = vals

    perm_max_abs_t = 0.0
    for conn_id in connection_ids:
        cols_needed = ["subject", conn_id] + [
            col for col in formula_vars if col in df_perm.columns
        ]
        df_conn = (
            df_perm[cols_needed]
            .rename(columns={conn_id: "power"})
            .dropna(subset=["power"])
            .reset_index(drop=True)
        )

        if len(df_conn) < 10:
            continue
        if df_conn["subject"].nunique() < 3:
            continue
        if not _predictor_has_variability(df_conn, predictor_of_interest):
            continue

        result = _fit_single_lmm(
            df_conn, fixed_formula, re_formula,
            predictor_of_interest, method, maxiter,
        )
        if result["converged"] and not np.isnan(result["t_statistic"]):
            perm_max_abs_t = max(perm_max_abs_t, abs(result["t_statistic"]))

    return perm_max_abs_t


def apply_max_t_permutation(
    results_df: pd.DataFrame,
    df_wide: pd.DataFrame,
    connection_ids: List[str],
    formula: str = "power ~ onoff + (1|subject)",
    predictor_of_interest: str = "onoff",
    method: str = "powell",
    maxiter: int = 500,
    n_permutations: int = 5000,
    alpha: float = 0.05,
    random_state: int = 42,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """
    Apply max-t permutation correction for FWER control.

    Less conservative than Bonferroni but controls FWER. Suitable for
    ROI-level analyses where the number of tests is small (~28 connections).
    Uses the same within-subject restricted permutation as NBS:
    only `predictor_of_interest` is permuted; all nuisance covariates
    retain their observed values (Nichols & Holmes, 2002, HBM).

    Parameters
    ----------
    results_df : pd.DataFrame
        Results from run_lmm_per_connection.
    df_wide : pd.DataFrame
        Wide-format data for re-fitting.
    connection_ids : List[str]
        Connection column names.
    formula : str
        LMM formula.
    predictor_of_interest : str
        Predictor to permute.
    method : str
        LMM optimization method.
    maxiter : int
        Max iterations.
    n_permutations : int
        Number of permutations.
    alpha : float
        Significance level.
    random_state : int
        Master random seed (fully determines all permutation seeds).
    n_jobs : int
        Parallel workers (-1 = all CPUs).

    Returns
    -------
    pd.DataFrame
        Input DataFrame with added columns: p_value_perm, significant_perm.
    """
    from joblib import Parallel, delayed

    df = results_df.copy()

    # All permutation seeds from master seed (reproducible)
    rng = np.random.RandomState(random_state)
    perm_seeds = rng.randint(0, 2**31 - 1, size=n_permutations).tolist()

    # Parse formula once
    fixed_formula, re_formula = _parse_random_effects(formula)
    formula_vars = _extract_formula_variables(formula)

    print(
        f"  Max-t permutation [{predictor_of_interest}]: "
        f"Running {n_permutations} within-subject permutations "
        f"({n_jobs} parallel jobs) ..."
    )

    null_max_t = Parallel(n_jobs=n_jobs, verbose=0, backend="loky")(
        delayed(_run_max_t_permutation)(
            seed, df_wide, connection_ids, predictor_of_interest,
            formula_vars, fixed_formula, re_formula, method, maxiter,
        )
        for seed in perm_seeds
    )
    null_max_t = np.array(null_max_t, dtype=float)

    print(
        f"  Max-t: Null distribution — "
        f"mean={null_max_t.mean():.3f}, "
        f"95th pct={np.percentile(null_max_t, 95):.3f}, "
        f"max={null_max_t.max():.3f}"
    )

    # p-value with Phipson & Smyth (2010) continuity correction
    df["p_value_perm"] = np.nan
    df["significant_perm"] = False

    valid_mask = df["converged"] & df["t_statistic"].notna()
    for idx in df[valid_mask].index:
        obs_abs_t = abs(df.loc[idx, "t_statistic"])
        p_val = (np.sum(null_max_t >= obs_abs_t) + 1) / (n_permutations + 1)
        df.loc[idx, "p_value_perm"] = p_val
        df.loc[idx, "significant_perm"] = p_val < alpha

    n_sig = df["significant_perm"].sum()
    print(f"  Max-t: {n_sig}/{valid_mask.sum()} significant at alpha={alpha}")

    return df


def _extract_unique_nodes(connection_ids: List[str]) -> set:
    """
    Extract unique node names from connection ID strings.

    Parameters
    ----------
    connection_ids : List[str]
        Connection IDs like ["Fp1-F3", "Fp1-F7", ...].

    Returns
    -------
    set
        Set of unique node names.
    """
    nodes = set()
    for conn_id in connection_ids:
        parts = conn_id.split(PAIR_SEPARATOR)
        if len(parts) == 2:
            nodes.add(parts[0].strip())
            nodes.add(parts[1].strip())
    return nodes


def _find_supra_threshold_components(
    t_values: Dict[str, float],
    nodes: set,
    threshold: float,
) -> Tuple[List[List[str]], List[List[str]]]:
    """
    Find connected components in the supra-threshold t-statistic graph.

    Positive and negative effects are handled in separate graphs so that
    a positive and negative edge sharing a node do not form one chimeric
    component (Zalesky et al., 2010).  Returning them separately also
    allows ``apply_nbs_correction`` to use direction-specific null
    distributions, which avoids inflating the null by mixing both
    directions and thus improves statistical power.

    Edge keys are stored as frozensets so that lookup is direction-agnostic:
    an edge stored as (A, B) is found regardless of whether NetworkX returns
    it as (A, B) or (B, A) when iterating subgraph.edges().

    Parameters
    ----------
    t_values : Dict[str, float]
        Mapping connection_id -> t-statistic.
    nodes : set
        All node names (channels or ROIs).
    threshold : float
        Absolute t-value threshold for edge inclusion.

    Returns
    -------
    Tuple[List[List[str]], List[List[str]]]
        (pos_components, neg_components) where each element is a list of
        components and each component is a list of connection_id strings
        (edges) belonging to that component.
    """
    import networkx as nx

    G_pos = nx.Graph()
    G_pos.add_nodes_from(nodes)

    G_neg = nx.Graph()
    G_neg.add_nodes_from(nodes)

    # Store edge_id lookup with frozenset keys so direction doesn't matter
    edge_map_pos: Dict[frozenset, str] = {}
    edge_map_neg: Dict[frozenset, str] = {}

    for conn_id, t_val in t_values.items():
        if np.isnan(t_val):
            continue
        parts = conn_id.split(PAIR_SEPARATOR)
        if len(parts) != 2:
            continue
        n1, n2 = parts[0].strip(), parts[1].strip()
        key = frozenset({n1, n2})

        if t_val >= threshold:
            G_pos.add_edge(n1, n2)
            edge_map_pos[key] = conn_id
        elif t_val <= -threshold:
            G_neg.add_edge(n1, n2)
            edge_map_neg[key] = conn_id

    def _extract_components(graph: "nx.Graph", edge_map: Dict) -> List[List[str]]:
        comps: List[List[str]] = []
        for comp_nodes in nx.connected_components(graph):
            if len(comp_nodes) < 2:
                continue
            subgraph = graph.subgraph(comp_nodes)
            comp_edges = [
                edge_map[frozenset({n1, n2})]
                for n1, n2 in subgraph.edges()
                if frozenset({n1, n2}) in edge_map
            ]
            if comp_edges:
                comps.append(comp_edges)
        return comps

    pos_components = _extract_components(G_pos, edge_map_pos)
    neg_components = _extract_components(G_neg, edge_map_neg)

    return pos_components, neg_components


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
