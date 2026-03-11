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
) -> pd.DataFrame:
    """
    Apply Network-Based Statistic (NBS) correction for channel-level connectivity.

    NBS (Zalesky et al., 2010, NeuroImage) is the graph analogue of
    cluster-based permutation testing. Steps:
    1. Threshold the observed t-statistic matrix at `primary_threshold`.
    2. Identify connected components in the supra-threshold graph.
    3. Record the size (number of edges) of each component.
    4. Build a null distribution by permuting the predictor label within
       subjects, re-fitting LMMs, re-thresholding, and recording the
       largest component size.
    5. Compute p-values as the proportion of null max-component sizes >= the
       observed component size.

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

    Returns
    -------
    pd.DataFrame
        Input DataFrame with added columns:
        p_value_nbs, significant_nbs, nbs_component_id
    """
    import networkx as nx

    rng = np.random.RandomState(random_state)
    df = results_df.copy()

    # ── Build observed graph ─────────────────────────────────────────────
    nodes = _extract_unique_nodes(connection_ids)
    obs_t = dict(zip(df["connection_id"], df["t_statistic"]))

    obs_components = _find_supra_threshold_components(
        obs_t, nodes, primary_threshold
    )
    obs_component_sizes = [len(comp_edges) for comp_edges in obs_components]

    if not obs_component_sizes:
        print("  NBS: No supra-threshold components found. "
              "Consider lowering primary_threshold.")
        df["p_value_nbs"] = np.nan
        df["significant_nbs"] = False
        df["nbs_component_id"] = -1
        return df

    print(f"  NBS: {len(obs_components)} observed component(s), "
          f"sizes = {obs_component_sizes}")

    # ── Null distribution via permutation ────────────────────────────────
    fixed_formula, re_formula = _parse_random_effects(formula)
    null_max_sizes = np.zeros(n_permutations)

    # Memory-efficient batch processing
    batch_size = 100  # Process permutations in batches to reduce peak memory
    n_batches = (n_permutations + batch_size - 1) // batch_size
    
    print(f"  NBS: Running {n_permutations} permutations in {n_batches} batches...")
    
    for batch_idx in range(n_batches):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, n_permutations)
        print(f"    Batch {batch_idx + 1}/{n_batches}: permutations {batch_start + 1}-{batch_end}")
        
        for perm_i in range(batch_start, batch_end):
            # Permute predictor within subjects
            df_perm = df_wide.copy()
            for subj in df_perm["subject"].unique():
                mask = df_perm["subject"] == subj
                vals = df_perm.loc[mask, predictor_of_interest].values.copy()
                rng.shuffle(vals)
                df_perm.loc[mask, predictor_of_interest] = vals

            # Re-fit LMMs (fast: only extract t-values)
            perm_t = {}
            for conn_id in connection_ids:
                df_conn = df_perm[["subject", conn_id]].copy()
                for col in _extract_formula_variables(formula):
                    if col in df_perm.columns and col != "power":
                        df_conn[col] = df_perm[col].values
                df_conn = df_conn.rename(columns={conn_id: "power"})
                df_conn = df_conn.dropna(subset=["power"])
                if len(df_conn) < 10:
                    continue
                try:
                    result = _fit_single_lmm(
                        df_conn, fixed_formula, re_formula,
                        predictor_of_interest, method, maxiter,
                    )
                    if result["converged"] and not np.isnan(result["t_statistic"]):
                        perm_t[conn_id] = result["t_statistic"]
                except Exception:
                    continue

            # Find largest component in permuted data
            perm_comps = _find_supra_threshold_components(
                perm_t, nodes, primary_threshold
            )
            if perm_comps:
                null_max_sizes[perm_i] = max(len(c) for c in perm_comps)
            
            # Clear intermediate data
            del df_conn, perm_t
        
        # Force garbage collection between batches
        del df_perm
        import gc
        gc.collect()

    # ── Compute p-values for each observed component ─────────────────────
    df["p_value_nbs"] = np.nan
    df["significant_nbs"] = False
    df["nbs_component_id"] = -1

    for comp_idx, comp_edges in enumerate(obs_components):
        comp_size = len(comp_edges)
        p_val = (np.sum(null_max_sizes >= comp_size) + 1) / (n_permutations + 1)

        for edge in comp_edges:
            mask = df["connection_id"] == edge
            df.loc[mask, "p_value_nbs"] = p_val
            df.loc[mask, "significant_nbs"] = p_val < alpha
            df.loc[mask, "nbs_component_id"] = comp_idx

    n_sig = df["significant_nbs"].sum()
    print(f"  NBS: {n_sig} edges significant at alpha={alpha}")

    return df


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
) -> pd.DataFrame:
    """
    Apply max-t permutation correction for FWER control.

    Less conservative than Bonferroni but controls FWER. Suitable for
    ROI-level analyses with moderate number of tests.

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
        Random seed.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with added columns: p_value_perm, significant_perm.
    """
    rng = np.random.RandomState(random_state)
    df = results_df.copy()
    fixed_formula, re_formula = _parse_random_effects(formula)

    # Observed max |t|
    obs_t = df.set_index("connection_id")["t_statistic"].to_dict()

    # Null distribution of max |t|
    null_max_t = np.zeros(n_permutations)

    # Memory-efficient batch processing
    batch_size = 100
    n_batches = (n_permutations + batch_size - 1) // batch_size
    
    print(f"  Max-t permutation: Running {n_permutations} permutations in {n_batches} batches...")
    
    for batch_idx in range(n_batches):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, n_permutations)
        print(f"    Batch {batch_idx + 1}/{n_batches}: permutations {batch_start + 1}-{batch_end}")
        
        for perm_i in range(batch_start, batch_end):
            df_perm = df_wide.copy()
            for subj in df_perm["subject"].unique():
                mask = df_perm["subject"] == subj
                vals = df_perm.loc[mask, predictor_of_interest].values.copy()
                rng.shuffle(vals)
                df_perm.loc[mask, predictor_of_interest] = vals

            perm_max = 0.0
            for conn_id in connection_ids:
                df_conn = df_perm[["subject", conn_id]].copy()
                for col in _extract_formula_variables(formula):
                    if col in df_perm.columns and col != "power":
                        df_conn[col] = df_perm[col].values
                df_conn = df_conn.rename(columns={conn_id: "power"})
                df_conn = df_conn.dropna(subset=["power"])
                if len(df_conn) < 10:
                    continue
                try:
                    result = _fit_single_lmm(
                        df_conn, fixed_formula, re_formula,
                        predictor_of_interest, method, maxiter,
                    )
                    if result["converged"] and not np.isnan(result["t_statistic"]):
                        perm_max = max(perm_max, abs(result["t_statistic"]))
                except Exception:
                    continue

            null_max_t[perm_i] = perm_max
            del df_conn
        
        # Force garbage collection between batches
        del df_perm
        import gc
        gc.collect()

    # Compute p-values
    df["p_value_perm"] = np.nan
    df["significant_perm"] = False

    valid_mask = df["converged"] & df["t_statistic"].notna()
    for idx in df[valid_mask].index:
        obs_abs_t = abs(df.loc[idx, "t_statistic"])
        p_val = (np.sum(null_max_t >= obs_abs_t) + 1) / (n_permutations + 1)
        df.loc[idx, "p_value_perm"] = p_val
        df.loc[idx, "significant_perm"] = p_val < alpha

    n_sig = df["significant_perm"].sum()
    print(f"  Max-t perm: {n_sig}/{valid_mask.sum()} significant at alpha={alpha}")

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
) -> List[List[str]]:
    """
    Find connected components in the supra-threshold graph.

    Parameters
    ----------
    t_values : Dict[str, float]
        Mapping connection_id -> t-statistic.
    nodes : set
        All node names.
    threshold : float
        Absolute t-value threshold for edge inclusion.

    Returns
    -------
    List[List[str]]
        List of components, each component is a list of edge (connection_id) strings.
    """
    import networkx as nx

    G = nx.Graph()
    G.add_nodes_from(nodes)

    edge_map = {}
    for conn_id, t_val in t_values.items():
        if np.isnan(t_val):
            continue
        if abs(t_val) >= threshold:
            parts = conn_id.split(PAIR_SEPARATOR)
            if len(parts) == 2:
                n1, n2 = parts[0].strip(), parts[1].strip()
                G.add_edge(n1, n2)
                edge_map[(n1, n2)] = conn_id

    components = []
    for comp_nodes in nx.connected_components(G):
        if len(comp_nodes) < 2:
            continue
        subgraph = G.subgraph(comp_nodes)
        comp_edges = []
        for n1, n2 in subgraph.edges():
            key = (n1, n2) if (n1, n2) in edge_map else (n2, n1)
            if key in edge_map:
                comp_edges.append(edge_map[key])
        if comp_edges:
            components.append(comp_edges)

    return components


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
