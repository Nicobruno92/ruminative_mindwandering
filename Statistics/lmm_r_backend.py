"""
R-backend WLS-LMM via rpy2 + lmerTest.

Drop-in replacement for the two functions used by andrillon_pipeline.py:

    run_lmm_per_channel(power_data, df_behavioral, formula,
                        predictor_of_interest, ..., obs_weights=None)

    fit_reduced_model_per_channel(power_data, df_behavioral, formula,
                                  predictor_of_interest, ..., obs_weights=None)

Both return the same types as their counterparts in lmm_model.py.

The key difference: when `obs_weights` is supplied, lme4::lmer is called
with the `weights` argument, which implements exact WLS-LMM — i.e. the
residual variance for observation i is sigma^2 / w_i.  This correctly
propagates into the BLUP estimates of the random effects, which the
sqrt(w) pre-multiplication approximation does not.

R session lifecycle
-------------------
rpy2 embeds an R session in the Python process.  The session is
initialized lazily on the first call to any public function and is
reused for every subsequent call in the same process.  Under loky
(the joblib default), each worker spawns a separate process, so each
worker initializes R once and reuses it across the permutations it
handles.  This keeps startup overhead to O(n_workers), not O(n_perms).

Dependencies
------------
Python:  rpy2 >= 3.5    (conda install -c conda-forge rpy2)
R:       lme4, lmerTest  (Rscript -e "install.packages(c('lme4','lmerTest'))")
"""

import logging
import re
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

# Shared with the Python backend so both report residual shape on the same
# scale and against the same flag thresholds.
from lmm_model import RESIDUAL_OUTLIER_SD

logger = logging.getLogger(__name__)

# ── R session state (per-process) ──────────────────────────────────────────
_R_INITIALIZED: bool = False
_r_lme4 = None       # lme4 namespace  (lmerControl lives here)
_r_lmerTest = None   # lmerTest namespace (lmer with Satterthwaite p-values)
_r_base = None       # base namespace (summary, rownames, AIC, BIC, …)
_r_stats = None      # stats namespace (residuals, fitted, logLik)
_pandas2ri = None    # pandas ↔ R converter


# ── R session initialization ────────────────────────────────────────────────

def _init_r_session() -> None:
    """Initialize rpy2 and R packages (idempotent, lazy)."""
    global _R_INITIALIZED, _r_lme4, _r_lmerTest, _r_base, _r_stats, _pandas2ri

    if _R_INITIALIZED:
        return

    # Point rpy2 at the conda-env R when R_HOME is not already set.
    # Without this, rpy2 may pick up a system R that lacks lme4/lmerTest.
    import os, shutil
    if 'R_HOME' not in os.environ:
        conda_prefix = os.environ.get('CONDA_PREFIX', '')
        conda_r = os.path.join(conda_prefix, 'lib', 'R')
        if os.path.isdir(conda_r):
            os.environ['R_HOME'] = conda_r
        else:
            # Fallback: ask the Rscript on PATH
            rscript = shutil.which('Rscript')
            if rscript:
                import subprocess
                r_home = subprocess.check_output(
                    [rscript, '-e', 'cat(R.home())'], text=True
                ).strip()
                if r_home:
                    os.environ['R_HOME'] = r_home

    try:
        import rpy2.robjects as ro                          # noqa: F401
        from rpy2.robjects import pandas2ri
        from rpy2.robjects.packages import importr
    except ImportError as exc:
        raise ImportError(
            "rpy2 is not installed in this environment.\n"
            "Install with:  conda install -c conda-forge rpy2"
        ) from exc

    try:
        _r_lme4 = importr('lme4')
        _r_lmerTest = importr('lmerTest')
    except Exception as exc:
        raise RuntimeError(
            "R package lmerTest (or its dependency lme4) is not installed.\n"
            "Install from R:  install.packages(c('lme4', 'lmerTest'))\n"
            f"R_HOME={os.environ.get('R_HOME', 'not set')}\n"
            f"Original error: {exc}"
        ) from exc

    _r_base = importr('base')
    _r_stats = importr('stats')
    # Store the pandas2ri module for use in _df_to_r (no .activate() — deprecated)
    _pandas2ri = pandas2ri

    _R_INITIALIZED = True
    logger.info("R session initialized — lme4 + lmerTest loaded (R_HOME=%s)",
                os.environ.get('R_HOME', 'inherited'))


# ── Internal helpers ─────────────────────────────────────────────────────────

def _df_to_r(df: pd.DataFrame):
    """Convert a pandas DataFrame to an R data.frame (rpy2 3.6+ compatible)."""
    from rpy2.robjects import pandas2ri
    # rpy2 3.6: use converter.context() instead of the deprecated activate()
    with pandas2ri.converter.context():
        import rpy2.robjects as ro
        return ro.conversion.py2rpy(df)


def _extract_coef_row(
    r_summary,
    predictor: str,
) -> Tuple[float, float]:
    """
    Extract (t_value, p_value) for `predictor` from an R lmerTest summary.

    lmerTest summary$coefficients is a matrix with columns:
        [Estimate, Std.Error, df, t value, Pr(>|t|)]   (indices 0–4)

    Row names match the formula terms exactly, including interaction
    notation (e.g. "onoff:confidence").

    Returns (nan, nan) if the predictor is not found (e.g. dropped due to
    perfect collinearity), so the caller can mark the channel as failed.
    """
    coef_matrix = r_summary.rx2('coefficients')
    rownames = list(_r_base.rownames(coef_matrix))
    coef_array = np.array(coef_matrix)   # shape (n_terms, 5)

    if predictor not in rownames:
        return np.nan, np.nan

    row_idx = rownames.index(predictor)
    t_val = float(coef_array[row_idx, 3])
    p_val = float(coef_array[row_idx, 4])
    return t_val, p_val


def _statsmodels_formula_to_r(formula: str) -> str:
    """
    Convert a statsmodels-style formula to an R lme4 formula string.

    The two syntaxes are almost identical.  The only difference handled
    here is that statsmodels uses ``(1|subject)`` inside the Python
    string while R's formula parser reads the same string directly.
    No transformation is needed — this function exists as an explicit
    boundary so future differences can be handled here.
    """
    return formula


# ── Public API ───────────────────────────────────────────────────────────────

def run_lmm_per_channel(
    power_data: np.ndarray,
    df_behavioral: pd.DataFrame,
    formula: str,
    predictor_of_interest: str,
    method: str = 'powell',       # ignored (lme4 uses its own optimizer)
    maxiter: int = 200,           # mapped to lmerControl(optCtrl=list(maxfun=maxiter))
    random_state: int = 42,       # ignored (R RNG not used for fitting)
    return_diagnostics: bool = False,
    abort_on_high_failure: bool = True,
    obs_weights: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Fit lme4::lmer per channel and extract t/p statistics.

    Mirrors the interface of ``lmm_model.run_lmm_per_channel``.
    When ``obs_weights`` is supplied, lmer is called with ``weights=``
    for exact WLS-LMM (residual variance = sigma^2 / w_i).

    Parameters
    ----------
    power_data : np.ndarray, shape (n_obs, n_channels)
    df_behavioral : pd.DataFrame
        Must contain 'subject' plus all formula variables.
    formula : str
        statsmodels/R formula, e.g. ``"power ~ onoff + (1|subject)"``.
    predictor_of_interest : str
        Term whose t/p to extract (e.g. ``"onoff"`` or ``"onoff:confidence"``).
    method : str
        Ignored — lme4 uses bobyqa/Nelder_Mead internally.
    maxiter : int
        Passed to lmerControl optCtrl maxfun.
    random_state : int
        Ignored.
    return_diagnostics : bool
        If True, return a diagnostics dict (convergence info).
    abort_on_high_failure : bool
        If True and >50% of channels fail, raise RuntimeError.
    obs_weights : np.ndarray or None, shape (n_obs,)
        Per-observation weights for WLS-LMM.  None = OLS-LMM.

    Returns
    -------
    t_stats : np.ndarray, shape (n_channels,)
    p_values : np.ndarray, shape (n_channels,)
    diagnostics : dict
    """
    _init_r_session()
    import rpy2.robjects as ro
    from rpy2.robjects import FloatVector

    n_channels = power_data.shape[1]
    t_stats = np.full(n_channels, np.nan)
    p_values = np.full(n_channels, np.nan)

    diagnostics: Dict = {
        'n_converged': 0,
        'n_warnings': 0,
        'n_failed': 0,
        'n_insufficient_data': 0,
        'convergence_rate': 0.0,
        'failed_channels': [],
        'convergence_warnings': [],
        'aic': [], 'bic': [], 'log_likelihood': [],
        'conditional_r2': [], 'residual_variance': [],
        'residual_skew': [], 'residual_exkurt': [],
        'pct_residual_outliers': [], 'breusch_pagan_p': [],
    }

    r_formula_str = _statsmodels_formula_to_r(formula)
    # Strip random effects for variable extraction (for NaN dropping)
    formula_vars = _extract_formula_vars(formula)

    n_subjects = df_behavioral['subject'].nunique()
    if n_subjects < 2:
        raise ValueError(f"Insufficient subjects: {n_subjects}")

    # lmerControl with maxfun
    r_ctrl = _r_lme4.lmerControl(
        optCtrl=_r_base.list(maxfun=float(maxiter))
    )

    for ch_idx in range(n_channels):
        df_ch = df_behavioral.copy()
        df_ch['power'] = power_data[:, ch_idx]

        dropna_cols = ['power', 'subject'] + formula_vars
        df_ch = df_ch.dropna(subset=dropna_cols)
        df_ch['subject'] = df_ch['subject'].astype(str)

        if len(df_ch) < 10 or df_ch['subject'].nunique() < 2:
            diagnostics['n_insufficient_data'] += 1
            continue

        # Align weights to surviving rows
        w_ch: Optional[np.ndarray] = None
        if obs_weights is not None:
            w_ch = obs_weights[df_ch.index.values]
            df_ch = df_ch.reset_index(drop=True)
            # Guard against non-positive weights after row dropping
            if np.any(w_ch <= 0) or np.any(~np.isfinite(w_ch)):
                diagnostics['n_failed'] += 1
                diagnostics['failed_channels'].append(ch_idx)
                continue
        else:
            df_ch = df_ch.reset_index(drop=True)

        try:
            r_df = _df_to_r(df_ch)
            r_formula = ro.Formula(r_formula_str)

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                if w_ch is not None:
                    model = _r_lmerTest.lmer(
                        r_formula,
                        data=r_df,
                        weights=FloatVector(w_ch.tolist()),
                        REML=True,
                        control=r_ctrl,
                    )
                else:
                    model = _r_lmerTest.lmer(
                        r_formula,
                        data=r_df,
                        REML=True,
                        control=r_ctrl,
                    )
                has_warning = any(
                    'convergence' in str(w.message).lower()
                    or 'singular' in str(w.message).lower()
                    for w in caught
                )

            r_summary = _r_base.summary(model)
            t_val, p_val = _extract_coef_row(r_summary, predictor_of_interest)

            if not np.isfinite(t_val):
                diagnostics['n_failed'] += 1
                diagnostics['failed_channels'].append(ch_idx)
                continue

            t_stats[ch_idx] = t_val
            p_values[ch_idx] = p_val

            if has_warning:
                diagnostics['n_warnings'] += 1
                if return_diagnostics:
                    diagnostics['convergence_warnings'].append(
                        f"Channel {ch_idx}: convergence/singularity warning"
                    )
            else:
                diagnostics['n_converged'] += 1

            # Basic model quality metrics
            if return_diagnostics:
                _collect_model_metrics(model, r_summary, df_ch, diagnostics)

        except Exception as exc:
            diagnostics['n_failed'] += 1
            diagnostics['failed_channels'].append(ch_idx)
            if return_diagnostics:
                diagnostics['convergence_warnings'].append(
                    f"Channel {ch_idx}: R error — {exc}"
                )

    n_attempted = n_channels - diagnostics['n_insufficient_data']
    diagnostics['convergence_rate'] = (
        diagnostics['n_converged'] / max(1, n_attempted)
    )

    if abort_on_high_failure and n_attempted > 0:
        failure_rate = diagnostics['n_failed'] / n_attempted
        if failure_rate > 0.5:
            raise RuntimeError(
                f"R backend: {diagnostics['n_failed']}/{n_attempted} channels "
                f"failed to fit ({100*failure_rate:.1f}%). "
                "Check formula, data, and R package versions."
            )

    return t_stats, p_values, diagnostics


def fit_reduced_model_per_channel(
    power_data: np.ndarray,
    df_behavioral: pd.DataFrame,
    formula: str,
    predictor_of_interest: str,
    method: str = 'powell',
    maxiter: int = 200,
    random_state: int = 42,
    obs_weights: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit reduced lme4::lmer (without predictor) and return residuals + fitted.

    Used as the first step of Freedman-Lane permutation.  The reduced model
    omits `predictor_of_interest` but keeps all other fixed and random effects.
    When ``obs_weights`` is supplied the reduced model also uses WLS so that
    the residuals permuted in subsequent steps are on the correct weighted scale.

    Returns
    -------
    residuals : np.ndarray, shape (n_obs, n_channels)
    fitted_values : np.ndarray, shape (n_obs, n_channels)
    """
    _init_r_session()
    import rpy2.robjects as ro
    from rpy2.robjects import FloatVector

    n_observations, n_channels = power_data.shape
    residuals = np.full((n_observations, n_channels), np.nan)
    fitted_values = np.full((n_observations, n_channels), np.nan)

    reduced_formula = _create_reduced_formula(formula, predictor_of_interest)
    r_formula_str = _statsmodels_formula_to_r(reduced_formula)
    formula_vars = _extract_formula_vars(reduced_formula)

    r_ctrl = _r_lme4.lmerControl(
        optCtrl=_r_base.list(maxfun=float(maxiter))
    )

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        for ch_idx in range(n_channels):
            df_ch = df_behavioral.copy()
            df_ch['power'] = power_data[:, ch_idx]

            original_index = df_ch.index.values
            dropna_cols = ['power', 'subject'] + formula_vars
            df_ch = df_ch.dropna(subset=dropna_cols)
            surviving_idx = df_ch.index.values

            df_ch['subject'] = df_ch['subject'].astype(str)

            if len(df_ch) < 10 or df_ch['subject'].nunique() < 2:
                continue

            w_ch: Optional[np.ndarray] = None
            if obs_weights is not None:
                w_ch = obs_weights[surviving_idx]
                if np.any(w_ch <= 0) or np.any(~np.isfinite(w_ch)):
                    continue

            df_ch = df_ch.reset_index(drop=True)

            try:
                r_df = _df_to_r(df_ch)
                r_formula = ro.Formula(r_formula_str)

                if w_ch is not None:
                    model = _r_lmerTest.lmer(
                        r_formula,
                        data=r_df,
                        weights=FloatVector(w_ch.tolist()),
                        REML=True,
                        control=r_ctrl,
                    )
                else:
                    model = _r_lmerTest.lmer(
                        r_formula,
                        data=r_df,
                        REML=True,
                        control=r_ctrl,
                    )

                r_resid = np.array(_r_stats.residuals(model))
                r_fitted = np.array(_r_stats.fitted(model))

                residuals[surviving_idx, ch_idx] = r_resid
                fitted_values[surviving_idx, ch_idx] = r_fitted

            except Exception as _exc:
                logger.debug("R reduced model failed for channel %d: %s", ch_idx, _exc)

    return residuals, fitted_values


# ── Normalization helper (called from andrillon_pipeline) ───────────────────

def compute_obs_weights(
    df_behavioral: pd.DataFrame,
    weight_col: str,
    normalization: Optional[str] = 'subject',
    subject_col: str = 'subject',
) -> np.ndarray:
    """
    Compute per-observation weights from a behavioral column.

    Parameters
    ----------
    df_behavioral : pd.DataFrame
    weight_col : str
        Column to use as weights (e.g. 'confidence').
    normalization : str or None
        'subject' — divide each observation by its subject's mean weight
                    so mean weight per subject = 1.0.  Prevents subjects
                    with systematically lower confidence from contributing
                    less to fixed-effect estimation.
        'global'  — divide all weights by the global mean.
        None      — use raw values (must be > 0).
    subject_col : str
        Column name for subject identifier.

    Returns
    -------
    weights : np.ndarray, shape (n_obs,)
        Positive, finite weight per observation aligned to df_behavioral index.

    Raises
    ------
    ValueError
        If any weight is non-positive or non-finite after normalization.
    """
    raw = df_behavioral[weight_col].values.astype(float)

    # Replace NaN with 0 so they get the floor treatment below
    raw = np.where(np.isfinite(raw), raw, 0.0)

    if normalization == 'subject':
        # Divide by subject mean → mean weight per subject = 1.0.
        # Removes between-subject mean differences but not scale differences.
        weights = raw.copy()
        for subj in df_behavioral[subject_col].unique():
            mask = df_behavioral[subject_col].values == subj
            subj_mean = np.nanmean(raw[mask])
            if subj_mean > 0:
                weights[mask] = raw[mask] / subj_mean

    elif normalization == 'subject_zscore':
        # Z-score within each subject (removes both mean AND scale differences).
        # A probe at the subject's mean gets z=0; one SD above gets z=1.
        # This makes weights comparable across subjects with different
        # confidence ranges (e.g. always-high vs. high-variance subjects).
        # Z-scores are then shifted globally so the minimum = ε, keeping all
        # weights strictly positive while preserving relative ordering.
        weights = np.zeros_like(raw)
        for subj in df_behavioral[subject_col].unique():
            mask = df_behavioral[subject_col].values == subj
            vals = raw[mask]
            subj_mean = np.nanmean(vals)
            subj_std = np.nanstd(vals, ddof=1)
            if subj_std > 0:
                weights[mask] = (vals - subj_mean) / subj_std
            else:
                weights[mask] = 0.0
        global_min = weights.min()
        weights = weights - global_min + 0.01

    elif normalization == 'subject_minmax':
        # Min-max scaling within each subject → [ε, 1].
        # The most confident probe for each subject gets weight 1; the least
        # confident gets weight ε.  Bounded range avoids inflating influence
        # of outlier-high confidence probes (unlike z-score which can exceed 4).
        # Purely ordinal: only the within-subject ranking of confidence matters.
        # Subjects with constant confidence (sd=0) get uniform weight 1.
        _MINMAX_EPS = 0.01
        weights = np.ones_like(raw)
        for subj in df_behavioral[subject_col].unique():
            mask = df_behavioral[subject_col].values == subj
            vals = raw[mask]
            lo, hi = np.nanmin(vals), np.nanmax(vals)
            if hi > lo:
                weights[mask] = _MINMAX_EPS + (1.0 - _MINMAX_EPS) * (vals - lo) / (hi - lo)
            # else: constant confidence → uniform weight 1 (no weighting effect)

    elif normalization == 'global':
        global_mean = np.nanmean(raw)
        weights = raw / global_mean if global_mean > 0 else raw.copy()

    else:
        weights = raw.copy()

    # Floor: probes with zero/near-zero confidence (e.g. confidence=0) get a
    # small positive weight rather than being discarded. 1% of the mean weight
    # means they contribute but have negligible influence on the estimates.
    n_zero = int(np.sum(weights <= 0))
    if n_zero > 0:
        floor = np.nanmean(weights[weights > 0]) * 0.01
        weights = np.where(weights <= 0, floor, weights)
        logger.info(
            "  %d observations had weight <= 0 in '%s'; floored to %.4f "
            "(1%% of mean positive weight).",
            n_zero, weight_col, floor,
        )

    if np.any(~np.isfinite(weights)):
        n_bad = int(np.sum(~np.isfinite(weights)))
        raise ValueError(
            f"{n_bad} observations have non-finite weights in '{weight_col}' "
            "after normalization. Check for NaN values."
        )

    return weights


# ── Private helpers ──────────────────────────────────────────────────────────

def _extract_formula_vars(formula: str) -> List[str]:
    """Extract all variable names referenced in formula fixed effects."""
    # Remove random effects (1|subject) etc.
    fixed = re.sub(r'\([^)]*\|[^)]*\)', '', formula)
    # Remove left-hand side
    if '~' in fixed:
        fixed = fixed.split('~', 1)[1]
    # Split on operators and clean up
    tokens = re.split(r'[+\*:|~\s\(\)]+', fixed)
    return [t.strip() for t in tokens if t.strip() and t.strip() != '1']


def _create_reduced_formula(formula: str, predictor_to_remove: str) -> str:
    """
    Remove predictor_to_remove from formula.

    Mirrors the logic in lmm_model._create_reduced_formula to avoid
    importing from that module (which has heavy imports).

    For interaction predictors (e.g. 'onoff:confidence'), removes the
    interaction term but keeps both main effects.
    """
    parts = formula.split('~')
    left_side = parts[0].strip()
    right_side = parts[1].strip()

    random_effects = re.findall(r'\([^)]*\|[^)]*\)', right_side)
    fixed_effects = re.sub(r'\s*\+?\s*\([^)]*\|[^)]*\)', '', right_side).strip()

    expanded_terms: List[str] = []
    for raw_term in fixed_effects.split('+'):
        raw_term = raw_term.strip()
        if not raw_term:
            continue
        if '*' in raw_term:
            vars_in_star = [v.strip() for v in raw_term.split('*')]
            for v in vars_in_star:
                if v and v not in expanded_terms:
                    expanded_terms.append(v)
            interaction = ':'.join(vars_in_star)
            if interaction not in expanded_terms:
                expanded_terms.append(interaction)
        else:
            if raw_term not in expanded_terms:
                expanded_terms.append(raw_term)

    reduced_terms = [t for t in expanded_terms if t != predictor_to_remove and t]
    reduced_fixed = ' + '.join(reduced_terms) if reduced_terms else '1'

    if random_effects:
        return f"{left_side} ~ {reduced_fixed} + {' + '.join(random_effects)}"
    return f"{left_side} ~ {reduced_fixed}"


def _collect_model_metrics(model, r_summary, df_ch: pd.DataFrame, diagnostics: Dict) -> None:
    """Append AIC, BIC, logLik, conditional R² to diagnostics lists."""
    try:
        aic_val = float(np.array(_r_base.AIC(model))[0])
        bic_val = float(np.array(_r_base.BIC(model))[0])
        llf_val = float(np.array(_r_stats.logLik(model))[0])
    except Exception:
        aic_val = bic_val = llf_val = np.nan

    diagnostics['aic'].append(aic_val)
    diagnostics['bic'].append(bic_val)
    diagnostics['log_likelihood'].append(llf_val)

    try:
        r_resid = np.array(_r_stats.residuals(model))
        total_var = float(np.var(df_ch['power']))
        residual_var = float(np.var(r_resid))
        cond_r2 = 1.0 - residual_var / total_var if total_var > 0 else 0.0
        diagnostics['conditional_r2'].append(cond_r2)
        diagnostics['residual_variance'].append(residual_var)

        # Residual shape, mirroring the Python backend. These are O(n) so —
        # unlike the Shapiro-Wilk test this replaces — they are cheap enough
        # to compute on every channel of every permutation.
        resid_sd = float(np.std(r_resid))
        diagnostics['residual_skew'].append(float(stats.skew(r_resid)))
        diagnostics['residual_exkurt'].append(float(stats.kurtosis(r_resid)))
        if resid_sd > 0:
            n_outliers = int(
                np.sum(
                    np.abs(r_resid - r_resid.mean())
                    > RESIDUAL_OUTLIER_SD * resid_sd
                )
            )
            diagnostics['pct_residual_outliers'].append(
                float(100.0 * n_outliers / r_resid.size)
            )
        else:
            diagnostics['pct_residual_outliers'].append(0.0)
    except Exception:
        diagnostics['conditional_r2'].append(np.nan)
        diagnostics['residual_variance'].append(np.nan)
        diagnostics['residual_skew'].append(np.nan)
        diagnostics['residual_exkurt'].append(np.nan)
        diagnostics['pct_residual_outliers'].append(np.nan)

    diagnostics['breusch_pagan_p'].append(np.nan)
