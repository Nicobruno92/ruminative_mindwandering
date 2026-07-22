"""Pure response-construction and transformation functions for the GLMM track.

These are deliberately free of any I/O or R dependency so they can be unit
tested in isolation. See
``docs/superpowers/specs/2026-07-22-glmm-objective-markers-design.md``.
"""

# =============================================================================
# IMPORTS
# =============================================================================

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "glmm_config.yaml"

# =============================================================================
# HELPERS
# =============================================================================


def load_glmm_config(path: Path | str | None = None) -> dict:
    """Load the GLMM configuration YAML.

    Parameters
    ----------
    path : Path | str | None
        Config location. Defaults to ``glmm_config.yaml`` beside this module.

    Returns
    -------
    dict
        Parsed configuration.
    """
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    return yaml.safe_load(open(config_path))


def build_binomial_response(
    df: pd.DataFrame, success_col: str, total_col: str
) -> pd.DataFrame:
    """Add ``_succ``/``_fail`` columns for a ``cbind()`` binomial response.

    Rows whose denominator is zero carry no information about the proportion
    and are dropped. In the ``n10`` dataset this removes the 847 probes with
    no no-go trials from the commission model.

    Parameters
    ----------
    df : pd.DataFrame
        Probe-level dataframe.
    success_col : str
        Column holding the event count (numerator).
    total_col : str
        Column holding the trial count (denominator).

    Returns
    -------
    pd.DataFrame
        Copy of ``df`` with ``_succ`` and ``_fail`` added and zero-denominator
        rows removed.

    Raises
    ------
    ValueError
        If any numerator exceeds its denominator.
    """
    out = df[df[total_col] > 0].copy()
    if (out[success_col] > out[total_col]).any():
        n_bad = int((out[success_col] > out[total_col]).sum())
        raise ValueError(
            f"{n_bad} rows have {success_col} > {total_col}; "
            "the binomial response is undefined."
        )
    out["_succ"] = out[success_col].astype(int)
    out["_fail"] = (out[total_col] - out[success_col]).astype(int)
    return out


def empirical_logit(
    successes: np.ndarray, totals: np.ndarray, correction: float = 0.5
) -> np.ndarray:
    """Haldane-corrected empirical logit.

    ``log((y + c) / (n - y + c))`` stays finite at ``y = 0`` and ``y = n``,
    which plain ``logit`` does not. This is the standard treatment for
    proportions with small denominators -- here as small as 1-4 no-go trials.

    Parameters
    ----------
    successes : np.ndarray
        Event counts.
    totals : np.ndarray
        Trial counts.
    correction : float
        Continuity constant added to both numerator and denominator.

    Returns
    -------
    np.ndarray
        Empirical logit values.
    """
    successes = np.asarray(successes, dtype=float)
    totals = np.asarray(totals, dtype=float)
    return np.log((successes + correction) / (totals - successes + correction))


def log_transform(values: np.ndarray) -> np.ndarray:
    """Natural log of a strictly positive response.

    Parameters
    ----------
    values : np.ndarray
        Response values; must all be > 0.

    Returns
    -------
    np.ndarray
        Log-transformed values.

    Raises
    ------
    ValueError
        If any value is <= 0.
    """
    values = np.asarray(values, dtype=float)
    if np.any(values <= 0):
        raise ValueError("log_transform requires strictly positive values.")
    return np.log(values)
