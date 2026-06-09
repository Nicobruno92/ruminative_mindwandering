"""
Spatial decoding (per-electrode searchlight) core, shared by the LOSO and
Within-Subject MW classification pipelines.

Each per-electrode model is the existing classification engine restricted to a
single channel's marker columns. This module owns only the spatial orchestration:
channel parsing, the channel loop, multiple-comparisons correction, MNE montage
construction, and topomap rendering. The ML engine itself is supplied by the
caller as a `channel_eval` callable (see module docstring of each driver).

Column convention: per-channel feature columns are named ``{channel}_{marker}``.
"""

from __future__ import annotations

import os
from typing import Callable

import numpy as np
import pandas as pd


def parse_channels_from_columns(columns: list[str]) -> list[str]:
    """
    Extract the sorted, unique set of channel names from ``{channel}_{marker}`` columns.

    The channel token is everything before the first underscore. This is robust to
    markers that themselves contain underscores (e.g. ``psd_bands_delta_mean``).

    Parameters
    ----------
    columns : list of str
        Feature column names in ``{channel}_{marker}`` form.

    Returns
    -------
    list of str
        Sorted unique channel names.
    """
    channels = {col.split("_", 1)[0] for col in columns if "_" in col}
    return sorted(channels)


def select_channel_columns(X: pd.DataFrame, channel: str) -> pd.DataFrame:
    """
    Restrict a per-channel feature matrix to a single channel's marker columns.

    Uses an exact ``{channel}_`` prefix so that ``P1`` does not match ``P10_*``.

    Parameters
    ----------
    X : pd.DataFrame
        Per-channel feature matrix.
    channel : str
        Electrode name.

    Returns
    -------
    pd.DataFrame
        Sub-matrix with only ``{channel}_*`` columns.

    Raises
    ------
    ValueError
        If the channel matches no columns.
    """
    prefix = f"{channel}_"
    cols = [c for c in X.columns if c.startswith(prefix)]
    if not cols:
        raise ValueError(f"Channel '{channel}' matched no columns in X.")
    return X[cols]


def fdr_correct(p_values: np.ndarray, alpha: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """
    Benjamini-Hochberg FDR correction.

    Parameters
    ----------
    p_values : np.ndarray
        Raw p-values (1-D).
    alpha : float
        Target false-discovery rate.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (adjusted_p_values, reject_mask) in the original input order.
    """
    p = np.asarray(p_values, dtype=float)
    n = p.size
    order = np.argsort(p)
    ranked = p[order]
    # BH-adjusted p in sorted order, with monotone (cumulative-min from the top) enforcement.
    adj_sorted = ranked * n / (np.arange(n) + 1)
    adj_sorted = np.minimum.accumulate(adj_sorted[::-1])[::-1]
    adj_sorted = np.clip(adj_sorted, 0.0, 1.0)
    adj = np.empty_like(adj_sorted)
    adj[order] = adj_sorted
    reject = adj <= alpha
    return adj, reject
