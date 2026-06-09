"""Shared utilities for behavioral classification pipeline.

Provides data loading, merging, and feature column resolution for the
behavior classification analyses (group classification, inclusion/exclusion
classification).

All paths are resolved relative to the current working directory (repo root),
so scripts must be run from the repository root.
"""

# === Imports ===
from pathlib import Path
from typing import Dict, List

import pandas as pd


# === Data Loading ===

def load_probe_data(config: dict) -> pd.DataFrame:
    """Load probe-level aggregated data (all probes, all subjects).

    Parameters
    ----------
    config : dict
        Configuration dict loaded from ``Behavior/Classification/config.yaml``.

    Returns
    -------
    pd.DataFrame
        Probe-level data with 2460 rows and all probe dimension columns,
        subject metadata, and group/inclusion_exclusion labels.
    """
    path = Path(config["data"]["probe_data_path"])
    return pd.read_csv(path)


def load_pca_data(config: dict) -> pd.DataFrame:
    """Load PCA results restricted to off-task probes (PC1, PC2, PC3 only).

    Off-task probes are those below the median onoff split; this file
    contains 888 rows. On-task probes will have NaN PCs after merging.

    Parameters
    ----------
    config : dict
        Configuration dict loaded from ``Behavior/Classification/config.yaml``.

    Returns
    -------
    pd.DataFrame
        Columns: subject, task, probe_number, PC1, PC2, PC3.
    """
    path = Path(config["data"]["pca_results_path"])
    df = pd.read_csv(path)
    return df[["subject", "task", "probe_number", "PC1", "PC2", "PC3"]]


def load_objective_markers(config: dict) -> pd.DataFrame:
    """Load SART objective performance markers per probe window.

    Returns the merge-key columns plus the marker columns listed under
    ``features.objective_marker_cols`` in the config.

    Parameters
    ----------
    config : dict
        Configuration dict loaded from ``Behavior/Classification/config.yaml``.

    Returns
    -------
    pd.DataFrame
        Columns: subject, task, probe_number, <objective_marker_cols>.
    """
    path = Path(config["data"]["objective_markers_path"])
    marker_cols: List[str] = config["features"]["objective_marker_cols"]
    return pd.read_csv(path)[["subject", "task", "probe_number"] + marker_cols]


def merge_all_features(config: dict) -> pd.DataFrame:
    """Load and left-join probe data, PCA components, and objective markers.

    PCA components are left-joined, so on-task probes (not in pca_results.csv)
    will have NaN for PC1/PC2/PC3. Objective markers span all 2460 probes.
    The returned DataFrame has the same number of rows as the probe data.

    Parameters
    ----------
    config : dict
        Configuration dict loaded from ``Behavior/Classification/config.yaml``.

    Returns
    -------
    pd.DataFrame
        Merged DataFrame with no duplicate column names.
    """
    df = load_probe_data(config)
    merge_keys: List[str] = ["subject", "task", "probe_number"]

    if config["features"]["include_pca"]:
        pca = load_pca_data(config)
        df = df.merge(pca, on=merge_keys, how="left")

    if config["features"]["include_objective_markers"]:
        obj = load_objective_markers(config)
        df = df.merge(obj, on=merge_keys, how="left")

    return df


def get_feature_cols(config: dict) -> List[str]:
    """Return ordered list of feature column names based on config flags.

    Order: probe dimensions → objective markers → PCA components.

    Parameters
    ----------
    config : dict
        Configuration dict loaded from ``Behavior/Classification/config.yaml``.

    Returns
    -------
    List[str]
        Feature column names in the order they should be passed to classifiers.
    """
    cols: List[str] = list(config["features"]["probe_dimensions"])
    if config["features"]["include_objective_markers"]:
        cols += config["features"]["objective_marker_cols"]
    if config["features"]["include_pca"]:
        cols += ["PC1", "PC2", "PC3"]
    return cols
