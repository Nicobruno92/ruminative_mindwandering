"""Shared utilities for behavioral classification pipeline.

Provides data loading, merging, and feature column resolution for the
behavior classification analyses (group classification, inclusion/exclusion
classification).

All paths are resolved relative to the current working directory (repo root),
so scripts must be run from the repository root.
"""

# === Imports ===
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml


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


# === Feature Building ===

def build_subject_level_features(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    positive_class: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate features per subject (mean). Returns X, y, groups.

    Parameters
    ----------
    df : DataFrame with probe-level rows
    feature_cols : column names to use as features
    target_col : column with the classification target (e.g. 'group')
    positive_class : value encoded as 1 (e.g. 'Risk of Depression')

    Returns
    -------
    X : (n_subjects, n_features) float array
    y : (n_subjects,) int binary array
    groups : (n_subjects,) str array of subject IDs
    """
    agg = df.groupby("subject")[feature_cols].mean()
    y_raw = df.groupby("subject")[target_col].first()
    y = (y_raw == positive_class).astype(int).values
    groups = agg.index.astype(str).values
    return agg.values.astype(float), y, groups


def build_probe_level_features(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    positive_class: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return probe-level feature matrix. Label = subject's class (constant within subject).

    Parameters
    ----------
    df : DataFrame with probe-level rows
    feature_cols : column names to use as features
    target_col : column with the classification target (e.g. 'group')
    positive_class : value encoded as 1 (e.g. 'Risk of Depression')

    Returns
    -------
    X : (n_probes, n_features)
    y : (n_probes,) — same label for all probes of the same subject
    groups : (n_probes,) — subject IDs (LOSO fold key)
    """
    df = df.copy()
    subj_labels = df.groupby("subject")[target_col].first()
    df["_label"] = df["subject"].map(subj_labels)
    y = (df["_label"] == positive_class).astype(int).values
    groups = df["subject"].astype(str).values
    return df[feature_cols].values.astype(float), y, groups


def build_block_level_features(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    positive_class: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate features per subject × task block (mean). One row per block.

    Parameters
    ----------
    df : DataFrame already filtered to the relevant conditions (e.g. intervention only)
    feature_cols : column names to use as features
    target_col : column with the classification target (e.g. 'inclusion_exclusion')
    positive_class : value encoded as 1 (e.g. 'inclusion')

    Returns
    -------
    X : (n_blocks, n_features)
    y : (n_blocks,) binary
    groups : (n_blocks,) subject IDs (LOSO fold key)
    """
    agg = df.groupby(["subject", "task"])[feature_cols].mean()
    y_raw = df.groupby(["subject", "task"])[target_col].first()
    y = (y_raw == positive_class).astype(int).values
    groups = agg.index.get_level_values("subject").astype(str).values
    return agg.values.astype(float), y, groups
